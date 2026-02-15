import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import f1_score, accuracy_score

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_COLUMNS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]
NUM_EMOTIONS = len(EMOTION_COLUMNS)

# 1. CLASSES (Model, Loss, Dataset, Metrics)


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sent_preds = []
        self.sent_targets = []
        self.emo_preds = []
        self.emo_targets = []

    def update_sent(self, logits, labels):
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        self.sent_preds.extend(preds)
        self.sent_targets.extend(labels.cpu().numpy())

    def update_emo(self, logits, labels):
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int().cpu().numpy()
        self.emo_preds.extend(preds)
        self.emo_targets.extend(labels.cpu().numpy())

    def calculate(self):
        sent_acc = accuracy_score(self.sent_targets, self.sent_preds)
        sent_f1 = f1_score(self.sent_targets, self.sent_preds, average='binary')
        emo_f1_micro = f1_score(self.emo_targets, self.emo_preds, average='micro')
        return {
            'sent_acc': sent_acc,
            'sent_f1': sent_f1,
            'emo_f1_micro': emo_f1_micro
        }

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask_self = torch.eye(batch_size, dtype=torch.float32).to(device)
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float().to(device) - mask_self
        exp_logits = torch.exp(logits) * (1 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask_pos * log_prob).sum(1) / (mask_pos.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()
class FocalLossMultiLabel(nn.Module):
    # ADD pos_weight to the init
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None):
        super(FocalLossMultiLabel, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # PASS the pos_weight to the internal BCE loss
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MTLSBERT(nn.Module):
    def __init__(self, model_name, num_emotions=28):
        super(MTLSBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.sentiment_head = nn.Linear(self.hidden_size, 2)
        self.emotion_head = nn.Linear(self.hidden_size, num_emotions)
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(),
            nn.Linear(self.hidden_size, 128)
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask, task_type):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs, attention_mask)
        if task_type == 'sentiment':
            return self.sentiment_head(embeddings), F.normalize(self.contrastive_head(embeddings), dim=1)
        elif task_type == 'emotion':
            return self.emotion_head(embeddings), None

class MTLDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, task_type):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task_type = task_type

    def __len__(self): return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = str(row['text'])
        inputs = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        item = {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze(), 'task_type': self.task_type}

        if self.task_type == 'sentiment':
            item['labels'] = torch.tensor(row['label'], dtype=torch.long)
        elif self.task_type == 'emotion':
            label_vector = [float(row[col]) for col in EMOTION_COLUMNS]
            item['labels'] = torch.tensor(label_vector, dtype=torch.float)
        return item

#data prep
def prepare_data():
    print("--- 1. Loading Datasets (SetFit Versions) ---")

    # 1. Load full Emotion dataset
    ds_emo = load_dataset("SetFit/go_emotions", split="train")
    emo_train_size = len(ds_emo) # 43,410

    # 2. Load Sentiment, SHUFFLE, then match the Emotion size perfectly
    ds_sent_full = load_dataset("SetFit/amazon_polarity", split="train")
    ds_sent = ds_sent_full.shuffle(seed=42).select(range(emo_train_size))

    # 3. Clean validation sets (2000 is plenty for sentiment validation)
    val_s_ds = load_dataset("SetFit/amazon_polarity", split="test").shuffle(seed=42).select(range(2000))
    val_e_ds = load_dataset("SetFit/go_emotions", split="test")

    df_sent = ds_sent.to_pandas()
    df_emo = ds_emo.to_pandas()
    df_val_s = val_s_ds.to_pandas()
    df_val_e = val_e_ds.to_pandas()

    if 'text' in df_sent.columns: pass
    elif 'content' in df_sent.columns:
        df_sent = df_sent.rename(columns={'content': 'text'})
        df_val_s = df_val_s.rename(columns={'content': 'text'})

    print("--- 2. Calculating Class Weights ---")
    pos_counts = df_emo[EMOTION_COLUMNS].sum().values
    total_samples = len(df_emo)
    neg_counts = total_samples - pos_counts

    pos_weights = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float)

    print(" Sample Weights (First 5):", pos_weights[:5])
    print(f" Neutral Weight (Last): {pos_weights[-1]:.4f}")

    df_emo_final = df_emo.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f" Final Emotion Dataset Size: {len(df_emo_final)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader_s = DataLoader(MTLDataset(df_sent, tokenizer, MAX_LEN, 'sentiment'), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader_e = DataLoader(MTLDataset(df_emo_final, tokenizer, MAX_LEN, 'emotion'), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    val_loader_s = DataLoader(MTLDataset(df_val_s, tokenizer, MAX_LEN, 'sentiment'), batch_size=BATCH_SIZE, shuffle=False)
    val_loader_e = DataLoader(MTLDataset(df_val_e, tokenizer, MAX_LEN, 'emotion'), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader_s, train_loader_e, val_loader_s, val_loader_e, pos_weights


def training(tr_s, tr_e, val_s, val_e, emotion_pos_weights):
    model = MTLSBERT(MODEL_NAME).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-3)

    # Scheduler remains watching the Loss
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    criterion_s = nn.CrossEntropyLoss()

    # We now pass the emotion_pos_weights here
    criterion_e = FocalLossMultiLabel(
        alpha=0.25,
        gamma=2.0,
        pos_weight=emotion_pos_weights.to(DEVICE)
    ).to(DEVICE)

    criterion_con = SupConLoss()

    tracker = MetricsTracker()
    history = {'train_loss': [], 'val_loss': [], 'sent_f1': [], 'emo_f1': []}
    best_score = 0.0

    print(f"\nStarting Weighted Focal Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        run_loss, steps = 0, 0
        loop = tqdm(zip(tr_s, tr_e), total=min(len(tr_s), len(tr_e)), leave=False)

        for batch_s, batch_e in loop:
            optimizer.zero_grad()

            #Sentiment Task
            s_in, s_mask, s_lbl = batch_s['input_ids'].to(DEVICE), batch_s['attention_mask'].to(DEVICE), batch_s['labels'].to(DEVICE)
            s_logits, s_proj = model(s_in, s_mask, 'sentiment')
            loss_s = criterion_s(s_logits, s_lbl) + 0.5 * criterion_con(s_proj, s_lbl)

            # Emotion Task
            e_in, e_mask, e_lbl = batch_e['input_ids'].to(DEVICE), batch_e['attention_mask'].to(DEVICE), batch_e['labels'].to(DEVICE)
            e_logits, _ = model(e_in, e_mask, 'emotion')
            loss_e = criterion_e(e_logits, e_lbl)

            # Backprop
            loss = loss_s + loss_e
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            steps += 1
            loop.set_postfix(loss=run_loss/steps)

        #validate
        model.eval()
        val_loss = 0
        tracker.reset()
        with torch.no_grad():
            for batch in val_s:
                in_ids, mask, lbl = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
                logits, _ = model(in_ids, mask, 'sentiment')
                val_loss += criterion_s(logits, lbl).item()
                tracker.update_sent(logits, lbl)

            for batch in val_e:
                in_ids, mask, lbl = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['labels'].to(DEVICE)
                logits, _ = model(in_ids, mask, 'emotion')
                val_loss += criterion_e(logits, lbl).item()
                tracker.update_emo(logits, lbl)

        avg_val_loss = val_loss / (len(val_s) + len(val_e))
        metrics = tracker.calculate()

        # Step scheduler based on Loss
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f} | Sent F1: {metrics['sent_f1']:.4f} | Emo F1: {metrics['emo_f1_micro']:.4f} | LR: {current_lr:.6f}")

        # Checkpointing
        current_score = (metrics['sent_f1'] + metrics['emo_f1_micro']) / 2
        if current_score > best_score:
            best_score = current_score
            torch.save(model.state_dict(), "best_mtl_model.pth")
            print(f"  --> Best Score! Model Saved.")
    plot_results(history['train_loss'], history['val_loss'], history)

def plot_results(train_losses, val_losses, metrics_history):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r--', label='Validation Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics_history['sent_f1'], 'g-', label='Sentiment F1')
    plt.plot(epochs, metrics_history['emo_f1'], 'm-', label='Emotion F1')
    plt.title('Evaluation Metrics')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 1. Prepare Data with Weights
    tr_s, tr_e, val_s, val_e, weights = prepare_data()

    # 2. Run Training passing weights
    training(tr_s, tr_e, val_s, val_e, weights)