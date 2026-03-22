import os
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

class PairwiseRankingDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]

        text = (
            f"[QUESTION] {item['question']} "
            f"[CANDIDATE_A] {item['candidate_a']} "
            f"[CANDIDATE_B] {item['candidate_b']}"
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def split_train_dev(records, dev_ratio=0.1, seed=42):
    random.seed(seed)
    records = records[:]
    random.shuffle(records)

    n_dev = int(len(records) * dev_ratio)
    dev = records[:n_dev]
    train = records[n_dev:]
    return train, dev


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            total_loss += loss.item() * labels.size(0)

    acc = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return avg_loss, acc


def main():
    PAIRWISE_TRAIN_FILE = r"./distilled_reranker_data/pairwise_train.jsonl"
    OUTPUT_DIR = r"./distilled_reranker_ckpt"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 384
    BATCH_SIZE = 8
    EPOCHS = 3
    LR = 2e-5
    DEV_RATIO = 0.1
    SEED = 42

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("📦 加载 pairwise 数据...")
    records = load_jsonl(PAIRWISE_TRAIN_FILE)
    print(f"总样本数: {len(records)}")

    train_records, dev_records = split_train_dev(records, dev_ratio=DEV_RATIO, seed=SEED)
    print(f"训练集: {len(train_records)} | 验证集: {len(dev_records)}")

    print("📦 加载 tokenizer / model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    train_dataset = PairwiseRankingDataset(train_records, tokenizer, max_length=MAX_LENGTH)
    dev_dataset = PairwiseRankingDataset(dev_records, tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    best_dev_acc = -1.0

    for epoch in range(EPOCHS):
        print(f"\n========== Epoch {epoch + 1}/{EPOCHS} ==========")
        model.train()

        running_loss = 0.0
        total = 0
        correct = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = torch.argmax(logits, dim=-1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            running_loss += loss.item() * labels.size(0)

            if (step + 1) % 100 == 0:
                print(f"step {step + 1}/{len(train_loader)}")

        train_loss = running_loss / total if total > 0 else 0.0
        train_acc = correct / total if total > 0 else 0.0

        dev_loss, dev_acc = evaluate(model, dev_loader, device)

        print(f"Train loss={train_loss:.4f} | Train acc={train_acc:.4f}")
        print(f"Dev   loss={dev_loss:.4f} | Dev   acc={dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print(f"✅ 保存最佳模型, dev_acc={dev_acc:.4f}")
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "base_model": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "best_dev_acc": best_dev_acc
        }, f, indent=2, ensure_ascii=False)

    print("\n🎉 训练完成")
    print(f"最佳模型保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()