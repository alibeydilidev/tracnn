import sys
import os
os.makedirs("outputs", exist_ok=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch.optim import Adam
from models.tracnn import TraCNN
from data.utils import get_dataloaders
import yaml

with open("training/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_dataloaders()
model = TraCNN(
    input_channels=config["input_channels"],
    trans_dim=config["trans_dim"],
    fused_dim=config["fused_dim"],
    num_classes=config["num_classes"]
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        img = batch["image"].to(device)
        seq = batch["sequence"].to(device)
        labels = batch["label"].to(device)

        outputs = model(img, seq)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            seq = batch["sequence"].to(device)
            labels = batch["label"].to(device)

            outputs = model(img, seq)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, acc

def run_training():
    best_val_loss = float("inf")
    patience = 5
    counter = 0

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")
        print("-" * 50)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), config["model_save_path"].replace(".pt", "_best.pt"))
            print(f"Best model saved to {config['model_save_path'].replace('.pt', '_best.pt')} at epoch {epoch}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

if __name__ == "__main__":
    run_training()

    # Save the model weights after training
    torch.save(model.state_dict(), config["model_save_path"])
    print(f"Model saved to {config['model_save_path']}")

    print("\n--- Final Test Evaluation ---")
    # Using validation loader as test set for now
    test_loss, test_acc = validate(model, val_loader, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")