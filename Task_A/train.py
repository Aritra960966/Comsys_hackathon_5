import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import csv
import os

from config import DEVICE, AUX_WEIGHT


def train_one_epoch(model, loader, criterion, optimizer, aux_weight=AUX_WEIGHT):
    """Train the model for one epoch"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    loop = tqdm(loader, desc="Training", leave=False)
    for gray, rgb, labels in loop:
        gray, rgb, labels = gray.to(DEVICE), rgb.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        out, logits_gray, logits_rgb = model(gray, rgb)

        # Combined loss: main + auxiliary losses
        loss_main = criterion(out, labels)
        loss_aux1 = criterion(logits_gray, labels)
        loss_aux2 = criterion(logits_rgb, labels)
        loss = loss_main + aux_weight * (loss_aux1 + loss_aux2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=correct / total if total else 0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, aux_weight=AUX_WEIGHT):
    """Evaluate the model"""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    loop = tqdm(loader, desc="Validating", leave=False)
    with torch.no_grad():
        for gray, rgb, labels in loop:
            gray, rgb, labels = gray.to(DEVICE), rgb.to(DEVICE), labels.to(DEVICE)

            out, logits_gray, logits_rgb = model(gray, rgb)

            # Combined loss: main + auxiliary losses
            loss_main = criterion(out, labels)
            loss_aux1 = criterion(logits_gray, labels)
            loss_aux2 = criterion(logits_rgb, labels)
            loss = loss_main + aux_weight * (loss_aux1 + loss_aux2)

            total_loss += loss.item() * labels.size(0)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=loss.item(), acc=correct / total if total else 0)

    avg_loss = total_loss / total
    avg_acc = correct / total
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    return avg_loss, avg_acc, precision, recall, f1


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, 
                patience=4, results_file="results.csv"):
    """Main training loop with early stopping"""
    
    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize CSV file for results
    with open(results_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'Precision', 'Recall', 'F1'])
    
    # Early stopping variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train and validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, precision, recall, f1 = evaluate(model, val_loader, criterion)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")

        # Save results to CSV
        with open(results_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, precision, recall, f1])

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "checkpoint_best.pth")
            print(f"✓ New best model saved: checkpoint_best.pth")
        else:
            early_stop_counter += 1
            torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pth")
            print(f"  Early stop counter: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print("⚠ Early stopping triggered.")
            break
    
    # Save final model
    torch.save(model.state_dict(), "TASK_A_MODEL.pth")
    print(" Final model saved as: TASK_A_MODEL.pth")
    
    return model