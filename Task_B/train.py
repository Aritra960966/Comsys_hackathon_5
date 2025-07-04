"""
Training module for TaskB Face Recognition
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from model import HybridLoss
from utils import evaluate_model


def train_model(model, train_loader, val_loader, optimizer, device, config):
    criterion = HybridLoss(margin=config['MARGIN'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_sim_f1 = 0
    best_sim_macro_f1 = 0
    best_dist_f1 = 0
    best_dist_macro_f1 = 0
    best_val_loss = float('inf')

    early_stop_counter = 0
    metrics_history = []

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, config['EPOCHS'] + 1):
        model.train()
        total_loss = 0
        num_batches = 0

        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['EPOCHS']}")
        print(f"{'='*50}")

        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, (anchor, comp, label) in enumerate(progress_bar):
            anchor, comp, label = anchor.to(device), comp.to(device), label.to(device)

            emb1, emb2 = model(anchor), model(comp)
            loss = criterion(emb1, emb2, label)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step(epoch + batch_idx / len(train_loader))

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/num_batches:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_loader)

        val_metrics = evaluate_model(model, val_loader, device, config['THRESHOLD'])

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for anchor, comp, label in val_loader:
                anchor, comp, label = anchor.to(device), comp.to(device), label.to(device)
                emb1, emb2 = model(anchor), model(comp)
                loss = criterion(emb1, emb2, label)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        torch.cuda.empty_cache()

        print(f"\nEpoch {epoch} Results:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Distance-based - Acc: {val_metrics['dist_acc']:.4f}, F1: {val_metrics['dist_f1']:.4f}, Macro F1: {val_metrics['dist_macro_f1']:.4f}, AUC: {val_metrics['dist_auc']:.4f}")
        print(f"Similarity-based - Acc: {val_metrics['sim_acc']:.4f}, F1: {val_metrics['sim_f1']:.4f}, Macro F1: {val_metrics['sim_macro_f1']:.4f}, AUC: {val_metrics['sim_auc']:.4f}")

        metrics = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        metrics_history.append(metrics)

        df = pd.DataFrame([metrics])
        if epoch == 1:
            df.to_csv(config['CSV_LOG_PATH'], index=False)
        else:
            df.to_csv(config['CSV_LOG_PATH'], mode='a', header=False, index=False)

        sim_improved = False
        dist_improved = False
        val_loss_improved = False

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            val_loss_improved = True

        if val_metrics['sim_f1'] > best_sim_f1:
            best_sim_f1 = val_metrics['sim_f1']
            sim_improved = True

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_sim_f1': best_sim_f1,
                'best_sim_macro_f1': val_metrics['sim_macro_f1'],
                'metrics': val_metrics,
                'config': config,
                'model_type': 'similarity'
            }, config['SIMILARITY_MODEL_PATH'])

            print(f"New best similarity model saved with F1: {best_sim_f1:.4f}, Macro F1: {val_metrics['sim_macro_f1']:.4f}")

        if val_metrics['dist_f1'] > best_dist_f1:
            best_dist_f1 = val_metrics['dist_f1']
            dist_improved = True

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dist_f1': best_dist_f1,
                'best_dist_macro_f1': val_metrics['dist_macro_f1'],
                'metrics': val_metrics,
                'config': config,
                'model_type': 'distance'
            }, config['DISTANCE_MODEL_PATH'])

            print(f"New best distance model saved with F1: {best_dist_f1:.4f}, Macro F1: {val_metrics['dist_macro_f1']:.4f}")

        if val_metrics['sim_macro_f1'] > best_sim_macro_f1:
            best_sim_macro_f1 = val_metrics['sim_macro_f1']

        if val_metrics['dist_macro_f1'] > best_dist_macro_f1:
            best_dist_macro_f1 = val_metrics['dist_macro_f1']

        current_best_f1 = max(val_metrics['sim_f1'], val_metrics['dist_f1'])
        overall_best_f1 = max(best_sim_f1, best_dist_f1)

        if current_best_f1 >= overall_best_f1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_sim_f1': best_sim_f1,
                'best_dist_f1': best_dist_f1,
                'best_sim_macro_f1': best_sim_macro_f1,
                'best_dist_macro_f1': best_dist_macro_f1,
                'metrics': val_metrics,
                'config': config,
                'model_type': 'overall_best'
            }, config['MODEL_PATH'])

        if sim_improved or dist_improved or val_loss_improved:
            early_stop_counter = 0
        else:
            early_stop_counter += 1

            if epoch % 5 == 0:
                ckpt_path = f"checkpoints/epoch_{epoch:03d}_simF1_{val_metrics['sim_f1']:.4f}_distF1_{val_metrics['dist_f1']:.4f}_valLoss_{avg_val_loss:.4f}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                    'config': config
                }, ckpt_path)

            if early_stop_counter >= config['PATIENCE']:
                print(f"Early stopping triggered after {config['PATIENCE']} epochs without improvement in F1 scores or validation loss")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

    # Save final model
    final_path = config['MODEL_PATH'].replace('.pth', '_final.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_metrics': val_metrics,
        'best_sim_f1': best_sim_f1,
        'best_dist_f1': best_dist_f1,
        'best_sim_macro_f1': best_sim_macro_f1,
        'best_dist_macro_f1': best_dist_macro_f1,
        'metrics_history': metrics_history,
        'config': config
    }, final_path)

    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Best Similarity F1: {best_sim_f1:.4f}")
    print(f"Best Distance F1: {best_dist_f1:.4f}")
    print(f"Best Similarity Macro F1: {best_sim_macro_f1:.4f}")
    print(f"Best Distance Macro F1: {best_dist_macro_f1:.4f}")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*50}")

    return {
        'best_sim_f1': best_sim_f1,
        'best_dist_f1': best_dist_f1,
        'best_sim_macro_f1': best_sim_macro_f1,
        'best_dist_macro_f1': best_dist_macro_f1,
        'metrics_history': metrics_history
    }


if __name__ == "__main__":
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from config import CONFIG
    from dataset import FacePairDataset, train_transform, val_transform
    from model import FaceEmbedder
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = FacePairDataset(CONFIG['TRAIN_DIR'], train_transform, is_training=True)
    val_dataset = FacePairDataset(CONFIG['VAL_DIR'], val_transform, is_training=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    model = FaceEmbedder(
        embedding_dim=CONFIG['EMBEDDING_DIM'],
        dropout_rate=CONFIG['DROPOUT_RATE']
    ).to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['LEARNING_RATE'],
        weight_decay=CONFIG['WEIGHT_DECAY']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train model
    results = train_model(model, train_loader, val_loader, optimizer, device, CONFIG)
    
    print("Training process completed successfully!")