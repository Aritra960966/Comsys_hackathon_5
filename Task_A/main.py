import os
import argparse

from config import *
from dataset import create_dataloaders
from model import create_model
from train import train_model


def main():
    parser = argparse.ArgumentParser(description='Train AMR-CD Model')
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR,
                        help='Path to training dataset directory')
    parser.add_argument('--val_dir', type=str, default=VAL_DIR,
                        help='Path to validation dataset directory')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for optimizer')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                        help='Early stopping patience')
    parser.add_argument('--results_file', type=str, default=RESULTS_FILE,
                        help='CSV file to save training results')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory '{args.train_dir}' not found.")
        return
    
    if not os.path.exists(args.val_dir):
        print(f"Error: Validation directory '{args.val_dir}' not found.")
        return
    
    print("=== AMR-CD Model Training ===")
    print(f"Device: {DEVICE}")
    print(f"Training directory: {args.train_dir}")
    print(f"Validation directory: {args.val_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Early stopping patience: {args.patience}")
    
    # Create data loaders
    print("\nCreating data loaders...")
    try:
        train_loader, val_loader = create_dataloaders(
            args.train_dir, 
            args.val_dir, 
            args.batch_size
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    # Create model
    print("\nCreating model...")
    try:
        model = create_model(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f" Model created successfully")
        print(f" Total parameters: {total_params:,}")
        print(f" Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"Error creating model: {e}")
        return
    
    # Start training
    print("\n" + "="*50)
    try:
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            patience=args.patience,
            results_file=args.results_file
        )
        print("\n" + "="*50)
        print(" Training completed successfully!")
        print(f" Results saved to: {args.results_file}")
        print(f" Best model saved to: checkpoint_best.pth")
        print(f" Final model saved to: TASK_A_MODEL.pth")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return


if __name__ == "__main__":
    main()