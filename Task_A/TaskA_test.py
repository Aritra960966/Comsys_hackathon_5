import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import csv
import os
import argparse

from config import DEVICE, BATCH_SIZE, TEST_DIR
from dataset import create_test_dataloader
from model import load_model


def test_model(model, test_loader, save_predictions=False, output_file="test_results.csv"):
    """Test the model and return detailed metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    correct = 0
    total = 0
    
    print("Running inference on test set...")
    loop = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for gray, rgb, labels in loop:
            gray, rgb, labels = gray.to(DEVICE), rgb.to(DEVICE), labels.to(DEVICE)
            
            # Get model predictions
            outputs, _, _ = model(gray, rgb)
            
            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Update accuracy
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(acc=correct/total if total > 0 else 0)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    # Calculate per-class metrics
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\n=== TASK A TEST RESULTS ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    
    print(f"\n=== BINARY CLASSIFICATION METRICS ===")
    print(f"Precision (Binary): {precision:.4f}")
    print(f"Recall (Binary):    {recall:.4f}")
    print(f"F1-Score (Binary):  {f1:.4f}")
    
    print(f"\n=== MACRO AVERAGE METRICS ===")
    print(f"Precision (Macro): {precision_macro:.4f}")
    print(f"Recall (Macro):    {recall_macro:.4f}")
    print(f"F1-Score (Macro):  {f1_macro:.4f}")
    
    print(f"\n=== WEIGHTED AVERAGE METRICS ===")
    print(f"Precision (Weighted): {precision_weighted:.4f}")
    print(f"Recall (Weighted):    {recall_weighted:.4f}")
    print(f"F1-Score (Weighted):  {f1_weighted:.4f}")
    
    # Detailed classification report
    print(f"\n=== CLASSIFICATION REPORT ===")
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1']))
    
    
    # Save predictions if requested
    if save_predictions:
        print(f"\nSaving predictions to {output_file}...")
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sample_ID', 'True_Label', 'Predicted_Label', 'Confidence_Class_0', 'Confidence_Class_1'])
            
            for i, (true_label, pred_label, prob) in enumerate(zip(all_labels, all_preds, all_probs)):
                writer.writerow([i, true_label, pred_label, prob[0], prob[1]])
        
        print(f"Predictions saved to {output_file}")
    
    # Save summary metrics to CSV
    summary_file = output_file.replace('.csv', '_summary.csv')
    print(f"\nSaving summary metrics to {summary_file}...")
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Accuracy', f'{accuracy:.4f}'])
        writer.writerow(['Precision (Binary)', f'{precision:.4f}'])
        writer.writerow(['Recall (Binary)', f'{recall:.4f}'])
        writer.writerow(['F1-Score (Binary)', f'{f1:.4f}'])
        writer.writerow(['Precision (Macro)', f'{precision_macro:.4f}'])
        writer.writerow(['Recall (Macro)', f'{recall_macro:.4f}'])
        writer.writerow(['F1-Score (Macro)', f'{f1_macro:.4f}'])
        writer.writerow(['Precision (Weighted)', f'{precision_weighted:.4f}'])
        writer.writerow(['Recall (Weighted)', f'{recall_weighted:.4f}'])
        writer.writerow(['F1-Score (Weighted)', f'{f1_weighted:.4f}'])
        writer.writerow(['Total Samples', total])
        writer.writerow(['Correct Predictions', correct])
    
    print(f" Summary metrics saved to {summary_file}")
    
    return {
        'accuracy': accuracy,
        'precision_binary': precision,
        'recall_binary': recall,
        'f1_binary': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def main():
    parser = argparse.ArgumentParser(description='Test Task A AMR-CD Model')
    parser.add_argument('--model_path', type=str, default='TASK_A_MODEL.pt',
                        help='Path to the trained Task A model')
    parser.add_argument('--test_dir', type=str, default=TEST_DIR,
                        help='Path to test dataset directory')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for testing')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predictions to CSV file')
    parser.add_argument('--output_file', type=str, default='TaskA_test_results.csv',
                        help='Output file for predictions')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        print("Available model files:")
        for file in os.listdir('.'):
            if file.endswith('.pth') or file.endswith('.pt'):
                print(f"  - {file}")
        return
    
    # Check if test directory exists
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory '{args.test_dir}' not found.")
        return
    
    print(f"=== TASK A MODEL TESTING ===")
    print(f"Loading model from: {args.model_path}")
    print(f"Test directory: {args.test_dir}")
    print(f"Device: {DEVICE}")
    
    # Load model
    try:
        model = load_model(args.model_path, DEVICE)
        print(" Task A model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create test dataloader
    try:
        test_loader = create_test_dataloader(args.test_dir, args.batch_size)
        print(f"Test dataset loaded: {len(test_loader.dataset)} samples")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return
    
    # Run testing
    results = test_model(
        model=model,
        test_loader=test_loader,
        save_predictions=args.save_predictions,
        output_file=args.output_file
    )
    
    print("\n=== TASK A TESTING COMPLETED ===")


if __name__ == "__main__":
    main()