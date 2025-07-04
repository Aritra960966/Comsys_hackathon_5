"""
Utilities module for TaskB Face Recognition
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from tqdm import tqdm


def mixup_pairs(x1, x2, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size).to(x1.device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]
    y_a, y_b = y, y[index]

    return mixed_x1, mixed_x2, y_a, y_b, lam


def mixup_loss(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@torch.no_grad()
def optimize_thresholds(model, dataloader, device, threshold_range=(0.1, 0.9), steps=50, metric='f1'):
    model.eval()
    all_embeddings1, all_embeddings2, labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            anchor, comp, batch_labels = [x.to(device) for x in batch]
            emb1 = model(anchor)
            emb2 = model(comp)
            all_embeddings1.append(emb1)
            all_embeddings2.append(emb2)
            labels.append(batch_labels)

    all_embeddings1 = torch.cat(all_embeddings1)
    all_embeddings2 = torch.cat(all_embeddings2)
    labels = torch.cat(labels).cpu().numpy()

    similarities = F.cosine_similarity(all_embeddings1, all_embeddings2).cpu().numpy()
    distances = F.pairwise_distance(all_embeddings1, all_embeddings2).cpu().numpy()

    thresholds = np.linspace(threshold_range[0], threshold_range[1], steps)

    best_sim_threshold = threshold_range[0]
    best_sim_score = 0
    best_sim_metrics = {}

    best_dist_threshold = threshold_range[0]
    best_dist_score = 0
    best_dist_metrics = {}

    sim_threshold_results = []
    dist_threshold_results = []

    print("Optimizing similarity threshold...")
    for threshold in tqdm(thresholds, desc="Similarity thresholds"):
        sim_preds = (similarities > threshold).astype(int)

        acc = accuracy_score(labels, sim_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, sim_preds, average='binary', zero_division=0)
        _, _, macro_f1, _ = precision_recall_fscore_support(labels, sim_preds, average='macro', zero_division=0)

        metrics = {
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'macro_f1': macro_f1
        }
        sim_threshold_results.append(metrics)

        if metric == 'f1':
            score = f1
        elif metric == 'macro_f1':
            score = macro_f1
        elif metric == 'accuracy':
            score = acc
        else:
            score = f1

        if score > best_sim_score:
            best_sim_score = score
            best_sim_threshold = threshold
            best_sim_metrics = metrics.copy()

    print("Optimizing distance threshold...")
    for threshold in tqdm(thresholds, desc="Distance thresholds"):
        dist_preds = (distances < threshold).astype(int)

        acc = accuracy_score(labels, dist_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, dist_preds, average='binary', zero_division=0)
        _, _, macro_f1, _ = precision_recall_fscore_support(labels, dist_preds, average='macro', zero_division=0)

        metrics = {
            'threshold': threshold,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'macro_f1': macro_f1
        }
        dist_threshold_results.append(metrics)

        if metric == 'f1':
            score = f1
        elif metric == 'macro_f1':
            score = macro_f1
        elif metric == 'accuracy':
            score = acc
        else:
            score = f1

        if score > best_dist_score:
            best_dist_score = score
            best_dist_threshold = threshold
            best_dist_metrics = metrics.copy()

    return {
        'similarity': {
            'optimal_threshold': best_sim_threshold,
            'best_metrics': best_sim_metrics,
            'all_results': sim_threshold_results
        },
        'distance': {
            'optimal_threshold': best_dist_threshold,
            'best_metrics': best_dist_metrics,
            'all_results': dist_threshold_results
        }
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device, sim_threshold=0.5, dist_threshold=0.5):
    model.eval()
    all_embeddings1, all_embeddings2, labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            anchor, comp, batch_labels = [x.to(device) for x in batch]

            emb1 = model(anchor)
            emb2 = model(comp)

            all_embeddings1.append(emb1)
            all_embeddings2.append(emb2)
            labels.append(batch_labels)

    all_embeddings1 = torch.cat(all_embeddings1)
    all_embeddings2 = torch.cat(all_embeddings2)
    labels = torch.cat(labels)

    similarities = F.cosine_similarity(all_embeddings1, all_embeddings2)
    distances = F.pairwise_distance(all_embeddings1, all_embeddings2)

    sim_preds = (similarities > sim_threshold).long()
    dist_preds = (distances < dist_threshold).long()

    # Distance-based metrics
    dist_acc = accuracy_score(labels.cpu(), dist_preds.cpu())
    dist_prec, dist_rec, dist_f1, _ = precision_recall_fscore_support(
        labels.cpu(), dist_preds.cpu(), average='binary', zero_division=0
    )
    _, _, dist_macro_f1, _ = precision_recall_fscore_support(
        labels.cpu(), dist_preds.cpu(), average='macro', zero_division=0
    )

    # Similarity-based metrics
    sim_acc = accuracy_score(labels.cpu(), sim_preds.cpu())
    sim_prec, sim_rec, sim_f1, _ = precision_recall_fscore_support(
        labels.cpu(), sim_preds.cpu(), average='binary', zero_division=0
    )
    _, _, sim_macro_f1, _ = precision_recall_fscore_support(
        labels.cpu(), sim_preds.cpu(), average='macro', zero_division=0
    )

    try:
        dist_auc = roc_auc_score(labels.cpu(), -distances.cpu())
        sim_auc = roc_auc_score(labels.cpu(), similarities.cpu())
    except ValueError:
        dist_auc = 0.5
        sim_auc = 0.5

    return {
        'dist_acc': dist_acc, 'dist_prec': dist_prec, 'dist_rec': dist_rec,
        'dist_f1': dist_f1, 'dist_macro_f1': dist_macro_f1, 'dist_auc': dist_auc,
        'sim_acc': sim_acc, 'sim_prec': sim_prec, 'sim_rec': sim_rec,
        'sim_f1': sim_f1, 'sim_macro_f1': sim_macro_f1, 'sim_auc': sim_auc,
        'sim_threshold': sim_threshold, 'dist_threshold': dist_threshold
    }