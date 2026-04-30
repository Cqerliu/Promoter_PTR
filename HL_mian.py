import os
import random
import copy
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
import numpy as np
from encode import PromoterEncoder, ProEncoder
from downTCPPR_module import TransformerFeature, MLP, CNNFeature, FeatureFusion
import pandas as pd
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from data_split import create_promoter_subsets_with_replacement
from visualization import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

warnings.filterwarnings('ignore')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class PrototypeLoss(nn.Module):
    def __init__(self, lambda_align=0.15):
        super().__init__()
        self.focal = FocalLoss(gamma=2)
        self.align_loss = nn.CosineEmbeddingLoss(margin=0.0)
        self.lambda_align = lambda_align

    def forward(self, logits, dna_feat, rnap_feat, targets):
        cls_loss = self.focal(logits, targets)
        align_targets = targets.float() * 2 - 1

        if dna_feat.dim() > 2: dna_feat = dna_feat.max(dim=1)[0]
        if rnap_feat.dim() > 2: rnap_feat = rnap_feat.max(dim=1)[0]

        feat_loss = self.align_loss(dna_feat, rnap_feat, align_targets)
        return cls_loss + self.lambda_align * feat_loss, cls_loss, feat_loss


class CustomDataset(Dataset):
    def __init__(self, x, fc, y):
        self.x = x
        self.fc = fc
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.fc[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)


def clear_memory(model_list=None):
    if model_list is not None:
        for model in model_list:
            if model is not None:
                del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def deduplicate_samples(promoter_seqs, rnap_seqs, labels, predictions, probabilities=None):
    valid_indices = []
    counts = {}
    valid_count = 0

    for i, p_seq in enumerate(promoter_seqs):
        if p_seq is not None and str(p_seq).strip() != "":
            valid_indices.append(i)
            valid_count += 1
            counts[p_seq] = counts.get(p_seq, 0) + 1

    seen_promoters = set()
    unique_indices = []

    for i in valid_indices:
        p_seq = promoter_seqs[i]
        if p_seq not in seen_promoters:
            seen_promoters.add(p_seq)
            unique_indices.append(i)

    dedup_promoter = [promoter_seqs[i] for i in unique_indices]
    dedup_rnap = [rnap_seqs[i] for i in unique_indices]
    dedup_labels = [labels[i] for i in unique_indices]
    dedup_preds = [predictions[i] for i in unique_indices]
    dedup_probs = [probabilities[i] for i in unique_indices] if probabilities is not None else None

    dedup_counts = [counts[p_seq] for p in dedup_promoter]
    dedup_proportions = [c / valid_count for c in dedup_counts]

    return {
        'promoter': dedup_promoter,
        'rnap': dedup_rnap,
        'labels': dedup_labels,
        'predictions': dedup_preds,
        'probabilities': dedup_probs,
        'counts': dedup_counts,
        'proportions': dedup_proportions,
        'original_count': len(promoter_seqs),
        'valid_count': valid_count,
        'unique_count': len(unique_indices)
    }


def evaluate_model(dataloader, transformer, cnn, fusion, mlp, device):
    transformer.eval()
    cnn.eval()
    fusion.eval()
    mlp.eval()

    all_predicted = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_X, batch_fc, batch_y in dataloader:
            batch_X, batch_fc, batch_y = batch_X.to(device), batch_fc.to(device), batch_y.to(device)

            p_feat = transformer(batch_X)
            r_feat = cnn(batch_fc)
            f_feat, _, _ = fusion(p_feat, r_feat)
            logits = mlp(f_feat)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            predicted = torch.argmax(logits, dim=1).cpu().numpy()

            all_predicted.extend(predicted)
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_predicted), np.array(all_probs)

def find_optimal_threshold(y_true, y_prob, thresholds=np.arange(0.1, 0.9, 0.01)):
    best_thresh = 0.5
    best_acc = 0.0
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = th
    return best_thresh
# ==================================================================


def calculate_metrics_dict(labels, preds, probs):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else 0.0
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}


def process_subset(subset_idx, promoter_subset, rnap_subset, device, base_seed=41):
    print(f"\n===== Process promoter subset {subset_idx + 1} =====")
    promoter_sequences = promoter_subset['sequence'].values
    y = promoter_subset['label'].values.astype(np.int64)
    subset_size = len(y)
    rnap_sequences = rnap_subset['sequence'].values

    shuffled_indices = np.random.permutation(subset_size)
    promoter_sequences, rnap_sequences, y = promoter_sequences[shuffled_indices], rnap_sequences[shuffled_indices], y[shuffled_indices]

    cv_indices, test_indices = train_test_split(
        np.arange(subset_size), test_size=0.2, stratify=y, random_state=base_seed
    )

    test_promoter = promoter_sequences[test_indices]
    test_rnap = rnap_sequences[test_indices]
    test_y = y[test_indices]

    cv_promoter = promoter_sequences[cv_indices]
    cv_rnap = rnap_sequences[cv_indices]
    cv_y = y[cv_indices]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=base_seed)

    val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

    global_best_cv_val_acc = 0.0
    best_model_states = None
    best_x_test = None
    best_fc_test = None

    fold_opt_thresholds = []
    all_fold_kmers = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(cv_promoter, cv_y)):
        print(f"\n----- Fold  {fold + 1}  -----")

        train_promoter = cv_promoter[train_idx]
        val_promoter = cv_promoter[val_idx]
        train_rnap = cv_rnap[train_idx]
        val_rnap = cv_rnap[val_idx]
        train_y = cv_y[train_idx]
        val_y = cv_y[val_idx]

        encoder = PromoterEncoder(kmer=5, vector_size=100)
        x_train, _ = encoder.run_pipeline(sequences=train_promoter, labels=train_y)
        x_train = x_train.astype(np.float32)

        def encode_data(seqs, labels, base_encoder):
            enc = PromoterEncoder(kmer=5, vector_size=100)
            enc.word_vectors = base_encoder.word_vectors
            enc.sequence_length = base_encoder.sequence_length
            enc.label_encoder = base_encoder.label_encoder
            enc.set_sequences_labels(sequences=seqs, labels=labels)
            enc.generate_kmers()
            x_data, _ = enc.get_transformer_input()
            return x_data.astype(np.float32)

        x_val = encode_data(val_promoter, val_y, encoder)
        x_test_current = encode_data(test_promoter, test_y, encoder)

        fc_encode = ProEncoder(VECTOR_REPETITION_CNN=x_train.shape[1])
        fc_train = np.array([fc_encode.encode_conjoint_cnn(s).squeeze(0).cpu().numpy() for s in train_rnap], dtype=np.float32)
        fc_val = np.array([fc_encode.encode_conjoint_cnn(s).squeeze(0).cpu().numpy() for s in val_rnap], dtype=np.float32)
        fc_test_current = np.array([fc_encode.encode_conjoint_cnn(s).squeeze(0).cpu().numpy() for s in test_rnap], dtype=np.float32)

        train_dataloader = DataLoader(CustomDataset(x_train, fc_train, train_y), batch_size=64, shuffle=True)
        val_dataloader = DataLoader(CustomDataset(x_val, fc_val, val_y), batch_size=64, shuffle=False)

        transformer = TransformerFeature(input_dim=x_train.shape[2], max_seq_len=x_train.shape[1], depth=4, heads=5, dim_head=16, use_projection=True).to(device)
        cnn = CNNFeature(input_length=fc_train.shape[2], input_channels=fc_train.shape[1], feature_dim=x_train.shape[2], conv_filters=[256, 128], conv_kernels=[5, 7], pool_sizes=[2, 2]).to(device)
        fusion = FeatureFusion(feature_dim=x_train.shape[2], heads=4, dim_head=16).to(device)
        mlp = MLP(input_dim=x_train.shape[2], num_classes=2).to(device)
        mlp_dna_only = MLP(input_dim=x_train.shape[2], num_classes=2).to(device)

        criterion = PrototypeLoss(lambda_align=0.1).to(device)
        optimizer = torch.optim.AdamW(
            list(transformer.parameters()) + list(cnn.parameters()) + list(fusion.parameters()) + list(mlp.parameters()) + list(mlp_dna_only.parameters()),
            lr=1e-4, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        num_epochs = 50
        best_fold_val_acc = 0.0
        best_fold_states = None

        for epoch in range(num_epochs):
            transformer.train()
            cnn.train()
            fusion.train()
            mlp.train()
            mlp_dna_only.train()

            running_loss = 0.0
            all_train_preds = []
            all_train_labels = []

            for batch_X, batch_fc, batch_y in train_dataloader:
                batch_X, batch_fc, batch_y = batch_X.to(device), batch_fc.to(device), batch_y.to(device)
                optimizer.zero_grad()

                p_feat = transformer(batch_X)
                r_feat = cnn(batch_fc)
                r_feat_f = torch.zeros_like(r_feat) if random.random() < 0.3 else r_feat
                f_feat, _, _ = fusion(p_feat, r_feat_f)

                logits = mlp(f_feat)
                loss_fusion, _, _ = criterion(logits, p_feat, r_feat, batch_y)

                logits_dna = mlp_dna_only(p_feat.max(dim=1)[0])
                loss_dna = F.cross_entropy(logits_dna, batch_y)

                total_loss = loss_fusion + 0.6 * loss_dna
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += total_loss.item()
                all_train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_train_labels.extend(batch_y.cpu().numpy())
                scheduler.step()

            train_loss = running_loss / len(train_dataloader)
            train_acc = accuracy_score(all_train_labels, all_train_preds)
            val_labels, val_preds, _ = evaluate_model(val_dataloader, transformer, cnn, fusion, mlp, device)
            val_acc = accuracy_score(val_labels, val_preds)

            status = " "
            if val_acc > best_fold_val_acc:
                best_fold_val_acc = val_acc
                best_fold_states = {
                    'transformer': copy.deepcopy(transformer.state_dict()),
                    'cnn': copy.deepcopy(cnn.state_dict()),
                    'fusion': copy.deepcopy(fusion.state_dict()),
                    'mlp': copy.deepcopy(mlp.state_dict())
                }
                status = " [Best Model Updated!]"

            print(f"Epoch [{epoch+1:02d}/{num_epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}{status}")

        transformer.load_state_dict(best_fold_states['transformer'])
        cnn.load_state_dict(best_fold_states['cnn'])
        fusion.load_state_dict(best_fold_states['fusion'])
        mlp.load_state_dict(best_fold_states['mlp'])

        val_lbls, val_prds, val_prbs = evaluate_model(val_dataloader, transformer, cnn, fusion, mlp, device)
        opt_th = find_optimal_threshold(val_lbls, val_prbs)
        fold_opt_thresholds.append(opt_th)  

        val_prds_opt = (val_prbs >= opt_th).astype(int)
        v_res = calculate_metrics_dict(val_lbls, val_prds_opt, val_prbs)

        for k in val_metrics:
            val_metrics[k].append(v_res[k])

        print(f"  [Optimal threshold of current fold={opt_th:.3f}] Acc: {v_res['accuracy']:.3f}, Pre: {v_res['precision']:.3f}, Rec: {v_res['recall']:.3f}, F1: {v_res['f1']:.3f}, AUC: {v_res['auc']:.3f}")

        if v_res['accuracy'] > global_best_cv_val_acc:
            global_best_cv_val_acc = v_res['accuracy']
            best_model_states = copy.deepcopy(best_fold_states)
            best_x_test = x_test_current.copy()
            best_fc_test = fc_test_current.copy()

        print(f"  Extract k-mer importance of the {fold+1}-th fold")
        fold_kmer = get_kmer_importance(
            encoder=encoder,
            model={'transformer': transformer, 'cnn': cnn, 'fusion': fusion, 'mlp': mlp},
            x_data=x_test_current,
            fc_data=fc_test_current,
            device=device,
            top_k=50,
            promoter_sequences=test_promoter
        )
        if fold_kmer is not None:
            all_fold_kmers.extend(fold_kmer)

        clear_memory([transformer, cnn, fusion, mlp, mlp_dna_only])
        del x_train, x_val, x_test_current, fc_train, fc_val, fc_test_current, train_dataloader, val_dataloader
        clear_memory()

    best_threshold = np.mean(fold_opt_thresholds)
    print(f"----- Final average threshold={best_threshold:.3f} -----")
    # ===========================================================================

    print(f"\n----- Test with global optimal model + optimal threshold = {best_threshold:.3f} -----")

    transformer = TransformerFeature(input_dim=best_x_test.shape[2], max_seq_len=best_x_test.shape[1], depth=4, heads=5, dim_head=16, use_projection=True).to(device)
    cnn = CNNFeature(input_length=best_fc_test.shape[2], input_channels=best_fc_test.shape[1], feature_dim=best_x_test.shape[2], conv_filters=[256, 128], conv_kernels=[5, 7], pool_sizes=[2, 2]).to(device)
    fusion = FeatureFusion(feature_dim=best_x_test.shape[2], heads=4, dim_head=16).to(device)
    mlp = MLP(input_dim=best_x_test.shape[2], num_classes=2).to(device)

    transformer.load_state_dict(best_model_states['transformer'])
    cnn.load_state_dict(best_model_states['cnn'])
    fusion.load_state_dict(best_model_states['fusion'])
    mlp.load_state_dict(best_model_states['mlp'])

    test_dataloader = DataLoader(CustomDataset(best_x_test, best_fc_test, test_y), batch_size=64, shuffle=False)
    test_lbls, _, test_prbs = evaluate_model(test_dataloader, transformer, cnn, fusion, mlp, device)
    test_prds = (test_prbs >= best_threshold).astype(int) 
    test_metrics = calculate_metrics_dict(test_lbls, test_prds, test_prbs)

    print(f"  [Test使用平均阈值={best_threshold:.3f}] Acc: {test_metrics['accuracy']:.3f}, Pre: {test_metrics['precision']:.3f}, Rec: {test_metrics['recall']:.3f}, F1: {test_metrics['f1']:.3f}, AUC: {test_metrics['auc']:.3f}")

    fold_test_data = {
        'promoter': test_promoter.tolist(),
        'rnap': test_rnap.tolist(),
        'labels': test_lbls.tolist(),
        'predictions': test_prds.tolist(),
        'probabilities': test_prbs.tolist()
    }


    subset_output = {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'fold_test_data': fold_test_data,
        'top_kmer': all_fold_kmers
    }

    return subset_output


def main():
    seed_val = 41
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    species_vis_dir = "visualizations3/withRNAP_HL4001"
    raw_data_root = 'orignal_motif/HL_4001'
    os.makedirs(species_vis_dir, exist_ok=True)
    os.makedirs(raw_data_root, exist_ok=True)

    try:
        promoter_data = pd.read_csv('data/eukaryote/HL/promoter/predict/HL_400.tsv', sep='\t')
        rnap_data = pd.read_csv('data/eukaryote/HL/polymerase/HL_prow.tsv', sep='\t')
    except Exception as e:
        print(f"load error: {e}")
        return
    species_name = "HL"

    promoter_subsets = create_promoter_subsets_with_replacement(promoter_data, rnap_data, seed_val)
    num_subsets = len(promoter_subsets)

    all_val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    all_test_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    all_test_data = {'promoter': [], 'rnap': [], 'labels': [], 'predictions': [], 'probabilities': []}
    all_raw_kmers_history = []
    all_subsets_kmer_scores = defaultdict(list)

    for i, subset in enumerate(promoter_subsets):
        out = process_subset(i, subset, rnap_data, device, seed_val)

        all_test_data['promoter'].extend(out['fold_test_data']['promoter'])
        all_test_data['rnap'].extend(out['fold_test_data']['rnap'])
        all_test_data['labels'].extend(out['fold_test_data']['labels'])
        all_test_data['predictions'].extend(out['fold_test_data']['predictions'])
        all_test_data['probabilities'].extend(out['fold_test_data']['probabilities'])

        if out['top_kmer'] is not None and len(out['top_kmer']) > 0:
            for item in out['top_kmer']:
                kmer_seq = item[0]
                kmer_score = float(item[1])
                all_raw_kmers_history.append({
                    'subset_idx': i, 'fold': 0, 'kmer': kmer_seq, 'score': kmer_score
                })
                all_subsets_kmer_scores[kmer_seq].append(kmer_score)

        for metric in all_val_metrics:
            all_val_metrics[metric].append(np.mean(out['val_metrics'][metric]))
            all_test_metrics[metric].append(out['test_metrics'][metric])

        print(f"\n[子集 {i+1}/{num_subsets}] CV Average Val Acc: {np.mean(out['val_metrics']['accuracy']):.3f}, Independent Test Acc: {out['test_metrics']['accuracy']:.3f}")

    print("\n\n" + "="*45)
    print("     Overall Average Validation Results of 5-Fold Cross-Validation")
    print("="*45)
    print(f"  Accuracy : {np.mean(all_val_metrics['accuracy']):.4f} ± {np.std(all_val_metrics['accuracy']):.4f}")
    print(f"  Precision: {np.mean(all_val_metrics['precision']):.4f} ± {np.std(all_val_metrics['precision']):.4f}")
    print(f"  Recall   : {np.mean(all_val_metrics['recall']):.4f} ± {np.std(all_val_metrics['recall']):.4f}")
    print(f"  F1 Score : {np.mean(all_val_metrics['f1']):.4f} ± {np.std(all_val_metrics['f1']):.4f}")
    print(f"  AUC              : {np.mean(all_val_metrics['auc']):.4f} ± {np.std(all_val_metrics['auc']):.4f}")

    print("\n" + "="*45)
    print("          Overall Average Test Set Results")
    print("="*45)
    print(f"  Accuracy : {np.mean(all_test_metrics['accuracy']):.4f}")
    print(f"  Precision: {np.mean(all_test_metrics['precision']):.4f}")
    print(f"  Recall   : {np.mean(all_test_metrics['recall']):.4f}")
    print(f"  F1 Score : {np.mean(all_test_metrics['f1']):.4f}")
    print(f"  AUC              : {np.mean(all_test_metrics['auc']):.4f}")

    final_dedup = deduplicate_samples(
        all_test_data['promoter'], all_test_data['rnap'],
        all_test_data['labels'], all_test_data['predictions'], all_test_data['probabilities']
    )
    print(f"\nGlobal test set aggregation completed: Accumulated independent test samples {final_dedup['valid_count']} → Global unique sequences {final_dedup['unique_count']}")

    results_df = pd.DataFrame({
        "promoter_sequence": final_dedup['promoter'],
        "true_label": final_dedup['labels'],
        "predicted_label": final_dedup['predictions'],
        "probability": final_dedup['probabilities'],
        "occurrence_count": final_dedup['counts'],
        "proportion_in_test": final_dedup['proportions']
    })

    results_df['proportion_in_test'] = results_df['proportion_in_test'].apply(lambda x: f"{x:.4%}")
    results_df["prediction_type"] = results_df.apply(
        lambda x: "TP" if x.true_label == 1 and x.predicted_label == 1
        else "TN" if x.true_label == 0 and x.predicted_label == 0
        else "FP" if x.true_label == 0 and x.predicted_label == 1
        else "FN", axis=1
    )
    results_df = results_df.sort_values(by="occurrence_count", ascending=False)
    save_path = os.path.join(species_vis_dir, f"{species_name}_test_prediction_results.tsv")
    results_df.to_csv(save_path, sep="\t", index=False)
    print(f"Aggregated global retained test set prediction results saved to: {save_path}")

    if len(all_raw_kmers_history) > 0:
        df_raw = pd.DataFrame(all_raw_kmers_history)
        cols_order = ['subset_idx', 'fold', 'kmer', 'score']
        final_cols = [c for c in cols_order if c in df_raw.columns]
        df_raw = df_raw[final_cols]
        csv_path = os.path.join(raw_data_root, "withHL_raw_kmer_importance.csv")
        df_raw.to_csv(csv_path, index=False)
        print(f"\nRaw k-mer data has been saved to: {csv_path}")
        print(f"   Total {len(df_raw)} records\n")
    else:
        print("\nNo K-mer data collected. Skipping CSV saving.\n")

if __name__ == '__main__':
    main()