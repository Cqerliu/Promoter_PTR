import os
import random
from tqdm import tqdm
import copy
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from encode import PromoterEncoder, ProEncoder
from downTCPPR_module_TF import TransformerFeature, MLP, CNNFeature, FeatureFusion
import pandas as pd
import warnings
from collections import defaultdict
from datasplit_TF import create_promoter_subsets_with_replacement
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from tf_plots import *

warnings.filterwarnings('ignore')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class TFFreqMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=100):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class PrototypeLoss(nn.Module):
    def __init__(self, lambda_align=0.05):
        super().__init__()
        self.focal = FocalLoss(gamma=2, label_smoothing=0.1)
        self.align_loss = nn.CosineEmbeddingLoss(margin=0.3)
        self.lambda_align = lambda_align

    def forward(self, logits, dna_feat, rnap_feat, tf_feat, targets):
        cls_loss = self.focal(logits, targets)
        align_targets = targets.float() * 2 - 1

        dna_vec = dna_feat.max(dim=1)[0] if dna_feat.dim() > 2 else dna_feat
        rnap_vec = rnap_feat.max(dim=1)[0] if rnap_feat.dim() > 2 else rnap_feat
        tf_vec = tf_feat.max(dim=1)[0] if tf_feat.dim() > 2 else tf_feat

        feat_loss1 = self.align_loss(dna_vec, rnap_vec, align_targets)
        feat_loss2 = self.align_loss(dna_vec, tf_vec, align_targets)
        feat_loss = (feat_loss1 + feat_loss2) / 2
        return cls_loss + self.lambda_align * feat_loss, cls_loss, feat_loss


class CustomDataset(Dataset):
    def __init__(self, x, fc, tf, y):
        self.x = x
        self.fc = fc
        self.tf = tf
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).float(), \
            torch.from_numpy(self.fc[idx]).float(), \
            torch.from_numpy(self.tf[idx]).float(), \
            torch.tensor(self.y[idx], dtype=torch.long)


def clear_memory(model_list=None):
    if model_list is not None:
        for model in model_list:
            if model is not None:
                del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def deduplicate_samples(promoter_seqs, rnap_seqs, labels, predictions, probabilities=None, features=None):
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
    dedup_rnap = [rnap_seqs[i] for i in unique_indices] if rnap_seqs is not None else None
    dedup_labels = [labels[i] for i in unique_indices]
    dedup_preds = [predictions[i] for i in unique_indices]
    dedup_probs = [probabilities[i] for i in unique_indices] if probabilities is not None else None
    dedup_features = [features[i] for i in unique_indices] if features is not None else None
    dedup_counts = [counts[p_seq] for p_seq in dedup_promoter]
    dedup_proportions = [c / valid_count for c in dedup_counts]

    return {
        'promoter': dedup_promoter, 'rnap': dedup_rnap, 'labels': dedup_labels,
        'predictions': dedup_preds, 'probabilities': dedup_probs, 'counts': dedup_counts,
        'proportions': dedup_proportions, 'features': dedup_features,
        'original_count': len(promoter_seqs), 'valid_count': valid_count, 'unique_count': len(unique_indices)
    }


def get_kmer_importance(encoder, model, x_data, fc_data, tf_data, device, top_k=None, promoter_sequences=None,
                        batch_size=128):
    local_pfi_seed = 42
    rng_pfi = np.random.Generator(np.random.PCG64(local_pfi_seed))

    for sub_model in model.values():
        sub_model.eval()

    fc_tensor = torch.from_numpy(fc_data).float().to(device) if fc_data is not None else None
    tf_tensor = torch.from_numpy(tf_data).float().to(device) if tf_data is not None else None

    total_samples = x_data.shape[0]
    original_probs = np.zeros(total_samples)

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_x = torch.from_numpy(x_data[i:i + batch_size]).float().to(device)
            p_feat = model['transformer'](batch_x)
            t_feat = model['tf_mlp'](tf_tensor[i:i + batch_size]).unsqueeze(1) if tf_tensor is not None else None
            pt_feat, _ = model['fusion_pt'](p_feat, t_feat)

            if fc_tensor is not None:
                r_feat = model['cnn'](fc_tensor[i:i + batch_size])
                final_feat, _ = model['fusion_final'](pt_feat, r_feat)
            else:
                final_feat = pt_feat

            final_feat = final_feat.max(dim=1)[0]
            original_probs[i:i + batch_size] = torch.softmax(model['mlp'](final_feat), dim=1)[:, 1].cpu().numpy()

            del batch_x, p_feat, final_feat
            torch.cuda.empty_cache()

    kmer_importance = {}
    seq_len = x_data.shape[1]
    kmer_size = 5

    for pos in range(seq_len):
        pos_importance_list = []
        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                batch_x_pert = x_data[i:i + batch_size].copy()
                for idx in range(batch_x_pert.shape[0]):
                    batch_x_pert[idx, pos] = rng_pfi.permutation(batch_x_pert[idx, pos])
                batch_x_pert = torch.from_numpy(batch_x_pert).float().to(device)

                p_feat_pert = model['transformer'](batch_x_pert)
                t_feat = model['tf_mlp'](tf_tensor[i:i + batch_size]).unsqueeze(1) if tf_tensor is not None else None
                pt_feat_pert, _ = model['fusion_pt'](p_feat_pert, t_feat)

                if fc_tensor is not None:
                    r_feat = model['cnn'](fc_tensor[i:i + batch_size])
                    final_feat_pert, _ = model['fusion_final'](pt_feat_pert, r_feat)
                else:
                    final_feat_pert = pt_feat_pert

                final_feat_pert = final_feat_pert.max(dim=1)[0]
                batch_pert_probs = torch.softmax(model['mlp'](final_feat_pert), dim=1)[:, 1].cpu().numpy()

                batch_importance = np.abs(original_probs[i:i + batch_size] - batch_pert_probs)
                pos_importance_list.extend(batch_importance)

                del batch_x_pert, p_feat_pert, final_feat_pert
                torch.cuda.empty_cache()

        importance = np.mean(pos_importance_list)

        if promoter_sequences is not None and len(promoter_sequences) > 0:
            sample_seq = promoter_sequences[0]
            start_idx = max(0, min(pos, len(sample_seq) - kmer_size))
            kmer_seq = sample_seq[start_idx:start_idx + kmer_size]
        else:
            kmer_seq = f'pos_{pos}'

        kmer_importance[kmer_seq] = importance

    sorted_importance = sorted(kmer_importance.items(), key=lambda x: x[1], reverse=True)
    return sorted_importance[:top_k] if top_k is not None else sorted_importance


def evaluate_model(dataloader, transformer, cnn, tf_mlp, fusion_pt, fusion_final, mlp, device, return_features=False):
    transformer.eval()
    cnn.eval()
    tf_mlp.eval()
    fusion_pt.eval()
    fusion_final.eval()
    mlp.eval()

    all_predicted = []
    all_labels = []
    all_probs = []
    all_features = []

    with torch.no_grad():
        for batch_X, batch_fc, batch_tf, batch_y in dataloader:
            batch_X, batch_fc, batch_tf, batch_y = batch_X.to(device), batch_fc.to(device), batch_tf.to(
                device), batch_y.to(device)

            p_feat = transformer(batch_X)
            r_feat = cnn(batch_fc)
            t_feat = tf_mlp(batch_tf).unsqueeze(1)

            pt_feat, _ = fusion_pt(p_feat, t_feat)
            final_feat, _ = fusion_final(pt_feat, r_feat)

            final_feat_pool = final_feat.max(dim=1)[0]
            if return_features:
                all_features.append(final_feat_pool.cpu().numpy())

            logits = mlp(final_feat_pool)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            predicted = torch.argmax(logits, dim=1).cpu().numpy()

            all_predicted.extend(predicted)
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs)

    if return_features:
        return np.array(all_labels), np.array(all_predicted), np.array(all_probs), np.concatenate(all_features, axis=0)
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


def calculate_metrics_dict(labels, preds, probs):
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else 0.0
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}


def process_subset(subset_idx, promoter_subset, rnap_subset, tf_total, device, base_seed=41, tf_names=None,
                   kmer_save_dir=None, species_vis_dir=None, species_name=None):
    print(f"\n===== process promoter subset  {subset_idx + 1} =====")

    promoter_sequences = promoter_subset['sequence'].values
    y = promoter_subset['label'].values.astype(np.int64)
    subset_size = len(y)
    rnap_sequences = rnap_subset['sequence'].values

    shuffled_indices = np.random.permutation(subset_size)
    promoter_sequences, rnap_sequences, y = promoter_sequences[shuffled_indices], rnap_sequences[shuffled_indices], y[
        shuffled_indices]
    tf_data = tf_total[shuffled_indices]

    cv_indices, test_indices = train_test_split(np.arange(subset_size), test_size=0.2, stratify=y,
                                                random_state=base_seed)

    test_promoter = promoter_sequences[test_indices]
    test_rnap = rnap_sequences[test_indices]
    test_y = y[test_indices]
    test_tf = tf_data[test_indices]

    cv_promoter = promoter_sequences[cv_indices]
    cv_rnap = rnap_sequences[cv_indices]
    cv_y = y[cv_indices]
    cv_tf = tf_data[cv_indices]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=base_seed)
    val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

    global_best_cv_val_acc = 0.0
    best_model_states = None
    best_encoder = None
    best_x_test = None
    best_fc_test = None
    best_tf_test = None
    fold_opt_thresholds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(cv_promoter, cv_y)):
        print(f"\n----- Fold  {fold + 1}  -----")

        train_promoter, val_promoter = cv_promoter[train_idx], cv_promoter[val_idx]
        train_rnap, val_rnap = cv_rnap[train_idx], cv_rnap[val_idx]
        train_y, val_y = cv_y[train_idx], cv_y[val_idx]
        train_tf, val_tf = cv_tf[train_idx], cv_tf[val_idx]

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
        fc_train = np.array([fc_encode.encode_conjoint_cnn(s).squeeze(0).cpu().numpy() for s in train_rnap],
                            dtype=np.float32)
        fc_val = np.array([fc_encode.encode_conjoint_cnn(s).squeeze(0).cpu().numpy() for s in val_rnap],
                          dtype=np.float32)
        fc_test_current = np.array([fc_encode.encode_conjoint_cnn(s).squeeze(0).cpu().numpy() for s in test_rnap],
                                   dtype=np.float32)

        train_dataloader = DataLoader(CustomDataset(x_train, fc_train, train_tf, train_y), batch_size=64, shuffle=True)
        val_dataloader = DataLoader(CustomDataset(x_val, fc_val, val_tf, val_y), batch_size=64, shuffle=False)

        hidden_dim = x_train.shape[2]
        tf_in_dim = train_tf.shape[1]

        transformer = TransformerFeature(input_dim=x_train.shape[2], max_seq_len=x_train.shape[1], depth=4, heads=5,
                                         dim_head=16, use_projection=True).to(device)
        cnn = CNNFeature(input_length=fc_train.shape[2], input_channels=fc_train.shape[1], feature_dim=x_train.shape[2],
                         conv_filters=[256, 128], conv_kernels=[5, 7], pool_sizes=[2, 2]).to(device)
        tf_mlp = TFFreqMLP(in_dim=tf_in_dim, out_dim=hidden_dim).to(device)
        fusion_pt = FeatureFusion(feature_dim=x_train.shape[2], heads=4, dim_head=16).to(device)
        fusion_final = FeatureFusion(feature_dim=x_train.shape[2], heads=4, dim_head=16).to(device)
        mlp = MLP(input_dim=x_train.shape[2], num_classes=2).to(device)
        mlp_dna_only = MLP(input_dim=x_train.shape[2], num_classes=2).to(device)

        criterion = PrototypeLoss(lambda_align=0.05).to(device)
        optimizer = torch.optim.AdamW(
            list(transformer.parameters()) + list(cnn.parameters()) + list(tf_mlp.parameters()) +
            list(fusion_pt.parameters()) + list(fusion_final.parameters()) + list(mlp.parameters()) + list(
                mlp_dna_only.parameters()),
            lr=1e-4, weight_decay=0.005)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        num_epochs = 50
        best_fold_val_acc = 0.0
        best_fold_states = None

        for epoch in range(num_epochs):
            transformer.train()
            cnn.train()
            tf_mlp.train()
            fusion_pt.train()
            fusion_final.train()
            mlp.train()
            mlp_dna_only.train()

            running_loss = 0.0
            for batch_X, batch_fc, batch_tf, batch_y in train_dataloader:
                batch_X, batch_fc, batch_tf, batch_y = batch_X.to(device), batch_fc.to(device), batch_tf.to(
                    device), batch_y.to(device)
                optimizer.zero_grad()

                p_feat = transformer(batch_X)
                r_feat = cnn(batch_fc)
                t_feat = tf_mlp(batch_tf).unsqueeze(1)

                t_feat_f = torch.zeros_like(t_feat) if random.random() < 0.3 else t_feat
                r_feat_f = torch.zeros_like(r_feat) if random.random() < 0.3 else r_feat

                pt_feat, _ = fusion_pt(p_feat, t_feat_f)
                final_feat, _ = fusion_final(pt_feat, r_feat_f)

                final_feat_pool = final_feat.max(dim=1)[0]
                logits = mlp(F.dropout(final_feat_pool, p=0.05, training=True))
                loss_fusion, _, _ = criterion(logits, p_feat, r_feat, t_feat, batch_y)

                logits_dna = mlp_dna_only(F.dropout(p_feat.max(dim=1)[0], p=0.05, training=True))
                loss_dna = F.cross_entropy(logits_dna, batch_y, label_smoothing=0.1)

                total_loss = loss_fusion + 0.6 * loss_dna
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += total_loss.item()
                scheduler.step()

            val_labels, val_preds, _ = evaluate_model(val_dataloader, transformer, cnn, tf_mlp, fusion_pt, fusion_final,
                                                      mlp, device)
            val_acc = accuracy_score(val_labels, val_preds)

            status = " "
            if val_acc > best_fold_val_acc:
                best_fold_val_acc = val_acc
                best_fold_states = {
                    'transformer': copy.deepcopy(transformer.state_dict()),
                    'cnn': copy.deepcopy(cnn.state_dict()),
                    'tf_mlp': copy.deepcopy(tf_mlp.state_dict()),
                    'fusion_pt': copy.deepcopy(fusion_pt.state_dict()),
                    'fusion_final': copy.deepcopy(fusion_final.state_dict()),
                    'mlp': copy.deepcopy(mlp.state_dict())
                }
                status = " [Best Model Updated!]"

            print(
                f"Epoch [{epoch + 1:02d}/{num_epochs}] | Train Loss: {running_loss / len(train_dataloader):.4f} | Val Acc: {val_acc:.4f}{status}")

        transformer.load_state_dict(best_fold_states['transformer'])
        cnn.load_state_dict(best_fold_states['cnn'])
        tf_mlp.load_state_dict(best_fold_states['tf_mlp'])
        fusion_pt.load_state_dict(best_fold_states['fusion_pt'])
        fusion_final.load_state_dict(best_fold_states['fusion_final'])
        mlp.load_state_dict(best_fold_states['mlp'])

        val_lbls, val_prds, val_prbs = evaluate_model(val_dataloader, transformer, cnn, tf_mlp, fusion_pt, fusion_final,
                                                      mlp, device)
        opt_th = find_optimal_threshold(val_lbls, val_prbs)
        fold_opt_thresholds.append(opt_th)

        val_prds_opt = (val_prbs >= opt_th).astype(int)
        v_res = calculate_metrics_dict(val_lbls, val_prds_opt, val_prbs)
        for k in val_metrics:
            val_metrics[k].append(v_res[k])

        print(
            f"  [the best threshold={opt_th:.3f}] Acc: {v_res['accuracy']:.3f}, Pre: {v_res['precision']:.3f}, F1: {v_res['f1']:.3f}")

        if v_res['accuracy'] > global_best_cv_val_acc:
            global_best_cv_val_acc = v_res['accuracy']
            best_model_states = copy.deepcopy(best_fold_states)
            best_encoder = copy.deepcopy(encoder)
            best_x_test = x_test_current.copy()
            best_fc_test = fc_test_current.copy()
            best_tf_test = test_tf.copy()

        clear_memory([transformer, cnn, tf_mlp, fusion_pt, fusion_final, mlp, mlp_dna_only])
        del x_train, x_val, x_test_current, fc_train, fc_val, fc_test_current, encoder
        clear_memory()

    best_threshold = np.mean(fold_opt_thresholds)
    print(f"\n-----uese the best model + avg threshold={best_threshold:.3f} to test -----")

    transformer = TransformerFeature(input_dim=best_x_test.shape[2], max_seq_len=best_x_test.shape[1], depth=4, heads=5,
                                     dim_head=16, use_projection=True).to(device)
    cnn = CNNFeature(input_length=best_fc_test.shape[2], input_channels=best_fc_test.shape[1],
                     feature_dim=best_x_test.shape[2], conv_filters=[256, 128], conv_kernels=[5, 7],
                     pool_sizes=[2, 2]).to(device)
    tf_mlp = TFFreqMLP(in_dim=best_tf_test.shape[1], out_dim=best_x_test.shape[2]).to(device)
    fusion_pt = FeatureFusion(feature_dim=best_x_test.shape[2], heads=4, dim_head=16).to(device)
    fusion_final = FeatureFusion(feature_dim=best_x_test.shape[2], heads=4, dim_head=16).to(device)
    mlp = MLP(input_dim=best_x_test.shape[2], num_classes=2).to(device)

    transformer.load_state_dict(best_model_states['transformer'])
    cnn.load_state_dict(best_model_states['cnn'])
    tf_mlp.load_state_dict(best_model_states['tf_mlp'])
    fusion_pt.load_state_dict(best_model_states['fusion_pt'])
    fusion_final.load_state_dict(best_model_states['fusion_final'])
    mlp.load_state_dict(best_model_states['mlp'])

    test_dataloader = DataLoader(CustomDataset(best_x_test, best_fc_test, best_tf_test, test_y), batch_size=64,
                                 shuffle=False)

    test_lbls, test_preds, test_prbs, test_feats = evaluate_model(
        test_dataloader, transformer, cnn, tf_mlp, fusion_pt, fusion_final, mlp, device, return_features=True
    )
    test_prds_final = (test_prbs >= best_threshold).astype(int)
    test_metrics = calculate_metrics_dict(test_lbls, test_prds_final, test_prbs)

    print(
        f"  [Test使用平均阈值={best_threshold:.3f}] Acc: {test_metrics['accuracy']:.3f}, AUC: {test_metrics['auc']:.3f}")

    print(f"\n  [Analysis] PFI ...")
    subset_pfi_rng = np.random.Generator(np.random.PCG64(42))
    curr_model_dict = {
        'transformer': transformer, 'cnn': cnn, 'tf_mlp': tf_mlp,
        'fusion_pt': fusion_pt, 'fusion_final': fusion_final, 'mlp': mlp
    }

    all_subset_raw_kmers = []
    tp_indices = [i for i in range(len(test_lbls)) if test_lbls[i] == 1 and test_prds_final[i] == 1]
    if len(tp_indices) > 0:
        raw_kmers = get_kmer_importance(
            best_encoder, curr_model_dict,
            best_x_test[tp_indices], best_fc_test[tp_indices], best_tf_test[tp_indices],
            device, promoter_sequences=test_promoter[tp_indices]
        )
        if raw_kmers is not None:
            all_subset_raw_kmers.extend(
                [{'subset_idx': subset_idx, 'fold': -1, 'kmer': k, 'score': s} for k, s in raw_kmers]
            )

    subset_tf_pfi = defaultdict(list)
    for ch in tqdm(range(best_tf_test.shape[1]), desc="TF PFI ", leave=False):
        s_tf = best_tf_test.copy()
        s_tf[:, ch] = s_tf[subset_pfi_rng.permutation(len(s_tf)), ch]
        _, _, s_pb = evaluate_model(DataLoader(CustomDataset(best_x_test, best_fc_test, s_tf, test_y), batch_size=64),
                                    transformer, cnn, tf_mlp, fusion_pt, fusion_final, mlp, device)
        subset_tf_pfi[tf_names[ch]].append(np.mean(np.abs(test_prbs - s_pb)))

    subset_rnap_pfi = defaultdict(list)
    for ch in tqdm(range(best_fc_test.shape[1]), desc="RNAP PFI ", leave=False):
        s_fc = best_fc_test.copy()
        s_fc[:, ch, :] = s_fc[subset_pfi_rng.permutation(len(s_fc)), ch, :]
        _, _, s_pb = evaluate_model(DataLoader(CustomDataset(best_x_test, s_fc, best_tf_test, test_y), batch_size=64),
                                    transformer, cnn, tf_mlp, fusion_pt, fusion_final, mlp, device)
        subset_rnap_pfi[f"Channel_{ch + 1}"].append(np.mean(np.abs(test_prbs - s_pb)))

    subset_macro_pfi = defaultdict(list)
    for mod in tqdm(['Promoter', 'RNAP', 'TF'], desc="Macro PFI ", leave=False):
        s_x, s_fc, s_tf = best_x_test.copy(), best_fc_test.copy(), best_tf_test.copy()
        if mod == 'Promoter':
            s_x = s_x[subset_pfi_rng.permutation(len(s_x))]
        elif mod == 'RNAP':
            s_fc = s_fc[subset_pfi_rng.permutation(len(s_fc))]
        else:
            s_tf = s_tf[subset_pfi_rng.permutation(len(s_tf))]
        _, _, s_pb = evaluate_model(DataLoader(CustomDataset(s_x, s_fc, s_tf, test_y), batch_size=64), transformer, cnn,
                                    tf_mlp, fusion_pt, fusion_final, mlp, device)
        subset_macro_pfi[mod].append(np.mean(np.abs(test_prbs - s_pb)))

    if subset_idx == 0 and species_vis_dir is not None and species_name is not None:
        print("\n  -> [Visualization] heatmap...")

        if len(tp_indices) > 0:
            sample_idx = tp_indices[:15]

            transformer.eval()
            cnn.eval()
            tf_mlp.eval()
            fusion_pt.eval()
            fusion_final.eval()

            with torch.no_grad():
                batch_x = torch.from_numpy(best_x_test[sample_idx]).float().to(device)
                batch_fc = torch.from_numpy(best_fc_test[sample_idx]).float().to(device)
                batch_tf = torch.from_numpy(best_tf_test[sample_idx]).float().to(device)

                p_feat = transformer(batch_x)
                r_feat = cnn(batch_fc)
                t_feat = tf_mlp(batch_tf).unsqueeze(1)

                # pt_feat 形状为 (Batch, SeqLen, Dim)
                pt_feat, _ = fusion_pt(p_feat, t_feat)
                final_feat, _ = fusion_final(pt_feat, r_feat)

                focus_p = p_feat.norm(dim=-1).mean(dim=0).cpu().numpy()
                focus_pt = pt_feat.norm(dim=-1).mean(dim=0).cpu().numpy()
                focus_final = final_feat.norm(dim=-1).mean(dim=0).cpu().numpy()

            evolution_save_path = os.path.join(species_vis_dir, f"{species_name}_focus_evolution_heatmap.png")
            plot_focus_evolution_heatmap(focus_p, focus_pt, focus_final, evolution_save_path)
        else:
            print("  [prompt] no True Positive sample.")
    # =====================================================================

    fold_test_data = {
        'promoter': test_promoter.tolist(),
        'rnap': test_rnap.tolist(),
        'labels': test_lbls.tolist(),
        'predictions': test_prds_final.tolist(),
        'probabilities': test_prbs.tolist(),
        'features': test_feats.tolist()
    }

    clear_memory([transformer, cnn, tf_mlp, fusion_pt, fusion_final, mlp])
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'fold_test_data': fold_test_data,
        'top_kmer': all_subset_raw_kmers,
        'tf_importance': {k: np.mean(v) for k, v in subset_tf_pfi.items()},
        'rnap_importance': {k: np.mean(v) for k, v in subset_rnap_pfi.items()},
        'macro_importance': {k: np.mean(v) for k, v in subset_macro_pfi.items()}
    }


def main():
    seed_val = 41
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    species_vis_dir = "visualizations_PTRmap/withRNAP_HL600"
    os.makedirs(species_vis_dir, exist_ok=True)

    try:
        promoter_data = pd.read_csv('data/eukaryote/HL/promoter/predict/HL_600.tsv', sep='\t')
        rnap_data = pd.read_csv('data/eukaryote/HL/polymerase/HL_prow.tsv', sep='\t')
        TF_df = pd.read_csv('data/eukaryote/HL/promoter/predict/TF_600.tsv', sep='\t')
        TF_data = TF_df.values.astype(np.float32)
        tf_motif_cols = TF_df.columns[3:].tolist()
    except Exception as e:
        print(f"load error: {e}")
        return
    species_name = "HL"

    promoter_subsets = create_promoter_subsets_with_replacement(promoter_data, rnap_data, TF_data, seed_val)
    num_subsets = len(promoter_subsets)

    all_val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    all_test_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    all_test_data = {'promoter': [], 'rnap': [], 'labels': [], 'predictions': [], 'probabilities': [], 'features': []}

    all_raw_kmers_history = []
    all_subsets_kmer_scores = defaultdict(list)
    all_tf_scores = defaultdict(list)
    all_rnap_scores = defaultdict(list)
    all_macro_scores = defaultdict(list)

    for i, subset in enumerate(promoter_subsets):
        tf_subset = np.array(subset['tf_features'].tolist())

        out = process_subset(i, subset, rnap_data, tf_subset, device, seed_val,
                             tf_names=TF_df.columns.tolist(),
                             species_vis_dir=species_vis_dir, species_name=species_name)

        for key in all_test_data:
            all_test_data[key].extend(out['fold_test_data'][key])

        if out['top_kmer'] is not None and len(out['top_kmer']) > 0:
            all_raw_kmers_history.extend(out['top_kmer'])
            for kmer_dict in out['top_kmer']:
                all_subsets_kmer_scores[kmer_dict['kmer']].append(kmer_dict['score'])

        for k, v in out['tf_importance'].items():
            all_tf_scores[k].append(v)
        for k, v in out['rnap_importance'].items():
            all_rnap_scores[k].append(v)
        for k, v in out['macro_importance'].items():
            all_macro_scores[k].append(v)

        for metric in all_val_metrics:
            all_val_metrics[metric].append(np.mean(out['val_metrics'][metric]))
            all_test_metrics[metric].append(out['test_metrics'][metric])

    final_dedup = deduplicate_samples(
        all_test_data['promoter'], all_test_data['rnap'],
        all_test_data['labels'], all_test_data['predictions'],
        all_test_data['probabilities'], features=all_test_data['features']
    )

    summary_lines = []
    summary_lines.append(f"  Accuracy: {np.mean(all_val_metrics['accuracy']):.4f} ± {np.std(all_val_metrics['accuracy']):.4f}")
    summary_lines.append(f"  Precision: {np.mean(all_val_metrics['precision']):.4f} ± {np.std(all_val_metrics['precision']):.4f}")
    summary_lines.append(f"  Recall   : {np.mean(all_val_metrics['recall']):.4f} ± {np.std(all_val_metrics['recall']):.4f}")
    summary_lines.append(f"  F1 Score : {np.mean(all_val_metrics['f1']):.4f} ± {np.std(all_val_metrics['f1']):.4f}")
    summary_lines.append(f"  AUC        : {np.mean(all_val_metrics['auc']):.4f} ± {np.std(all_val_metrics['auc']):.4f}")

    summary_lines.append("\n" + "=" * 50)
    summary_lines.append("         Test avg result")
    summary_lines.append("=" * 50)
    summary_lines.append(f"  Accuracy : {np.mean(all_test_metrics['accuracy']):.4f}")
    summary_lines.append(f"  Precision: {np.mean(all_test_metrics['precision']):.4f}")
    summary_lines.append(f"  Recall   : {np.mean(all_test_metrics['recall']):.4f}")
    summary_lines.append(f"  F1 Score : {np.mean(all_test_metrics['f1']):.4f}")
    summary_lines.append(f"  AUC       : {np.mean(all_test_metrics['auc']):.4f}")

    summary_text = "\n".join(summary_lines)

    print("\n\n" + summary_text)

    summary_file_path = os.path.join(species_vis_dir, f"{species_name}_evaluation_summary.txt")
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    # =========================================================================================

    results_df = pd.DataFrame({
        "promoter_sequence": final_dedup['promoter'],
        "true_label": final_dedup['labels'],
        "predicted_label": final_dedup['predictions'],
        "probability": final_dedup['probabilities'],
        "occurrence_count": final_dedup['counts'],
    })
    save_path = os.path.join(species_vis_dir, f"{species_name}_test_prediction_results.tsv")
    results_df.to_csv(save_path, sep="\t", index=False)

    macro_final = {k: np.mean(v) for k, v in all_macro_scores.items()}
    plot_macro_modality_importance(macro_final, os.path.join(species_vis_dir, f"{species_name}_macro_donut.png"))

    tf_f_df = pd.DataFrame([{'tf_name': k, 'importance': np.mean(v)} for k, v in all_tf_scores.items()])
    tf_motif_df = tf_f_df[tf_f_df['tf_name'].isin(tf_motif_cols)].reset_index(drop=True)
    plot_top_tf_ranking(tf_motif_df, os.path.join(species_vis_dir, f"{species_name}_top_tf_lollipop.png"), top_n=20)

    trimodal_list = []
    if all_subsets_kmer_scores:
        trimodal_list.append(pd.DataFrame(
            [{'feature_name': f"DNA: {k}", 'importance': np.mean(v), 'modality': 'Promoter'}
             for k, v in all_subsets_kmer_scores.items()]))
    if not tf_motif_df.empty:
        tf_temp = tf_motif_df.copy()
        tf_temp['feature_name'] = "TF: " + tf_temp['tf_name']
        tf_temp['modality'] = 'TF'
        trimodal_list.append(tf_temp[['feature_name', 'importance', 'modality']])
    if all_rnap_scores:
        trimodal_list.append(pd.DataFrame(
            [{'feature_name': f"RNAP: {k}", 'importance': np.mean(v), 'modality': 'RNAP'}
             for k, v in all_rnap_scores.items()]))

    if trimodal_list:
        global_df = pd.concat(trimodal_list).sort_values('importance', ascending=False).reset_index(drop=True)
        plot_trimodal_global_ranking(global_df,
                                     os.path.join(species_vis_dir, f"{species_name}_global_trimodal_lollipop.png"),
                                     top_n=25)

    test_results_df = pd.DataFrame({'true_label': final_dedup['labels'], 'probability': final_dedup['probabilities']})
    plot_prediction_density(test_results_df, os.path.join(species_vis_dir, f"{species_name}_prediction_density.png"))

    plot_top_features_violin(tf_motif_df, TF_data, promoter_data['label'].values, TF_df.columns.tolist(),
                             os.path.join(species_vis_dir, f"{species_name}_tf_violin.png"), top_n=8)

    if final_dedup.get('features') is not None and len(final_dedup['features']) > 0:
        latent_matrix = np.array(final_dedup['features'])
        plot_latent_space_umap(latent_matrix, final_dedup['labels'],
                               os.path.join(species_vis_dir, f"{species_name}_umap_latent_space.png"))

    if not tf_motif_df.empty:
        top_tfs = tf_motif_df.sort_values('importance', ascending=False).head(20)['tf_name'].tolist()
        tf_importance_dict = dict(zip(tf_motif_df['tf_name'], tf_motif_df['importance']))
        synergy_save_prefix = os.path.join(species_vis_dir, f"{species_name}_tf_synergy")

        print("\n  -> Clustermap, Network, UpSet...")
        plot_tf_synergy_comprehensive(
            tf_matrix=TF_data,
            tf_names=TF_df.columns.tolist(),
            top_tfs=top_tfs,
            save_prefix=synergy_save_prefix,
            use_jaccard=True,
            net_threshold=0.15,
            upset_min_size=15,
            tf_importance_dict=tf_importance_dict
        )

if __name__ == '__main__':
    main()