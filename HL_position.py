import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

PLOT_CONFIG = {
    "font": {"family": "Times New Roman", "title_size": 14, "label_size": 12, "tick_size": 10},
    "dpi": 300,
    "colors": {
        "enhanced": "#3498db", 
        "new": "#e74c3c", 
        "pos_enhanced": "#3498db",
        "pos_new": "#e74c3c",
        "density": "#000000",
    },
}

TARGET_SPECIES = "HL"
SEQ_LENGTH = 400  
BIN_SIZE = 10
TOP_N_PLOT = 10
TOP_N_MERGED = 5
TOP_N_SCORE = 10

TSS_OFFSET = 299
X_MIN = -299
X_MAX = 100

BASE_SPECIES_DIR = os.path.join("orignal_motif/HL_4001")
DOMAIN = "eukaryote"
RAW_SEQ_FILE = os.path.join("data", DOMAIN, TARGET_SPECIES, "promoter", "predict", "HL_400.tsv")


def create_dir_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def plot_combined_scores(enhanced_df, new_df, species, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    enh = enhanced_df.sort_values('delta', ascending=False).head(TOP_N_SCORE).sort_values('delta', ascending=True)
    bars1 = ax1.barh(enh['kmer'], enh['delta'], color=PLOT_CONFIG["colors"]["enhanced"], edgecolor="white",
                     linewidth=0.8, height=0.6)
    max_val_enh = max(enh['delta'])
    for bar, v in zip(bars1, enh['delta']):
        ax1.text(v + max_val_enh * 0.03, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va='center', fontsize=7,
                 color='black')

    ax1.set_xlabel("Score Increase (Delta)", fontsize=8, labelpad=2)
    ax1.grid(alpha=0.2, axis='x')
    ax1.set_xlim(0, max_val_enh * 1.15)
    ax1.tick_params(axis='both', labelsize=7)
    ax1.text(0.5, -0.2, f"(a) Enhanced K-mers (Top {TOP_N_SCORE})",
             transform=ax1.transAxes, ha='center', va='center', fontsize=8)

    new = new_df.sort_values('score_with', ascending=False).head(TOP_N_SCORE).sort_values('score_with', ascending=True)
    bars2 = ax2.barh(new['kmer'], new['score_with'], color=PLOT_CONFIG["colors"]["new"], edgecolor="white",
                     linewidth=0.8, height=0.6)
    max_val_new = max(new['score_with'])
    for bar, v in zip(bars2, new['score_with']):
        ax2.text(v + max_val_new * 0.03, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va='center', fontsize=7,
                 color='black')

    ax2.set_xlabel("Importance Score", fontsize=8, labelpad=2)
    ax2.grid(alpha=0.2, axis='x')
    ax2.set_xlim(0, max_val_new * 1.15)
    ax2.tick_params(axis='both', labelsize=7)
    ax2.text(0.5, -0.2, f"(b) Newly Added K-mers (Top {TOP_N_SCORE})",
             transform=ax2.transAxes, ha='center', va='center', fontsize=8)

    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_positions_combined(enhanced_list, new_list, seq_file, species, save_path):
    all_kmers = []
    for k, s in enhanced_list: all_kmers.append((k, s, "enhanced"))
    for k, s in new_list: all_kmers.append((k, s, "new"))
    df = pd.read_csv(seq_file, sep='\t')
    sequences = df[df['label'] == 1]['sequence'].tolist()

    tss_offset = TSS_OFFSET
    x_min, x_max = X_MIN, X_MAX

    bins = np.arange(x_min, x_max + BIN_SIZE, BIN_SIZE)
    fig, axes = plt.subplots(4, 5, figsize=(30, 16), dpi=300)
    axes = axes.flatten()
    for idx, (kmer, score, typ) in enumerate(all_kmers):
        if idx >= len(axes): break
        ax = axes[idx]
        pos = []
        for s in sequences:
            matches = [m.start() - tss_offset for m in re.finditer(f'(?={kmer})', s.upper())]
            pos.extend(matches)
        total = len(pos)
        bar_color = PLOT_CONFIG["colors"]["pos_enhanced"] if typ == "enhanced" else PLOT_CONFIG["colors"]["pos_new"]
        counts, _ = np.histogram(pos, bins=bins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax.bar(bin_centers, counts, width=BIN_SIZE * 0.85, color=bar_color, alpha=0.95)
        ax_twin = ax.twinx()
        sns.kdeplot(pos, ax=ax_twin, color=PLOT_CONFIG["colors"]["density"], linewidth=3, label="Density")
        ax_twin.set_yticks([])
        ax_twin.legend(loc="upper right", fontsize=8)
        ax.axvline(0, color="black", linewidth=3, zorder=10)

        ax.set_xlim(x_min - 5, x_max + 5)
        ax.set_xlabel("Position relative to TSS")
        ax.grid(alpha=0.2)
        ax.text(0.5, -0.22, f"{kmer}\nScore={score:.4f} | Count={total}",
                transform=ax.transAxes, ha='center', va='center', fontsize=8)

    for j in range(len(all_kmers), len(axes)): axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_group_merged(
        kmers_list,
        full_merged_df,
        seq_file,
        save_path,
        group_size,
        is_enhanced=True
):
    df = pd.read_csv(seq_file, sep='\t')
    sequences = df[df['label'] == 1]['sequence'].tolist()

    tss_offset = TSS_OFFSET
    x_min, x_max = X_MIN, X_MAX

    bins = np.arange(x_min, x_max + BIN_SIZE, BIN_SIZE)
    color = PLOT_CONFIG["colors"]["enhanced"] if is_enhanced else PLOT_CONFIG["colors"]["new"]

    groups = [kmers_list[i:i + group_size] for i in range(0, len(kmers_list), group_size)]
    n_groups = len(groups)

    group_txt_path = save_path.replace(".png", "_Groups.txt")
    with open(group_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"=== {'Enhanced' if is_enhanced else 'New'} K-mer Groups ===\n")
        f.write(f"Group size: {group_size}\n\n")
        for g_idx, group in enumerate(groups):
            f.write(f"Group {g_idx + 1}:\n")
            for kmer, score_val in group:
                row = full_merged_df[full_merged_df['kmer'] == kmer].iloc[0]
                cat = row['Category']
                delta_val = row['delta']
                f.write(f"  {kmer:8s} | Score: {score_val:8.4f} | Delta: {delta_val:8.4f} | {cat}\n")
            f.write("\n")

    if n_groups == 5:
        nrows, ncols = 2, 3
        figsize = (24, 10)
    elif n_groups == 2:
        nrows, ncols = 1, 2
        figsize = (18, 6)
    else:
        nrows = (n_groups + 1) // 2
        ncols = 2
        figsize = (18, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=300)
    axes = axes.flatten()

    for g_idx, group in enumerate(groups):
        ax = axes[g_idx]
        all_pos = []
        for kmer, _ in group:
            for seq in sequences:
                matches = [m.start() - tss_offset for m in re.finditer(f'(?={kmer})', seq.upper())]
                all_pos.extend(matches)

        counts, _ = np.histogram(all_pos, bins=bins)
        centers = 0.5 * (bins[:-1] + bins[:-1])
        ax.bar(centers, counts, width=BIN_SIZE * 0.85, color=color, alpha=0.95, edgecolor='white')

        ax_twin = ax.twinx()
        sns.kdeplot(all_pos, ax=ax_twin, color=PLOT_CONFIG["colors"]["density"], linewidth=3)
        ax_twin.set_yticks([])

        ax.axvline(0, color='crimson', linewidth=3, zorder=10)
        names = " + ".join([k for k, _ in group])
        ax.set_xlim(x_min - 5, x_max + 5)
        ax.set_xlabel("Position relative to TSS", )
        ax.grid(alpha=0.2)

        ax.text(0.5, -0.35, f"Group {g_idx + 1}\n{names}\nTotal={len(all_pos)}",
                transform=ax.transAxes, ha='center', va='center', fontsize=8)

    for idx in range(n_groups, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'{"Enhanced" if is_enhanced else "New"} K-mers | {group_size} Merged per Group', fontsize=8, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def main():
    out_dir = f"motif_final_output/{TARGET_SPECIES}"
    create_dir_if_not_exists(out_dir)

    raw_file = os.path.join(BASE_SPECIES_DIR, f"with{TARGET_SPECIES}_raw_kmer_importance.csv")
    no_rnap_file = os.path.join(BASE_SPECIES_DIR, f"without{TARGET_SPECIES}_raw_kmer_importance.csv")
    df_w_raw, df_n_raw = pd.read_csv(raw_file), pd.read_csv(no_rnap_file)

    try:
        df_w = df_w_raw.groupby('kmer', as_index=False)['score'].mean()
        df_n = df_n_raw.groupby('kmer', as_index=False)['score'].mean()
    except:
        df_w = df_w_raw.groupby('kmer_sequence', as_index=False)['importance_score'].mean()
        df_n = df_n_raw.groupby('kmer_sequence', as_index=False)['importance_score'].mean()
        df_w.columns = ['kmer', 'score']
        df_n.columns = ['kmer', 'score']

    df_w.rename(columns={'score': 'score_with'}, inplace=True)
    df_n.rename(columns={'score': 'score_no'}, inplace=True)
    df_merged = pd.merge(df_w, df_n, on='kmer', how='left').fillna(0)
    df_merged['delta'] = df_merged['score_with'] - df_merged['score_no']

    positive_deltas = df_merged[df_merged['delta'] > 0]['delta']
    dynamic_threshold = np.percentile(positive_deltas, 75)

    def classify(row):
        if row['score_no'] == 0 and row['score_with'] > 0:
            return 'Newly_Added'
        if row['score_no'] > 0 and row['delta'] >= dynamic_threshold:
            return 'Enhanced'
        return 'Stable_Or_Decreased'

    df_merged['Category'] = df_merged.apply(classify, axis=1)
    enhanced = df_merged[df_merged['Category'] == 'Enhanced'].copy()
    new_added = df_merged[df_merged['Category'] == 'Newly_Added'].copy()

    plot_combined_scores(enhanced, new_added, TARGET_SPECIES, f"{out_dir}/Top_Important_Kmers_Combined.png")
    enh_top_n = enhanced.sort_values('delta', ascending=False).head(TOP_N_PLOT)[['kmer', 'score_with']].values.tolist()
    new_top_n = new_added.sort_values('score_with', ascending=False).head(TOP_N_PLOT)[
        ['kmer', 'score_with']].values.tolist()
    plot_positions_combined(enh_top_n, new_top_n, RAW_SEQ_FILE, TARGET_SPECIES, f"{out_dir}/Position_Combined.png")

    plot_group_merged(enh_top_n, df_merged, RAW_SEQ_FILE, f"{out_dir}/Enhanced_Group_Merged5.png", TOP_N_MERGED,
                      is_enhanced=True)
    plot_group_merged(new_top_n, df_merged, RAW_SEQ_FILE, f"{out_dir}/New_Group_Merg5.png", TOP_N_MERGED,
                      is_enhanced=False)

    print("completed")


if __name__ == "__main__":
    main()