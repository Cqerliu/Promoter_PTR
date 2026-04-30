import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from scipy.stats import gaussian_kde
from matplotlib.colors import to_hex

# ===================== Basic configuration =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Times New Roman'

plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
# =============================================================

# ===================== Data Loading and Classification =====================
def load_and_classify(before_file, after_file):
    df_before = pd.read_csv(before_file, sep='\t', encoding='utf-8')
    df_after = pd.read_csv(after_file, sep='\t', encoding='utf-8')

    df_before = df_before[['promoter_sequence', 'true_label', 'predicted_label']].rename(
        columns={'predicted_label': 'pred_before'})
    df_after = df_after[['promoter_sequence', 'predicted_label']].rename(columns={'predicted_label': 'pred_after'})

    df_merged = pd.merge(df_before, df_after, on='promoter_sequence', how='inner')
    df_merged['correct_before'] = df_merged['pred_before'] == df_merged['true_label']
    df_merged['correct_after'] = df_merged['pred_after'] == df_merged['true_label']

    conditions = [
        (df_merged['correct_before'] == False) & (df_merged['correct_after'] == True),
        (df_merged['correct_before'] == True) & (df_merged['correct_after'] == False),
        (df_merged['correct_before'] == False) & (df_merged['correct_after'] == False),
        (df_merged['correct_before'] == True) & (df_merged['correct_after'] == True)
    ]
    labels = ['Improved', 'Deteriorated', 'Consistently wrong', 'Consistently correct']
    df_merged['classification'] = np.select(conditions, labels, default='unknown')

    return df_merged

# ===================== Feature extract =====================
def extract_deep_bio_features(seq):
    seq = seq.upper()
    seq_len = len(seq)
    if seq_len == 0: return pd.Series([0, 0, 0, 0, 0])
    c, g = seq.count('C'), seq.count('G')
    gc_content = (c + g) / seq_len
    gc_skew = (g - c) / (g + c) if (g + c) > 0 else 0
    ta_steps = len(re.findall(r'(?=TA)', seq)) / (seq_len - 1) if seq_len > 1 else 0
    cg_steps = len(re.findall(r'(?=CG)', seq)) / (seq_len - 1) if seq_len > 1 else 0
    at_tracts = re.findall(r'[AT]+', seq)
    max_at_tract = max([len(t) for t in at_tracts]) if at_tracts else 0
    return pd.Series([gc_content, abs(gc_skew), ta_steps, cg_steps, max_at_tract])

def save_sequence_subsets(df_merged, output_dir):
    seq_dir = os.path.join(output_dir, 'classified_sequences')
    os.makedirs(seq_dir, exist_ok=True)

    subsets = {'Promoter': 1, 'Non-Promoter': 0}
    class_names = ['Consistently correct', 'Improved', 'Deteriorated', 'Consistently wrong']

    for label_name, label_val in subsets.items():
        df_sub = df_merged[df_merged['true_label'] == label_val].copy()
        for cls in class_names:
            df_cls = df_sub[df_sub['classification'] == cls]
            if len(df_cls) == 0:
                continue
            out_path = os.path.join(seq_dir, f'{label_name}_{cls}.tsv')
            df_cls[['promoter_sequence', 'true_label', 'pred_before', 'pred_after', 'classification']].to_csv(
                out_path, sep='\t', index=False, encoding='utf-8')
    print(f"All class sequences have been saved to: {seq_dir}")

# ===================== Visualization Drawing and Statistics=====================
def plot_stratified_mechanisms(df_merged, output_dir='./promoter_analysis_results/HL_4001/'):
    subsets = {'Promoter': 1, 'Non-Promoter': 0}
    plot_order = ['Consistently correct','Improved', 'Deteriorated', 'Consistently wrong']
    palette_dict = {
        'Consistently correct': '#f39c12',
        'Improved': '#2ecc71',
        'Deteriorated': '#e74c3c',
        'Consistently wrong': '#3498db'
    }

    stats_output = []
    stats_output.append("==================== Statistics of Model Prediction Classification Quantity ====================")

    for label_name, label_val in subsets.items():
        feat_dir = os.path.join(output_dir, f'stratified_analysis_{label_name}')
        os.makedirs(feat_dir, exist_ok=True)
        df_sub = df_merged[df_merged['true_label'] == label_val].copy()

        stats_output.append(f"\n【{label_name} group】 (Total: {len(df_sub)})")
        counts = df_sub['classification'].value_counts()
        for cls in plot_order:
            count = counts.get(cls, 0)
            stats_output.append(f"  - {cls}: {count} 个 ({count / len(df_sub) * 100:.2f}%)")

        df_sub = df_sub[df_sub['classification'].isin(plot_order)]
        if len(df_sub) < 5: continue

        features = df_sub['promoter_sequence'].apply(extract_deep_bio_features)
        features.columns = ['GC_Content', 'Abs_GC_Skew', 'TA_Flexibility', 'CpG_Density', 'Max_AT_Tract']
        df_sub = pd.concat([df_sub, features], axis=1)
        medians = df_sub.groupby('classification').median(numeric_only=True)

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=300)

        # === A: TA frequency ===
        sns.violinplot(x='classification', y='TA_Flexibility', data=df_sub, palette=palette_dict,
                       order=plot_order, ax=axes[0,0], hue='classification', legend=False)
        for cls in plot_order:
            if cls in medians.index:
                axes[0,0].axhline(y=medians.loc[cls, 'TA_Flexibility'], color=palette_dict[cls], linestyle='--', linewidth=1, alpha=0.8)
        axes[0,0].set_xlabel('')
        axes[0,0].text(0.5, -0.15, f'(a) TpA frequency',
                      transform=axes[0,0].transAxes, ha='center', va='center', fontsize=8)

        # === B: Max AT ===
        sns.histplot(data=df_sub, x='Max_AT_Tract', hue='classification', hue_order=plot_order,
                     palette=palette_dict, discrete=True, stat='density', common_norm=False,
                     multiple='dodge', shrink=0.7, alpha=0.5, ax=axes[0,1])
        sns.kdeplot(data=df_sub, x='Max_AT_Tract', hue='classification', hue_order=plot_order,
                    palette=palette_dict, linewidth=1.5, bw_adjust=1.2, common_norm=False, ax=axes[0,1], legend=False)
        for line in axes[0,1].get_lines():
            x_data, y_data = line.get_xdata(), line.get_ydata()
            if len(y_data) > 0:
                peak_x = x_data[np.argmax(y_data)]
                axes[0,1].axvline(x=peak_x, color=line.get_color(), linestyle='--', linewidth=1, alpha=0.9)
        axes[0,1].set_xlim(2, 18)
        axes[0,1].set_ylabel('Density', fontsize=8)
        axes[0,1].set_xlabel('')
        axes[0,1].text(0.5, -0.15, f'(b) Max AT',
                      transform=axes[0,1].transAxes, ha='center', va='center', fontsize=8)

        # === C: CpG Density ===
        sns.boxplot(x='classification', y='CpG_Density', data=df_sub, palette=palette_dict,
                    order=plot_order, ax=axes[1,0], showfliers=False, width=0.4,
                    boxprops=dict(alpha=0.6), hue='classification', legend=False)
        sns.stripplot(x='classification', y='CpG_Density', data=df_sub, order=plot_order,
                      color='black', alpha=0.1, size=1, jitter=0.2, ax=axes[1,0])
        for cls in plot_order:
            if cls in medians.index:
                axes[1,0].axhline(y=medians.loc[cls, 'CpG_Density'], color=palette_dict[cls], linestyle='--', linewidth=1, alpha=0.8)
        axes[1,0].set_xlabel('')
        axes[1,0].text(0.5, -0.15, f'(c) CpG density',
                      transform=axes[1,0].transAxes, ha='center', va='center', fontsize=8)

        # === D: GC Content ===
        sns.kdeplot(data=df_sub, x='GC_Content', hue='classification', hue_order=plot_order,
                    fill=True, common_norm=False, palette=palette_dict, alpha=0.5, linewidth=1.5, ax=axes[1,1])
        for cls in plot_order:
            data = df_sub[df_sub['classification'] == cls]['GC_Content'].dropna()
            if len(data) > 1:
                kde = gaussian_kde(data)
                x_vals = np.linspace(data.min(), data.max(), 1000)
                peak_x = x_vals[np.argmax(kde(x_vals))]
                axes[1,1].axvline(x=peak_x, color=palette_dict[cls], linestyle='--', linewidth=1, alpha=0.9)
        axes[1,1].set_xlabel('')
        axes[1,1].text(0.5, -0.15, f'(d) GC content',
                      transform=axes[1,1].transAxes, ha='center', va='center',fontsize=8)

        plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)
        plt.savefig(os.path.join(feat_dir, f'{label_name}_mechanism_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generation of {label_name} sequence analysis plot completed")

    stats_file_path = os.path.join(output_dir, 'Category Count Statistics.txt')
    with open(stats_file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(stats_output))
    print(f"\nCount statistics saved to: {stats_file_path}")
    print("\n".join(stats_output))

def main(before_file, after_file):
    df_merged = load_and_classify(before_file, after_file)
    output_dir = './promoter_analysis_results/HL_4001/'
    save_sequence_subsets(df_merged, output_dir)
    plot_stratified_mechanisms(df_merged, output_dir)
    print("\nAll analysis and sequence export tasks completed!")

if __name__ == '__main__':
    BEFORE_FILE = 'visualizations3/withoutRNAP_HL4001/HL_test_prediction_results.tsv'
    AFTER_FILE = 'visualizations3/withRNAP_HL4001/HL_test_prediction_results.tsv'
    main(BEFORE_FILE, AFTER_FILE)