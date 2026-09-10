from matplotlib.patches import Patch
import umap
from sklearn.preprocessing import minmax_scale
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from upsetplot import plot as upset_plot
import matplotlib.cm as cm

plt.rcParams['font.family'] = 'serif'          
plt.rcParams['font.serif'] = ['Times New Roman'] 
plt.rcParams['axes.unicode_minus'] = False       
sns.set_theme(style="white", font="Times New Roman") 
MOD_COLORS = {'Promoter': '#00A087', 'RNAP': '#E64B35', 'TF': '#3C5488'}


def set_clean_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)


def plot_trimodal_global_ranking(df_trimodal, save_path, top_n=25, title="Global Feature Importance (Multi-modal)"):
    if df_trimodal.empty: return
    df_plot = df_trimodal.sort_values('importance', ascending=False).head(top_n).reset_index(drop=True)

    df_plot['z_score'] = (df_plot['importance'] - df_plot['importance'].mean()) / (df_plot['importance'].std() + 1e-6)
    df_plot['plot_val'] = df_plot['z_score'] - df_plot['z_score'].min() + 0.1

    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.4 * top_n)))
    y_pos = np.arange(len(df_plot))
    colors = [MOD_COLORS.get(m, '#7F7F7F') for m in df_plot['modality']][::-1]
    plot_vals = df_plot['plot_val'][::-1].values

    ax.hlines(y=y_pos, xmin=0, xmax=plot_vals, color=colors, alpha=0.6, linewidth=3, zorder=1)
    ax.scatter(plot_vals, y_pos, color=colors, s=120, alpha=1.0, edgecolor='white', linewidth=1.5, zorder=2)

    ax.grid(axis='x', linestyle='--', color='#E5E5E5', linewidth=1.0, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['feature_name'][::-1], fontsize=11, color='#2C3E50')
    ax.set_xlabel('Relative Importance (Effect Size)', fontsize=13, fontweight='bold', color='#333333')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#222222')

    set_clean_style(ax)

    legend_elements = [Patch(facecolor=MOD_COLORS[k], label=k, alpha=0.8) for k in ['Promoter', 'RNAP', 'TF']]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_macro_modality_importance(modality_scores, save_path):
    fig, ax = plt.subplots(figsize=(7, 7))
    mods = list(modality_scores.keys())
    scores = np.array(list(modality_scores.values()))

    percentages = scores / scores.sum() * 100
    colors = [MOD_COLORS.get(m, '#7F7F7F') for m in mods]

    wedges, texts, autotexts = ax.pie(
        percentages, labels=mods, colors=colors, startangle=140,
        autopct='%1.1f%%', pctdistance=0.75,
        textprops=dict(color="#222222", fontsize=13, fontweight='bold'),
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=3)
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')

    ax.set_title('Macro Modality Contribution', fontsize=17, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, transparent=False, facecolor='white')
    plt.close()


def plot_top_tf_ranking(tf_importance_df, save_path, top_n=20):
    if tf_importance_df.empty: return
    df_plot = tf_importance_df.sort_values('importance', ascending=False).head(top_n).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(5, 0.4 * top_n)))
    y_pos = np.arange(len(df_plot))
    plot_vals = df_plot['importance'][::-1].values
    solid_color = MOD_COLORS['TF']

    ax.hlines(y=y_pos, xmin=0, xmax=plot_vals, color=solid_color, alpha=0.5, linewidth=4, zorder=1)
    ax.scatter(plot_vals, y_pos, color=solid_color, s=100, edgecolor='white', linewidth=1, zorder=2)

    ax.grid(axis='x', linestyle='--', color='#E5E5E5', linewidth=1.0, zorder=0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['tf_name'][::-1], fontsize=11, color='#2C3E50')
    ax.set_xlabel('Mean Probability Drop (PFI)', fontsize=13, fontweight='bold')
    ax.set_title("Top Transcriptional Factors Importance", fontsize=15, fontweight='bold', pad=15)

    set_clean_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_top_features_violin(tf_importance_df, tf_matrix, labels, tf_names, save_path, top_n=8):
    if tf_importance_df.empty: return
    top_tfs = tf_importance_df.sort_values('importance', ascending=False).head(top_n)['tf_name'].tolist()
    tf_idx_map = {name: i for i, name in enumerate(tf_names)}

    plot_data = []
    for tf in top_tfs:
        if tf in tf_idx_map:
            idx = tf_idx_map[tf]
            vals = tf_matrix[:, idx]
            for val, lbl in zip(vals, labels):
                plot_data.append({'TF': tf, 'Value': val, 'Class': 'Promoter (1)' if lbl == 1 else 'Non-Promoter (0)'})

    df_plot = pd.DataFrame(plot_data)
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.violinplot(data=df_plot, x='Value', y='TF', hue='Class', split=True, inner="quartile",
                   palette={'Promoter (1)': '#E64B35', 'Non-Promoter (0)': '#4DBBD5'}, ax=ax, linewidth=1.2)

    ax.set_title("Top TFs: Differential Distribution between Classes", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("TF Motif Score", fontsize=12, fontweight='bold')
    ax.set_ylabel("")
    set_clean_style(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_prediction_density(results_df, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=results_df[results_df['true_label'] == 1], x='probability', fill=True, color='#E64B35',
                label='Promoter (Positive)', ax=ax, alpha=0.6, linewidth=2)
    sns.kdeplot(data=results_df[results_df['true_label'] == 0], x='probability', fill=True, color='#3C5488',
                label='Non-Promoter (Negative)', ax=ax, alpha=0.6, linewidth=2)

    ax.axvline(0.5, color='#333333', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_title('Model Confidence Distribution (Prediction Density)', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')

    set_clean_style(ax)
    ax.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='white')
    plt.close()


def plot_latent_space_umap(latent_features, labels, save_path):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(latent_features)

    fig, ax = plt.subplots(figsize=(8, 6))

    label_names = np.array(['Promoter' if l == 1 else 'Non-Promoter' for l in labels])

    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1],
        hue=label_names,
        palette={'Promoter': '#E64B35', 'Non-Promoter': '#4DBBD5'},
        s=30, alpha=0.8, edgecolor='white', linewidth=0.5, ax=ax
    )

    ax.set_title("UMAP Projection of Latent Fusion Features", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax.set_ylabel("UMAP Dimension 2", fontsize=12)
    set_clean_style(ax)
    ax.legend(title='True Label', frameon=False, loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor='white')
    plt.close()


def plot_tf_synergy_comprehensive(tf_matrix, tf_names, top_tfs, save_prefix,
                                  use_jaccard=True, net_threshold=0.2, upset_min_size=5,
                                  tf_importance_dict=None):

    tf_idx_map = {name: i for i, name in enumerate(tf_names)}
    valid_tfs = [tf for tf in top_tfs if tf in tf_idx_map]
    indices = [tf_idx_map[tf] for tf in valid_tfs]

    if len(indices) < 2:
        print("skip")
        return

    top_tf_count = tf_matrix[:, indices]
    top_tf_binary = (top_tf_count >= 1).astype(np.float32)
    df_bin = pd.DataFrame(top_tf_binary, columns=valid_tfs)

    if use_jaccard:
        jaccard_dist = pdist(df_bin.T.values, metric="jaccard")
        jaccard_sim = 1 - squareform(jaccard_dist)
        df_corr = pd.DataFrame(jaccard_sim, index=valid_tfs, columns=valid_tfs)
        cbar_label = "Jaccard Co-occurrence"
        center_val = None
        cmap = "rocket"
    else:
        df_corr = df_bin.corr()
        cbar_label = "Pearson Correlation"
        center_val = 0
        cmap = "vlag"


    print("  -> Clustermap...")
    g = sns.clustermap(
        df_corr, cmap=cmap, center=center_val, figsize=(10, 10),
        linewidths=0.5, annot=False, cbar_kws={'label': cbar_label}
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    g.fig.suptitle("Co-occurrence of Top TFs", fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(f"{save_prefix}_clustermap.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("  -> Network...")
    G = nx.Graph()
    tf_freq = df_bin.sum(axis=0)

    raw_node_weights = {}
    for tf in df_corr.columns:
        if tf_importance_dict is not None and tf in tf_importance_dict:
            raw_node_weights[tf] = tf_importance_dict[tf]
        else:
            raw_node_weights[tf] = tf_freq[tf]  
        G.add_node(tf, weight=raw_node_weights[tf])

    tfs = df_corr.columns
    for i in range(len(tfs)):
        for j in range(i + 1, len(tfs)):
            edge_weight = df_corr.iloc[i, j]
            if edge_weight >= net_threshold:
                G.add_edge(tfs[i], tfs[j], weight=edge_weight)

    weights_list = list(raw_node_weights.values())
    max_w, min_w = max(weights_list), min(weights_list)
    visual_min, visual_max = 300, 2000

    node_sizes = []
    for node in G.nodes():
        w = G.nodes[node]['weight']
        if max_w == min_w:
            scaled_size = visual_min + (visual_max - visual_min) / 2
        else:
            scaled_size = visual_min + (w - min_w) / (max_w - min_w) * (visual_max - visual_min)
        node_sizes.append(scaled_size)

    sorted_edges = sorted(G.edges(data=True), key=lambda t: t[2]['weight'])
    edges_list = [(u, v) for u, v, d in sorted_edges]
    edge_weights = [d['weight'] for u, v, d in sorted_edges]

    edge_widths = [w * 6 for w in edge_weights]

    min_edge_w = min(edge_weights) if edge_weights else 0
    max_edge_w = max(edge_weights) if edge_weights else 1

    from matplotlib.colors import LinearSegmentedColormap
    edge_mono_cmap = LinearSegmentedColormap.from_list("Edge_Mono_Greys", ["#D2D2D2", "#000000"])

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.8, seed=42)

    nx.draw_networkx_edges(
        G, pos, edgelist=edges_list, width=edge_widths,
        edge_color=edge_weights, edge_cmap=edge_mono_cmap,
        edge_vmin=min_edge_w, edge_vmax=max_edge_w,
        alpha=0.85
    )

    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color='#3C5488',
        alpha=0.95, edgecolors='white', linewidths=1.2
    )

    nx.draw_networkx_labels(
        G, pos, font_size=10, font_weight='bold',
        font_family='Times New Roman'
    )

    title_suffix = "Size by Importance" if tf_importance_dict else "Size by Frequency"
    plt.title(f"TF Co-occurrence Network\n({cbar_label} >= {net_threshold} | {title_suffix})", fontsize=14)
    plt.axis('off')
    plt.savefig(f"{save_prefix}_network.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print("  ->UpSet Plot...")
    try:
        df_bool = df_bin.astype(bool)
        upset_data = df_bool.groupby(list(df_bool.columns)).size()

        all_false_idx = tuple([False] * len(df_bool.columns))
        if all_false_idx in upset_data.index:
            upset_data = upset_data.drop(index=all_false_idx)

        if not upset_data.empty:
            plt.figure(figsize=(12, 6))
            upset_plot(upset_data, min_subset_size=upset_min_size,
                       show_counts=True, facecolor="darkblue", element_size=40)
            plt.suptitle(f"Intersection of TF Occurrences (min count={upset_min_size})", fontsize=16, y=1.05)
            plt.savefig(f"{save_prefix}_upset.png", dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            print("  skip")
    except Exception as e:
        print(f"  error: {e}")



def plot_focus_evolution_heatmap(feat_stage1, feat_stage2, feat_stage3, save_path):

    s1 = minmax_scale(gaussian_filter1d(feat_stage1, sigma=2))
    s2 = minmax_scale(gaussian_filter1d(feat_stage2, sigma=2))
    s3 = minmax_scale(gaussian_filter1d(feat_stage3, sigma=2))

    seq_len = len(feat_stage1)  

    kmer_size = 5
    offset = (kmer_size - 1) // 2 
    orig_len = seq_len + (kmer_size - 1)  

    if orig_len <= 300:
        orig_end = 50
    else:
        orig_end = 100
    orig_start = orig_end - orig_len + 1  

  
    full_data = np.full((3, orig_len), np.nan)


    full_data[0, offset: offset + seq_len] = s1
    full_data[1, offset: offset + seq_len] = s2
    full_data[2, offset: offset + seq_len] = s3

    cmap = cm.get_cmap("magma").copy()
    cmap.set_bad(color="black")


    if orig_len <= 150:
        step = 25
    elif orig_len <= 350:
        step = 50
    else:
        step = 100

    candidate_coords = list(range(-1000, 1001, step))
    tick_coords = [orig_start]  

    for c in candidate_coords:
        if orig_start < c < orig_end and c != 0:
            tick_coords.append(c)

    tick_coords.append(0)  # TSS (0)
    tick_coords.append(orig_end)  

    tick_coords = sorted(list(set(tick_coords)))

    
    tick_positions = [(c - orig_start + 0.5) for c in tick_coords]

    tick_labels = []
    for c in tick_coords:
        if c == 0:
            tick_labels.append("TSS (0)")
        elif c > 0:
            tick_labels.append(f"+{c}")
        else:
            tick_labels.append(f"{c}")


    plt.figure(figsize=(18, 4.5))

    ax = sns.heatmap(
        full_data,
        cmap=cmap,
        cbar_kws={"shrink": 0.8, "label": "Normalized Feature Activation\n(Focus Intensity)", "pad": 0.02},
        yticklabels=["1. DNA Self-Focus", "2. + TF Fusion", "3. + RNAP Fusion"]
    )

   
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontweight='bold', fontsize=12)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=11)
    ax.set_xlabel("Promoter Sequence Position (bp relative to TSS)", fontweight='bold', fontsize=13, labelpad=10)

    tss_position = 0 - orig_start + 0.5
    ax.axvline(x=tss_position, color='white', linestyle='--', linewidth=1.5, alpha=0.7)

    # plt.title("Evolution of Spatial Sequence Focus Across Multi-Modal Fusion Stages",
    #           fontsize=16, pad=20, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()