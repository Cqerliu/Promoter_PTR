import numpy as np
import pandas as pd


def create_promoter_subsets_with_replacement(promoter_data, rnap_data, tf_data, base_seed=41):
    rnap_total = len(rnap_data)
    prom_total = len(promoter_data)

    if rnap_total == 0:
        raise ValueError("RNAP is NONE")

    ratio = prom_total / rnap_total
    num_subsets = int(ratio) + 1
    print(f"promoter: {prom_total}, RNAP: {rnap_total}, ratio: {ratio:.2f}")
    print(f"{num_subsets} subset")

    rnap_pos_count = rnap_data['label'].value_counts().get(1, 0)
    rnap_neg_count = rnap_data['label'].value_counts().get(0, 0)
    print(f"RNAP：P={rnap_pos_count},N={rnap_neg_count}, total={rnap_total}")
    prom_pos = promoter_data[promoter_data['label'] == 1].copy()
    prom_neg = promoter_data[promoter_data['label'] == 0].copy()
    prom_pos_count = len(prom_pos)
    prom_neg_count = len(prom_neg)
    print(f"promoter original data：P={prom_pos_count}, N={prom_neg_count}, 总计={prom_total}")

    if prom_pos_count == 0 and prom_neg_count == 0:
        raise ValueError("promoter is none")
    if prom_pos_count == 0:
        print("error：P promoter is none")
    if prom_neg_count == 0:
        print("error：N promoter is none")

    prom_pos['unique_id'] = range(prom_pos_count) if prom_pos_count > 0 else []
    prom_neg['unique_id'] = range(prom_neg_count) if prom_neg_count > 0 else []

    tf_df = pd.DataFrame(tf_data)

    used_pos_ids = set()
    used_neg_ids = set()
    subsets = []

    for i in range(num_subsets):
        current_seed = base_seed + i
        np.random.seed(current_seed)
        current_pos = pd.DataFrame()
        current_neg = pd.DataFrame()

        if prom_pos_count > 0:
            unused_pos = prom_pos[~prom_pos['unique_id'].isin(used_pos_ids)]
            if not unused_pos.empty:
                need = min(rnap_pos_count, len(unused_pos))
                current_pos = pd.concat([unused_pos.sample(n=need, replace=False, random_state=current_seed),
                                         prom_pos.sample(n=rnap_pos_count - need, replace=True,
                                                         random_state=current_seed)])
            else:
                current_pos = prom_pos.sample(n=rnap_pos_count, replace=True, random_state=current_seed)
        if prom_neg_count > 0:
            unused_neg = prom_neg[~prom_neg['unique_id'].isin(used_neg_ids)]
            if not unused_neg.empty:
                need = min(rnap_neg_count, len(unused_neg))
                current_neg = pd.concat([unused_neg.sample(n=need, replace=False, random_state=current_seed),
                                         prom_neg.sample(n=rnap_neg_count - need, replace=True,
                                                         random_state=current_seed)])
            else:
                current_neg = prom_neg.sample(n=rnap_neg_count, replace=True, random_state=current_seed)

        if not current_pos.empty:
            used_pos_ids.update(current_pos['unique_id'].values)
        if not current_neg.empty:
            used_neg_ids.update(current_neg['unique_id'].values)

        # 拼接启动子
        current_subset = pd.concat([current_pos, current_neg], ignore_index=True)

        current_tf = tf_df.loc[current_subset.index].values

        current_subset['tf_features'] = current_tf.tolist()

        subsets.append(current_subset)

        print(
            f"produce {i + 1}/{num_subsets}：P={len(current_pos)}, N={len(current_neg)}, 总计={len(current_subset)}")
        print(
            f"  used P: {len(used_pos_ids)}/{prom_pos_count}, used N: {len(used_neg_ids)}/{prom_neg_count}")

    if prom_pos_count > 0 and len(used_pos_ids) < prom_pos_count:
        missing_pos = prom_pos[~prom_pos['unique_id'].isin(used_pos_ids)]
        pos_per_subset = len(missing_pos) // num_subsets
        remaining_pos = len(missing_pos) % num_subsets

        start_idx = 0
        for i in range(num_subsets):
            add_count = pos_per_subset + (1 if i < remaining_pos else 0)
            if add_count <= 0:
                continue
            end_idx = start_idx + add_count
            pos_to_add = missing_pos.iloc[start_idx:end_idx]
            start_idx = end_idx
            current_subset = subsets[i]
            current_pos = current_subset[current_subset['label'] == 1]
            new_pos = pd.concat([current_pos.iloc[add_count:], pos_to_add])
            if len(new_pos) < rnap_pos_count:
                additional = rnap_pos_count - len(new_pos)
                new_pos = pd.concat(
                    [new_pos, prom_pos.sample(n=additional, replace=True, random_state=base_seed + 2000 + i)])
            current_neg = current_subset[current_subset['label'] == 0]
            new_subset = pd.concat([new_pos, current_neg], ignore_index=True)
            subsets[i] = new_subset
            used_pos_ids.update(pos_to_add['unique_id'].values)

    if prom_neg_count > 0 and len(used_neg_ids) < prom_neg_count:
        missing_neg = prom_neg[~prom_neg['unique_id'].isin(used_neg_ids)]
        neg_per_subset = len(missing_neg) // num_subsets
        remaining_neg = len(missing_neg) % num_subsets

        start_idx = 0
        for i in range(num_subsets):
            add_count = neg_per_subset + (1 if i < remaining_neg else 0)
            if add_count <= 0:
                continue
            end_idx = start_idx + add_count
            neg_to_add = missing_neg.iloc[start_idx:end_idx]
            start_idx = end_idx
            current_subset = subsets[i]
            current_neg = current_subset[current_subset['label'] == 0]
            new_neg = pd.concat([current_neg.iloc[add_count:], neg_to_add])
            if len(new_neg) < rnap_neg_count:
                additional = rnap_neg_count - len(new_neg)
                new_neg = pd.concat(
                    [new_neg, prom_neg.sample(n=additional, replace=True, random_state=base_seed + 3000 + i)])
            current_pos = current_subset[current_subset['label'] == 1]
            new_subset = pd.concat([current_pos, new_neg], ignore_index=True)
            subsets[i] = new_subset
            used_neg_ids.update(neg_to_add['unique_id'].values)

    assert (prom_pos_count == 0 or len(used_pos_ids) == prom_pos_count), \
        f"Promoter positive samples are not fully utilized: actually used {len(used_pos_ids)}, expected {prom_pos_count}"
    assert (prom_neg_count == 0 or len(used_neg_ids) == prom_neg_count), \
        f"Promoter positive samples are not fully utilized: actually used {len(used_neg_ids)}, expected {prom_neg_count}"
    return subsets