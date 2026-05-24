#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CGFN Dataset Partitioning & 5-Level Evaluation Framework
This pipeline automates the Functional Mutational Lineage clustering, 
stratified partitioning, and evaluation metric generation (Table 1 & 2) 
required for rigorous computational biology validation.
"""

import math
import random
import os
import argparse
import logging
import difflib
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Logging Setup
# ==========================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# 2. Global Configuration & Paths
# ==========================================
# These will be updated dynamically in main() based on argparse
DATA_ROOT = Path("data")
POS_DATA_CSV = DATA_ROOT / "all_Positive.csv"
POS_NPZ_DIR = DATA_ROOT / "features" / "positive"

OUT_SPLITS = DATA_ROOT / "splits_lineage"
ANALYSIS_DIR = OUT_SPLITS / "lineage_analysis"
FAMILY_SPLIT_DIR = OUT_SPLITS / "balanced_split_62out"

FAMILY_TABLE_BASE = FAMILY_SPLIT_DIR / "all_sequences_by_lineage.csv"
NONMR_TRAIN_CSV = FAMILY_SPLIT_DIR / "balanced_train_set.csv"
NONMR_TEST_CSV = FAMILY_SPLIT_DIR / "balanced_test_set.csv"

FULL_TEST_N = 100
NONMR_TEST_N = 70
RANDOM_SEEDS = [42, 123, 456, 789, 1024]
DOMINANT_PREFIX = "MRPEIW"

QUOTA_BUCKETS = {"small_1_2": 30, "medium_3_5": 25, "large_6_10": 25, "xlarge_ge11": 20}

# ==========================================
# 3. Base Utility Functions
# ==========================================
def write_list(p: Path, items):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(map(str, items)) + "\n", encoding="utf-8")
    return str(p.resolve())

def _read_id_seq_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cand_id = next((c for c in df.columns if "id" in c.lower()), None)
    cand_seq = next((c for c in df.columns if "seq" in c.lower()), None)
    if not cand_id: raise ValueError(f"Missing ID column in {path}")
    if not cand_seq: raise ValueError(f"Missing seq column in {path}")
    df = df.rename(columns={cand_id: "ID", cand_seq: "seq"})
    df["ID"] = df["ID"].astype(str).str.strip()
    df["seq"] = df["seq"].astype(str).str.strip().str.upper()
    return df

# ==========================================
# 4. Functional Mutational Lineage Identification
# ==========================================
def improved_precise_family_identification(sequences: List[str]) -> Tuple[Dict, Dict, Dict]:
    n_sequences = len(sequences)
    assigned_to_family = {}  
    families = defaultdict(list)  
    mutation_info = {}  
    family_counter = 0
    
    logger.info(f"Starting lineage identification for {n_sequences} sequences...")
    sorted_indices = sorted(range(n_sequences), key=lambda i: len(sequences[i]), reverse=True)
    
    for i in sorted_indices:
        if i in assigned_to_family: continue
        parent_seq = sequences[i]
        family_id = f"lineage_{family_counter}"
        families[family_id].append(i)
        assigned_to_family[i] = family_id
        family_mutation_positions = set()
        
        for j in range(n_sequences):
            if j in assigned_to_family or j == i: continue
            current_seq = sequences[j]
            if len(current_seq) != len(parent_seq): continue
            diff_positions = [pos for pos in range(len(parent_seq)) if parent_seq[pos] != current_seq[pos]]
            if len(diff_positions) == 1:
                families[family_id].append(j)
                assigned_to_family[j] = family_id
                family_mutation_positions.add(diff_positions[0])
        
        mutation_info[family_id] = {'parent': parent_seq, 'mutation_positions': family_mutation_positions, 'size': len(families[family_id])}
        family_counter += 1
    
    logger.info("Processing double point mutations...")
    for i in range(n_sequences):
        if i in assigned_to_family: continue
        current_seq = sequences[i]
        best_family, best_match_score = None, -1
        
        for family_id, info in mutation_info.items():
            parent = info['parent']
            if len(current_seq) != len(parent): continue
            diff_positions = [pos for pos in range(len(parent)) if parent[pos] != current_seq[pos]]
            if len(diff_positions) == 2:
                known_positions = info['mutation_positions']
                match_score = len(set(diff_positions) & known_positions)
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_family = family_id
        
        if best_family and best_match_score >= 1:
            families[best_family].append(i)
            assigned_to_family[i] = best_family
            diff_positions = [pos for pos in range(len(mutation_info[best_family]['parent'])) if mutation_info[best_family]['parent'][pos] != current_seq[pos]]
            mutation_info[best_family]['mutation_positions'].update(diff_positions)
            mutation_info[best_family]['size'] += 1
    
    remaining_indices = [i for i in range(n_sequences) if i not in assigned_to_family]
    for idx in remaining_indices:
        seq = sequences[idx]
        family_id = f"lineage_{family_counter}"
        families[family_id] = [idx]
        mutation_info[family_id] = {'parent': seq, 'mutation_positions': set(), 'size': 1}
        assigned_to_family[idx] = family_id
        family_counter += 1
    
    return dict(families), mutation_info, assigned_to_family

def analyze_family_distribution(families, mutation_info, sequences, assigned_to_family, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    family_sizes = [len(indices) for indices in families.values()]
    total_sequences, total_families = len(sequences), len(families)
    size_groups = {
        '1 seq': [1], '2 seqs': [2], '3-5 seqs': [3, 4, 5], 
        '6-10 seqs': [6, 7, 8, 9, 10], '11-20 seqs': range(11, 21), 
        '21-50 seqs': range(21, 51), '51-100 seqs': range(51, 101), 
        '100+ seqs': range(101, 1000)
    }
    group_stats = {}
    for group_name, sizes in size_groups.items():
        families_in_group = [size for size in family_sizes if size in sizes]
        if families_in_group:
            group_stats[group_name] = {
                'lineage_count': len(families_in_group), 
                'sequence_count': sum(families_in_group), 
                'lineage_percent': len(families_in_group) / total_families * 100, 
                'sequence_percent': sum(families_in_group) / total_sequences * 100
            }
    return group_stats

def balanced_test_set_allocation(families, mutation_info, sequences, test_size=100):
    family_list = [{'id': fid, 'indices': idx, 'size': len(idx), 'proportion': len(idx) / len(sequences)} for fid, idx in families.items()]
    family_list.sort(key=lambda x: x['size'], reverse=True)
    strata_quotas = {'small': 30, 'medium': 25, 'large': 25, 'xlarge': 20}
    test_indices, allocation_details = [], {}
    
    for cat, sizes, quota_pct in [('small', [1, 2], strata_quotas['small']), ('medium', [3, 4, 5], strata_quotas['medium']), ('large', [6, 7, 8, 9, 10], strata_quotas['large']), ('xlarge', range(11, 1000), strata_quotas['xlarge'])]:
        fams = [f for f in family_list if f['size'] in sizes]
        quota = int(test_size * quota_pct / 100)
        if not fams: continue
        if cat == 'small':
            for f in random.sample(fams, min(quota, len(fams))):
                test_indices.append(random.choice(f['indices']))
                allocation_details[f['id']] = 1
        else:
            total_size = sum(f['size'] for f in fams)
            for f in fams:
                f_quota = max(1, int(f['size'] / total_size * quota))
                max_cap = min(15, f['size'] // 3) if cat == 'xlarge' else (f['size'] - 1 if cat == 'large' else f['size'])
                f_quota = min(f_quota, max_cap)
                if f_quota > 0:
                    test_indices.extend(random.sample(f['indices'], f_quota))
                    allocation_details[f['id']] = allocation_details.get(f['id'], 0) + f_quota
                    
    if len(test_indices) < test_size:
        all_avail, weights = [], []
        for f in family_list:
            avail = [idx for idx in f['indices'] if idx not in test_indices]
            if avail:
                all_avail.extend(avail)
                weights.extend([f['proportion']] * len(avail))
        if all_avail:
            add_idx = random.choices(all_avail, weights=[w/sum(weights) for w in weights], k=test_size - len(test_indices))
            test_indices.extend(add_idx)
            for idx in add_idx:
                for f in family_list:
                    if idx in f['indices']: allocation_details[f['id']] = allocation_details.get(f['id'], 0) + 1; break
                        
    if len(test_indices) > test_size: test_indices = random.sample(test_indices, test_size)
    return test_indices, [i for i in range(len(sequences)) if i not in test_indices], allocation_details

def complete_family_analysis_and_balanced_split(csv_file, test_size, analysis_dir, split_dir):
    df = _read_id_seq_csv(Path(csv_file))
    sequences, ids = df['seq'].tolist(), df['ID'].tolist()
    families, mutation_info, assigned_to_family = improved_precise_family_identification(sequences)
    group_stats = analyze_family_distribution(families, mutation_info, sequences, assigned_to_family, analysis_dir)
    test_indices, train_indices, allocation_details = balanced_test_set_allocation(families, mutation_info, sequences, test_size)
    
    os.makedirs(split_dir, exist_ok=True)
    df.iloc[test_indices].to_csv(os.path.join(split_dir, "balanced_test_set.csv"), index=False)
    df.iloc[train_indices].to_csv(os.path.join(split_dir, "balanced_train_set.csv"), index=False)
    
    df_with_family = df.copy()
    df_with_family['family_id'] = [assigned_to_family[i] for i in range(len(sequences))]
    df_with_family['family_size'] = df_with_family['family_id'].map({fid: len(idx) for fid, idx in families.items()})
    df_with_family['in_test_set'] = [i in set(test_indices) for i in range(len(sequences))]
    df_with_family['family_num'] = df_with_family['family_id'].str.extract('(\d+)').astype(int)
    df_with_family = df_with_family.sort_values(['family_num', 'in_test_set', 'seq'], ascending=[True, False, True]).drop('family_num', axis=1)
    df_with_family.to_csv(os.path.join(split_dir, "all_sequences_by_lineage.csv"), index=False)
    return test_indices, train_indices, allocation_details

# ==========================================
# 5. 5-Level Partitioning Module
# ==========================================
def assign_quota_bucket(family_size: int) -> str:
    if family_size <= 2: return "small_1_2"
    elif 3 <= family_size <= 5: return "medium_3_5"
    elif 6 <= family_size <= 10: return "large_6_10"
    else: return "xlarge_ge11"

def max_take_from_family(family_size: int) -> int:
    return min(math.floor(family_size / 3), 15)

def sample_xlarge_with_cap(df_bucket: pd.DataFrame, target_n: int, rng: random.Random) -> pd.DataFrame:
    selected_ids, family_taken = [], {}
    for _, row in df_bucket.sample(frac=1, random_state=rng.randint(1, 10**9)).iterrows():
        fid, fsize = row["family_id"], int(row["family_size"])
        if family_taken.get(fid, 0) < max_take_from_family(fsize):
            selected_ids.append(row["ID"])
            family_taken[fid] = family_taken.get(fid, 0) + 1
        if len(selected_ids) >= target_n: break
    return df_bucket[df_bucket["ID"].isin(selected_ids)].copy()

def sample_bucket_with_quota(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    df = df.copy()
    df["quota_bucket"] = df["family_size"].apply(assign_quota_bucket)
    selected_parts, remaining_total, leftover_pool = [], FULL_TEST_N, []
    
    for bucket, target_n in QUOTA_BUCKETS.items():
        bucket_df = df[df["quota_bucket"] == bucket].copy()
        if bucket == "xlarge_ge11":
            if len(bucket_df) > 0:
                max_poss = sum(min(len(g), max_take_from_family(int(g["family_size"].iloc[0]))) for _, g in bucket_df.groupby("family_id"))
                if max_poss <= target_n:
                    chosen_ids = []
                    for _, g in bucket_df.groupby("family_id"):
                        chosen_ids.extend(g["ID"].tolist()[:min(len(g), max_take_from_family(int(g["family_size"].iloc[0])))])
                    chosen = bucket_df[bucket_df["ID"].isin(chosen_ids)].copy()
                else:
                    chosen = sample_xlarge_with_cap(bucket_df, target_n, rng)
            else: chosen = bucket_df.copy()
            selected_parts.append(chosen)
            remaining_total -= len(chosen)
            leftover_pool.extend(bucket_df.loc[~bucket_df["ID"].isin(set(chosen["ID"].tolist())), "ID"].tolist())
        else:
            ids = bucket_df["ID"].tolist()
            chosen_ids = ids if len(ids) <= target_n else rng.sample(ids, target_n)
            chosen = bucket_df[bucket_df["ID"].isin(chosen_ids)].copy()
            selected_parts.append(chosen)
            remaining_total -= len(chosen)
            leftover_pool.extend([x for x in ids if x not in set(chosen_ids)])

    if remaining_total > 0:
        selected_parts.append(df[df["ID"].isin(rng.sample(leftover_pool, remaining_total))].copy())
    return pd.concat(selected_parts, axis=0).drop_duplicates(subset=["ID"]).copy()

def build_random_splits(table_path: Path, tag_prefix: str):
    df = pd.read_csv(table_path)
    all_ids = df["ID"].astype(str).str.strip().tolist()
    for s in RANDOM_SEEDS:
        test_ids_r = random.Random(s).sample(all_ids, FULL_TEST_N)
        train_ids_r = [i for i in all_ids if i not in set(test_ids_r)]
        write_list(OUT_SPLITS / f"{tag_prefix}_random_s{s}/train_pos.txt", [str(POS_NPZ_DIR / f"feature_{i}.npz") for i in train_ids_r])
        write_list(OUT_SPLITS / f"{tag_prefix}_random_s{s}/test_pos.txt", [str(POS_NPZ_DIR / f"feature_{i}.npz") for i in test_ids_r])

def build_random_quota_splits(table_path: Path, tag_prefix: str):
    df = pd.read_csv(table_path)
    df["ID"], df["seq"] = df["ID"].astype(str).str.strip(), df["seq"].astype(str).str.strip().str.upper()
    for s in RANDOM_SEEDS:
        test_ids_r = sample_bucket_with_quota(df, random.Random(s))["ID"].tolist()
        train_ids_r = [i for i in df["ID"].tolist() if i not in set(test_ids_r)]
        write_list(OUT_SPLITS / f"{tag_prefix}_random_s{s}/train_pos.txt", [str(POS_NPZ_DIR / f"feature_{i}.npz") for i in train_ids_r])
        write_list(OUT_SPLITS / f"{tag_prefix}_random_s{s}/test_pos.txt", [str(POS_NPZ_DIR / f"feature_{i}.npz") for i in test_ids_r])

def build_lineage_splits(table_path: Path, tag_prefix: str):
    """Replicates Table 1 logic: Maximize lineage diversity, 1 representative per lineage."""
    df = pd.read_csv(table_path)
    df["ID"] = df["ID"].astype(str).str.strip()
    for s in RANDOM_SEEDS:
        rng = random.Random(s)
        families = df['family_id'].unique().tolist()
        rng.shuffle(families)
        test_ids = []
        for fid in families:
            f_ids = df[df['family_id'] == fid]['ID'].tolist()
            test_ids.append(rng.choice(f_ids))
            if len(test_ids) == FULL_TEST_N: break
        if len(test_ids) < FULL_TEST_N:
            rem = FULL_TEST_N - len(test_ids)
            leftover = df[~df["ID"].isin(test_ids)]["ID"].tolist()
            test_ids.extend(rng.sample(leftover, rem))
        train_ids = [i for i in df["ID"].tolist() if i not in set(test_ids)]
        write_list(OUT_SPLITS / f"{tag_prefix}_lineage_s{s}/train_pos.txt", [str(POS_NPZ_DIR / f"feature_{i}.npz") for i in train_ids])
        write_list(OUT_SPLITS / f"{tag_prefix}_lineage_s{s}/test_pos.txt", [str(POS_NPZ_DIR / f"feature_{i}.npz") for i in test_ids])

def build_lineage_quota_splits(table_path: Path, tag_prefix: str):
    df = pd.read_csv(table_path)
    df["ID"], df["seq"] = df["ID"].astype(str).str.strip(), df["seq"].astype(str).str.strip().str.upper()
    for s in RANDOM_SEEDS:
        test_ids = sample_bucket_with_quota(df, random.Random(s))["ID"].tolist()
        train_ids = [i for i in df["ID"].tolist() if i not in set(test_ids)]
        write_list(OUT_SPLITS / f"{tag_prefix}_lineage_quota_s{s}/train_pos.txt", [str(POS_NPZ_DIR / f"feature_{i}.npz") for i in train_ids])
        write_list(OUT_SPLITS / f"{tag_prefix}_lineage_quota_s{s}/test_pos.txt", [str(POS_NPZ_DIR / f"feature_{i}.npz") for i in test_ids])

def build_nonmr_splits():
    tr, te = _read_id_seq_csv(NONMR_TRAIN_CSV), _read_id_seq_csv(NONMR_TEST_CSV)
    all_valid = pd.concat([tr, te])
    all_valid = all_valid[~all_valid["seq"].str.startswith(DOMINANT_PREFIX)].copy()
    all_ids = all_valid["ID"].tolist()
    for s in RANDOM_SEEDS:
        test_ids = random.Random(s).sample(all_ids, NONMR_TEST_N)
        train_ids = [i for i in all_ids if i not in set(test_ids)]
        write_list(OUT_SPLITS / f"nonMR_lineage_s{s}/train_pos.txt", [str(POS_NPZ_DIR / f"feature_{i}.npz") for i in train_ids])
        write_list(OUT_SPLITS / f"nonMR_lineage_s{s}/test_pos.txt", [str(POS_NPZ_DIR / f"feature_{i}.npz") for i in test_ids])

# ==========================================
# 6. Zenodo Metadata Generation
# ==========================================
def generate_zenodo_csv():
    logger.info("📦 Compiling unified Zenodo dataset partitions...")
    df_main = pd.read_csv(FAMILY_TABLE_BASE)
    df_main.rename(columns={"sequence": "seq"}, inplace=True, errors="ignore")
    df_main["ID"] = df_main["ID"].astype(str).str.strip()
    df_zenodo = df_main[["ID", "seq", "family_id", "family_size"]].copy()
    
    def get_test_ids(folder_name):
        txt_path = OUT_SPLITS / folder_name / "test_pos.txt"
        if not txt_path.exists(): return set()
        with open(txt_path, 'r') as f:
            return {Path(line.strip()).stem.replace("feature_", "") for line in f if line.strip()}

    for s in RANDOM_SEEDS:
        df_zenodo[f"R_Base_Seed{s}"] = df_zenodo["ID"].apply(lambda x: "Test" if x in get_test_ids(f"base_random_s{s}") else "Train")
        df_zenodo[f"R_Quota_Seed{s}"] = df_zenodo["ID"].apply(lambda x: "Test" if x in get_test_ids(f"quota_random_s{s}") else "Train")
        df_zenodo[f"L_Base_Seed{s}"] = df_zenodo["ID"].apply(lambda x: "Test" if x in get_test_ids(f"base_lineage_s{s}") else "Train")
        df_zenodo[f"L_Quota_Seed{s}"] = df_zenodo["ID"].apply(lambda x: "Test" if x in get_test_ids(f"quota_lineage_quota_s{s}") else "Train")
        nonmr_ids = get_test_ids(f"nonMR_lineage_s{s}")
        def label_nonmr(seq_id):
            if seq_id in nonmr_ids: return "Test"
            elif not df_zenodo.loc[df_zenodo["ID"] == seq_id, "seq"].iloc[0].startswith(DOMINANT_PREFIX): return "Train"
            else: return "Excluded_MRPEIW"
        df_zenodo[f"nonMR_Seed{s}"] = df_zenodo["ID"].apply(label_nonmr)

    out_file = OUT_SPLITS / "CGFN_Dataset_Partitions.csv"
    df_zenodo.to_csv(out_file, index=False)
    logger.info(f"🎉 Zenodo dataset partition table generated: {out_file.resolve()}")
    return out_file

# ==========================================
# 7. Table 1 & Table 2 Analysis Module
# ==========================================
def compute_fast_identity(seq1, seq2):
    s = difflib.SequenceMatcher(None, seq1, seq2)
    matches = sum(triple.size for triple in s.get_matching_blocks())
    return matches / len(seq1)

def analyze_and_print_tables(zenodo_csv_path):
    print("\n" + "="*90)
    print("📊 Table 1. Summary of test set compositional characteristics")
    print("="*90)
    df = pd.read_csv(zenodo_csv_path)
    total_families = df['family_id'].nunique()
    settings = ["R_Base", "R_Quota", "L_Base", "L_Quota", "nonMR"]
    
    res_t1 = {}
    res_t2 = {}
    
    for setting in settings:
        metrics_t1 = {'positives': [], 'covered_groups': [], 'coverage_pct': [], 'mrpeiw_pct': [], 'non_mrpeiw_pct': [], 'small_pct': [], 'xlarge_pct': [], 'mean_length': [], 'median_size': []}
        metrics_t2 = {'median_nn': [], 'nn_80': [], 'nn_90': [], 'same_lineage': []}
        
        for s in RANDOM_SEEDS:
            col_name = f"{setting}_Seed{s}"
            if col_name not in df.columns: continue
            
            test_df = df[df[col_name] == 'Test']
            train_df = df[df[col_name] == 'Train']
            n_pos = len(test_df)
            if n_pos == 0: continue
            
            # --- T1 Metrics ---
            metrics_t1['positives'].append(n_pos)
            metrics_t1['covered_groups'].append(test_df['family_id'].nunique())
            metrics_t1['coverage_pct'].append(test_df['family_id'].nunique() / total_families * 100)
            n_mrpeiw = test_df['seq'].str.startswith(DOMINANT_PREFIX).sum()
            metrics_t1['mrpeiw_pct'].append(n_mrpeiw / n_pos * 100)
            metrics_t1['non_mrpeiw_pct'].append((n_pos - n_mrpeiw) / n_pos * 100)
            metrics_t1['small_pct'].append((test_df['family_size'] <= 2).sum() / n_pos * 100)
            metrics_t1['xlarge_pct'].append((test_df['family_size'] >= 11).sum() / n_pos * 100)
            metrics_t1['mean_length'].append(test_df['seq'].str.len().mean())
            metrics_t1['median_size'].append(test_df['family_size'].median())
            
            # --- T2 Metrics (NN Identity & Leakage) ---
            train_seqs = train_df['seq'].tolist()
            train_fams = train_df['family_id'].tolist()
            test_seqs = test_df['seq'].tolist()
            test_fams = test_df['family_id'].tolist()
            
            identities = []
            same_lineage = []
            
            for t_seq, t_fam in zip(test_seqs, test_fams):
                max_id = -1.0
                best_fam = None
                t_len = len(t_seq)
                for tr_seq, tr_fam in zip(train_seqs, train_fams):
                    if min(t_len, len(tr_seq)) / t_len <= max_id: continue
                    ident = compute_fast_identity(t_seq, tr_seq)
                    if ident > max_id:
                        max_id = ident
                        best_fam = tr_fam
                        if max_id == 1.0:
                            if tr_fam == t_fam: best_fam = tr_fam; break
                identities.append(max_id)
                same_lineage.append(1 if best_fam == t_fam else 0)
                
            metrics_t2['median_nn'].append(np.median(identities))
            metrics_t2['nn_80'].append(np.mean(np.array(identities) >= 0.80) * 100)
            metrics_t2['nn_90'].append(np.mean(np.array(identities) >= 0.90) * 100)
            metrics_t2['same_lineage'].append(np.mean(same_lineage) * 100)

        # Aggregate T1
        agg_t1 = {}
        for k, v in metrics_t1.items():
            agg_t1[k] = f"{np.mean(v):.1f} ± {np.std(v, ddof=0):.1f}" if v else "N/A"
        res_t1[setting] = agg_t1
        
        # Aggregate T2
        agg_t2 = {}
        if len(metrics_t2['median_nn']) > 0:
            agg_t2['median_nn'] = f"{np.mean(metrics_t2['median_nn']):.4f}" 
            agg_t2['nn_80'] = f"{np.mean(metrics_t2['nn_80']):.2f} ± {np.std(metrics_t2['nn_80'], ddof=0):.2f}"
            agg_t2['nn_90'] = f"{np.mean(metrics_t2['nn_90']):.2f} ± {np.std(metrics_t2['nn_90'], ddof=0):.2f}"
            agg_t2['same_lineage'] = f"{np.mean(metrics_t2['same_lineage']):.2f} ± {np.std(metrics_t2['same_lineage'], ddof=0):.2f}"
        else:
            for k in metrics_t2: agg_t2[k] = "N/A"
        res_t2[setting] = agg_t2

    # Print T1
    labels_t1 = ["Test positives, n", "Covered lineage groups, n", "Lineage coverage, %", "MRPEIW positives, %", "non-MRPEIW positives, %", "Small-lineage positives, %", "Extra large-lineage pos, %", "Mean length", "Median lineage size"]
    keys_t1 = ['positives', 'covered_groups', 'coverage_pct', 'mrpeiw_pct', 'non_mrpeiw_pct', 'small_pct', 'xlarge_pct', 'mean_length', 'median_size']
    print(f"{'Settings':<35} | {'R_Base':<12} | {'R_Quota':<12} | {'L_Base':<12} | {'L_Quota':<12} | {'nonMR_lineage':<12}")
    print("-" * 105)
    for key, label in zip(keys_t1, labels_t1):
        print(f"{label:<35} | {res_t1['R_Base'].get(key):<12} | {res_t1['R_Quota'].get(key):<12} | {res_t1['L_Base'].get(key):<12} | {res_t1['L_Quota'].get(key):<12} | {res_t1['nonMR'].get(key):<12}")
    
    print("\n" + "="*90)
    print("📊 Table 2. Summary of training-test proximity and leakage risk")
    print("="*90)
    labels_t2 = ["Median NN identity", "NN identity >= 0.80, %", "NN identity >= 0.90, %", "Same-lineage NN rate, %"]
    keys_t2 = ['median_nn', 'nn_80', 'nn_90', 'same_lineage']
    print(f"{'Settings':<30} | {'R_Base':<15} | {'R_Quota':<15} | {'L_Base':<15} | {'L_Quota':<15} | {'nonMR_lineage':<15}")
    print("-" * 105)
    for key, label in zip(keys_t2, labels_t2):
        print(f"{label:<30} | {res_t2['R_Base'].get(key):<15} | {res_t2['R_Quota'].get(key):<15} | {res_t2['L_Base'].get(key):<15} | {res_t2['L_Quota'].get(key):<15} | {res_t2['nonMR'].get(key):<15}")
    print("="*90)

# ==========================================
# 8. Main Entry Point
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Run 5-level CGFN dataset partition pipeline.")
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Root directory containing input CSVs and features.")
    args = parser.parse_args()

    # Dynamically update global paths based on argparse
    global DATA_ROOT, POS_DATA_CSV, POS_NPZ_DIR, OUT_SPLITS, ANALYSIS_DIR
    global FAMILY_SPLIT_DIR, FAMILY_TABLE_BASE, NONMR_TRAIN_CSV, NONMR_TEST_CSV

    DATA_ROOT = Path(args.data_dir)
    POS_DATA_CSV = DATA_ROOT / "all_Positive.csv"
    POS_NPZ_DIR = DATA_ROOT / "features" / "positive"

    OUT_SPLITS = DATA_ROOT / "splits_lineage"
    ANALYSIS_DIR = OUT_SPLITS / "lineage_analysis"
    FAMILY_SPLIT_DIR = OUT_SPLITS / "balanced_split_62out"

    FAMILY_TABLE_BASE = FAMILY_SPLIT_DIR / "all_sequences_by_lineage.csv"
    NONMR_TRAIN_CSV = FAMILY_SPLIT_DIR / "balanced_train_set.csv"
    NONMR_TEST_CSV = FAMILY_SPLIT_DIR / "balanced_test_set.csv"

    OUT_SPLITS.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("🚀 [PHASE 1] Running Functional Mutational Lineage identification and balanced allocation")
    logger.info("="*60)
    if not POS_DATA_CSV.exists():
        logger.error(f"Input file not found: {POS_DATA_CSV}. Please ensure data is placed correctly.")
        return
        
    complete_family_analysis_and_balanced_split(str(POS_DATA_CSV), FULL_TEST_N, str(ANALYSIS_DIR), str(FAMILY_SPLIT_DIR))
    
    logger.info("="*60)
    logger.info("🚀 [PHASE 2] Executing 5-Level Evaluation * 5 Seed Partitions")
    logger.info("="*60)
    build_random_splits(FAMILY_TABLE_BASE, "base")
    build_random_quota_splits(FAMILY_TABLE_BASE, "quota")
    build_lineage_splits(FAMILY_TABLE_BASE, "base")
    build_lineage_quota_splits(FAMILY_TABLE_BASE, "quota")
    build_nonmr_splits()

    logger.info("="*60)
    logger.info("🚀 [PHASE 3] Compiling unified Zenodo export table")
    logger.info("="*60)
    zenodo_file = generate_zenodo_csv()
    
    analyze_and_print_tables(zenodo_file)
    logger.info("✅ End-to-end pipeline execution completed successfully!")

if __name__ == "__main__":
    # Ensure absolute reproducibility for publication
    random.seed(42) 
    np.random.seed(42)
    main()