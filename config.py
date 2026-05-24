#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimized Configuration File - Includes Dynamic Strategy Switches and Data List Support 
(Publication-Ready Version)
"""

# ==================== Data Paths ====================
# [IMPORTANT FOR REPRODUCIBILITY] 
# The raw dataset and feature files are hosted on Zenodo due to GitHub's file size limits.
# Please download the dataset from Zenodo (DOI: 10.5281/zenodo.XXXXXXX) 
# and extract it into the 'data/' directory at the repository root before running.

train_positive_dir = "data/partitions/train/positive/"
train_negative_dir = "data/partitions/train/negative/"
test_positive_dir  = "data/partitions/test/positive/"
test_negative_dir  = "data/partitions/test/negative/"

# Output Directory for trained models, metrics, and heatmaps
model_dir = "results/model_output"

# Metadata lists for the Lineage-isolated 5-level evaluation framework
train_positive_list = "data/metadata/full_lineage_base/train_pos.txt"
test_positive_list  = "data/metadata/full_lineage_base/test_pos.txt"
test_negative_list  = "data/metadata/full_lineage_base/fixed_test_neg_100.txt"

# ==================== Advanced Training & Evaluation Strategies ====================
# 1. Repeated CV Stacking (Controls GRU Bagging iterations)
use_repeated_cv_stacking = True
repeated_cv_runs = 3  # Effective only when use_repeated_cv_stacking is True; otherwise forced to 1

# 2. Confidence Locking
use_confidence_locking = True
confidence_lock_threshold = 0.98

# 3. Optimal Threshold Search
use_optimal_threshold = True

# ==================== Basic Parameters ====================
epochs = 45
final_epochs = 150
batch_size = 32
learning_rate = 0.001
random_seed = 2024
verbose = True
kfold_splits = 5
prediction_batch_size = 96
final_early_stopping_patience = 15

# ==================== CNN Parameters ====================
cnn_learning_rate = 0.001
cnn_l2_regularization = 0.003
cnn_dropout_rate = 0.4

# ==================== GRU Parameters ====================
gru_units_large = (64, 32)  
gru_l2_reg = 0.0005      
gru_dropout = 0.3        
gru_class_weight = {0: 1.0, 1: 1.0} 
gru_initial_lr = 0.0003   
gru_batch_size = 64      
gru_epochs = 120         
gru_augmentation_factor = 1.0 
gru_patience = 40
gru_lr_schedule = {
    'warmup_epochs': 5,
    'decay_start_epoch': 30,
    'decay_factor': 0.8,
    'min_lr': 1e-6
}

# ==================== Fusion Parameters ====================
fusion_units = 64
fusion_dropout = 0.3  
fusion_learning_rate = 0.0005  
fusion_epochs = 50

gate_activation = 'sigmoid'
gate_l2_reg = 0.0005  
feature_projection_units = 64  

use_learnable_prior = True
initial_cnn_prior = 0.0
initial_gru_prior = 0.0

ema_alpha = 0.9  
ema_clip_range = 0.2  

reliability_weights = {
    'agreement': 0.2,
    'confidence': 0.5,
    'feature_quality': 0.3
}

fusion_strategy = {
    'early_stage': {'gate_weight': 0.7, 'reliability_weight': 0.3},
    'mid_stage': {'gate_weight': 0.5, 'reliability_weight': 0.5},
    'late_stage': {'gate_weight': 0.3, 'reliability_weight': 0.7},
    'inference': {'gate_weight': 0.3, 'reliability_weight': 0.7}
}

relative_advantage_boost = 0.15

# ==================== Training Schedule & Data Split ====================
fusion_early_stopping_patience = 25
fusion_reduce_lr_patience = 6
fusion_reduce_lr_factor = 0.7

fusion_train_ratio = 0.0 
min_fusion_samples = 50   
max_fusion_ratio = 0.4    
min_base_samples = 150    
use_cv_fusion = True    

# ==================== Other Parameters ====================
heatmap_method = "occlusion"
heatmap_max_total_samples = 10
enable_fusion_debug = True
max_debug_samples = 100
debug_sample_types = ['cnn_wrong_gru_right', 'both_wrong', 'cnn_right_gru_wrong']
confidence_threshold = 0.7
min_confidence_gap = 0.2

use_meta_learner = False
meta_learner_units = 64
meta_learner_lr = 0.0001
use_adversarial_training = False