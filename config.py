#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration File for Confidence-aware Gated Fusion Network (CGFN)
(Publication-Ready Version)
"""

# ==================== Data Paths (Multi-view Aligned Lists) ====================
# NOTE: Please ensure your dataset is placed in the './data/' directory, 
# or modify the paths below to match your local environment.

train_pos_esm = "./data/splits/nonMR_lineage/train_pos.txt"
train_pos_aaindex = "./data/splits/nonMR_lineage/train_pos_aaindex.txt"
test_pos_esm = "./data/splits/nonMR_lineage/test_pos.txt"
test_pos_aaindex = "./data/splits/nonMR_lineage/test_pos_aaindex.txt"

train_neg_esm = "./data/negative_lists/train_neg_pool_esm2.txt"
train_neg_aaindex = "./data/negative_lists/train_neg_pool_aaindex.txt"
test_neg_esm = "./data/negative_lists/fixed_test_neg_70_esm2.txt"
test_neg_aaindex = "./data/negative_lists/fixed_test_neg_70_aaindex.txt"

model_dir = "./results"

# ==================== Global Experimental Settings ====================
random_seed = 1024
verbose = True
prediction_batch_size = 96

# ==================== CNN Branch Parameters (Evolutionary View) ====================
epochs = 45               
final_epochs = 150         
batch_size = 32
learning_rate = 0.001       
cnn_learning_rate = 0.001  
cnn_l2_regularization = 0.003
cnn_dropout_rate = 0.4

# ==================== GRU Branch Parameters (Biophysical View) ====================
gru_epochs = 120
gru_batch_size = 64
gru_initial_lr = 0.0005
gru_l2_reg = 0.0001
gru_class_weight = {0: 1.0, 1: 1.0}

gru_lr_schedule = {
    'warmup_epochs': 5,
    'decay_start_epoch': 20,    
    'decay_factor': 0.8,
    'min_lr': 1e-6
}

# ==================== 7-Dim Gated Fusion Parameters ====================
fusion_epochs = 50
fusion_batch_size = 64
fusion_learning_rate = 0.0003
fusion_units = 64
fusion_dropout = 0.3  
gate_l2_reg = 0.0005  

# ==================== Visualization & Debugging Tools ====================
enable_fusion_debug = True
max_debug_samples = 100
debug_sample_types = ['cnn_wrong_gru_right', 'both_wrong', 'cnn_right_gru_wrong']

# Heatmap generation is currently bypassed in the dual-stream architecture script
generate_heatmaps = False  
heatmap_method = "occlusion"
heatmap_max_total_samples = 10
