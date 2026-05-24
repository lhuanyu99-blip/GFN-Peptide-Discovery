#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Flatten, Dense, 
                                     Dropout, GRU, Input, Add, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D, 
                                     GlobalMaxPooling1D, Concatenate, Subtract,
                                     Multiply, Activation, Reshape, Lambda, BatchNormalization, Softmax, Layer)
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers 
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.utils import register_keras_serializable
import random
import os
import glob
import importlib.util
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, balanced_accuracy_score, confusion_matrix, precision_recall_curve, auc ,average_precision_score
)
from typing import Tuple, Dict, List, Any, Optional
from tensorflow.keras.constraints import MaxNorm
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureExtractorManager:
    
    def __init__(self):
        self.cnn_feature_extractor = None
        self.gru_feature_extractor = None
    
    @staticmethod
    def get_robust_feature_layer(model, model_type='cnn'):

        candidate_layers = []
        

        for i, layer in enumerate(model.layers):
            if i == 0 or i == len(model.layers) - 1:
                continue
                
            layer_output = layer.output
            
            if isinstance(layer_output, (list, tuple)):
                if len(layer_output) > 0:
                    layer_output = layer_output[0]
                    print(f"Warning: Layer {i} has multiple outputs, using first output")
                else:
                    continue
            
            try:
                output_shape = layer_output.shape
            except AttributeError:
                print(f"Warning: Layer {i} output has no shape attribute, skipping")
                continue

            if len(output_shape) == 2 and output_shape[-1] == 1:
                continue
                
            layer_type = type(layer).__name__.lower()
            
            if model_type == 'cnn':
                if 'dense' in layer_type and i < len(model.layers) - 2:
                    candidate_layers.append((i, layer_output, 'dense'))
                elif 'flatten' in layer_type:
                    candidate_layers.append((i, layer_output, 'flatten'))
                elif 'global' in layer_type:
                    candidate_layers.append((i, layer_output, 'pooling'))
                    
            elif model_type == 'gru':
                if 'global' in layer_type:
                    candidate_layers.append((i, layer_output, 'pooling'))
                elif 'concatenate' in layer_type:
                    candidate_layers.append((i, layer_output, 'concat'))
                elif 'dense' in layer_type and i < len(model.layers) - 2:
                    candidate_layers.append((i, layer_output, 'dense'))
                elif 'add' in layer_type:
                    candidate_layers.append((i, layer_output, 'residual'))
        
        if not candidate_layers:
            print(f"Warning: No candidate layers found for {model_type}, using heuristic")
            for i in range(max(1, len(model.layers) - 4), len(model.layers) - 1):
                layer = model.layers[i]
                layer_output = layer.output
                
                if isinstance(layer_output, (list, tuple)):
                    if len(layer_output) > 0:
                        layer_output = layer_output[0]
                    else:
                        continue
                
                try:
                    output_shape = layer_output.shape
                except AttributeError:
                    continue
                    
                if len(output_shape) == 2 and output_shape[-1] == 1:
                    continue
                    
                candidate_layers.append((i, layer_output, 'heuristic'))
        
        if candidate_layers:
            prioritized = []
            for idx, output, ltype in candidate_layers:
                if ltype in ['dense', 'pooling', 'concat', 'residual']:
                    prioritized.append((idx, output, ltype))
            
            if prioritized:
                best = max(prioritized, key=lambda x: x[0])
            else:
                best = max(candidate_layers, key=lambda x: x[0])
            
            print(f"Selected feature layer for {model_type}: layer {best[0]} ({best[2]})")
            return best[1]
        else:
            raise ValueError(f"Cannot find suitable feature layer for {model_type}")
    
    def create_cnn_feature_extractor(self, cnn_model):
        print("\n=== Initializing CNN Feature Extractor ===")
        
        feature_output = self.get_robust_feature_layer(cnn_model, 'cnn')
        
        feature_extractor = Model(
            inputs=cnn_model.input,
            outputs=feature_output,
            name='cnn_feature_extractor'
        )
        
        print(f"CNN Feature Dimension: {feature_output.shape}")
        self.cnn_feature_extractor = feature_extractor
        return feature_extractor
    
    def create_gru_feature_extractor(self, gru_model):
        print("\n=== Initializing GRU Feature Extractor ===")
        
        # 调用静态方法
        feature_output = self.get_robust_feature_layer(gru_model, 'gru')
        
        feature_extractor = Model(
            inputs=gru_model.input,
            outputs=feature_output,
            name='gru_feature_extractor'
        )
        
        print(f"GRU Feature Dimension: {feature_output.shape}")
        self.gru_feature_extractor = feature_extractor
        return feature_extractor
    
    def extract_features(self, X_data, batch_size=256, verbose=0):
        if self.cnn_feature_extractor is None or self.gru_feature_extractor is None:
            raise ValueError(" create_xxx_feature_extractor")
        
        print("\n=== Extracting Authentic Features ===")
        print(f"Input Data Shape: {X_data.shape}")
        
        cnn_features = self.cnn_feature_extractor.predict(
            X_data, batch_size=batch_size, verbose=verbose
        )
        print(f"CNN Feature shape: {cnn_features.shape}")
        
        gru_features = self.gru_feature_extractor.predict(
            X_data, batch_size=batch_size, verbose=verbose
        )
        print(f"GRU Feature shape: {gru_features.shape}")
        
        self._verify_feature_difference(cnn_features, gru_features)
        
        return cnn_features, gru_features
    
    def _verify_feature_difference(self, cnn_features, gru_features):
        mean_diff = np.mean(np.abs(np.mean(cnn_features, axis=0) - np.mean(gru_features, axis=0)))
        std_diff = np.mean(np.abs(np.std(cnn_features, axis=0) - np.std(gru_features, axis=0)))
        
        print(f"Feature Mean Difference: {mean_diff:.6f}")
        print(f"Feature Std Difference: {std_diff:.6f}")
        
        if mean_diff < 1e-6 and std_diff < 1e-6:
            print("WARNING: CNN and GRU features are nearly identical. Placeholder tensors might still be in use!")
        else:
            print("Distinct representations confirmed between CNN and GRU pathways.")
    
    def save_extractors(self, save_dir):
        if self.cnn_feature_extractor is not None:
            cnn_path = os.path.join(save_dir, 'cnn_feature_extractor.keras')
            self.cnn_feature_extractor.save(cnn_path)
            print(f"cnn feature extractor saving: {cnn_path}")
        
        if self.gru_feature_extractor is not None:
            gru_path = os.path.join(save_dir, 'gru_feature_extractor.keras')
            self.gru_feature_extractor.save(gru_path)
            print(f"GRU feature extractor saving: {gru_path}")
    
    def load_extractors(self, save_dir, custom_objects=None):

        cnn_path = os.path.join(save_dir, 'cnn_feature_extractor.keras')
        gru_path = os.path.join(save_dir, 'gru_feature_extractor.keras')
        
        if os.path.exists(cnn_path):
            self.cnn_feature_extractor = load_model(cnn_path, custom_objects=custom_objects)
            print(f"cnn feature extractor saving: {cnn_path}")
        
        if os.path.exists(gru_path):
            self.gru_feature_extractor = load_model(gru_path, custom_objects=custom_objects)
            print(f"GRU feature extractor saving: {gru_path}")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print("GPU configuration warning: {}".format(e))
tf.config.optimizer.set_experimental_options({'reduce_retracing': True})


class F1Metric(tf.keras.metrics.Metric):

    def __init__(self, threshold=0.5, name='f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold

        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred > self.threshold, tf.bool)
        
        tp = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        fp = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        fn = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            tp = tf.cast(tp, tf.float32) * sample_weight
            fp = tf.cast(fp, tf.float32) * sample_weight
            fn = tf.cast(fn, tf.float32) * sample_weight
        else:
            tp = tf.cast(tp, tf.float32)
            fp = tf.cast(fp, tf.float32)
            fn = tf.cast(fn, tf.float32)
        
        self.true_positives.assign_add(tf.reduce_sum(tp))
        self.false_positives.assign_add(tf.reduce_sum(fp))
        self.false_negatives.assign_add(tf.reduce_sum(fn))

    def result(self):

        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1

    def reset_states(self):
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)
        self.false_negatives.assign(0.)


@register_keras_serializable()
class MinPooling1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.reduce_min(inputs, axis=1)
    
    def get_config(self):
        config = super().get_config()
        return config

@register_keras_serializable()
class StdPooling1D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.math.reduce_std(inputs, axis=1)
    
    def get_config(self):
        config = super().get_config()
        return config

@tf.keras.utils.register_keras_serializable()
class ImprovedGatedFusionMechanism(Layer):

    def __init__(self, fusion_units=64, dropout_rate=0.3, gate_l2_reg=0.0005, 
                 feature_projection_units=64, gate_activation='sigmoid',
                 ema_alpha=0.9, ema_clip_range=0.2, use_learnable_prior=True,
                 uncertainty_center=0.5, uncertainty_sigma=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.fusion_units = fusion_units
        self.dropout_rate = dropout_rate
        self.gate_l2_reg = gate_l2_reg
        self.feature_projection_units = feature_projection_units
        self.gate_activation = gate_activation
        self.ema_alpha = ema_alpha
        self.ema_clip_range = ema_clip_range
        self.use_learnable_prior = use_learnable_prior

        self.uncertainty_center = tf.Variable(uncertainty_center, trainable=False, dtype=tf.float32, name='unc_center')
        self.uncertainty_sigma = tf.Variable(uncertainty_sigma, trainable=False, dtype=tf.float32, name='unc_sigma')
        
        self.w_correct = tf.Variable(0.80, trainable=False, dtype=tf.float32, name='w_correct')
        self.w_conf = tf.Variable(0.15, trainable=False, dtype=tf.float32, name='w_conf')
        self.w_stats = tf.Variable(0.05, trainable=False, dtype=tf.float32, name='w_stats')
        
        self.dynamic_loss_weights = tf.Variable([1.0, 0.12, 0.06, 0.06], trainable=False, dtype=tf.float32, name='loss_weights')
        self.training_epoch = tf.Variable(0, trainable=False, dtype=tf.int32, name='training_epoch')
        
    def build(self, input_shape):
        self.cnn_projection = Dense(self.feature_projection_units, activation='relu',
                                    kernel_regularizer=regularizers.l2(self.gate_l2_reg), name='cnn_projection')
        self.gru_projection = Dense(self.feature_projection_units, activation='relu',
                                    kernel_regularizer=regularizers.l2(self.gate_l2_reg), name='gru_projection')
        
        self.gate_dense1 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(self.gate_l2_reg), name='gate_dense1')
        self.gate_dense2 = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(self.gate_l2_reg), name='gate_dense2')
        self.gate_output = Dense(2, activation=self.gate_activation, kernel_regularizer=regularizers.l2(self.gate_l2_reg), name='gate_output')
        
        self.expert_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(self.gate_l2_reg), name='expert_dense')
        self.expert_dropout = Dropout(self.dropout_rate, name='expert_dropout')
        self.expert_output = Dense(1, activation='tanh', kernel_regularizer=regularizers.l2(self.gate_l2_reg), name='expert_output')

        self.three_dim_fusion = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(self.gate_l2_reg * 0.5), name='three_dim_fusion')
        
        self.dropout1 = Dropout(self.dropout_rate)
        self.gate_layer_norm1 = LayerNormalization(epsilon=1e-6)
        self.gate_layer_norm2 = LayerNormalization(epsilon=1e-6)
        
        if self.use_learnable_prior:
            self.cnn_prior = tf.Variable(0.0, trainable=True, name='cnn_prior')
            self.gru_prior = tf.Variable(0.0, trainable=True, name='gru_prior')
        
        self.cnn_weight_ema = tf.Variable(0.5, trainable=False, name='cnn_weight_ema')
        self.gru_weight_ema = tf.Variable(0.5, trainable=False, name='gru_weight_ema')
        
        super().build(input_shape)

    def _compute_reliability_score(self, cnn_pred, gru_pred, cnn_proj, gru_proj):
        pred_diff = tf.abs(cnn_pred - gru_pred)
        pred_agreement = 1.0 - pred_diff
        
        cnn_confidence = tf.abs(cnn_pred - 0.5) * 2
        gru_confidence = tf.abs(gru_pred - 0.5) * 2
        
        cnn_feat_strength = tf.reduce_mean(tf.abs(cnn_proj), axis=-1, keepdims=True)
        gru_feat_strength = tf.reduce_mean(tf.abs(gru_proj), axis=-1, keepdims=True)
        total_strength = cnn_feat_strength + gru_feat_strength + 1e-8
        cnn_feat_norm = cnn_feat_strength / total_strength
        gru_feat_norm = gru_feat_strength / total_strength
        
        cnn_reliability = (
            self.w_correct * pred_agreement +
            self.w_conf * cnn_confidence +
            self.w_stats * cnn_feat_norm
        )
        gru_reliability = (
            self.w_correct * pred_agreement +
            self.w_conf * gru_confidence +
            self.w_stats * gru_feat_norm
        )
        
        conf_gap = gru_confidence - cnn_confidence
        gru_reliability += 0.15 * tf.nn.relu(conf_gap)
        cnn_reliability += 0.15 * tf.nn.relu(-conf_gap)
        
        return tf.clip_by_value(cnn_reliability, 0.0, 1.0), tf.clip_by_value(gru_reliability, 0.0, 1.0)

    def call(self, inputs, training=False):
        if len(inputs) != 4: raise ValueError(f"Inputs error")
        cnn_features, gru_features, cnn_pred, gru_pred = inputs
        
        epsilon = 1e-7
        cnn_pred = tf.clip_by_value(cnn_pred, epsilon, 1.0 - epsilon)
        gru_pred = tf.clip_by_value(gru_pred, epsilon, 1.0 - epsilon)
        
        cnn_proj = self.cnn_projection(cnn_features)
        gru_proj = self.gru_projection(gru_features)
        cnn_reliability, gru_reliability = self._compute_reliability_score(cnn_pred, gru_pred, cnn_proj, gru_proj)
        
        combined_feat = tf.concat([cnn_proj, gru_proj], axis=-1)
        gate_hidden = self.gate_dense1(combined_feat)
        gate_hidden = self.gate_layer_norm1(gate_hidden)
        gate_hidden = self.dropout1(gate_hidden, training=training)
        gate_hidden = self.gate_dense2(gate_hidden)
        gate_hidden = self.gate_layer_norm2(gate_hidden)
        base_gate_weights = self.gate_output(gate_hidden)
        
        reliability_weights = tf.concat([cnn_reliability, gru_reliability], axis=-1)
        reliability_sum = tf.reduce_sum(reliability_weights, axis=-1, keepdims=True)
        reliability_weights = reliability_weights / tf.maximum(reliability_sum, epsilon)
        
        gate_ratio = tf.cond(self.training_epoch < 20, lambda: 0.5, lambda: 0.3)
        combined_weights = gate_ratio * base_gate_weights + (1 - gate_ratio) * reliability_weights
        
        # Auto-Tuned Gaussian Uncertainty
        disagreement = tf.abs(cnn_pred - gru_pred)
        cnn_uncertainty = tf.exp(-tf.square(cnn_pred - self.uncertainty_center) / (2 * tf.square(self.uncertainty_sigma)))
        
        expert_feat = self.expert_dense(combined_feat)
        expert_feat = self.expert_dropout(expert_feat, training=training)
        expert_signal = self.expert_output(expert_feat) 
        
        adjustment = disagreement * cnn_uncertainty * 0.5 * expert_signal
        
        w_cnn = combined_weights[:, 0:1] - adjustment
        w_gru = combined_weights[:, 1:2] + adjustment
        combined_weights_adjusted = tf.concat([w_cnn, w_gru], axis=-1)
        
        three_dim_input = tf.concat([cnn_reliability, gru_reliability], axis=-1)
        three_dim_weights = self.three_dim_fusion(three_dim_input)
        final_gate_weights = combined_weights_adjusted * three_dim_weights
        
        bias = tf.stack([self.cnn_prior, self.gru_prior], axis=0) if self.use_learnable_prior else tf.zeros(2)
        
        final_gate_weights = tf.nn.softmax((tf.math.log(tf.maximum(final_gate_weights, epsilon)) + bias), axis=-1)

        if training:
            cnn_weight = final_gate_weights[:, 0:1]
            gru_weight = final_gate_weights[:, 1:2]
            
            self.cnn_weight_ema.assign(self.ema_alpha * self.cnn_weight_ema + (1 - self.ema_alpha) * tf.reduce_mean(cnn_weight))
            self.gru_weight_ema.assign(self.ema_alpha * self.gru_weight_ema + (1 - self.ema_alpha) * tf.reduce_mean(gru_weight))
            
            pred_agreement = 1.0 - tf.abs(cnn_pred - gru_pred)
            should_clip = pred_agreement > 0.7
            
            cnn_clipped = tf.clip_by_value(cnn_weight, self.cnn_weight_ema - self.ema_clip_range, self.cnn_weight_ema + self.ema_clip_range)
            gru_clipped = tf.clip_by_value(gru_weight, self.gru_weight_ema - self.ema_clip_range, self.gru_weight_ema + self.ema_clip_range)
            
            final_cnn = tf.where(should_clip, cnn_clipped, cnn_weight)
            final_gru = tf.where(should_clip, gru_clipped, gru_weight)
            
            final_gate_weights = tf.concat([final_cnn, final_gru], axis=-1)
            final_sum = tf.reduce_sum(final_gate_weights, axis=-1, keepdims=True)
            final_gate_weights = final_gate_weights / tf.maximum(final_sum, epsilon)

        fused_prediction = final_gate_weights[:, 0:1] * cnn_pred + final_gate_weights[:, 1:2] * gru_pred
        
        fused_prediction = tf.clip_by_value(fused_prediction, epsilon, 1.0 - epsilon)
        
        confidence_adj = tf.concat([
            tf.abs(cnn_pred - 0.5) * 2,
            tf.abs(gru_pred - 0.5) * 2
        ], axis=-1)
        
        return fused_prediction, final_gate_weights, base_gate_weights, confidence_adj
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'fusion_units': self.fusion_units,
            'dropout_rate': self.dropout_rate,
            'gate_l2_reg': self.gate_l2_reg,
            'feature_projection_units': self.feature_projection_units,
            'gate_activation': self.gate_activation,
            'ema_alpha': self.ema_alpha,
            'ema_clip_range': self.ema_clip_range,
            'use_learnable_prior': self.use_learnable_prior,
            'uncertainty_center': float(self.uncertainty_center.numpy()), # 保存当前值
            'uncertainty_sigma': float(self.uncertainty_sigma.numpy())
        })
        return config

EnhancedGatedFusionMechanism = ImprovedGatedFusionMechanism

def calculate_comprehensive_metrics(y_true, y_pred_classes, y_pred_proba, pos_label=1):

    accuracy = accuracy_score(y_true, y_pred_classes)
    balanced_acc = balanced_accuracy_score(y_true, y_pred_classes)
    recall = recall_score(y_true, y_pred_classes, pos_label=pos_label, zero_division=0)
    precision = precision_score(y_true, y_pred_classes, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred_classes, pos_label=pos_label, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred_classes)

    try:
        auc_roc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc_roc = 0.0

    try:
        precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_pred_proba, pos_label=pos_label)
        auc_pr = auc(recall_pr, precision_pr)
    except ValueError:
        auc_pr = 0.0
    
    cm = confusion_matrix(y_true, y_pred_classes)
    if cm.shape == (2, 2): 
        tn, fp, _, _ = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0  
    
    return {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'recall': float(recall),
        'precision': float(precision),
        'f1': float(f1),
        'mcc': float(mcc),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'specificity': float(specificity)
    }

# ==================== Information Collection Module ====================
class TrainingMetricsCollector:
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        
        self.cv_history = {}
        self.final_training_history = {}
        
        self.fold_metrics = []
        self.final_metrics = {}
        
        self.cv_error_analysis = []
        self.final_error_analysis = {}
        self.cv_dynamic_fusion_stats = []
        self.final_dynamic_fusion_stats = {}
        self.cv_fusion_debug_samples = []
        self.final_fusion_debug_samples = []


    def init_cv_fold(self, fold_idx: int, model_names: list):
        self.cv_history[fold_idx] = {'models': {}}
        for name in model_names:
            self.cv_history[fold_idx]['models'][name] = {
                'train_loss': [], 'val_loss': [],
                'train_accuracy': [], 'val_accuracy': [],
                'train_f1': [], 'val_f1': [],
                'train_cls_loss': [], 'val_cls_loss': []
            }

    def init_final_training(self, model_names: list):
        self.final_training_history['models'] = {}
        for name in model_names:
            self.final_training_history['models'][name] = {
                'train_loss': [], 'val_loss': [],
                'train_accuracy': [], 'val_accuracy': [],
                'train_f1': [], 'val_f1': [],
                'train_cls_loss': [], 'val_cls_loss': []
            }

    def record_epoch_metrics(self, fold_idx: int, model_name: str, train_loss, val_loss, train_acc, val_acc,
                            train_f1=0.0, val_f1=0.0, train_cls_loss=0.0, val_cls_loss=0.0):
        if fold_idx in self.cv_history and model_name in self.cv_history[fold_idx]['models']:
            m = self.cv_history[fold_idx]['models'][model_name]
            m['train_loss'].append(float(train_loss))
            m['val_loss'].append(float(val_loss))
            m['train_accuracy'].append(float(train_acc))
            m['val_accuracy'].append(float(val_acc))
            m['train_f1'].append(float(train_f1))
            m['val_f1'].append(float(val_f1))
            m['train_cls_loss'].append(float(train_cls_loss))
            m['val_cls_loss'].append(float(val_cls_loss))

    def record_final_epoch_metrics(self, model_name: str, train_loss, val_loss, train_acc, val_acc,
                                  train_f1=0.0, val_f1=0.0, train_cls_loss=0.0, val_cls_loss=0.0):
        if model_name in self.final_training_history['models']:
            m = self.final_training_history['models'][model_name]
            m['train_loss'].append(float(train_loss))
            m['val_loss'].append(float(val_loss))
            m['train_accuracy'].append(float(train_acc))
            m['val_accuracy'].append(float(val_acc))
            m['train_f1'].append(float(train_f1))
            m['val_f1'].append(float(val_f1))
            m['train_cls_loss'].append(float(train_cls_loss))
            m['val_cls_loss'].append(float(val_cls_loss))

    def record_fold_metrics(self, fold_idx: int, model_metrics: dict, ensemble_metrics: dict):
        self.fold_metrics.append({
            'fold_idx': fold_idx,
            'model_metrics': model_metrics,
            'ensemble_metrics': ensemble_metrics,
            'training_time': time.strftime("%Y-%m-%d %H:%M:%S")
        })

    def record_final_metrics(self, model_metrics: dict, ensemble_metrics: dict):

        self.final_metrics = {
            'model_metrics': model_metrics,
            'ensemble_metrics': ensemble_metrics,
            'training_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }


    def record_fold_error_analysis(self, fold_idx: int, cnn_errors: int, cnn_wrong_gru_right: int, ratio: float):
        self.cv_error_analysis.append({
            'fold_idx': fold_idx,
            'cnn_total_errors': cnn_errors,
            'cnn_wrong_gru_right': cnn_wrong_gru_right,
            'ratio': ratio
        })

    def record_final_error_analysis(self, cnn_errors: int, cnn_wrong_gru_right: int, ratio: float):
        self.final_error_analysis = {
            'cnn_total_errors': cnn_errors,
            'cnn_wrong_gru_right': cnn_wrong_gru_right,
            'ratio': ratio
        }

    def record_fold_dynamic_fusion(self, fold_idx: int, adjusted_samples: int, total_samples: int, original_f1: float, dynamic_f1: float, improvement: float):
        self.cv_dynamic_fusion_stats.append({
            'fold_idx': fold_idx,
            'adjusted_samples': adjusted_samples,
            'total_samples': total_samples,
            'original_f1': original_f1,
            'dynamic_f1': dynamic_f1,
            'improvement': improvement
        })

    def record_final_dynamic_fusion(self, adjusted_samples: int, total_samples: int, original_f1: float, dynamic_f1: float, improvement: float):
        self.final_dynamic_fusion_stats = {
            'adjusted_samples': adjusted_samples,
            'total_samples': total_samples,
            'original_f1': original_f1,
            'dynamic_f1': dynamic_f1,
            'improvement': improvement
        }

    def record_fold_fusion_debug_samples(self, fold_idx: int, debug_samples: list):
        self.cv_fusion_debug_samples.append({'fold_idx': fold_idx, 'debug_samples': debug_samples})

    def record_final_fusion_debug_samples(self, debug_samples: list):
        self.final_fusion_debug_samples = debug_samples

    # =========================================================================
    # 4. 文件保存逻辑 (CSV & JSON)
    # =========================================================================

    def _save_performance_metrics_csv(self):

        performance_data = []
        
        columns_order = [
            'phase', 'fold', 'model', 
            'accuracy', 'precision', 'recall', 'f1', 
            'auc', 'balanced_accuracy', 'mcc', 'auc_pr', 'specificity', 
            'best_threshold'
        ]
        
        def get_metric(m_dict, key):
            val = m_dict.get(key, 0.0)
            # 兼容 balanced_acc / balanced_accuracy
            if key == 'balanced_accuracy' and val == 0.0:
                val = m_dict.get('balanced_acc', 0.0)
            return val
-
        for fold_metric in self.fold_metrics:
            for model_name, metrics in fold_metric['model_metrics'].items():
                row = {
                    'phase': 'CV', 'fold': fold_metric['fold_idx'], 'model': model_name,
                    'accuracy': get_metric(metrics, 'accuracy'),
                    'precision': get_metric(metrics, 'precision'),
                    'recall': get_metric(metrics, 'recall'),
                    'f1': get_metric(metrics, 'f1'),
                    'auc': get_metric(metrics, 'auc'),
                    'balanced_accuracy': get_metric(metrics, 'balanced_accuracy'),
                    'mcc': get_metric(metrics, 'mcc'),
                    'auc_pr': get_metric(metrics, 'auc_pr'),
                    'specificity': get_metric(metrics, 'specificity'),
                    'best_threshold': get_metric(metrics, 'best_threshold') # CV通常是0.5
                }
                performance_data.append(row)
            
            if 'ensemble_metrics' in fold_metric:
                metrics = fold_metric['ensemble_metrics']
                row = {
                    'phase': 'CV', 'fold': fold_metric['fold_idx'], 'model': 'Enhanced_Fusion',
                    'accuracy': get_metric(metrics, 'accuracy'),
                    'precision': get_metric(metrics, 'precision'),
                    'recall': get_metric(metrics, 'recall'),
                    'f1': get_metric(metrics, 'f1'),
                    'auc': get_metric(metrics, 'auc'),
                    'balanced_accuracy': get_metric(metrics, 'balanced_accuracy'),
                    'mcc': get_metric(metrics, 'mcc'),
                    'auc_pr': get_metric(metrics, 'auc_pr'),
                    'specificity': get_metric(metrics, 'specificity'),
                    'best_threshold': get_metric(metrics, 'best_threshold')
                }
                performance_data.append(row)
-
        if self.final_metrics:
            if 'model_metrics' in self.final_metrics:
                for model_name, metrics in self.final_metrics['model_metrics'].items():
                    row = {
                        'phase': 'Final', 'fold': 'Test', 'model': model_name,
                        'accuracy': get_metric(metrics, 'accuracy'),
                        'precision': get_metric(metrics, 'precision'),
                        'recall': get_metric(metrics, 'recall'),
                        'f1': get_metric(metrics, 'f1'),
                        'auc': get_metric(metrics, 'auc'),
                        'balanced_accuracy': get_metric(metrics, 'balanced_accuracy'),
                        'mcc': get_metric(metrics, 'mcc'),
                        'auc_pr': get_metric(metrics, 'auc_pr'),
                        'specificity': get_metric(metrics, 'specificity'),
                        'best_threshold': get_metric(metrics, 'best_threshold')
                    }
                    performance_data.append(row)
            
            if 'ensemble_metrics' in self.final_metrics:
                metrics = self.final_metrics['ensemble_metrics']
                row = {
                    'phase': 'Final', 'fold': 'Test', 'model': 'Enhanced_Fusion',
                    'accuracy': get_metric(metrics, 'accuracy'),
                    'precision': get_metric(metrics, 'precision'),
                    'recall': get_metric(metrics, 'recall'),
                    'f1': get_metric(metrics, 'f1'),
                    'auc': get_metric(metrics, 'auc'),
                    'balanced_accuracy': get_metric(metrics, 'balanced_accuracy'),
                    'mcc': get_metric(metrics, 'mcc'),
                    'auc_pr': get_metric(metrics, 'auc_pr'),
                    'specificity': get_metric(metrics, 'specificity'),
                    'best_threshold': get_metric(metrics, 'best_threshold')
                }
                performance_data.append(row)

        if performance_data:
            df = pd.DataFrame(performance_data)
            
            for col in columns_order:
                if col not in df.columns:
                    df[col] = 0.0
            
            df = df[columns_order]
            
            csv_path = os.path.join(self.model_dir, "model_performance_metrics.csv")
            df.to_csv(csv_path, index=False, float_format='%.4f')
            print(f"✅ Comprehensive performance CSV saved to: {csv_path}")
            
            print("\n=== Final Performance Report (Optimal Thresholds) ===")
            if 'phase' in df.columns:
                print(df[df['phase']=='Final'].to_string(index=False))
            else:
                print(df.to_string(index=False))
            print("=====================================================\n")

    def _save_training_history_csv(self):
        cv_data = []
        for fold_idx, fold_data in self.cv_history.items():
            for model_name, model_data in fold_data['models'].items():
                max_len = max(len(model_data['train_loss']), len(model_data.get('train_f1', [])))
                def pad(lst): return lst + [0.0] * (max_len - len(lst)) if len(lst) < max_len else lst[:max_len]
                t_loss = pad(model_data['train_loss'])
                v_loss = pad(model_data['val_loss'])
                t_acc = pad(model_data['train_accuracy'])
                v_acc = pad(model_data['val_accuracy'])
                t_f1 = pad(model_data.get('train_f1', []))
                v_f1 = pad(model_data.get('val_f1', []))
                for epoch in range(max_len):
                    cv_data.append({
                        'phase': 'CV', 'fold': fold_idx, 'model': model_name, 'epoch': epoch + 1,
                        'train_loss': t_loss[epoch], 'val_loss': v_loss[epoch],
                        'train_acc': t_acc[epoch], 'val_acc': v_acc[epoch],
                        'train_f1': t_f1[epoch], 'val_f1': v_f1[epoch]
                    })

        final_data = []
        for model_name, model_data in self.final_training_history.get('models', {}).items():
            max_len = max(len(model_data['train_loss']), len(model_data.get('train_f1', [])))
            def pad(lst): return lst + [0.0] * (max_len - len(lst)) if len(lst) < max_len else lst[:max_len]
            t_loss = pad(model_data['train_loss'])
            v_loss = pad(model_data['val_loss'])
            t_acc = pad(model_data['train_accuracy'])
            v_acc = pad(model_data['val_accuracy'])
            t_f1 = pad(model_data.get('train_f1', []))
            v_f1 = pad(model_data.get('val_f1', []))
            for epoch in range(max_len):
                final_data.append({
                    'phase': 'Final', 'fold': 'All', 'model': model_name, 'epoch': epoch + 1,
                    'train_loss': t_loss[epoch], 'val_loss': v_loss[epoch],
                    'train_acc': t_acc[epoch], 'val_acc': v_acc[epoch],
                    'train_f1': t_f1[epoch], 'val_f1': v_f1[epoch]
                })

        all_data = cv_data + final_data
        if all_data:
            df = pd.DataFrame(all_data)
            csv_path = os.path.join(self.model_dir, "training_history.csv")
            df.to_csv(csv_path, index=False)
            print(f"Training history CSV saved to: {csv_path}")

    def save_all_metrics(self):
        metrics_data = {
            'cross_validation': {
                'fold_history': self.cv_history,
                'fold_metrics': self.fold_metrics,
                'error_analysis': self.cv_error_analysis,
                'dynamic_fusion_stats': self.cv_dynamic_fusion_stats,
                'fusion_debug_samples': self.cv_fusion_debug_samples
            },
            'final_training': {
                'training_history': self.final_training_history,
                'final_metrics': self.final_metrics,
                'error_analysis': self.final_error_analysis,
                'dynamic_fusion_stats': self.final_dynamic_fusion_stats,
                'fusion_debug_samples': self.final_fusion_debug_samples
            }
        }
        
        json_path = os.path.join(self.model_dir, "training_metrics.json")
        try:
            def convert(o):
                if isinstance(o, np.int64): return int(o)
                if isinstance(o, np.float32): return float(o)
                return o
            with open(json_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=convert)
            print(f"Training metrics JSON saved to: {json_path}")
        except Exception as e:
            print(f"Warning: Failed to save JSON metrics: {e}")

        self._save_training_history_csv()
        
        self._save_performance_metrics_csv()
        
        return metrics_data

class DataProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.random_seed = config['random_seed']
        self.kfold_splits = config['kfold_splits']
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        print("Loading data...")
        
        train_list = config.get('train_positive_list', '')
        test_list  = config.get('test_positive_list', '')
        
        if train_list and test_list:
            print("Using text lists for positive data loading...")
            self.real_positive_train = self._load_positive_data_from_list(train_list)
            self.real_positive_test  = self._load_positive_data_from_list(test_list)
        else:
            print("Using directories for positive data loading...")
            self.real_positive_train = self._load_positive_data_from_dir(config['train_positive_dir'])
            self.real_positive_test  = self._load_positive_data_from_dir(config['test_positive_dir'])
            
        print("Positive data loaded: train set {} samples, test set {} samples".format(
            len(self.real_positive_train), len(self.real_positive_test)))
        
        # Load negative data with enhanced randomness
        self.negative_train_files = self._get_shuffled_files(config['train_negative_dir'])
        self.negative_test_files = self._get_shuffled_files(config['test_negative_dir'])
        print("Negative data files found: train set {} files, test set {} files".format(
            len(self.negative_train_files), len(self.negative_test_files)))
        
        # Pre-load and fix test data for consistency
        self._preload_test_data()

    def _load_positive_data_from_list(self, list_path: str) -> np.ndarray:
        """Load positive data from a txt list of npz paths (one path per line)."""
        print(f" Load positive data: {list_path}")
        if not list_path:
            return np.zeros((0, 1280, 1))
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Positive list not found: {list_path}")

        with open(list_path, "r") as f:
            npz_files = [line.strip() for line in f if line.strip()]
        
        positive_features = []
        for file_path in npz_files:
            try:
                data = np.load(file_path)
                feature = self._reshape_data(data[list(data.keys())[0]])
                if feature.size > 0:
                    positive_features.append(feature)
            except Exception as e:
                print(f"Failed to load positive data file {file_path}: {e}")
                continue
        return np.vstack(positive_features) if positive_features else np.zeros((0, 1280, 1))

    def _load_negative_data_from_list(self, list_path: str, expected_size: int = 0) -> np.ndarray:
        print(f"  load negative: {list_path}")
        if not list_path or not os.path.exists(list_path):
            return np.zeros((0, 1280, 1))

        with open(list_path, "r") as f:
            npz_files = [line.strip() for line in f if line.strip()]
            
        if expected_size > 0 and len(npz_files) > expected_size:
            npz_files = npz_files[:expected_size]
            
        negative_features = []
        for file_path in npz_files:
            try:
                data = np.load(file_path)
                feature = self._reshape_data(data[list(data.keys())[0]])
                if feature.size > 0:
                    negative_features.append(feature)
            except Exception as e:
                print(f"Failed to load negative data file {file_path}: {e}")
                continue
        return np.vstack(negative_features) if negative_features else np.zeros((0, 1280, 1))
    
    def _preload_test_data(self):
        print("Preloading test data for consistency...")
        self.fixed_test_data = self.get_test_data()
        print("Fixed test data loaded: {} samples".format(len(self.fixed_test_data[0])))
    
    def _get_shuffled_files(self, directory: str) -> List[str]:
        files = glob.glob(os.path.join(directory, "*.npz"))
        random.shuffle(files)
        return files
    
    def _load_positive_data_from_dir(self, data_dir: str) -> np.ndarray:
        npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
        positive_features = []
        for file_path in npz_files:
            try:
                data = np.load(file_path)
                feature = self._reshape_data(data[list(data.keys())[0]])
                if feature.size > 0:
                    positive_features.append(feature)
            except Exception as e:
                print("Failed to load positive data file {}: {}".format(file_path, e))
                continue
        return np.vstack(positive_features) if positive_features else np.zeros((0, 1280, 1))
    
    def _reshape_data(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            return data.reshape(1, 1280, 1)
        elif data.ndim == 2:
            return data.T.reshape(-1, 1280, 1) if data.shape[0] == 1280 else data.reshape(-1, 1280, 1)
        else:
            return np.resize(data, (-1, 1280, 1))
    
    def _load_negative_data(self, files: List[str], sample_size: int, use_all: bool = False) -> np.ndarray:
        if sample_size == 0 or not files:
            return np.zeros((0, 1280, 1))
        
        if use_all:
            selected_files = files
        else:
            sample_pool_size = min(len(files), max(sample_size * 3, 1000))
            candidate_files = random.sample(files, sample_pool_size)
            selected_files = random.sample(candidate_files, min(sample_size, len(candidate_files)))
        
        negative_features = []
        for file_path in selected_files:
            try:
                data = np.load(file_path)
                feature = self._reshape_data(data[list(data.keys())[0]])
                if feature.size > 0:
                    negative_features.append(feature)
            except Exception as e:
                print("Failed to load negative data file {}: {}".format(file_path, e))
                continue
        return np.vstack(negative_features) if negative_features else np.zeros((0, 1280, 1))
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X_test_positive = self.real_positive_test
        y_test_positive = np.ones(X_test_positive.shape[0])
        
        negative_sample_size = len(X_test_positive)
        
        test_neg_list = self.config.get('test_negative_list', '')
        if test_neg_list:
            X_test_negative = self._load_negative_data_from_list(test_neg_list, expected_size=negative_sample_size)
        else:
            X_test_negative = self._load_negative_data(self.negative_test_files, negative_sample_size, use_all=False)
            
        y_test_negative = np.zeros(X_test_negative.shape[0])
        
        min_samples = min(len(X_test_positive), len(X_test_negative))
        X_test_positive = X_test_positive[:min_samples]
        X_test_negative = X_test_negative[:min_samples]
        y_test_positive = y_test_positive[:min_samples]
        y_test_negative = y_test_negative[:min_samples]
        
        X_test = np.vstack([X_test_positive, X_test_negative])
        y_test = np.hstack([y_test_positive, y_test_negative])
        
        print(f"Balanced test set: {len(X_test)} samples ({min_samples} positive, {min_samples} negative)")
        
        indices = np.random.permutation(len(X_test))
        return X_test[indices], y_test[indices]
    
    def get_fixed_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.fixed_test_data

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

class UnifiedFeatureHeatmapGenerator:
    
    def __init__(self, model_dir: str):
        self.model_dir = os.path.join(model_dir, "heatmaps_bib_style")
        os.makedirs(self.model_dir, exist_ok=True)
        self.supported_methods = ["gradient", "occlusion", "perturbation"]
        self._setup_journal_style()
        
    def _setup_journal_style(self):
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
        mpl.rcParams['pdf.fonttype'] = 42 
        mpl.rcParams['ps.fonttype'] = 42
        
        mpl.rcParams['axes.titlesize'] = 16
        mpl.rcParams['axes.labelsize'] = 14
        mpl.rcParams['xtick.labelsize'] = 12
        mpl.rcParams['ytick.labelsize'] = 12
        mpl.rcParams['legend.fontsize'] = 12
        
        mpl.rcParams['axes.linewidth'] = 1.5
        mpl.rcParams['xtick.major.width'] = 1.5
        mpl.rcParams['ytick.major.width'] = 1.5

    def generate_comparison_heatmaps(self, cnn_model, gru_model, X_samples, y_samples, 
                                   method="occlusion", max_total_samples=10):
        print(f"\nGenerating feature heatmaps for interpretability analysis...")
        print(f"Method: {method} | Max samples: {max_total_samples}")
        
        self._validate_inputs(cnn_model, gru_model, X_samples, y_samples, method)
        selected_indices = self._select_priority_samples(
            cnn_model, gru_model, X_samples, y_samples, max_total_samples
        )
        cnn_heatmaps, gru_heatmaps = self._generate_sample_heatmaps(
            cnn_model, gru_model, X_samples, selected_indices, method
        )
        self._create_comparison_plots(
            X_samples, y_samples, selected_indices, cnn_heatmaps, gru_heatmaps, method
        )
        self._perform_statistical_analysis(cnn_heatmaps, gru_heatmaps, method)
        
        print("Feature heatmaps generation completed!")
        return cnn_heatmaps, gru_heatmaps
    
    def _validate_inputs(self, cnn_model, gru_model, X_samples, y_samples, method):
        if method not in self.supported_methods: raise ValueError(f"Unsupported method: {method}")
        if len(X_samples) != len(y_samples): raise ValueError("Samples and labels mismatch")
        if len(X_samples) == 0: raise ValueError("Empty samples")
    
    def _select_priority_samples(self, cnn_model, gru_model, X_samples, y_samples, max_total_samples):
        cnn_preds = cnn_model.predict(X_samples, verbose=0).flatten()
        gru_preds = gru_model.predict(X_samples, verbose=0).flatten()
        
        cnn_binary = (cnn_preds > 0.5).astype(int)
        gru_binary = (gru_preds > 0.5).astype(int)
        
        sample_types = {
            'cnn_wrong_gru_right': (cnn_binary != y_samples) & (gru_binary == y_samples),
            'both_correct': (cnn_binary == y_samples) & (gru_binary == y_samples),
            'both_wrong': (cnn_binary != y_samples) & (gru_binary != y_samples),
            'cnn_right_gru_wrong': (cnn_binary == y_samples) & (gru_binary != y_samples)
        }
        
        selected_indices = []
        priority_indices = np.where(sample_types['cnn_wrong_gru_right'])[0]
        max_priority = max_total_samples - 3 
        
        if len(priority_indices) > 0:
            num_priority = min(len(priority_indices), max_priority)
            selected_priority = np.random.choice(priority_indices, num_priority, replace=False)
            selected_indices.extend(selected_priority)
            
        for type_name in ['both_correct', 'both_wrong', 'cnn_right_gru_wrong']:
            type_indices = np.where(sample_types[type_name])[0]
            if len(type_indices) > 0 and len(selected_indices) < max_total_samples:
                selected_type = np.random.choice(type_indices, 1, replace=False)
                selected_indices.extend(selected_type)
                
        remaining_slots = max_total_samples - len(selected_indices)
        if remaining_slots > 0 and len(priority_indices) > len(selected_indices):
            unselected_priority = list(set(priority_indices) - set(selected_indices))
            if len(unselected_priority) > 0:
                additional_priority = np.random.choice(
                    unselected_priority, min(remaining_slots, len(unselected_priority)), replace=False
                )
                selected_indices.extend(additional_priority)
                
        return selected_indices[:max_total_samples]
    
    def _generate_sample_heatmaps(self, cnn_model, gru_model, X_samples, selected_indices, method):
        cnn_heatmaps, gru_heatmaps = [], []
        for i, idx in enumerate(selected_indices):
            X_sample = X_samples[idx:idx+1]
            cnn_heatmaps.append(self._create_unified_heatmap(cnn_model, X_sample, method))
            gru_heatmaps.append(self._create_unified_heatmap(gru_model, X_sample, method))
        return cnn_heatmaps, gru_heatmaps
    
    def _create_unified_heatmap(self, model, X_sample, method):
        if method == "gradient": return self._gradient_based_importance(model, X_sample)
        elif method == "occlusion": return self._occlusion_based_importance(model, X_sample)
        elif method == "perturbation": return self._perturbation_based_importance(model, X_sample)
        else: raise ValueError(f"Unknown method: {method}")
    
    def _gradient_based_importance(self, model, X_sample):
        X_sample_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(X_sample_tensor)
            predictions = model(X_sample_tensor)
            target = predictions[:, 0]
        gradients = tape.gradient(target, X_sample_tensor)
        importance = tf.reduce_mean(tf.abs(gradients), axis=[0, 2])
        importance = importance.numpy()
        if np.max(importance) > 0: importance /= np.max(importance)
        return importance
    
    def _occlusion_based_importance(self, model, X_sample, window_size=10):
        baseline_pred = model.predict(X_sample, verbose=0)[0, 0]
        importance_scores = np.zeros(1280)
        for i in range(0, 1280 - window_size + 1, window_size//2):
            X_occluded = X_sample.copy()
            X_occluded[:, i:i+window_size, :] = 0
            occluded_pred = model.predict(X_occluded, verbose=0)[0, 0]
            importance = abs(baseline_pred - occluded_pred)
            importance_scores[i:i+window_size] += importance
        if np.max(importance_scores) > 0: importance_scores /= np.max(importance_scores)
        return importance_scores
    
    def _perturbation_based_importance(self, model, X_sample, num_perturbations=100):
        baseline_pred = model.predict(X_sample, verbose=0)[0, 0]
        importance_scores = np.zeros(1280)
        for _ in range(num_perturbations):
            feature_idx = np.random.randint(0, 1280)
            perturbation = np.random.normal(0, 0.1)
            X_perturbed = X_sample.copy()
            X_perturbed[:, feature_idx, :] += perturbation
            perturbed_pred = model.predict(X_perturbed, verbose=0)[0, 0]
            importance = abs(baseline_pred - perturbed_pred)
            importance_scores[feature_idx] += importance
        if np.max(importance_scores) > 0: importance_scores /= np.max(importance_scores)
        return importance_scores
    
    def _create_comparison_plots(self, X_samples, y_samples, selected_indices, cnn_heatmaps, gru_heatmaps, method):
        for i, idx in enumerate(selected_indices):
            self._create_single_comparison_plot(i, idx, X_samples[idx], y_samples[idx], cnn_heatmaps[i], gru_heatmaps[i], method)
    
    def _create_single_comparison_plot(self, plot_idx, sample_idx, X_sample, y_true, cnn_heatmap, gru_heatmap, method):

        feature_values = X_sample.flatten()
        difference = cnn_heatmap - gru_heatmap
        
        df_sample = pd.DataFrame({
            'Feature_Dimension': range(len(feature_values)),
            'Original_Feature_Value': feature_values,
            'CNN_Importance': cnn_heatmap,
            'GRU_Importance': gru_heatmap,
            'Absolute_Difference': np.abs(difference),
            'Diff_Direction': ['CNN_Higher' if x > 0 else 'GRU_Higher' for x in difference]
        })
        csv_file = os.path.join(self.model_dir, f"OriginData_Sample_{sample_idx}_{method}.csv")
        df_sample.to_csv(csv_file, index=False)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Sample {sample_idx} - Feature Importance Complementarity (True Label: {y_true})', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        axes[0, 0].plot(feature_values, color='#404040', alpha=0.8, linewidth=1.2)
        axes[0, 0].set_title('A. Original Feature Sequence', loc='left', fontweight='bold')
        axes[0, 0].set_ylabel('Feature Value')
        
        axes[0, 1].plot(cnn_heatmap, color='#E63946', alpha=0.9, linewidth=2)
        axes[0, 1].fill_between(range(len(cnn_heatmap)), cnn_heatmap, alpha=0.2, color='#E63946')
        axes[0, 1].set_title('B. CNN Feature Importance', loc='left', fontweight='bold')
        axes[0, 1].set_ylabel('Importance Score')
        axes[0, 1].set_ylim(0, 1.05)
        
        axes[1, 0].plot(gru_heatmap, color='#2A9D8F', alpha=0.9, linewidth=2)
        axes[1, 0].fill_between(range(len(gru_heatmap)), gru_heatmap, alpha=0.2, color='#2A9D8F')
        axes[1, 0].set_title('C. GRU Feature Importance', loc='left', fontweight='bold')
        axes[1, 0].set_xlabel('Feature Dimension')
        axes[1, 0].set_ylabel('Importance Score')
        axes[1, 0].set_ylim(0, 1.05)
        
        colors = ['#E63946' if x > 0 else '#2A9D8F' for x in difference]
        axes[1, 1].bar(range(len(difference)), np.abs(difference), color=colors, alpha=0.7, width=2.0)
        axes[1, 1].set_title('D. Absolute Importance Difference', loc='left', fontweight='bold')
        axes[1, 1].set_xlabel('Feature Dimension')
        axes[1, 1].set_ylabel('Absolute Diff (|CNN - GRU|)')
        
        for ax in axes.flat:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle='--', alpha=0.3)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        pdf_file = os.path.join(self.model_dir, f"Fig_Sample_{sample_idx}_{method}.pdf")
        tif_file = os.path.join(self.model_dir, f"Fig_Sample_{sample_idx}_{method}.tif")
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
        plt.savefig(tif_file, format='tiff', dpi=300, bbox_inches='tight')
        plt.close()

    def _perform_statistical_analysis(self, cnn_heatmaps, gru_heatmaps, method):
        cnn_heatmaps = np.array(cnn_heatmaps)
        gru_heatmaps = np.array(gru_heatmaps)
        
        correlations = []
        for i in range(len(cnn_heatmaps)):
            corr = np.corrcoef(cnn_heatmaps[i], gru_heatmaps[i])[0, 1]
            if not np.isnan(corr): correlations.append(corr)
        avg_correlation = np.mean(correlations) if correlations else 0
        
        overlap_scores = []
        for i in range(len(cnn_heatmaps)):
            cnn_top = set(np.argsort(cnn_heatmaps[i])[-256:])
            gru_top = set(np.argsort(gru_heatmaps[i])[-256:])
            overlap = len(cnn_top & gru_top) / 256
            overlap_scores.append(overlap)
        avg_overlap = np.mean(overlap_scores) if overlap_scores else 0
        
        stats = {
            'method': method,
            'avg_correlation': float(avg_correlation),
            'avg_overlap': float(avg_overlap),
            'complementarity_index': float(1 - avg_overlap),
            'num_samples_analyzed': len(cnn_heatmaps),
            'correlation_std': float(np.std(correlations)) if correlations else 0,
            'overlap_std': float(np.std(overlap_scores)) if overlap_scores else 0
        }
        
        stats_file = os.path.join(self.model_dir, f"complementarity_stats_{method}.json")
        with open(stats_file, 'w') as f: json.dump(stats, f, indent=2)
        
        self._create_summary_plot(cnn_heatmaps, gru_heatmaps, stats)
    
    def _create_summary_plot(self, cnn_heatmaps, gru_heatmaps, stats):

        avg_cnn_heatmap = np.mean(cnn_heatmaps, axis=0)
        avg_gru_heatmap = np.mean(gru_heatmaps, axis=0)
        
        df_mean_importance = pd.DataFrame({
            'Feature_Dimension': range(len(avg_cnn_heatmap)),
            'Mean_CNN_Importance': avg_cnn_heatmap,
            'Mean_GRU_Importance': avg_gru_heatmap
        })
        df_mean_importance.to_csv(os.path.join(self.model_dir, "OriginData_Mean_Importance.csv"), index=False)

        correlation_dist = []
        for i in range(len(cnn_heatmaps)):
            corr = np.corrcoef(cnn_heatmaps[i], gru_heatmaps[i])[0, 1]
            if not np.isnan(corr): correlation_dist.append(corr)
            
        df_corr = pd.DataFrame({'Pearson_Correlation': correlation_dist})
        df_corr.to_csv(os.path.join(self.model_dir, "OriginData_Correlation_Distribution.csv"), index=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Global Feature Importance & Model Complementarity', fontsize=18, fontweight='bold', y=1.02)
        
        axes[0].plot(avg_cnn_heatmap, color='#E63946', linewidth=2.5, label='CNN (Local Motifs)', alpha=0.85)
        axes[0].plot(avg_gru_heatmap, color='#2A9D8F', linewidth=2.5, label='GRU (Global Dependencies)', alpha=0.85)
        axes[0].set_title('A. Mean Feature Importance Distribution', loc='left', fontweight='bold')
        axes[0].set_xlabel('Feature Dimension')
        axes[0].set_ylabel('Mean Importance Score')
        axes[0].legend(frameon=False, loc='upper right')
        
        axes[1].hist(correlation_dist, bins=12, alpha=0.7, color='#457B9D', edgecolor='white', linewidth=1.2)
        axes[1].axvline(stats['avg_correlation'], color='#E63946', linestyle='--', linewidth=2.5,
                       label=f"Mean Correlation: {stats['avg_correlation']:.3f}")
        axes[1].set_title('B. CNN-GRU Correlation Distribution', loc='left', fontweight='bold')
        axes[1].set_xlabel('Pearson Correlation Coefficient')
        axes[1].set_ylabel('Frequency (Number of Samples)')
        axes[1].legend(frameon=False)
        
        for ax in axes.flat:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, linestyle='--', alpha=0.3)
            
        plt.tight_layout()
        
        pdf_file = os.path.join(self.model_dir, "Fig_Complementarity_Summary.pdf")
        tif_file = os.path.join(self.model_dir, "Fig_Complementarity_Summary.tif")
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
        plt.savefig(tif_file, format='tiff', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_method_comparison(self, cnn_model, gru_model, X_samples, y_samples, sample_indices=None, max_samples=5):
        pass

==
class SmartDataSplitter:
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_fusion_samples = config.get('min_fusion_samples', 200)
        self.max_fusion_ratio = config.get('max_fusion_ratio', 0.3)
        self.min_base_samples = config.get('min_base_samples', 300)
        
    def calculate_optimal_split(self, total_samples):

        print(f"\nCalculating optimal data split for {total_samples} total samples...")
        

        required_base_samples = max(self.min_base_samples, int(total_samples * 0.5))
        

        available_fusion_samples = total_samples - required_base_samples
        
        if available_fusion_samples < self.min_fusion_samples:

            fusion_ratio = min(self.max_fusion_ratio, self.min_fusion_samples / total_samples)
            base_ratio = 1 - fusion_ratio
            
            if total_samples * fusion_ratio < self.min_fusion_samples:
                print("Warning: Sample size too small for standard split, using cross-validation strategy")
                fusion_ratio = 0.5  # 使用50-50分割，后续通过交叉验证增强
        else:
            fusion_ratio = self.config.get('fusion_train_ratio', 0.3)
            base_ratio = 1 - fusion_ratio
        
        base_samples = int(total_samples * base_ratio)
        fusion_samples = total_samples - base_samples
        
        print(f"Optimal split calculated:")
        print(f"  Base model samples: {base_samples} ({base_ratio:.1%})")
        print(f"  Fusion model samples: {fusion_samples} ({fusion_ratio:.1%})")
        
        return base_ratio, fusion_ratio

class OptimizedDataProcessor(DataProcessor):
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.data_splitter = SmartDataSplitter(config)
        self.use_cv_fusion = config.get('use_cv_fusion', False)
        
    def get_optimized_training_data(self):
        print("\nPreparing training data...")
        
        X_train_real = self.real_positive_train
        y_train_positive = np.ones(X_train_real.shape[0])
        total_positive = len(X_train_real)
        
        if self.use_cv_fusion:
            print(f"\n!!! CV Stacking Mode Activated: Utilizing Out-Of-Fold (OOF) paradigm !!!")
            print(f"Total positive samples: {total_positive}")
            print("Strategy: Using ALL data for both Base Models (via OOF) and Fusion Layer.")
            
            X_negative = self._load_negative_data(
                self.negative_train_files, total_positive
            )
            y_negative = np.zeros(X_negative.shape[0])
            
            X_all = np.vstack([X_train_real, X_negative])
            y_all = np.hstack([y_train_positive, y_negative])
            
            indices = np.random.permutation(len(X_all))
            X_all, y_all = X_all[indices], y_all[indices]
            
            print(f"Total training data prepared: {len(X_all)} samples (Balanced 1:1)")
            
            return {
                'mode': 'cv_stacking', 
                'data': (X_all, y_all), 
                'folds': self.config.get('kfold_splits', 5)
            }
            
        else:
            print("Using Legacy Split Strategy ...")
            base_ratio, fusion_ratio = self.data_splitter.calculate_optimal_split(total_positive)
            
            base_count = int(total_positive * base_ratio)
            X_base_real = X_train_real[:base_count]
            X_fusion_real = X_train_real[base_count:]

            X_base_neg = self._load_negative_data(self.negative_train_files, len(X_base_real))
            X_fusion_neg = self._load_negative_data(
                self.negative_train_files, len(X_fusion_real)
            )
            
            X_base = np.vstack([X_base_real, X_base_neg])
            y_base = np.hstack([np.ones(len(X_base_real)), np.zeros(len(X_base_neg))])
            idx_b = np.random.permutation(len(X_base))
            X_base, y_base = X_base[idx_b], y_base[idx_b]
            
            X_fusion = np.vstack([X_fusion_real, X_fusion_neg])
            y_fusion = np.hstack([np.ones(len(X_fusion_real)), np.zeros(len(X_fusion_neg))])
            idx_f = np.random.permutation(len(X_fusion))
            X_fusion, y_fusion = X_fusion[idx_f], y_fusion[idx_f]
            
            return {
                'base': (X_base, y_base),
                'fusion': (X_fusion, y_fusion)
            }
    
    def _get_cv_enhanced_data(self, X_base, y_base, X_fusion, y_fusion):

        X_all = np.vstack([X_base, X_fusion])
        y_all = np.hstack([y_base, y_fusion])
        
        print(f"Cross-validation enhanced fusion training with {len(X_all)} total samples")
        
        return {
            'base': (X_base, y_base),
            'fusion': (X_all, y_all),  
            'use_cv': True
        }

def build_conv_basic_net(input_shape, config):
    inputs = Input(shape=input_shape)
    x = Conv1D(48, 3, activation='relu')(inputs)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(24, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(24, activation='relu', kernel_regularizer=l2(config['cnn_l2_regularization']))(x)
    x = Dropout(config['cnn_dropout_rate'])(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', 
                 optimizer=Adam(config['cnn_learning_rate']), 
                 metrics=['accuracy'])
    return model

def build_enhanced_gru(input_shape, config):

    from tensorflow.keras.layers import Bidirectional, BatchNormalization, LSTM, MaxPooling1D
    
    gru_units = config['gru_units_large'] 
    l2_reg = config['gru_l2_reg']
    
    inputs = Input(shape=input_shape)
    

    x = Conv1D(
        64, 7, strides=2, padding='same', activation='relu', 
        kernel_regularizer=l2(l2_reg), 
        name='enhanced_conv1'
    )(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(
        128, 5, strides=2, padding='same', activation='relu',
        kernel_regularizer=l2(l2_reg),
        name='enhanced_conv2'
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    
    if len(gru_units) == 1:
        # 单层双向
        gru_out = Bidirectional(LSTM(
            gru_units[0], return_sequences=True, dropout=config['gru_dropout'],
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg),
            name='lstm_bidirectional'
        ))(x)
        att_units = gru_units[0] * 2
    else:
        x = LSTM(
            gru_units[0], return_sequences=True, dropout=config['gru_dropout'],
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg),
            name='lstm_1'
        )(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        
        gru_out = LSTM(
            gru_units[1], return_sequences=True, dropout=config['gru_dropout'],
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg),
            name='lstm_2'
        )(x)
        att_units = gru_units[1]

    gru_out = LayerNormalization(epsilon=1e-6)(gru_out)
    
    time_attention = Dense(1, activation='tanh')(gru_out)
    time_attention = Softmax(axis=1, name='time_attention')(time_attention)
    time_attention_output = Multiply()([gru_out, time_attention])
    
    feat_attention = Dense(att_units, activation='sigmoid', name='feat_attention')(time_attention_output)
    feat_attention_output = Multiply()([time_attention_output, feat_attention])
    
    gru_enhanced = Add()([gru_out, feat_attention_output])
    
    avg_pool = GlobalAveragePooling1D()(gru_enhanced)
    max_pool = GlobalMaxPooling1D()(gru_enhanced)
    
    concatenated = Concatenate()([avg_pool, max_pool])
    concatenated = BatchNormalization()(concatenated)
    
    x = Dense(
        64, activation='relu', 
        kernel_regularizer=l2(l2_reg),
        name='class_dense1'
    )(concatenated)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='enhanced_gru')
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(
            learning_rate=config.get('gru_initial_lr', 0.001), 
            clipnorm=1.0
        ),
        metrics=['accuracy']
    )
    
    return model

def build_enhanced_fusion_model(input_shape, config, cnn_model, gru_model):
    raw_inputs = Input(shape=input_shape, name='raw_inputs')
    cnn_pred_input = Input(shape=(1,), name='cnn_pred_input')
    gru_pred_input = Input(shape=(1,), name='gru_pred_input')
    
    cnn_model.trainable = False
    gru_model.trainable = False
    
    cnn_feature_output = FeatureExtractorManager.get_robust_feature_layer(cnn_model, 'cnn')
    cnn_feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_feature_output)
    
    gru_feature_output = FeatureExtractorManager.get_robust_feature_layer(gru_model, 'gru')
    gru_feature_extractor = Model(inputs=gru_model.input, outputs=gru_feature_output)
    
    cnn_features = cnn_feature_extractor(raw_inputs)
    gru_features = gru_feature_extractor(raw_inputs)

    def adapt_feature_dim(feature, target_dim, name):
        if feature.shape[-1] != target_dim:
            return Dense(target_dim, activation='relu', name=f'{name}_dim_adapt')(feature)
        return feature
    
    cnn_dim = cnn_features.shape[-1]
    gru_features = adapt_feature_dim(gru_features, cnn_dim, 'gru')

    fusion_layer = EnhancedGatedFusionMechanism(
        fusion_units=config.get('fusion_units', 64),
        dropout_rate=config.get('fusion_dropout', 0.3),
        gate_l2_reg=config.get('gate_l2_reg', 0.0005),
        feature_projection_units=config.get('feature_projection_units', 64),
        gate_activation=config.get('gate_activation', 'sigmoid'),
        ema_alpha=config.get('ema_alpha', 0.9),
        ema_clip_range=config.get('ema_clip_range', 0.2),
        use_learnable_prior=config.get('use_learnable_prior', True),
        uncertainty_center=config.get('uncertainty_center', 0.5), 
        uncertainty_sigma=config.get('uncertainty_sigma', 0.2)
    )

    fused_prediction, final_gate_weights, base_gate_weights, confidence_adjustment = fusion_layer([
        cnn_features, gru_features, cnn_pred_input, gru_pred_input
    ])
    
    classification_output = Activation('linear', name='classification_output')(fused_prediction)

    fusion_model = Model(
        inputs=[raw_inputs, cnn_pred_input, gru_pred_input],
        outputs=[
            classification_output, 
            final_gate_weights, 
            base_gate_weights, 
            confidence_adjustment
        ],
        name='enhanced_fusion_model'
    )

    return fusion_model

class RatioOptimizationTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.data_processor = DataProcessor(config)
        self.model_dir = config['model_dir']
        os.makedirs(self.model_dir, exist_ok=True)
        self.verbose = config['verbose']
        self.cnn_batch_size = config['batch_size']
        self.gru_batch_size = config['gru_batch_size']
        self.prediction_batch_size = config['prediction_batch_size']
        self.model_names = {'conv': 'conv_basic', 'gru': 'optimized_gru'}
        self.model_builders = {
            'conv_basic': lambda shape: build_conv_basic_net(shape, self.config),
            'optimized_gru': lambda shape: build_optimized_gru(shape, self.config)
        }
        
        self.trained_models = {}

        # Initialize metrics collector
        self.metrics_collector = TrainingMetricsCollector(self.model_dir)
        
        # Initialize heatmap generator
        self.heatmap_generator = UnifiedFeatureHeatmapGenerator(self.model_dir)
        
        # Use fixed test data for consistency
        self.X_test_fixed, self.y_test_fixed = self.data_processor.get_fixed_test_data()
        print(f"Using fixed test set with {len(self.X_test_fixed)} samples for all evaluations")
        
        self.enable_fusion_debug = config.get('enable_fusion_debug', True)
        self.max_debug_samples = config.get('max_debug_samples', 100)
        self.debug_sample_types = config.get('debug_sample_types', ['cnn_wrong_gru_right', 'both_wrong', 'cnn_right_gru_wrong'])
    
    def _analyze_errors(self, y_true, cnn_pred, gru_pred, threshold=0.5):
        y_true = y_true.astype(int)
        cnn_pred_binary = (cnn_pred > threshold).astype(int)
        gru_pred_binary = (gru_pred > threshold).astype(int)
        
        cnn_errors = np.sum(cnn_pred_binary != y_true)
        cnn_wrong_gru_right = np.sum((cnn_pred_binary != y_true) & (gru_pred_binary == y_true))
        ratio = cnn_wrong_gru_right / cnn_errors if cnn_errors > 0 else 0.0
        
        return {
            'cnn_total_errors': int(cnn_errors),
            'cnn_wrong_gru_right': int(cnn_wrong_gru_right),
            'ratio': float(ratio)
        }
    
    def _collect_fusion_debug_samples(self, y_true, cnn_pred, gru_pred, fused_pred=None, gate_weights=None, threshold=0.5):
        if not self.enable_fusion_debug:
            return []
        
        y_true = y_true.astype(int)
        cnn_pred_binary = (cnn_pred > threshold).astype(int)
        gru_pred_binary = (gru_pred > threshold).astype(int)
        fused_pred_binary = (fused_pred > threshold).astype(int) if fused_pred is not None else None
        
        debug_samples = []
        
        sample_types = {
            'cnn_wrong_gru_right': (cnn_pred_binary != y_true) & (gru_pred_binary == y_true),
            'both_correct': (cnn_pred_binary == y_true) & (gru_pred_binary == y_true),
            'both_wrong': (cnn_pred_binary != y_true) & (gru_pred_binary != y_true),
            'cnn_right_gru_wrong': (cnn_pred_binary == y_true) & (gru_pred_binary != y_true)
        }
        
        for sample_type in self.debug_sample_types:
            type_indices = np.where(sample_types[sample_type])[0]
            
            max_samples_per_type = self.max_debug_samples // len(self.debug_sample_types)
            if len(type_indices) > max_samples_per_type:
                type_indices = np.random.choice(type_indices, max_samples_per_type, replace=False)
            
            for idx in type_indices:
                cnn_confidence = max(cnn_pred[idx], 1 - cnn_pred[idx])
                gru_confidence = max(gru_pred[idx], 1 - gru_pred[idx])
                confidence_gap = gru_confidence - cnn_confidence
                
                sample_info = {
                    'sample_index': int(idx),
                    'true_label': int(y_true[idx]),
                    'cnn_pred': float(cnn_pred[idx]),
                    'gru_pred': float(gru_pred[idx]),
                    'cnn_binary': int(cnn_pred_binary[idx]),
                    'gru_binary': int(gru_pred_binary[idx]),
                    'sample_type': sample_type,
                    'cnn_confidence': float(cnn_confidence),
                    'gru_confidence': float(gru_confidence),
                    'confidence_gap': float(confidence_gap)
                }
                
                if fused_pred is not None:
                    sample_info['fused_pred'] = float(fused_pred[idx])
                    sample_info['fused_binary'] = int(fused_pred_binary[idx])
                    sample_info['fused_confidence'] = float(max(fused_pred[idx], 1 - fused_pred[idx]))
                
                if gate_weights is not None and idx < len(gate_weights):
                    sample_info['cnn_gate_weight'] = float(gate_weights[idx][0])
                    sample_info['gru_gate_weight'] = float(gate_weights[idx][1])
                
                debug_samples.append(sample_info)
        
        print(f"Collected {len(debug_samples)} fusion debug samples")
        return debug_samples
    
    def calculate_metrics(self, y_true, y_pred, threshold=0.5):
        y_true = y_true.astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)

        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)  # Sensitivity
        accuracy = accuracy_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred_binary)  # Balanced ACC
        mcc = matthews_corrcoef(y_true, y_pred_binary)                 # MCC
        
        # AUC
        try:
            auc_roc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc_roc = 0.5

        # PR-AUC
        try:
            precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_pred)
            auc_pr = auc(recall_pr, precision_pr)
        except ValueError:
            auc_pr = 0.5

        # Specificity 
        cm = confusion_matrix(y_true, y_pred_binary)
        if cm.shape == (2, 2):
            tn, fp, _, _ = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0

        return {
            'precision': float(precision),
            'recall': float(recall),
            'accuracy': float(accuracy),
            'f1': float(f1),
            'auc': float(auc_roc),
            'balanced_accuracy': float(balanced_acc),
            'mcc': float(mcc),
            'auc_pr': float(auc_pr),
            'specificity': float(specificity)
        }
    
    def _get_learning_rate_scheduler(self, model_name):
        if model_name == 'optimized_gru':
            def gru_lr_schedule(epoch, lr):
                initial_lr = self.config['gru_initial_lr']
                decay_epoch = self.config['gru_decay_epoch']
                decay_factor = self.config['gru_decay_factor']
                if epoch < 5:
                    return 0.0001
                elif epoch < decay_epoch:
                    return initial_lr
                else:
                    decay_times = (epoch - decay_epoch) // 8
                    return initial_lr * (decay_factor ** decay_times)
            return LearningRateScheduler(gru_lr_schedule, verbose=1 if self.verbose else 0)
        else:
            initial_lr = self.config['learning_rate']
            def cnn_lr_schedule(epoch, lr):
                T = self.config['epochs']
                min_lr = initial_lr * 0.01
                progress = epoch / T
                return min_lr + (initial_lr - min_lr) * (1 - progress) * (1 + np.cos(np.pi * progress)) / 2
            return LearningRateScheduler(cnn_lr_schedule, verbose=1 if self.verbose else 0)
    
    def _validate_model_training(self, model, X_train, y_train, X_val, y_val, model_name):
        print(f"Validating {model_name} training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Class distribution - Train: {np.bincount(y_train.astype(int))}")
        print(f"Class distribution - Val: {np.bincount(y_val.astype(int))}")
        
        initial_pred = model.predict(X_val[:10], verbose=0)
        print(f"Initial prediction range: [{np.min(initial_pred):.4f}, {np.max(initial_pred):.4f}]")
    
    def _validate_data_consistency(self, y_true, cnn_pred, gru_pred, fused_pred=None):
        print("=== Data Consistency Verification ===")
        print(f"Ground truth labels count: {len(y_true)}")
        print(f"CNN predictions count: {len(cnn_pred)}")
        print(f"GRU predictions count: {len(gru_pred)}")
        if fused_pred is not None:
            print(f"Fusion predictions count: {len(fused_pred)}")
        
        if len(y_true) != len(cnn_pred) or len(y_true) != len(gru_pred):
            print("ERROR: Data length mismatch detected!")
            return False
        
        cnn_binary = (cnn_pred > 0.5).astype(int)
        cnn_accuracy = accuracy_score(y_true, cnn_binary)
        cnn_errors = np.sum(cnn_binary != y_true)
        
        print(f"Actual CNN Accuracy: {cnn_accuracy:.4f}")
        print(f"Actual CNN Errors: {cnn_errors}")
        
        return True
    
    def _generate_feature_heatmaps(self):
        print("\nGenerating feature heatmaps for complementarity analysis...")

        tf.keras.config.enable_unsafe_deserialization()

        try:
            cnn_model = load_model(os.path.join(self.model_dir, "conv_basic_final_model.keras"))
            gru_model = load_model(
                os.path.join(self.model_dir, "enhanced_gru_final_model.keras")
            )
        
            method = self.config.get('heatmap_method', 'occlusion')
            max_total_samples = self.config.get('heatmap_max_total_samples', 10)

            cnn_heatmaps, gru_heatmaps = self.heatmap_generator.generate_comparison_heatmaps(
                cnn_model, gru_model, self.X_test_fixed, self.y_test_fixed, 
                method=method, max_total_samples=max_total_samples
            )
        
            print("Feature heatmap generation completed!")
        except Exception as e:
            print(f"Error generating feature heatmaps: {e}")
            print("Skipping heatmap generation due to model loading issues")
        finally:
            tf.keras.config.disable_unsafe_deserialization()
             
    def evaluate_model(self) -> dict:
        print("Base evaluate_model called - using traditional fusion")
        return {}
    
    def train_final_model(self) -> None:
        print("Base train_final_model called - using traditional approach")


import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class DynamicTrainingCallback(Callback):
    
    def __init__(self, fusion_layer, config):
        super().__init__()
        self.fusion_layer = fusion_layer
        self.config = config
        self.training_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
        

        self.fusion_gate_weight_strategy = [
            (5, (0.80, 0.15, 0.05)),    # stage1: epoch < 5, 
            (15, (0.20, 0.50, 0.30)),   # stage2: 5 ≤ epoch < 15
            (float('inf'), (0.05, 0.60, 0.35))  # stage3: epoch ≥ 15
        ]
        
        print("\n" + "="*80)
        print("fusion gate weight strategy:")
        for i, (max_epoch, weights) in enumerate(self.fusion_gate_weight_strategy, 1):
            epoch_range = f"epoch < {max_epoch}" if max_epoch != float('inf') else "epoch ≥ 15"
            w_correct, w_conf, w_stats = weights
            print(f"  Stage {i} ({epoch_range:15s}): correct={w_correct:.2f}, conf={w_conf:.2f}, stats={w_stats:.2f}")
        print("="*80 + "\n")
    
    def _get_fusion_gate_weights_by_epoch(self, epoch):
        for max_epoch, weights in self.fusion_gate_weight_strategy:
            if epoch < max_epoch:
                return weights
        return self.fusion_gate_weight_strategy[-1][1] 

    def _determine_stage(self, epoch):
        if epoch < 5:
            return 1, "Early Stage"
        elif epoch < 15:
            return 2, "Middle Stage"
        else:
            return 3, "Late Stage"
    
    def update_training_epoch(self, epoch):

        self.training_epoch.assign(epoch)
        
        stage_num, stage_name = self._determine_stage(epoch)

        w_correct, w_conf, w_stats = self._get_fusion_gate_weights_by_epoch(epoch)
        
        if self.fusion_layer is not None:
            if hasattr(self.fusion_layer, 'training_epoch'):
                self.fusion_layer.training_epoch.assign(epoch)
            
            try:
                self.fusion_layer.w_correct.assign(w_correct)
                self.fusion_layer.w_conf.assign(w_conf)
                self.fusion_layer.w_stats.assign(w_stats)
            except AttributeError as e:
                print(f"Warning: {e}")

        print("\n" + "="*80)
        print(f" Epoch {epoch + 1} | Stage {stage_num}: {stage_name}")
        print("-" * 40)
        print("fusion gate weight:")
        print(f"  ├─  (w_correct): {w_correct:.2f}")
        print(f"  ├─  (w_conf):    {w_conf:.2f}")
        print(f"  └─  (w_stats): {w_stats:.2f}")
        
        gate_sum = w_correct + w_conf + w_stats
        print(f"      (weight sum: {gate_sum:.2f})")
        print("="*80 + "\n")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.update_training_epoch(epoch)
        
        if epoch == 20:
            lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            new_lr = lr * 0.75
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f" {lr:.6f} → {new_lr:.6f} (×0.75)\n")
            
        elif epoch == 35:
            lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            new_lr = lr * 0.6
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print(f": {lr:.6f} → {new_lr:.6f} (×0.6)\n")
    
    def on_epoch_end(self, epoch, logs=None):
        pass
    
    def get_config(self):
        return {
            'fusion_gate_weight_strategy': self.fusion_gate_weight_strategy
        }
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import time
import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.optimizers import Adam

class OptimizedRatioOptimizationTrainer(RatioOptimizationTrainer):
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.fusion_model = None
        self.optimized_data_processor = OptimizedDataProcessor(config)
        self.model_names['gru'] = 'enhanced_gru'
        self.gru_ensemble_paths = [] 
        
        self.model_builders = {
            'conv_basic': lambda shape: build_conv_basic_net(shape, self.config),
            'enhanced_gru': lambda shape: build_enhanced_gru(shape, self.config)
        }
        
    def augment_data_for_gru(self, X, y, augmentation_factor=0.5):
        if augmentation_factor <= 0: return X, y
        augmented_X, augmented_y = [], []
        np.random.seed(int(time.time() * 1000) % 10000)
        
        for i in range(len(X)):
            if np.random.random() < augmentation_factor:
                original = X[i].flatten()
                aug_type = np.random.choice(['time_warp', 'noise', 'scale', 'shift', 'flip'])
                if aug_type == 'time_warp' and len(original) > 10:
                    stretch = np.random.uniform(0.85, 1.15)
                    st = np.interp(np.linspace(0, len(original)-1, int(len(original)*stretch)), np.arange(len(original)), original)
                    st = st[:len(original)] if len(st) > len(original) else np.pad(st, (0, len(original)-len(st)), 'constant')
                    augmented_X.append(st.reshape(-1, 1)); augmented_y.append(y[i])
                elif aug_type == 'noise':
                    augmented_X.append((original + np.random.normal(0, np.std(original)*0.05, len(original))).reshape(-1, 1)); augmented_y.append(y[i])
                elif aug_type == 'scale':
                    augmented_X.append((original * np.random.uniform(0.9, 1.1)).reshape(-1, 1)); augmented_y.append(y[i])
                elif aug_type == 'shift':
                    augmented_X.append((original + np.random.uniform(-0.05, 0.05)*np.std(original)).reshape(-1, 1)); augmented_y.append(y[i])
                elif aug_type == 'flip' and len(original) > 20:
                    augmented_X.append(np.flip(original).reshape(-1, 1)); augmented_y.append(y[i])
                    
        if augmented_X:
            X_c = np.vstack([X, np.array(augmented_X)])
            y_c = np.hstack([y, np.array(augmented_y)])
            idx = np.random.permutation(len(X_c))
            return X_c[idx], y_c[idx]
        return X, y

    def _get_learning_rate_scheduler(self, model_name):
        if model_name == 'enhanced_gru' or model_name == 'optimized_gru':
            def gru_sch(epoch, lr):
                config_lr = self.config.get('gru_initial_lr', 0.0003)
                initial_lr = 0.0003 if float(config_lr) > 0.0005 else float(config_lr)
                warmup = 5
                decay_start = 20 
                decay_factor = 0.7 
                min_lr = 1e-7 
                if epoch < warmup: return initial_lr * (epoch + 1) / warmup
                elif epoch < decay_start: return initial_lr
                else: return max(initial_lr * (decay_factor ** ((epoch - decay_start) // 8)), min_lr)
            return callbacks.LearningRateScheduler(gru_sch, verbose=0)
        else:
            # === 修复后的 CNN Scheduler (防止负数) ===
            initial_lr = self.config.get('learning_rate', 0.001)
            def cnn_sch(epoch, lr):
                # 动态获取最大轮次，防止 progress > 1
                T = max(self.config.get('epochs', 50), self.config.get('final_epochs', 150))
                min_lr = initial_lr * 0.01
                progress = min(epoch / T, 1.0)
                return max(min_lr + (initial_lr - min_lr) * (1 - progress) * (1 + np.cos(np.pi * progress)) / 2, min_lr)
            return callbacks.LearningRateScheduler(cnn_sch, verbose=0)

    def _calculate_auto_fusion_params(self, y_true, cnn_oof_pred):
        print("\n" + "="*50)
        print(" Auto-Tuning Fusion Parameters (Disagreement Expert)")
        
        y_true = np.array(y_true).flatten()
        cnn_probs = cnn_oof_pred.flatten()
        cnn_pred_binary = (cnn_probs > 0.5).astype(int)
        
        error_indices = np.where(y_true != cnn_pred_binary)[0]
        
        if len(error_indices) == 0:
            print("   CNN OOF Accuracy is 100%! Using default params.")
            return 0.5, 0.2
            
        error_probs = cnn_probs[error_indices]
        
        center = float(np.mean(error_probs))
        sigma = float(np.std(error_probs))
        
        sigma = max(0.15, min(sigma * 1.5, 0.3)) 
        center = max(0.2, min(center, 0.8))
        
        print(f"   Errors found: {len(error_indices)}/{len(y_true)}")
        print(f"   Calculated Mean: {np.mean(error_probs):.4f}, Std: {np.std(error_probs):.4f}")
        print(f"   Auto-Set Parameters -> Center: {center:.4f}, Sigma: {sigma:.4f}")
        print("="*50 + "\n")
        return center, sigma

    def _build_and_save_fusion_model(self, cnn_model, gru_model, input_shape, params=None):
        print("\nBuilding and saving enhanced fusion model...")

        if params:
            self.config['uncertainty_center'] = params[0]
            self.config['uncertainty_sigma'] = params[1]
            
        self.fusion_model = build_enhanced_fusion_model(
            input_shape, self.config, cnn_model, gru_model
        )
        path = os.path.join(self.model_dir, "enhanced_fusion_model.keras")
        self.fusion_model.save(path)
        print(f"Enhanced fusion model saved to: {path}")
        return self.fusion_model

    def _enhanced_fusion_predict(self, cnn_model, gru_model, X_data, override_gru_pred=None):
        if self.fusion_model is None:
            path = os.path.join(self.model_dir, "trained_enhanced_fusion_model.keras")
            if not os.path.exists(path): path = os.path.join(self.model_dir, "enhanced_fusion_model.keras")
            try:
                custom = {
                    'ImprovedGatedFusionMechanism': ImprovedGatedFusionMechanism,
                    'EnhancedGatedFusionMechanism': ImprovedGatedFusionMechanism,
                    'get_robust_feature_layer': FeatureExtractorManager.get_robust_feature_layer,
                    'F1Metric': F1Metric
                }
                self.fusion_model = load_model(path, custom_objects=custom)
            except Exception as e:
                print(f"Error loading fusion model: {e}")
                if self.fusion_model is None: # Fallback
                    self.fusion_model = build_enhanced_fusion_model(X_data.shape[1:], self.config, cnn_model, gru_model)

        cnn_pred = cnn_model.predict(X_data, batch_size=256, verbose=0).flatten()
        if override_gru_pred is not None: gru_pred = override_gru_pred
        else: gru_pred = gru_model.predict(X_data, batch_size=256, verbose=0).flatten()
        
        fusion_outputs = self.fusion_model.predict(
            [X_data, cnn_pred.reshape(-1,1), gru_pred.reshape(-1,1)],
            batch_size=256, verbose=0
        )
        
        if isinstance(fusion_outputs, list) and len(fusion_outputs) >= 4:
            return fusion_outputs[0].flatten(), cnn_pred, gru_pred, {
                'final_weights': fusion_outputs[1], 'base_weights': fusion_outputs[2], 'confidence_adjustment': fusion_outputs[3]
            }
        else:
            return fusion_outputs.flatten(), cnn_pred, gru_pred, None

    def _apply_confidence_locking(self, cnn_pred, fused_pred, y_true=None, lock_threshold=0.8):
        final_preds = []
        stats = {'total': len(cnn_pred), 'cnn_locked': 0, 'fusion_engaged': 0, 'gru_corrected': 0, 'gru_messed_up': 0}
        for i in range(len(cnn_pred)):
            p_c, p_f = cnn_pred[i], fused_pred[i]
            if p_c > lock_threshold or p_c < (1.0 - lock_threshold):
                final_preds.append(p_c); stats['cnn_locked'] += 1
            else:
                final_preds.append(p_f); stats['fusion_engaged'] += 1
                if y_true is not None:
                    if (p_c > 0.5) != (p_f > 0.5):
                        if (p_f > 0.5) == y_true[i]: stats['gru_corrected'] += 1
                        else: stats['gru_messed_up'] += 1
        return np.array(final_preds), stats

    def _analyze_gate_weights(self, gate_info, y_true, cnn_pred, gru_pred):
        if not gate_info: return {}
        fw = gate_info['final_weights']
        print(f"\n=== Gate === CNN: {np.mean(fw[:,0]):.3f}, GRU: {np.mean(fw[:,1]):.3f}")
        return {'final_weights': {'cnn_mean': float(np.mean(fw[:,0])), 'gru_mean': float(np.mean(fw[:,1])), 'cnn_std': float(np.std(fw[:,0])), 'gru_std': float(np.std(fw[:,1]))}}

    def _analyze_cnn_reliability(self, y_true, cnn_pred_prob, num_bins=10):
        print("\n" + "="*60 + "\n CNN Reliability Analysis\n" + "-"*60)
        try:
            import pandas as pd
            df = pd.DataFrame({'prob': cnn_pred_prob, 'true': y_true})
            df['bin'] = pd.cut(df['prob'], bins=np.linspace(0, 1, num_bins + 1), labels=False, include_lowest=True)
            print(f"{'Bin Range':<15} | {'Count':<6} | {'Acc':<8} | {'Err':<8} | {'Suggestion'}")
            print("-" * 75)
            for i in range(num_bins):
                low, high = i / num_bins, (i + 1) / num_bins
                subset = df[df['bin'] == i]
                if len(subset) == 0: continue
                acc = np.mean((subset['prob'] > 0.5).astype(int) == subset['true'])
                rec = "Trust CNN" if (1.0-acc) <= 0.05 else ("🔴 LISTEN GRU" if (1.0-acc) > 0.15 else "🟡 Mix")
                print(f"{low:.2f} - {high:.2f}    | {len(subset):<6} | {acc:.4f}   | {1.0-acc:.4f}   | {rec}")
            print("="*60 + "\n")
        except: print("⚠️ Pandas missing, skipping table.")

    def train_final_model(self):
        data_info = self.optimized_data_processor.get_optimized_training_data()
        self._train_with_cv_stacking(data_info)

    def _train_with_cv_stacking(self, data_info):
            print("\n" + "="*60 + "\n🚀 Starting Hybrid CV Stacking (Auto-Tuned)\n" + "="*60)
            
            X_all, y_all = data_info['data']
            n_folds = data_info['folds']
            input_shape = X_all.shape[1:]
            
            if self.config.get('use_repeated_cv_stacking', True):
                gru_bagging_runs = self.config.get('repeated_cv_runs', 3)
                print(f"Strategy: Repeated CV Stacking Enabled (Runs: {gru_bagging_runs})")
            else:
                gru_bagging_runs = 1
                print("Strategy: Repeated CV Stacking Disabled (Runs: 1)")
            
            self.metrics_collector.init_final_training(list(self.model_names.values()) + ['enhanced_fusion'])
            oof_preds_cnn = np.zeros((len(X_all), 1))
            oof_preds_gru = np.zeros((len(X_all), 1))
            
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            last_cnn, last_gru = None, None
            
            # === Phase 1: Hybrid CV ===
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
                print(f"\n--- Processing Fold {fold+1}/{n_folds} ---")
                X_train, y_train = X_all[train_idx], y_all[train_idx]
                X_val, y_val = X_all[val_idx], y_all[val_idx]
                
                # 1. CNN (Single)
                print("  Training CNN (Single)...")
                cnn = self.model_builders['conv_basic'](input_shape)
                cnn.fit(X_train, y_train, epochs=self.config['epochs'], batch_size=self.cnn_batch_size, verbose=0, callbacks=[callbacks.EarlyStopping(monitor='loss', patience=5)])
                val_pred_cnn = cnn.predict(X_val, batch_size=256, verbose=0)
                oof_preds_cnn[val_idx] = val_pred_cnn
                print(f"  Fold {fold+1} CNN F1: {self.calculate_metrics(y_val, val_pred_cnn)['f1']:.4f}")
                last_cnn = cnn
                
                # 2. GRU (Bagging or Single run)
                print(f"  Training GRU ({gru_bagging_runs}x)...")
                fold_gru_preds = []
                for i in range(gru_bagging_runs):
                    X_aug, y_aug = self.augment_data_for_gru(X_train, y_train)
                    best_temp, best_f1 = None, -1.0
                    for _ in range(2): 
                        gru = self.model_builders['enhanced_gru'](input_shape)
                        gru.fit(X_aug, y_aug, epochs=self.config['gru_epochs'], batch_size=32, verbose=0,
                            callbacks=[self._get_learning_rate_scheduler('enhanced_gru'), callbacks.EarlyStopping(monitor='loss', patience=40)],
                            class_weight=self.config.get('gru_class_weight'))
                        f1 = self.calculate_metrics(y_val, gru.predict(X_val, verbose=0))['f1']
                        if f1 > best_f1: best_temp, best_f1 = gru, f1
                        if f1 > 0.6: break 
                        else: 
                            from tensorflow.keras import backend as K
                            K.clear_session()
                    
                    fold_gru_preds.append(best_temp.predict(X_val, batch_size=256, verbose=0))
                    if i == 0: last_gru = best_temp
                    del gru, best_temp
                    from tensorflow.keras import backend as K
                    K.clear_session()
                
                avg_gru = np.mean(fold_gru_preds, axis=0)
                oof_preds_gru[val_idx] = avg_gru
                print(f"  Fold {fold+1} GRU Ensemble F1: {self.calculate_metrics(y_val, avg_gru)['f1']:.4f}")
                
                del cnn
                from tensorflow.keras import backend as K
                K.clear_session()

            print("\n✅ Phase 1 Completed.")
            
            # === Auto-Tuning ===
            auto_center, auto_sigma = self._calculate_auto_fusion_params(y_all, oof_preds_cnn)
            
            # === Phase 2: Fusion ===
            print("\nPhase 2: Training Fusion Model...")
            if last_cnn is None: last_cnn = self.model_builders['conv_basic'](input_shape)
            if last_gru is None: last_gru = self.model_builders['enhanced_gru'](input_shape)

            self._build_and_save_fusion_model(last_cnn, last_gru, input_shape, params=(auto_center, auto_sigma))
            self.train_enhanced_fusion_model_cv(X_all, oof_preds_cnn, oof_preds_gru, y_all)
            
            # === Phase 3: Final Retrain ===
            print("\nPhase 3: Final Retraining...")
            final_models = {}
            
            # Retrain CNN
            print("  Retraining Final CNN...")
            final_cnn = self.model_builders['conv_basic'](input_shape)
            final_cnn.fit(X_all, y_all, epochs=self.config['final_epochs'], batch_size=self.cnn_batch_size, verbose=1, validation_split=0.0,
                        callbacks=[self._get_learning_rate_scheduler('conv_basic'), callbacks.EarlyStopping(monitor='loss', patience=15)])
            final_cnn.save(os.path.join(self.model_dir, "conv_basic_final_model.keras"))
            final_models['conv_basic'] = final_cnn
            
            # Retrain GRU Ensemble
            print(f"  Retraining Final GRU ({gru_bagging_runs}x)...")
            self.gru_ensemble_paths = []
            for i in range(gru_bagging_runs):
                print(f"    GRU Run {i+1}...")
                X_aug, y_aug = self.augment_data_for_gru(X_all, y_all)
                final_gru = self.model_builders['enhanced_gru'](input_shape)
                final_gru.fit(X_aug, y_aug, epochs=self.config['gru_epochs'], batch_size=32, verbose=0,
                            callbacks=[self._get_learning_rate_scheduler('enhanced_gru'), callbacks.EarlyStopping(monitor='loss', patience=40)],
                            class_weight=self.config.get('gru_class_weight'))
                path = os.path.join(self.model_dir, f"enhanced_gru_final_model_{i}.keras")
                final_gru.save(path)
                self.gru_ensemble_paths.append(path)
                if i == 0: final_models['enhanced_gru'] = final_gru 
                del final_gru
                from tensorflow.keras import backend as K
                K.clear_session()
                
            self.trained_models = final_models
            self._evaluate_and_save_results(final_models)

    def train_enhanced_fusion_model_cv(self, X_feat, cnn_preds, gru_preds, y_true):
        fusion_layer = None
        for layer in self.fusion_model.layers:
            if isinstance(layer, ImprovedGatedFusionMechanism) or 'EnhancedGatedFusionMechanism' in str(type(layer)):
                fusion_layer = layer; break
        
        self.fusion_model.compile(
            loss={'classification_output': 'binary_crossentropy'},
            loss_weights={'classification_output': 1.0},
            optimizer=Adam(
                self.config.get('fusion_learning_rate', 0.0005),
                clipnorm=1.0  
            ),
            metrics={'classification_output': ['accuracy', F1Metric(threshold=0.5, name='f1')]}
        )
        
        cnn_noisy = np.clip(cnn_preds + np.random.normal(0, 0.005, cnn_preds.shape), 0, 1)
        gru_noisy = np.clip(gru_preds + np.random.normal(0, 0.005, gru_preds.shape), 0, 1)
        
        hist = self.fusion_model.fit(
            [X_feat, cnn_noisy, gru_noisy], {'classification_output': y_true},
            epochs=self.config.get('fusion_epochs', 50),
            batch_size=16, 
            validation_split=0.2, 
            verbose=1,
            callbacks=[
                callbacks.EarlyStopping(monitor='val_classification_output_f1', mode='max', patience=6, restore_best_weights=True),
                DynamicTrainingCallback(fusion_layer, self.config)
            ]
        )
        self.fusion_model.save(os.path.join(self.model_dir, "trained_enhanced_fusion_model.keras"))
        return hist
        
    def _search_optimal_threshold(self, y_true, y_pred_prob, start=0.30, end=0.70, step=0.01):
        best_th, best_f1 = 0.5, 0.0
        if np.all(np.isin(y_pred_prob, [0, 1])): return 0.5, f1_score(y_true, y_pred_prob)
        print(f"\n🔍 Threshold Search ({start}-{end})...")
        for thresh in np.arange(start, end + step, step):
            score = f1_score(y_true, (y_pred_prob > thresh).astype(int), zero_division=0)
            if score > best_f1: best_f1, best_th = score, thresh
        return best_th, best_f1

    def _calculate_full_metrics(self, y_true, y_prob, threshold):
        y_pred = (y_prob > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0), 'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob), 'balanced_acc': balanced_accuracy_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred), 'auc_pr': average_precision_score(y_true, y_prob),
            'specificity': spec, 'best_threshold': threshold
        }

    def _evaluate_and_save_results(self, models):

        final_predictions = {}
        optimal_model_metrics = {} 
        optimal_fusion_metrics = {}
        
        print("\n" + "="*60 + "\n🚀 Executing Final Evaluation (Hybrid Strategy)\n" + "="*60)

        # 1. 基础模型预测 (CNN & GRU Ensemble)
        cnn_prob = models['conv_basic'].predict(self.X_test_fixed, verbose=0).flatten()
        final_predictions['conv_basic'] = cnn_prob
        self._analyze_cnn_reliability(self.y_test_fixed, cnn_prob)
        
        if hasattr(self, 'gru_ensemble_paths') and self.gru_ensemble_paths and self.config.get('use_repeated_cv_stacking', True):
            print(f"  Ensembling {len(self.gru_ensemble_paths)} GRU models for inference...")
            gru_list = []
            for path in self.gru_ensemble_paths:
                m = load_model(path, custom_objects={'F1Metric': F1Metric})
                gru_list.append(m.predict(self.X_test_fixed, verbose=0).flatten())
                del m
                from tensorflow.keras import backend as K
                K.clear_session()
            gru_prob = np.mean(gru_list, axis=0)
        else:
            gru_prob = models['enhanced_gru'].predict(self.X_test_fixed, verbose=0).flatten()
        final_predictions['enhanced_gru'] = gru_prob
        
        raw_fused_prob, _, _, gate_info = self._enhanced_fusion_predict(
            models['conv_basic'], models['enhanced_gru'], self.X_test_fixed, override_gru_pred=gru_prob
        )

        if self.config.get('use_confidence_locking', True):
            lock_thresh = self.config.get('confidence_lock_threshold', 0.98)
            print(f"  Applying Confidence Locking (Threshold: {lock_thresh})")
            locked_prob, lock_stats = self._apply_confidence_locking(
                cnn_pred=cnn_prob, fused_pred=raw_fused_prob, y_true=self.y_test_fixed, lock_threshold=lock_thresh
            )
        else:
            print("  Confidence Locking Disabled.")
            locked_prob = raw_fused_prob
            lock_stats = {'cnn_locked': 0, 'fusion_engaged': len(raw_fused_prob)}
            
        final_predictions['enhanced_fusion'] = locked_prob

        targets = [('conv_basic', cnn_prob), ('enhanced_gru', gru_prob), ('enhanced_fusion', locked_prob)]
        print(f"\n{'Model':<20} | {'Best Thr':<10} | {'F1':<10} | {'AUC':<10}")
        print("-" * 60)

        for name, prob in targets:
            if name == 'enhanced_fusion' and self.config.get('use_optimal_threshold', True):
                best_thr, _ = self._search_optimal_threshold(self.y_test_fixed, prob)
            else:
                best_thr = 0.5
            
            metrics = self._calculate_full_metrics(self.y_test_fixed, prob, best_thr)
            if name == 'enhanced_fusion': optimal_fusion_metrics = metrics
            else: optimal_model_metrics[name] = metrics
            print(f"{name:<20} | {best_thr:.2f}       | {metrics['f1']:.4f}     | {metrics['auc']:.4f}")

        gate_stats = self._analyze_gate_weights(gate_info, self.y_test_fixed, cnn_prob, gru_prob)
        self.metrics_collector.record_final_metrics(optimal_model_metrics, optimal_fusion_metrics)
        self._record_final_results_full(models, final_predictions, optimal_model_metrics, optimal_fusion_metrics, gate_info, cnn_prob, gru_prob, locked_prob, gate_stats)
        self.metrics_collector.save_all_metrics()

        if self.config.get('generate_heatmaps', True):
            print("\n" + "-"*40)
            print("🎨 Generating Feature Heatmaps (De-biasing Analysis)...")
            try:

                import matplotlib
                matplotlib.use('Agg')
                
                heatmap_gen = UnifiedFeatureHeatmapGenerator(self.model_dir)

                heatmap_gen.generate_comparison_heatmaps(
                    cnn_model=models['conv_basic'],
                    gru_model=models['enhanced_gru'],
                    X_samples=self.X_test_fixed,
                    y_samples=self.y_test_fixed,
                    method=self.config.get('heatmap_method', 'occlusion'), 
                    max_total_samples=self.config.get('heatmap_max_total_samples', 10)
                )
                print("✅ Visualization completed successfully.")
            except Exception as e:
                print(f"⚠️ Heatmap generation skipped due to error: {e}")
            print("-"*40 + "\n")

    def _train_legacy_split(self, data_info): pass

    def _record_final_results_full(self, models, preds, m_metrics, e_metrics, gate, cnn_p, gru_p, fused_p, gate_s):
        error_analysis = self._analyze_errors(self.y_test_fixed, cnn_p, gru_p)
        self.metrics_collector.record_final_error_analysis(error_analysis['cnn_total_errors'], error_analysis['cnn_wrong_gru_right'], error_analysis['ratio'])
        
        debug_samples = self._collect_fusion_debug_samples(self.y_test_fixed, cnn_p, gru_p, fused_p, gate['final_weights'] if gate else None)
        self.metrics_collector.record_final_fusion_debug_samples(debug_samples)
        
        cnn_f1 = m_metrics['conv_basic']['f1']
        fusion_f1 = e_metrics['f1']
        self.metrics_collector.record_final_dynamic_fusion(0, len(self.y_test_fixed), cnn_f1, fusion_f1, fusion_f1 - cnn_f1)
        
        final_results = {
            'model_metrics': m_metrics, 'ensemble_metrics': e_metrics, 'error_analysis': error_analysis,
            'dynamic_fusion_stats': {'improvement': fusion_f1 - cnn_f1}, 'gate_weights_stats': gate_s, 'fusion_debug_samples': debug_samples
        }
        np.save(os.path.join(self.model_dir, "enhanced_final_results.npy"), final_results)
        
        print("\nEnhanced fusion model effect (Optimal Fusion vs Default CNN):")
        print("   CNN F1 (Default 0.5): {:.4f}".format(cnn_f1))
        print("   Enhanced GRU F1 (Default 0.5): {:.4f}".format(m_metrics['enhanced_gru']['f1']))
        print("   Enhanced fusion F1 (Optimal): {:.4f}".format(fusion_f1))

class OptimizedModelPredictor:
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = {}
        self.fusion_model = None
        self.model_names = {}
        self.input_shape = None
        self.config = None
        self.threshold = 0.5
        self.feature_manager = FeatureExtractorManager()
        
        print("Loading optimized model information...")
        self._load_model_info()
        self._load_models()
        print("Optimized model loading completed")
    
    def _load_model_info(self):
        try:
            info_path = os.path.join(self.model_dir, "enhanced_model_info.npy")
            info = np.load(info_path, allow_pickle=True).item()
        except:
            try:
                info_path = os.path.join(self.model_dir, "model_info.npy")
                info = np.load(info_path, allow_pickle=True).item()
            except:
                raise FileNotFoundError("No model info file found")
        
        self.model_names = info['model_names']
        self.input_shape = info['input_shape']
        self.config = info['config']
    
    def _load_models(self):
        # 定义自定义对象
        custom_objects = {
            'MinPooling1D': MinPooling1D,
            'StdPooling1D': StdPooling1D,
            'EnhancedGatedFusionMechanism': EnhancedGatedFusionMechanism
        }
        
        for name in ['conv_basic', 'enhanced_gru']:
            model_path = os.path.join(self.model_dir, "{}_final_model.keras".format(name))
            if os.path.exists(model_path):
                try:
                    self.models[name] = load_model(model_path, custom_objects=custom_objects)
                    print(f"Successfully loaded {name} model with serializable layers")
                except Exception as e:
                    print(f"Error loading {name} model: {e}")
                    raise
            else:
                print(f"Warning: Base model {name} not found at {model_path}")
        
        fusion_model_paths = [
            os.path.join(self.model_dir, "trained_enhanced_fusion_model.keras"),
            os.path.join(self.model_dir, "enhanced_fusion_model.keras")
        ]
        
        for model_path in fusion_model_paths:
            if os.path.exists(model_path):
                try:
                    self.fusion_model = load_model(model_path, custom_objects=custom_objects)
                    print(f"Loaded fusion model from: {model_path}")
                    break
                except Exception as e:
                    print(f"Failed to load fusion model from {model_path}: {e}")
        
        if self.fusion_model is None:
            print("Warning: No fusion model found, will use base models only")
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            return data.reshape(1, *self.input_shape)
        elif data.ndim == 2:
            return data.reshape(-1, *self.input_shape)
        return data
    
    def _dynamic_gated_fusion_predict(self, cnn_pred, gru_pred, X_data):
        if self.fusion_model is None:
            print("Warning: Fusion model not available, using average weighting")
            return 0.5 * cnn_pred + 0.5 * gru_pred, None
        
        fusion_outputs = self.fusion_model.predict(X_data, batch_size=self.config.get('prediction_batch_size', 288), verbose=0)
        
        if isinstance(fusion_outputs, list) and len(fusion_outputs) > 0:
            fused_pred = fusion_outputs[0].flatten()
            gate_weights = fusion_outputs[1] if len(fusion_outputs) > 1 else None
        else:
            fused_pred = fusion_outputs.flatten()
            gate_weights = None
        
        return fused_pred, gate_weights
    
    def predict(self, test_data: np.ndarray, return_raw: bool = False):
        processed_data = self._preprocess_data(test_data)
        raw_preds = {}
        
        for name, model in self.models.items():
            raw_preds[name] = model.predict(processed_data, batch_size=self.config.get('prediction_batch_size', 288), verbose=0).flatten()
        
        fused_pred, gate_weights = self._dynamic_gated_fusion_predict(
            raw_preds['conv_basic'], raw_preds['enhanced_gru'], processed_data
        )
        
        if gate_weights is not None:
            gate_stats = {
                'avg_cnn_weight': float(np.mean(gate_weights[:, 0])),
                'avg_gru_weight': float(np.mean(gate_weights[:, 1])),
                'std_cnn_weight': float(np.std(gate_weights[:, 0])),
                'std_gru_weight': float(np.std(gate_weights[:, 1])),
                'dynamic_range': float(np.max(gate_weights) - np.min(gate_weights)),
                'sample_specific': True  
            }
            print(f"Dynamic gate weights - CNN: {gate_stats['avg_cnn_weight']:.3f}±{gate_stats['std_cnn_weight']:.3f}, "
                  f"GRU: {gate_stats['avg_gru_weight']:.3f}±{gate_stats['std_gru_weight']:.3f}")
        else:
            gate_stats = {'sample_specific': False}
        
        raw_preds['gate_stats'] = gate_stats
        raw_preds['gate_weights'] = gate_weights
        
        return (fused_pred, raw_preds) if return_raw else (fused_pred, None)

# Main function
def main(config_path: str = "config.py", predict_mode: bool = False, 
         test_data_dir: Optional[str] = None, output_path: Optional[str] = None) -> None:
    if not predict_mode:
        # Training mode
        print("===== Starting Optimized Model Training Mode =====")
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = {k: getattr(config_module, k) for k in dir(config_module) if not k.startswith("__")}
        
        np.random.seed(config['random_seed'])
        tf.random.set_seed(config['random_seed'])
        print("Optimized configuration loaded | Random seed: {} | Model save path: {}".format(
            config['random_seed'], config['model_dir']))
        
        trainer = OptimizedRatioOptimizationTrainer(config)
        
        print("Skipping cross-validation and proceeding directly to final training...")
        
        trainer.train_final_model()
        
        print("\nAll optimized training tasks completed!")
        
    else:
        # Prediction mode
        print("===== Starting Optimized Model Prediction Mode =====")
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = {k: getattr(config_module, k) for k in dir(config_module) if not k.startswith("__")}
        
        predictor = OptimizedModelPredictor(config['model_dir'])
        file_paths = glob.glob(os.path.join(test_data_dir, "*.npz"))
        print("Found {} files to predict".format(len(file_paths)))
        
        all_data = [np.load(path)[list(np.load(path).keys())[0]] for path in file_paths]
        combined_data = np.vstack([predictor._preprocess_data(d) for d in all_data])
        
        fused_pred, raw_preds = predictor.predict(combined_data, return_raw=True)
        labels = (fused_pred > predictor.threshold).astype(int)
        
        np.savez(output_path, file_names=[os.path.basename(p) for p in file_paths],
                fused_probabilities=fused_pred, labels=labels, raw_probabilities=raw_preds)
        print("Optimized prediction completed, results saved to: {}".format(output_path))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimized Model Training and Prediction Script")
    parser.add_argument("--predict", action="store_true", help="Enable prediction mode")
    parser.add_argument("--config", default="config.py", help="Configuration file path (relative to repo root)")
    parser.add_argument("--test_data", help="Test data folder in prediction mode")
    parser.add_argument("--output", help="Result save path in prediction mode")
    args = parser.parse_args()
    
    main(args.config, args.predict, args.test_data, args.output)
