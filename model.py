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
from tensorflow.keras.losses import BinaryCrossentropy
class FeatureExtractorManager:    
    def __init__(self):
        self.cnn_feature_extractor = None
        self.gru_feature_extractor = None 

    @staticmethod
    def get_robust_feature_layer(model, model_type='cnn'):

        target_name = f"{model_type}_feature_dense"
        
        for i, layer in enumerate(model.layers):
            if layer.name.startswith(target_name):
                print(f"🎯 [Target Lock] Selected exact feature layer for {model_type}: {layer.name} (Index {i})")
                return layer.output
                
        print(f"Warning: Exact name starting with '{target_name}' not found. Falling back to heuristics.")
        candidate_layers = []
        for i, layer in enumerate(model.layers):
            if i == 0 or i == len(model.layers) - 1:
                continue      
            
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
                
            layer_type = type(layer).__name__.lower()
            if model_type == 'cnn':
                if 'dense' in layer_type and i < len(model.layers) - 2:
                    candidate_layers.append((i, layer_output, 'dense'))
            elif model_type == 'gru':
                if 'dense' in layer_type and i < len(model.layers) - 2:
                    candidate_layers.append((i, layer_output, 'dense'))

        if candidate_layers:
            best = max(candidate_layers, key=lambda x: x[0])
            print(f"Selected fallback layer for {model_type}: layer {best[0]} ({best[2]})")
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
            raise ValueError("Extractors not initialized. Call create_xxx_feature_extractor first.")
        
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
        import numpy as np
        
        cnn_2d = cnn_features.reshape(cnn_features.shape[0], -1)
        gru_2d = gru_features.reshape(gru_features.shape[0], -1)
        
        norm_cnn = np.linalg.norm(cnn_2d, axis=1) + 1e-8
        norm_gru = np.linalg.norm(gru_2d, axis=1) + 1e-8

        cosine_sim = np.sum(cnn_2d * gru_2d, axis=1) / (norm_cnn * norm_gru)
        
        mean_sim = np.mean(cosine_sim)
        std_sim = np.std(cosine_sim)
        
        print(f"\n📊 Feature Orthogonality Analysis (Cosine Sim): {mean_sim:.4f} ± {std_sim:.4f}")
        
        if abs(mean_sim) > 0.7:
            print("⚠️ WARNING: High correlation detected! Models might be learning similar manifolds.")
        elif abs(mean_sim) > 0.4:
            print("🟡 NOTE: Moderate feature correlation. Router relies on fine-grained disagreement.")
        else:
            print("✅ Excellent! Distinct, orthogonal representations confirmed (Low Cosine Similarity).")

    def save_extractors(self, save_dir):
        if self.cnn_feature_extractor is not None:
            cnn_path = os.path.join(save_dir, 'cnn_feature_extractor.keras')
            self.cnn_feature_extractor.save(cnn_path)
            print(f"CNN feature extractor saved to: {cnn_path}")
        
        if self.gru_feature_extractor is not None:
            gru_path = os.path.join(save_dir, 'gru_feature_extractor.keras')
            self.gru_feature_extractor.save(gru_path)
            print(f"GRU feature extractor saved to: {gru_path}")
    
    def load_extractors(self, save_dir, custom_objects=None):
        cnn_path = os.path.join(save_dir, 'cnn_feature_extractor.keras')
        gru_path = os.path.join(save_dir, 'gru_feature_extractor.keras')
        
        if os.path.exists(cnn_path):
            self.cnn_feature_extractor = load_model(cnn_path, custom_objects=custom_objects)
            print(f"CNN feature extractor loaded from: {cnn_path}")
        
        if os.path.exists(gru_path):
            self.gru_feature_extractor = load_model(gru_path, custom_objects=custom_objects)
            print(f"GRU feature extractor loaded from: {gru_path}")

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
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import UnitNorm

@tf.keras.utils.register_keras_serializable()
class EnhancedGatedFusionMechanism(Layer):

    def __init__(self, fusion_units=16, dropout_rate=0.2, gate_l2_reg=0.005, 
                 entropy_reg_weight=0.0001, **kwargs):
        super().__init__(**kwargs)
        self.fusion_units = fusion_units
        self.dropout_rate = dropout_rate
        self.gate_l2_reg = gate_l2_reg
        self.entropy_reg_weight = entropy_reg_weight # 保持软路由

    def build(self, input_shape):
        self.cnn_norm = LayerNormalization(epsilon=1e-6)
        self.gru_norm = LayerNormalization(epsilon=1e-6)
        
        self.cnn_proj = Dense(self.fusion_units, activation='gelu', 
                              kernel_regularizer=regularizers.l2(self.gate_l2_reg),
                              kernel_constraint=UnitNorm(axis=0))
        self.gru_proj = Dense(self.fusion_units, activation='gelu', 
                              kernel_regularizer=regularizers.l2(self.gate_l2_reg),
                              kernel_constraint=UnitNorm(axis=0))
        
        self.cnn_feat_compressor = Dense(1, activation='tanh', kernel_regularizer=regularizers.l2(self.gate_l2_reg))
        self.gru_feat_compressor = Dense(1, activation='tanh', kernel_regularizer=regularizers.l2(self.gate_l2_reg))
        
        self.router_drop = Dropout(self.dropout_rate)
        
        self.router_dense = Dense(8, activation='gelu', kernel_regularizer=regularizers.l2(self.gate_l2_reg))
        self.router_out = Dense(2, activation='softmax', name='expert_gate')
        
        super().build(input_shape)

    def _to_logits(self, p, eps=1e-7):
        p = tf.clip_by_value(p, eps, 1.0 - eps)
        return tf.clip_by_value(tf.math.log(p / (1.0 - p)), -8.0, 8.0)

    def call(self, inputs, training=False):
        cnn_features, gru_features, cnn_pred, gru_pred = inputs
        epsilon = 1e-7
        cnn_pred = tf.clip_by_value(cnn_pred, epsilon, 1.0 - epsilon)
        gru_pred = tf.clip_by_value(gru_pred, epsilon, 1.0 - epsilon)
        
        h_cnn = tf.math.l2_normalize(self.cnn_proj(self.cnn_norm(cnn_features)), axis=-1)
        h_gru = tf.math.l2_normalize(self.gru_proj(self.gru_norm(gru_features)), axis=-1)
        cnn_feat_sum = self.cnn_feat_compressor(h_cnn)
        gru_feat_sum = self.gru_feat_compressor(h_gru)
            
        cnn_logit = self._to_logits(cnn_pred)
        gru_logit = self._to_logits(gru_pred)
        logit_diff = tf.abs(cnn_logit - gru_logit)
        
        cnn_conf = tf.abs(cnn_pred - 0.5) * 2.0
        gru_conf = tf.abs(gru_pred - 0.5) * 2.0
        
        router_context = tf.concat([
            cnn_logit,    
            gru_logit,    
            cnn_conf,     
            gru_conf,     
            logit_diff,   
            cnn_feat_sum, 
            gru_feat_sum
        ], axis=-1)       
        
        if training:
            router_context = self.router_drop(router_context, training=training)
            
        gate_weights = self.router_out(self.router_dense(router_context)) 
        w_cnn = gate_weights[:, 0:1]
        w_gru = gate_weights[:, 1:2]
        
        if training:
            gate_entropy = -tf.reduce_mean(tf.reduce_sum(gate_weights * tf.math.log(gate_weights + 1e-7), axis=-1))
            self.add_loss(self.entropy_reg_weight * gate_entropy) 
        
        final_logit = w_cnn * cnn_logit + w_gru * gru_logit
        final_pred = tf.math.sigmoid(final_logit)
        
        confidence_adj = tf.concat([cnn_conf, gru_conf], axis=-1)
        dummy_temps = tf.ones_like(confidence_adj)
        
        return final_pred, gate_weights, dummy_temps, confidence_adj

    def get_config(self):
        config = super().get_config()
        config.update({
            'fusion_units': self.fusion_units, 'dropout_rate': self.dropout_rate,
            'gate_l2_reg': self.gate_l2_reg, 'entropy_reg_weight': self.entropy_reg_weight
        })
        return config

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
            # balanced_acc / balanced_accuracy
            if key == 'balanced_accuracy' and val == 0.0:
                val = m_dict.get('balanced_acc', 0.0)
            return val
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
                    'best_threshold': get_metric(metrics, 'best_threshold') 
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

import numpy as np
import random
import os

class DualStreamDataProcessor:
    def __init__(self, config: dict):
        self.config = config
        random.seed(config['random_seed'])
        np.random.seed(config['random_seed'])
        
        print("\n===(Multi-view Pipeline) ===")
        

        self.pos_train_cnn, self.pos_train_gru = self._load_paired_data(
            config['train_pos_esm'], config['train_pos_aaindex']
        )
        self.pos_test_cnn, self.pos_test_gru = self._load_paired_data(
            config['test_pos_esm'], config['test_pos_aaindex']
        )
        print(f"✅ Positive: Train {len(self.pos_train_cnn)} 条, Test {len(self.pos_test_cnn)} 条")
        
        self.neg_train_pool_esm = self._read_list(config['train_neg_esm'])
        self.neg_train_pool_aaindex = self._read_list(config['train_neg_aaindex'])
        assert len(self.neg_train_pool_esm) == len(self.neg_train_pool_aaindex)
        print(f"✅ negtive: 共 {len(self.neg_train_pool_esm)} ")

        self.neg_test_cnn, self.neg_test_gru = self._load_paired_data(
            config['test_neg_esm'], config['test_neg_aaindex']
        )
        

        self._preload_fixed_test_set()

    def _read_list(self, list_path: str) -> list:
        if not list_path or not os.path.exists(list_path):
            raise FileNotFoundError(f"Missing list file: {list_path}")
        with open(list_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _load_paired_data(self, esm_list_path: str, aaindex_list_path: str, indices=None):
        esm_files = self._read_list(esm_list_path) if isinstance(esm_list_path, str) else esm_list_path
        aa_files = self._read_list(aaindex_list_path) if isinstance(aaindex_list_path, str) else aaindex_list_path
        
        if indices is not None:
            esm_files = [esm_files[i] for i in indices]
            aa_files = [aa_files[i] for i in indices]

        cnn_features, gru_features = [], []
        
        for esm_f, aa_f in zip(esm_files, aa_files):
            try:
                data_esm = np.load(esm_f)
                feat_esm = data_esm[list(data_esm.keys())[0]]
                
                if feat_esm.ndim == 1:
                    feat_esm = feat_esm.reshape(1, 1280, 1)
                elif feat_esm.ndim == 2:
                    feat_esm = feat_esm.reshape(1, 1280, 1)
                else:
                    feat_esm = np.resize(feat_esm, (1, 1280, 1))
                    
                cnn_features.append(feat_esm)
                
                data_aa = np.load(aa_f)
                feat_aa = data_aa['embedding'] 
                gru_features.append(feat_aa)
                
            except Exception as e:
                print(f"data fail: {esm_f} 或 {aa_f} -> {e}")
                continue
                
        return cnn_features, gru_features

    def get_dynamic_training_data(self, pos_multiplier=1.0):


        n_pos = len(self.pos_train_cnn)
        sample_size = int(n_pos * pos_multiplier)
        
        pool_size = len(self.neg_train_pool_esm)
        sampled_indices = random.sample(range(pool_size), min(sample_size, pool_size))
        
        neg_cnn, neg_gru = self._load_paired_data(
            self.neg_train_pool_esm, 
            self.neg_train_pool_aaindex, 
            indices=sampled_indices
        )
        
        X_cnn = np.vstack(self.pos_train_cnn + neg_cnn)
        
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        raw_gru = self.pos_train_gru + neg_gru
        X_gru = pad_sequences(raw_gru, padding='post', dtype='float32') # 补齐 0
        
        y = np.hstack([np.ones(n_pos), np.zeros(len(neg_cnn))])

        shuffle_idx = np.random.permutation(len(y))
        
        X_cnn_shuffled = X_cnn[shuffle_idx]
        X_gru_shuffled = X_gru[shuffle_idx]
        y_shuffled = y[shuffle_idx]

        try:
            print("\n" + "="*40)
            print(" 🔍 GRU Input Sanity Check")
            print("="*40)

            sample_col_0 = X_gru_shuffled[:100, :, 0].flatten()
            sample_col_0 = sample_col_0[sample_col_0 != 0] # 排除 padding 的 0
            unique_vals = np.unique(sample_col_0)
            print(f"Unique values in GRU column 0 (sample): {unique_vals[:20]}")
            
            if len(unique_vals) <= 22 and all(float(v).is_integer() for v in unique_vals[:5]):
                print("⚠️ DIAGNOSIS: Column 0 appears to be categorical AA Indices (0~20). Embedding is REQUIRED.")
            else:
                print("✅ DIAGNOSIS: Column 0 appears to be continuous physical/encoded features. DO NOT use Embedding.")
            print("="*40 + "\n")
        except Exception as e:
            print(f"Sanity Check Error: {e}")

        return X_cnn_shuffled, X_gru_shuffled, y_shuffled


    def _preload_fixed_test_set(self):
        X_cnn = np.vstack(self.pos_test_cnn + self.neg_test_cnn)
        
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        raw_gru = self.pos_test_gru + self.neg_test_gru
        X_gru = pad_sequences(raw_gru, padding='post', dtype='float32')
        
        y = np.hstack([np.ones(len(self.pos_test_cnn)), np.zeros(len(self.neg_test_cnn))])
        
        self.fixed_test_data = (X_cnn, X_gru, y)
        
    def get_fixed_test_data(self):
        return self.fixed_test_data


def build_conv_basic_net(input_shape, config):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, 5, padding='same', activation='gelu', kernel_regularizer=l2(config['cnn_l2_regularization']))(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)  # 改为 2
    x = Dropout(0.3)(x) 

    x = Conv1D(32, 3, padding='same', activation='gelu', kernel_regularizer=l2(config['cnn_l2_regularization']))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)  # 改为 2
    x = Dropout(0.3)(x)

    x = Flatten()(x) 
    x = Dense(32, activation='gelu', kernel_regularizer=l2(config['cnn_l2_regularization']), name='cnn_feature_dense')(x)
    
    x = Dropout(config['cnn_dropout_rate'])(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    from tensorflow.keras.losses import BinaryCrossentropy
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=BinaryCrossentropy(label_smoothing=0.05), 
        optimizer=Adam(config['cnn_learning_rate'], clipnorm=1.0), 
        metrics=['accuracy']
    )
    return model

def build_enhanced_gru(input_shape, config): 
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Bidirectional, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Dense, Dropout, Embedding, Lambda, SpatialDropout1D
    from tensorflow.keras.models import Model
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    inputs = Input(shape=input_shape, name='gru_raw_inputs')
    l2_reg = config.get('gru_l2_reg', 0.0001)
    
    identity_feat = Lambda(lambda x: tf.cast(x[:, :, 0], tf.int32), name='slice_identity_cast')(inputs)
    phys_pos_feat = Lambda(lambda x: x[:, :, 1:], name='slice_phys_pos')(inputs)
    
    embed_id = Embedding(input_dim=21, output_dim=4, name='aa_identity_embed')(identity_feat)
    
    x = Concatenate(axis=-1, name='concat_features')([embed_id, phys_pos_feat])
    
    x = SpatialDropout1D(0.2, name='spatial_dropout')(x)
    
    x = Bidirectional(GRU(32, return_sequences=True, dropout=0.2, 
                          kernel_regularizer=l2(l2_reg)), name='physico_gru')(x)
    
    avg_pool = GlobalAveragePooling1D(name='gru_avg_pool')(x) 
    max_pool = GlobalMaxPooling1D(name='gru_max_pool')(x)     
    
    x = Concatenate(axis=-1, name='concat_pooling')([avg_pool, max_pool])
    x = Dense(32, activation='gelu', kernel_regularizer=l2(l2_reg), name='gru_feature_dense')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='enhanced_gru')
    model.compile(
        loss=BinaryCrossentropy(label_smoothing=0.05),
        optimizer=Adam(learning_rate=config.get('gru_initial_lr', 0.0005), clipnorm=1.0), 
        metrics=['accuracy']
    )
    return model

def build_enhanced_fusion_model(cnn_input_shape, gru_input_shape, config, cnn_model, gru_model):

    in_cnn = Input(shape=cnn_input_shape, name='cnn_raw_inputs')
    in_gru = Input(shape=gru_input_shape, name='gru_raw_inputs')
    in_c_pred = Input(shape=(1,), name='cnn_pred_input')
    in_g_pred = Input(shape=(1,), name='gru_pred_input')

    cnn_model.trainable = False
    gru_model.trainable = False
    
    cnn_feat_out = FeatureExtractorManager.get_robust_feature_layer(cnn_model, 'cnn')
    cnn_ext = Model(inputs=cnn_model.input, outputs=cnn_feat_out)
    
    gru_feat_out = FeatureExtractorManager.get_robust_feature_layer(gru_model, 'gru')
    gru_ext = Model(inputs=gru_model.input, outputs=gru_feat_out)
    
    cnn_features = cnn_ext(in_cnn)
    gru_features = gru_ext(in_gru)

    fusion_layer = EnhancedGatedFusionMechanism(
        fusion_units=config.get('fusion_units', 32),
        dropout_rate=config.get('fusion_dropout', 0.3),
        gate_l2_reg=config.get('gate_l2_reg', 0.0005)
    )

    fused_prediction, final_gate_weights, base_gate_weights, confidence_adj = fusion_layer([
        cnn_features, gru_features, in_c_pred, in_g_pred
    ])
    
    classification_output = Activation('linear', name='classification_output')(fused_prediction)

    fusion_model = Model(
        inputs=[in_cnn, in_gru, in_c_pred, in_g_pred],
        outputs=[
            classification_output, 
            final_gate_weights, 
            base_gate_weights, 
            confidence_adj
        ],
        name='enhanced_fusion_model'
    )
    return fusion_model

class RatioOptimizationTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.data_processor = None 
        
        self.model_dir = config['model_dir']
        os.makedirs(self.model_dir, exist_ok=True)
        self.verbose = config['verbose']
        self.cnn_batch_size = config['batch_size']
        self.gru_batch_size = config['gru_batch_size']
        self.prediction_batch_size = config['prediction_batch_size']
        self.model_names = {'conv': 'conv_basic', 'gru': 'optimized_gru'}
        self.model_builders = {
            'conv_basic': lambda shape: build_conv_basic_net(shape, self.config),
            'optimized_gru': lambda shape: build_enhanced_gru(shape, self.config) 
        }
        
        self.trained_models = {}
        self.metrics_collector = TrainingMetricsCollector(self.model_dir)
        
        self.enable_fusion_debug = config.get('enable_fusion_debug', True)
        self.max_debug_samples = config.get('max_debug_samples', 100)
        self.debug_sample_types = config.get('debug_sample_types', ['cnn_wrong_gru_right', 'both_wrong', 'cnn_right_gru_wrong'])
    
    def _analyze_errors(self, y_true, cnn_pred, gru_pred, fused_pred=None, threshold=0.5):
        y_true = y_true.astype(int)
        cnn_binary = (cnn_pred > threshold).astype(int)
        gru_binary = (gru_pred > threshold).astype(int)
        
        err_cnn = (cnn_binary != y_true).astype(int)
        err_gru = (gru_binary != y_true).astype(int)
        
        cnn_errors = np.sum(err_cnn)
        gru_errors = np.sum(err_gru)
        
        cnn_wrong_gru_right = np.sum((err_cnn == 1) & (err_gru == 0))
        cnn_right_gru_wrong = np.sum((err_cnn == 0) & (err_gru == 1))
        both_wrong = np.sum((err_cnn == 1) & (err_gru == 1))
        both_correct = np.sum((err_cnn == 0) & (err_gru == 0))
        
        disagreement_mask = (err_cnn != err_gru)
        disagreement_ratio = np.mean(disagreement_mask)
        ratio = cnn_wrong_gru_right / cnn_errors if cnn_errors > 0 else 0.0
        
        try:
            from scipy.stats import pearsonr
            error_correlation, _ = pearsonr(err_cnn, err_gru)
        except Exception:
            error_correlation = 0.0
            
        print("\n" + "="*60)
        print(" 🔍 Complementarity & Conditional Fusion Analysis")
        print("="*60)
        print(f" Both Correct: {both_correct} | Both Wrong: {both_wrong} (Hard limits)")
        print(f" CNN Only Wrong: {cnn_wrong_gru_right} | GRU Only Wrong: {cnn_right_gru_wrong}")
        print(f" Disagreement Ratio: {disagreement_ratio:.2%} (Target > 5-10%)")
        print(f" Error Correlation:  {error_correlation:.4f} (Target < 0.85)")
        
        if fused_pred is not None:
            fused_binary = (fused_pred > threshold).astype(int)
            fused_correct_mask = (fused_binary == y_true)
            p_fusion_given_disagree = np.mean(fused_correct_mask[disagreement_mask]) if np.sum(disagreement_mask) > 0 else 0.0
            p_fusion_given_cnn_w_gru_r = np.mean(fused_correct_mask[(err_cnn == 1) & (err_gru == 0)]) if cnn_wrong_gru_right > 0 else 0.0
            
            print("-" * 60)
            print(f" P(Fusion Correct | Disagreement): {p_fusion_given_disagree:.2%}")
            print(f" P(Fusion Correct | CNN Wrong, GRU Right): {p_fusion_given_cnn_w_gru_r:.2%}")
            
        print("="*60 + "\n")
        
        return {
            'cnn_total_errors': int(cnn_errors),
            'cnn_wrong_gru_right': int(cnn_wrong_gru_right),
            'ratio': float(ratio), 
            'error_correlation': float(error_correlation),
            'disagreement_ratio': float(disagreement_ratio)
        }
    
    def _collect_fusion_debug_samples(self, y_true, cnn_pred, gru_pred, fused_pred=None, gate_info=None, threshold=0.5):
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
                
                if gate_info is not None:
                    if 'final_weights' in gate_info and idx < len(gate_info['final_weights']):
                        sample_info['cnn_gate_weight'] = float(gate_info['final_weights'][idx][0])
                        sample_info['gru_gate_weight'] = float(gate_info['final_weights'][idx][1])
                    if 'instance_temps' in gate_info and idx < len(gate_info['instance_temps']):
                        sample_info['cnn_temp'] = float(gate_info['instance_temps'][idx][0])
                        sample_info['gru_temp'] = float(gate_info['instance_temps'][idx][1])
                
                debug_samples.append(sample_info)
        
        print(f"Collected {len(debug_samples)} fusion debug samples")
        return debug_samples
    
    def calculate_metrics(self, y_true, y_pred, threshold=0.5):
        from sklearn.metrics import brier_score_loss
        
        y_true = np.array(y_true).astype(int).flatten()
        y_pred = np.array(y_pred).flatten()
        
        y_pred_binary = (y_pred > threshold).astype(int)

        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)  
        accuracy = accuracy_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_true, y_pred_binary)  
        mcc = matthews_corrcoef(y_true, y_pred_binary)                 
        
        try:
            auc_roc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc_roc = 0.5

        try:
            precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_pred)
            auc_pr = auc(recall_pr, precision_pr)
        except ValueError:
            auc_pr = 0.5

        cm = confusion_matrix(y_true, y_pred_binary)
        if cm.shape == (2, 2):
            tn, fp, _, _ = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0

        brier = brier_score_loss(y_true, y_pred)
        
        def expected_calibration_error(y_true_1d, y_prob_1d, n_bins=10):
            ece = 0.0
            bins = np.linspace(0., 1., n_bins + 1)
            binned = np.digitize(y_prob_1d, bins) - 1
            for b in range(n_bins):
                mask = (binned == b)
                if np.any(mask):
                    acc = np.mean(y_true_1d[mask] == (y_prob_1d[mask] > 0.5))
                    conf = np.mean(y_prob_1d[mask])
                    ece += np.abs(acc - conf) * np.sum(mask) / len(y_true_1d)
            return float(ece)
            
        ece = expected_calibration_error(y_true, y_pred)

        return {
            'precision': float(precision), 'recall': float(recall), 'accuracy': float(accuracy),
            'f1': float(f1), 'auc': float(auc_roc), 'balanced_accuracy': float(balanced_acc),
            'mcc': float(mcc), 'auc_pr': float(auc_pr), 'specificity': float(specificity),
            'brier_score': float(brier), 'ece': float(ece) 
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

from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import time
import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import time
import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.optimizers import Adam

class OptimizedRatioOptimizationTrainer(RatioOptimizationTrainer):
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.fusion_model = None
        self.data_processor = DualStreamDataProcessor(config)
        self.model_names['gru'] = 'enhanced_gru'
        self.gru_ensemble_paths = [] 
        
        self.model_builders = {
            'conv_basic': lambda shape: build_conv_basic_net(shape, self.config),
            'enhanced_gru': lambda shape: build_enhanced_gru(shape, self.config)
        }
        
        self.X_cnn_test, self.X_gru_test, self.y_test_fixed = self.data_processor.get_fixed_test_data()

    def augment_data_for_gru(self, X, y, augmentation_factor=0.0):
        return X, y

    def _get_learning_rate_scheduler(self, model_name):
        if model_name == 'enhanced_gru' or model_name == 'optimized_gru':
            def gru_sch(epoch, lr):
                initial_lr = float(self.config.get('gru_initial_lr', 0.0005))
                sch_cfg = self.config.get('gru_lr_schedule', {})
                warmup = sch_cfg.get('warmup_epochs', 5)
                decay_start = sch_cfg.get('decay_start_epoch', 20)
                decay_factor = sch_cfg.get('decay_factor', 0.8)
                min_lr = sch_cfg.get('min_lr', 1e-6)
                
                if epoch < warmup: return initial_lr * (epoch + 1) / warmup
                elif epoch < decay_start: return initial_lr
                else: return max(initial_lr * (decay_factor ** ((epoch - decay_start) // 8)), min_lr)
            return callbacks.LearningRateScheduler(gru_sch, verbose=0)
        else:
            initial_lr = self.config.get('learning_rate', 0.001)
            def cnn_sch(epoch, lr):
                T = max(self.config.get('epochs', 50), self.config.get('final_epochs', 50))
                min_lr = initial_lr * 0.01
                progress = min(epoch / T, 1.0)
                return max(min_lr + (initial_lr - min_lr) * (1 - progress) * (1 + np.cos(np.pi * progress)) / 2, min_lr)
            return callbacks.LearningRateScheduler(cnn_sch, verbose=0)

    def _build_and_save_fusion_model(self, cnn_model, gru_model, cnn_shape, gru_shape):
        print("\nBuilding and saving enhanced fusion model...")
        self.fusion_model = build_enhanced_fusion_model(
            cnn_shape, gru_shape, self.config, cnn_model, gru_model
        )
        path = os.path.join(self.model_dir, "enhanced_fusion_model.keras")
        self.fusion_model.save(path)
        print(f"Enhanced fusion model saved to: {path}")
        return self.fusion_model

    def _enhanced_fusion_predict(self, cnn_model, gru_model, X_cnn, X_gru, override_gru_pred=None):
        if self.fusion_model is None:
            path = os.path.join(self.model_dir, "trained_enhanced_fusion_model.keras")
            if not os.path.exists(path): path = os.path.join(self.model_dir, "enhanced_fusion_model.keras")
            try:
                custom = {
                    'EnhancedGatedFusionMechanism': EnhancedGatedFusionMechanism,
                    'get_robust_feature_layer': FeatureExtractorManager.get_robust_feature_layer,
                    'F1Metric': F1Metric
                }
                self.fusion_model = load_model(path, custom_objects=custom)
            except Exception as e:
                print(f"Error loading fusion model: {e}")
                self.fusion_model = build_enhanced_fusion_model(X_cnn.shape[1:], X_gru.shape[1:], self.config, cnn_model, gru_model)

        cnn_pred = cnn_model.predict(X_cnn, batch_size=256, verbose=0).flatten()
        if override_gru_pred is not None: gru_pred = override_gru_pred
        else: gru_pred = gru_model.predict(X_gru, batch_size=256, verbose=0).flatten()
        
        fusion_outputs = self.fusion_model.predict(
            [X_cnn, X_gru, cnn_pred.reshape(-1,1), gru_pred.reshape(-1, 1)],
            batch_size=256, verbose=0
        )
        
        if isinstance(fusion_outputs, list) and len(fusion_outputs) >= 4:
            return fusion_outputs[0].flatten(), cnn_pred, gru_pred, {
                'final_weights': fusion_outputs[1], 
                'instance_temps': fusion_outputs[2], 
                'confidence_adjustment': fusion_outputs[3]
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
        w_cnn, w_gru = fw[:, 0], fw[:, 1]
        
        print(f"\n=== Routing & Calibration Evaluation ===")
        print(f"Overall Allocation - CNN: {np.mean(w_cnn):.3f} | GRU: {np.mean(w_gru):.3f}")
        
        cnn_correct = (cnn_pred > 0.5) == y_true
        if np.sum(cnn_correct) > 0: print(f"Mean CNN Weight when CNN is CORRECT: {np.mean(w_cnn[cnn_correct]):.3f}")
        if np.sum(~cnn_correct) > 0: print(f"Mean CNN Weight when CNN is WRONG  : {np.mean(w_cnn[~cnn_correct]):.3f}")
        
        if 'instance_temps' in gate_info:
            temps = gate_info['instance_temps']
            print(f"Mean Temperature   - CNN: {np.mean(temps[:, 0]):.3f} | GRU: {np.mean(temps[:, 1]):.3f}")
            
        return {
            'final_weights': {
                'cnn_mean': float(np.mean(w_cnn)), 'gru_mean': float(np.mean(w_gru)), 
                'cnn_std': float(np.std(w_cnn)), 'gru_std': float(np.std(w_gru))
            }
        }

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
        X_cnn_all, X_gru_all, y_all = self.data_processor.get_dynamic_training_data()
        self._train_with_cv_stacking(X_cnn_all, X_gru_all, y_all)

    def _train_with_cv_stacking(self, X_cnn_all, X_gru_all, y_all):
        print("\n" + "="*60 + "\n🚀 Starting Strict Holdout Meta-Learning (Dual-Stream)\n" + "="*60)
        
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(y_all))
        
        idx_base, idx_meta = train_test_split(indices, test_size=0.20, stratify=y_all, random_state=42)
        
        idx_in_train, idx_in_val = train_test_split(idx_base, test_size=0.125, stratify=y_all[idx_base], random_state=42)
        
        X_c_tr, X_g_tr, y_tr = X_cnn_all[idx_in_train], X_gru_all[idx_in_train], y_all[idx_in_train]
        X_c_val, X_g_val, y_val = X_cnn_all[idx_in_val], X_gru_all[idx_in_val], y_all[idx_in_val]
        X_c_meta, X_g_meta, y_meta = X_cnn_all[idx_meta], X_gru_all[idx_meta], y_all[idx_meta]

        cnn_shape = X_c_tr.shape[1:]
        gru_shape = X_g_tr.shape[1:]
        
        print(f"Inner Train Set: {len(X_c_tr)} | Inner Val (for ES): {len(X_c_val)} | Meta-OOF Set: {len(X_c_meta)}")
        self.metrics_collector.init_final_training(list(self.model_names.values()) + ['enhanced_fusion'])
        
        print("\n" + "-"*40 + "\n🤖 Training Canonical CNN Base Expert...\n" + "-" * 40)
        cnn = self.model_builders['conv_basic'](cnn_shape)
        cnn.fit(X_c_tr, y_tr, epochs=self.config['epochs'], batch_size=self.cnn_batch_size, verbose=1,
                validation_data=(X_c_val, y_val), 
                callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                           self._get_learning_rate_scheduler('conv_basic')])
                           
        val_cnn_f1 = self.calculate_metrics(y_meta, cnn.predict(X_c_meta, batch_size=256, verbose=0).flatten())['f1']
        print(f"✅ Canonical CNN Meta-OOF F1: {val_cnn_f1:.4f}")

        cnn.save(os.path.join(self.model_dir, "conv_basic_final_model.keras"))

        print("\n" + "-"*40 + "\n🧬 Training Canonical GRU Base Expert...\n" + "-" * 40)
        gru = self.model_builders['enhanced_gru'](gru_shape)
        gru_bs = self.config.get('gru_batch_size', 64)
        
        gru.fit(X_g_tr, y_tr, epochs=self.config.get('gru_epochs', 120), batch_size=gru_bs, verbose=1,
                validation_data=(X_g_val, y_val), class_weight=self.config.get('gru_class_weight'),
                callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                           self._get_learning_rate_scheduler('enhanced_gru')])
                           
        val_gru_f1 = self.calculate_metrics(y_meta, gru.predict(X_g_meta, batch_size=256, verbose=0).flatten())['f1']
        print(f"✅ Canonical GRU Meta-OOF F1: {val_gru_f1:.4f}")
        
        gru.save(os.path.join(self.model_dir, "enhanced_gru_final_model.keras"))

        print("\n--- Extracting STRICT Canonical Meta Features ---")
        cnn_feat_layer = FeatureExtractorManager.get_robust_feature_layer(cnn, 'cnn')
        cnn_ext = Model(inputs=cnn.input, outputs=cnn_feat_layer)
        oof_feats_cnn = cnn_ext.predict(X_c_meta, batch_size=256, verbose=0)
        oof_preds_cnn = cnn.predict(X_c_meta, batch_size=256, verbose=0)
        
        gru_feat_layer = FeatureExtractorManager.get_robust_feature_layer(gru, 'gru')
        gru_ext = Model(inputs=gru.input, outputs=gru_feat_layer)
        oof_feats_gru = gru_ext.predict(X_g_meta, batch_size=256, verbose=0)
        oof_preds_gru = gru.predict(X_g_meta, batch_size=256, verbose=0)
        
        print("\n--- Training Fusion Model on STRICT Canonical Meta Features ---")
        self.train_enhanced_fusion_model_cv(oof_feats_cnn, oof_feats_gru, oof_preds_cnn, oof_preds_gru, y_meta)
        
        print("\n 🧩 Assembling End-to-End Fusion Model for Inference...")
        final_models = {'conv_basic': cnn, 'enhanced_gru': gru}
        self.fusion_model = self._assemble_e2e_fusion_model(cnn, gru, cnn_shape, gru_shape)
        self.fusion_model.save(os.path.join(self.model_dir, "enhanced_fusion_model.keras"))
        
        self.trained_models = final_models
        self.gru_ensemble_paths = []
        self._evaluate_and_save_results(final_models)

    def train_enhanced_fusion_model_cv(self, oof_feat_cnn, oof_feat_gru, cnn_preds, gru_preds, y_true):
        self.trained_fusion_layer = EnhancedGatedFusionMechanism(
            fusion_units=self.config.get('fusion_units', 16),
            dropout_rate=self.config.get('fusion_dropout', 0.2),
            gate_l2_reg=self.config.get('gate_l2_reg', 0.005)
        )
        in_c_feat = Input(shape=(oof_feat_cnn.shape[1],), name='in_c_feat')
        in_g_feat = Input(shape=(oof_feat_gru.shape[1],), name='in_g_feat')
        in_c_pred = Input(shape=(1,), name='in_c_pred')
        in_g_pred = Input(shape=(1,), name='in_g_pred')
        
        outputs = self.trained_fusion_layer([in_c_feat, in_g_feat, in_c_pred, in_g_pred])
        train_model = Model([in_c_feat, in_g_feat, in_c_pred, in_g_pred], outputs[0]) 
        
        train_model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(self.config.get('fusion_learning_rate', 0.0003), clipnorm=1.0),
            metrics=['accuracy', F1Metric(threshold=0.5, name='f1')]
        )
        
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(y_true))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=y_true, random_state=42)
        
        hist = train_model.fit(
            [oof_feat_cnn[train_idx], oof_feat_gru[train_idx], cnn_preds[train_idx], gru_preds[train_idx]], 
            y_true[train_idx], 
            epochs=self.config.get('fusion_epochs', 50),
            batch_size=self.config.get('fusion_batch_size', 64), 
            validation_data=(
                [oof_feat_cnn[val_idx], oof_feat_gru[val_idx], cnn_preds[val_idx], gru_preds[val_idx]],
                y_true[val_idx]
            ),
            verbose=1,
            callbacks=[
                callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=15, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-6)
            ]
        )
        self.learned_optimal_threshold = 0.5 
        return hist

    def _assemble_e2e_fusion_model(self, final_cnn, final_gru, cnn_shape, gru_shape):
        in_cnn = Input(shape=cnn_shape, name='cnn_raw_inputs')
        in_gru = Input(shape=gru_shape, name='gru_raw_inputs')
        in_c_pred = Input(shape=(1,), name='cnn_pred_input')
        in_g_pred = Input(shape=(1,), name='gru_pred_input')
        
        final_cnn.trainable = False
        final_gru.trainable = False
        
        cnn_ext = Model(final_cnn.input, FeatureExtractorManager.get_robust_feature_layer(final_cnn, 'cnn'))
        gru_ext = Model(final_gru.input, FeatureExtractorManager.get_robust_feature_layer(final_gru, 'gru'))
        
        cnn_features = cnn_ext(in_cnn)
        gru_features = gru_ext(in_gru)
        
        outputs = self.trained_fusion_layer([cnn_features, gru_features, in_c_pred, in_g_pred])
        
        e2e_model = Model(
            inputs=[in_cnn, in_gru, in_c_pred, in_g_pred],
            outputs=[outputs[0], outputs[1], outputs[2], outputs[3]], 
            name='enhanced_fusion_model'
        )
        return e2e_model
        
    def _search_optimal_threshold(self, y_true, y_pred_prob, start=0.40, end=0.60, step=0.01):
        best_th, best_f1 = 0.5, 0.0
        if np.all(np.isin(y_pred_prob, [0, 1])): return 0.5, f1_score(y_true, y_pred_prob)
        for thresh in np.arange(start, end + step, step):
            score = f1_score(y_true, (y_pred_prob > thresh).astype(int), zero_division=0)
            if score > best_f1: best_f1, best_th = score, thresh
        return best_th, best_f1

    def _calculate_full_metrics(self, y_true, y_prob, threshold):
        y_pred = (y_prob > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(y_true, y_prob)
        
        def expected_calibration_error(y_true_1d, y_prob_1d, n_bins=10):
            ece = 0.0
            bins = np.linspace(0., 1., n_bins + 1)
            binned = np.digitize(y_prob_1d, bins) - 1
            for b in range(n_bins):
                mask = (binned == b)
                if np.any(mask):
                    acc = np.mean(y_true_1d[mask] == (y_prob_1d[mask] > 0.5))
                    conf = np.mean(y_prob_1d[mask])
                    ece += np.abs(acc - conf) * np.sum(mask) / len(y_true_1d)
            return float(ece)
            
        ece = expected_calibration_error(y_true, y_prob)

        return {
            'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0), 'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob), 'balanced_acc': balanced_accuracy_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred), 'auc_pr': average_precision_score(y_true, y_prob),
            'specificity': spec, 'best_threshold': threshold,
            'brier_score': float(brier), 'ece': float(ece) 
        }

    def _evaluate_and_save_results(self, models):
        final_predictions = {}
        optimal_model_metrics = {} 
        optimal_fusion_metrics = {}
        
        print("\n" + "="*60 + "\n🚀 Executing Final Evaluation (Dual-Stream Strategy)\n" + "="*60)

        cnn_prob = models['conv_basic'].predict(self.X_cnn_test, verbose=0).flatten()
        final_predictions['conv_basic'] = cnn_prob
        self._analyze_cnn_reliability(self.y_test_fixed, cnn_prob)
        
        gru_prob = models['enhanced_gru'].predict(self.X_gru_test, verbose=0).flatten()
        final_predictions['enhanced_gru'] = gru_prob
        
        raw_fused_prob, _, _, gate_info = self._enhanced_fusion_predict(
            models['conv_basic'], models['enhanced_gru'], self.X_cnn_test, self.X_gru_test, override_gru_pred=gru_prob
        )

        if self.config.get('use_confidence_locking', False):
            lock_thresh = self.config.get('confidence_lock_threshold', 0.98)
            print(f"  Applying Confidence Locking (Threshold: {lock_thresh})")
            locked_prob, lock_stats = self._apply_confidence_locking(
                cnn_pred=cnn_prob, fused_pred=raw_fused_prob, y_true=self.y_test_fixed, lock_threshold=lock_thresh
            )
        else:
            print("  Confidence Locking explicitly disabled. Using pure dynamic gated fusion.")
            locked_prob = raw_fused_prob
            lock_stats = {'cnn_locked': 0, 'fusion_engaged': len(raw_fused_prob)}
            
        final_predictions['enhanced_fusion'] = locked_prob

        targets = [('conv_basic', cnn_prob), ('enhanced_gru', gru_prob), ('enhanced_fusion', locked_prob)]
        print(f"\n{'Model':<20} | {'Best Thr':<10} | {'F1':<10} | {'AUC':<10}")
        print("-" * 60)

        for name, prob in targets:
            best_thr = 0.5 
            metrics = self._calculate_full_metrics(self.y_test_fixed, prob, best_thr)
            if name == 'enhanced_fusion': optimal_fusion_metrics = metrics
            else: optimal_model_metrics[name] = metrics
            print(f"{name:<20} | {best_thr:.2f}        | {metrics['f1']:.4f}     | {metrics['auc']:.4f}")

        gate_stats = self._analyze_gate_weights(gate_info, self.y_test_fixed, cnn_prob, gru_prob)
        self.metrics_collector.record_final_metrics(optimal_model_metrics, optimal_fusion_metrics)
        self._record_final_results_full(models, final_predictions, optimal_model_metrics, optimal_fusion_metrics, gate_info, cnn_prob, gru_prob, locked_prob, gate_stats)
        self.metrics_collector.save_all_metrics()

        if self.config.get('generate_heatmaps', True):
            print("\n⚠️ Heatmap generation deferred (Currently does not support multi-input dual streams)")

    def _train_legacy_split(self, data_info): pass

    def _record_final_results_full(self, models, preds, m_metrics, e_metrics, gate, cnn_p, gru_p, fused_p, gate_s):
        error_analysis = self._analyze_errors(self.y_test_fixed, cnn_p, gru_p, fused_pred=fused_p)
        self.metrics_collector.record_final_error_analysis(error_analysis['cnn_total_errors'], error_analysis['cnn_wrong_gru_right'], error_analysis['ratio'])
        debug_samples = self._collect_fusion_debug_samples(self.y_test_fixed, cnn_p, gru_p, fused_pred=fused_p, gate_info=gate)
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
