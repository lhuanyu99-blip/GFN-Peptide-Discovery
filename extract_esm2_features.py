#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ESM-2 Feature Extraction Pipeline for Confidence-Gated Fusion Network (CGFN)
This script processes raw FASTA sequences into high-dimensional representations 
using the ESM-2 pre-trained language model, serving as the foundational feature 
input for the downstream Functional Mutational Lineage evaluation framework.
"""

import torch
import esm
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Tuple

def read_fasta_file(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Reads a FASTA file and returns a list of (Sequence_ID, Sequence) tuples.
    Filters out invalid amino acid characters automatically.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    
    sequences = []
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    
    current_id = ""
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id and current_seq:
                    raw_seq = "".join(current_seq)
                    filtered_seq = ''.join([c for c in raw_seq if c in valid_amino_acids])
                    if filtered_seq:
                        sequences.append((current_id, filtered_seq))
                current_id = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line.upper())
        
        if current_id and current_seq:
            raw_seq = "".join(current_seq)
            filtered_seq = ''.join([c for c in raw_seq if c in valid_amino_acids])
            if filtered_seq:
                sequences.append((current_id, filtered_seq))
                
    print(f"Successfully loaded {len(sequences)} valid sequences from {fasta_path.name}")
    return sequences

class ESM2FeatureExtractor:
    def __init__(self, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_mixed_precision: bool = True,
                 max_length: int = 1024,
                 extract_type: str = "sequence"):
        """Initialize the ESM-2 Feature Extractor."""
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.max_length = max_length
        self.extract_type = extract_type.lower()
        
        print(f"Loading ESM-2 model (esm2_t33_650M_UR50D) to {device}...")
        try:
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        except Exception as e:
            print(f"Model initialization failed: {e}")
            raise e
            
        self.model = self.model.to(device)
        self.model.eval()
        
        if self.use_mixed_precision and "cuda" in device:
            self.model = self.model.half()
        
        self.batch_converter = self.alphabet.get_batch_converter()
        self.repr_layer = 33
        
        if self.extract_type not in ["both", "sequence", "token"]:
            raise ValueError("extract_type must be one of: 'both', 'sequence', 'token'")

    def _process_batch(self, batch_tokens: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Processes a batch of tokens to extract representations."""
        try:
            with torch.no_grad():
                if self.use_mixed_precision and "cuda" in self.device:
                    with torch.amp.autocast(device_type="cuda"):
                        results = self.model(batch_tokens, repr_layers=[self.repr_layer])
                else:
                    results = self.model(batch_tokens, repr_layers=[self.repr_layer])
        except Exception as e:
            print(f"Batch processing failed: {e}")
            raise e

        batch_features = []
        for i in range(batch_tokens.size(0)):
            try:
                valid_indices = (batch_tokens[i] != self.alphabet.padding_idx).nonzero().squeeze()[1:-1]
                feature = {}
                
                if self.extract_type in ["both", "token"]:
                    token_repr = results["representations"][self.repr_layer][i][valid_indices].cpu()
                    feature["token_features"] = token_repr
                
                if self.extract_type in ["both", "sequence"]:
                    token_repr = results["representations"][self.repr_layer][i][valid_indices].cpu()
                    seq_repr = token_repr.mean(0, keepdim=True)
                    feature["sequence_features"] = seq_repr
                
                batch_features.append(feature)
            except Exception as e:
                print(f"Failed to process sequence {i}: {e}")
                batch_features.append({"sequence_features": torch.zeros(0, 1280)})
        
        return batch_features

    def find_optimal_batch_size(self, sequences: List[str], initial_batch_size: int = 64) -> int:
        """Dynamically determines optimal batch size to prevent CUDA OOM."""
        test_seqs = [s[:self.max_length] for s in sequences[:initial_batch_size]]
        batch_size = initial_batch_size
        
        print("Calibrating optimal batch size for hardware constraints...")
        while batch_size >= 1:
            try:
                batch_data = [(str(i), seq) for i, seq in enumerate(test_seqs[:batch_size])]
                _, _, tokens = self.batch_converter(batch_data)
                tokens = tokens.to(self.device, dtype=torch.long)
                
                with torch.no_grad():
                    _ = self.model(tokens, repr_layers=[self.repr_layer])
                
                torch.cuda.empty_cache()
                print(f"Optimal batch size locked at: {batch_size}")
                return batch_size
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    batch_size = max(1, batch_size // 2)
                    torch.cuda.empty_cache()
                else:
                    raise e
            except Exception as e:
                raise e
                
        raise RuntimeError("Failed to find a viable batch size.")

def extract_and_save_features(extractor, sequences, output_dir, batch_size=None):
    """Extracts features and saves them as compressed .npz arrays."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    seq_ids = [seq_id for (seq_id, seq) in sequences]
    sequences_only = [seq for (seq_id, seq) in sequences]
    
    if batch_size is None:
        batch_size = extractor.find_optimal_batch_size(sequences_only)
    print(f"Executing extraction with batch size: {batch_size}")
    
    total_batches = (len(sequences_only) + batch_size - 1) // batch_size
    shape_stats = []
    saved_count = 0
    saved_paths = [] 
    
    for batch_idx in tqdm(range(total_batches), desc="Extracting Features"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(sequences_only))
        batch_seqs = sequences_only[start_idx:end_idx]
        batch_ids = seq_ids[start_idx:end_idx]
        
        batch_data = [(str(i), seq[:extractor.max_length]) for i, seq in enumerate(batch_seqs)]
        try:
            _, _, batch_tokens = extractor.batch_converter(batch_data)
        except Exception as e:
            print(f"Batch conversion failed: {e}")
            continue
            
        batch_tokens = batch_tokens.to(extractor.device, dtype=torch.long)
        
        try:
            batch_results = extractor._process_batch(batch_tokens)
        except Exception as e:
            print(f"Batch extraction failed: {e}")
            batch_results = [{"sequence_features": torch.zeros(0, 1280)}] * len(batch_seqs)
        
        for seq_id, feature in zip(batch_ids, batch_results):
            file_path = output_path / f"feature_{seq_id}.npz"
            
            save_dict = {}
            if "token_features" in feature and feature["token_features"].numel() > 0:
                save_dict["token_features"] = feature["token_features"].numpy()
            if "sequence_features" in feature and feature["sequence_features"].numel() > 0:
                save_dict["sequence_features"] = feature["sequence_features"].numpy()
            
            try:
                np.savez_compressed(file_path, **save_dict)
                saved_count += 1
                saved_paths.append(str(file_path.absolute()))
            except Exception as e:
                print(f"Failed to save features for ID: {seq_id}: {e}")
            
            stats = (len(seq_id),)
            if "token_features" in feature and feature["token_features"].numel() > 0:
                stats += (feature["token_features"].shape,)
            if "sequence_features" in feature and feature["sequence_features"].numel() > 0:
                stats += (feature["sequence_features"].shape,)
            shape_stats.append(stats)
        
        del batch_tokens, batch_results
        torch.cuda.empty_cache()
        
    return shape_stats, saved_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM-2 Feature Extractor for CGFN Dataset Preparation")
    parser.add_argument("--fasta_path", type=str, required=True, 
                        help="Path to the input FASTA file.")
    parser.add_argument("--out_npz_dir", type=str, required=True, 
                        help="Directory to save the generated .npz feature files.")
    parser.add_argument("--out_txt_list", type=str, required=True, 
                        help="Path to save the .txt list of generated feature paths.")
    parser.add_argument("--max_length", type=int, default=1024, 
                        help="Maximum sequence length (default: 1024).")
    
    args = parser.parse_args()
    
    try:
        extractor = ESM2FeatureExtractor(
            use_mixed_precision=True,
            max_length=args.max_length,
            extract_type="sequence"
        )
        
        print(f"\n[{'-'*50}]")
        print(f"Initiating processing for: {args.fasta_path}")
        
        sequences = read_fasta_file(args.fasta_path)
        if not sequences:
            print(f"WARNING: No valid sequences found in {args.fasta_path}. Exiting.")
            exit(0)
            
        shape_stats, saved_paths = extract_and_save_features(extractor, sequences, args.out_npz_dir)
        
        # Save absolute paths to the output text file
        with open(args.out_txt_list, 'w') as f:
            f.write("\n".join(saved_paths) + "\n")
            
        print(f"\n✅ Feature extraction complete. Saved to: {args.out_npz_dir}")
        print(f"✅ Metadata path list saved to: {args.out_txt_list}")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
