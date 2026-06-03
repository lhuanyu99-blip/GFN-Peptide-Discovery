# CGFN: Confidence-Gated Fusion Network for Robust Pro-Apoptotic Peptide Prediction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20366846.svg)](https://doi.org/10.5281/zenodo.20366846)

This repository contains the official implementation of the **Confidence-Gated Fusion Network (CGFN)**, an advanced deep learning framework designed for the precise identification of pro-apoptotic peptides and the characterization of their interactions with regulatory proteins (e.g., Bax and Bfl-1).

To address sequence redundancy and evaluation bias inherent in traditional random partitioning, CGFN incorporates a mathematically rigorous **5-Tier Evaluation Gradient** driven by strict **Functional Mutational Lineage** isolation. 

---

## 🔬 System Architecture

CGFN abandons conventional static ensembling in favor of a strictly orthogonal, dual-view architecture equipped with an Out-of-Distribution (OOD) aware dynamic router:

1. **Evolutionary View (CNN Branch):** Extracts dense local semantic motifs directly from pre-trained ESM-2 (1280-dimensional) contextual embeddings.
2. **Biophysical View (GRU Branch):** Captures long-range spatial topologies and physicochemical properties using 10D AAindex-based features. It employs `SpatialDropout1D` and a robust dual-pooling strategy (Global Average + Global Max) to prevent structural overfitting.
3. **7-Dim Confidence-aware Gated Fusion:** A dynamic, logit-space arbitration mechanism. By extracting a 7-dimensional context vector (including absolute confidence, inter-expert disagreement, and OOD manifold probes), the router intelligently arbitrates between modality-specific failure modes—*evolutionary hallucinations* versus *physical rigidity*—effectively intercepting negative transfer on challenging orphan variants.

---

## 📂 Repository Structure

Based on the root directory, the core assets are organized as follows:

```text
GFN-Peptide-Discovery/
 ├── data/                            # Default directory for dataset partitions & extracted NPZ features
 ├── CGFN_Dataset_Partitions.csv      # Integrated 5-level framework partition matrix
 ├── extract_features/                # Scripts for ESM-2 and AAindex extraction
 ├── pipeline_cgfn_dataset.py         # Automated sequence clustering & lineage partitioner
 ├── model_optimized.py               # Core neural architecture & Dual-stream Meta-Learning engine
 ├── config.py                        # Central hyperparameter configuration
 ├── requirements.txt                 # Optimized deployment dependency manifest
 ├── LICENSE                          # MIT License
 └── README.md                        # Documentation
```

---

⚡ Prerequisites & Installation
Ensure you have a CUDA-capable environment configured. Install the top-level dependencies via pip:
```text
git clone [https://github.com/lhuanyu99-blip/GFN-Peptide-Discovery.git](https://github.com/lhuanyu99-blip/GFN-Peptide-Discovery.git)
cd GFN-Peptide-Discovery
pip install -r requirements.txt
```

---

##🏋️‍♂️ Data Preparation & Feature Extraction

1. Generating ESM-2 Embeddings
Extract dense residue-level evolutionary representations using the pre-trained ESM-2 architecture:
```text
python extract_esm2_features.py --input Positive.fasta --output ./Positive_NPZ/
python extract_esm2_features.py --input neg_candidates.fasta --output ./Negative_NPZ/
```

2. Executing Lineage Partitioning

To recreate the structural and mutational partitions across different random seeds, execute the pipeline using the integrated matrix:
```text
python pipeline_cgfn_dataset.py --partitions CGFN_Dataset_Partitions.csv
```

---

##🚀 Model Training & Evaluation

The training engine strictly isolates data partitions to prevent test-set leakage. The decision threshold of the fusion layer is automatically calibrated on Out-Of-Fold (OOF) validation sub-matrices and permanently frozen before executing blind test evaluations.
To initiate training for a specific framework setting, adjust the target directory and random seed parameters in config.py and run:
```text
python model.py --config config.py
Reproducing the 5-Level Benchmarks
```
By mapping the target indices in CGFN_Dataset_Partitions.csv, you can evaluate across all 5 distinct experimental configurations defined in the manuscript:
Lineage Frameworks: L_Base (Core Mutation Isolation), L_Quota (Capacity Balanced), nonMR_lineage (Hard Non-MRPEIW sub-lineage).
Random Control Frameworks: R_Base (Standard Random Control), R_Quota (Capacity-Matched Random Control).

---

##🔮 Inference Mode (Prediction on Unseen Sequences)

Once models are trained, CGFN can screen novel peptide libraries. The prediction engine automatically restores the serializable pooling layers and injects the corresponding pre-calibrated validation threshold to ensure end-to-end integrity:
```text
python model.py --predict --test_data /path/to/novel_peptide_npz/ --output screening_results.npz
```

---

##🔒 Data and Code Availability Statement

Permanent Data Archive (Zenodo): The comprehensive datasets, rigorous 5-level partition matrices, and downstream AlphaFold3 structural screening results (including complex .pdb files and ranking metrics) are permanently hosted on Zenodo: https://doi.org/10.5281/zenodo.20366846.

Source Code & Architecture: Permanently open-sourced in this GitHub repository under the MIT License.
