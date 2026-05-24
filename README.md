# CGFN: Confidence-Gated Fusion Network for Robust Pro-Apoptotic Peptide Prediction

This repository contains the official implementation of the **Confidence-Gated Fusion Network (CGFN)**, an advanced deep learning framework designed for the precise identification of pro-apoptotic peptides and the characterization of their interactions with regulatory proteins (e.g., Bax and Bfl-1).

To address sequence redundancy and evaluation bias inherent in traditional random partitioning, CGFN incorporates a strict **5-Level Evaluation Framework** driven by **Functional Mutational Lineage** logic, alongside an **Anti-Leakage Validation-Calibrated Thresholding** strategy.

---

## 🔬 System Architecture

CGFN leverages a dual-pathway feature representation and dynamic gating mechanism:
1. **CNN Pathway:** Extracts dense local motifs from pre-trained ESM-2 (1280-dimensional) evolutionary embeddings.
2. **Enhanced BiGRU Pathway:** Captures long-range sequential dependencies and evolutionary trajectories using bagged bidirectional gated recurrent units stabilized by a dynamic learning rate warm-up schedule.
3. **Confidence-Gated Fusion Layer:** Dynamically scales representation weights based on Out-Of-Fold (OOF) disagreement statistics and uncertainty priors, optimizing the decision boundary via a strict anti-leakage protocol.

---

## 📂 Repository Structure

Based on the root directory, the core assets are organized as follows:


GFN-Peptide-Discovery/
 ├── CGFN_Dataset_Partitions.csv      # Integrated 5-level framework partition matrix
 ├── Positive.fasta                   # Raw positive pro-apoptotic peptide sequences
 ├── neg_candidates.fasta             # Raw negative background candidate sequences
 ├── extract_esm2_features.py         # ESM-2 pre-trained embedding extraction pipeline
 ├── pipeline_cgfn_dataset.py         # Automated sequence clustering & lineage partitioner
 ├── model.py                         # Core neural architecture & Anti-leakage training engine
 ├── config.py                        # Central hyperparameter and cross-validation configuration
 ├── requirements.txt                 # Optimized deployment dependency manifest
 ├── LICENSE                          # MIT License
 └── .gitignore                       # Git ignore file

---

## ⚡ Prerequisites & Installation
Ensure you have a CUDA-capable environment configured. Install the clean, top-level dependencies via pip:
git clone https://github.com/lhuanyu99-blip/GFN-Peptide-Discovery.git
cd GFN-Peptide-Discovery
pip install -r requirements.txt

---

##🏋️‍♂️ Data Preparation & Feature Extraction
1. Generating ESM-2 Embeddings
Extract dense residue-level evolutionary representations using the pre-trained ESM-2 architecture:
python extract_esm2_features.py --input Positive.fasta --output ./Positive_NPZ/
python extract_esm2_features.py --input neg_candidates.fasta --output ./Negative_NPZ/

---

2. Executing Lineage Partitioning
To recreate the structural and mutational partitions across different random seeds, execute the pipeline using the integrated matrix:
python pipeline_cgfn_dataset.py --partitions CGFN_Dataset_Partitions.csv

🚀 Model Training & Evaluation
The training engine strictly isolates data partitions to prevent test-set leakage. The decision threshold of the fusion layer is automatically calibrated on Out-Of-Fold (OOF) validation sub-matrices and permanently frozen before executing blind test evaluations.
To initiate training for a specific framework setting, adjust the target directory and random seed parameters in config.py and run:
python model.py --config config.py
Reproducing the 5-Level Benchmarks

By mapping the target indices in CGFN_Dataset_Partitions.csv, you can evaluate across all 5 distinct experimental configurations defined in the manuscript:
Lineage Frameworks: L_Base (Core Mutation Isolation), L_Quota (Capacity Balanced), nonMR_lineage (Hard Non-MRPEIW sub-lineage).
Random Control Frameworks: R_Base (Standard Random Control), R_Quota (Capacity-Matched Random Control).

🔮 Inference Mode (Prediction on Unseen Sequences)
Once models are trained, CGFN can screen novel peptide libraries. The prediction engine automatically restores the serializable pooling layers and injects the corresponding pre-calibrated validation threshold to ensure end-to-end integrity:
python model.py --predict --test_data /path/to/novel_peptide_npz/ --output screening_results.npz

🔒 Data and Code Availability Statement
Source Code & Architecture: Permanently open-sourced in this GitHub repository under the MIT License.
Core Dataset & Splits: The raw FASTA sequence files and the master partition coordinate file (CGFN_Dataset_Partitions.csv) are archived in this repository for zero-gatekeep validation.
Pre-trained Weights: The complete matrix of pre-trained neural network weights (.keras format) spanning all 5 evaluation levels across various random seeds are available from the corresponding author upon reasonable request, and will be formally hosted upon peer-review publication.
