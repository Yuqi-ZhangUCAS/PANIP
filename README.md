# PANIP
Protein-Nucleic Acid Interaction Prediction and Binding Residue Identification via Reciprocal Attention Learning

## **Requirement**

Checking and installing environmental requirements.This project depends on Uni-RNA to extract nucleic acid sequence features. Ensure that Uni-RNA is installed in the environment. Installation link: https://github.com/ComDec/unirna_tf

```bash
pip install -r requirements.txt
```

## **Predict interaction, binding sites, and affinity between the pairs of given protein and nucleic acid sequences**
Step 1: Obtain sequence features of proteins and nucleic acids. Input: ./data_prepare/example.csv. Output: ./data_prepare/example_feature.h5.
```bash
Python ./data_prepare/seq_data_process.py
```
Step 2: Obtain prediction results.
```bash
Python predict_RPI.py
```
