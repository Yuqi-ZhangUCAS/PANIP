# PANIP
Protein-Nucleic Acid Interaction Prediction and Binding Residue Identification via Reciprocal Attention Learning

## **Requirement**

Checking and installing environmental requirements.This project depends on Uni-RNA to extract nucleic acid sequence features. Ensure that Uni-RNA is installed in the environment. Installation link: https://github.com/ComDec/unirna_tf

```bash
pip install -r requirements.txt
```

## **Datasets**
The structural protein-nucleic acid interactions with annotated binding residue information are available from https://aideepmed.com/BioLiP/. The affinity data of protein-nucleic acid interactions are available from https://huggingface.co/datasets/Jesse7/CoPRA_data/tree/main and https://dpai.ccnu.edu.cn/PNAT/index.php/home/download. The aptamer data are available from https://www.science.org/doi/10.1126/science.adv6127#supplementary-materials.

## **Predict interaction, binding sites, and affinity between the pairs of given protein and nucleic acid sequences**
Step 1: Obtain sequence features of proteins and nucleic acids. Input: ./data_prepare/example.csv. Output: ./data_prepare/example_feature.h5.
```bash
Python ./data_prepare/seq_data_process.py
```
Step 2: Obtain prediction results.
```bash
Python predict_RPI.py
```
