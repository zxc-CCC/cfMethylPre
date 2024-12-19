# cfMethylPre
Classifying cancer patients using cfMethylPre.

![cfMethylPre](https://github.com/zxc-CCC/cfMethylPre/blob/main/cfMethylPre.png)

## Overview
Cancer remains a major global health challenge, requiring innovative tools for early detection and improved patient outcomes. To address this, we introduce cfMethylPre, a novel deep learning and transfer learning framework designed for cancer detection using cell-free DNA (cfDNA) methylation data. cfMethylPre combines pre-trained DNA sequence embeddings with methylation profiles to enhance feature representation. The model employs a transfer learning approach, pre-training on bulk DNA methylation data and then fine-tuning with cfDNA data to improve predictive accuracy. Additionally, cfMethylPre incorporates interpretability, allowing for insights into key features driving predictions. This framework demonstrates improved accuracy and robustness for cancer detection, paving the way for its application in precision oncology.

## Software dependencies
scanpy==1.9.6

pytorch==1.12.0+cu11.3

pytorch_geometric==2.4.0

R==4.2.3

mclust==5.4.10

## set up
First clone the repository.
```bash
git clone https://github.com/zxc-CCC/cfMethylPre.git
cd cfMethylPre-main
```

Then, we suggest creating a new environment：
```bash
conda create -n cfMethylPre python=3.10 
conda activate cfMethylPre
```
Additionally, install the packages required:
```bash
pip install -r requiements.txt
```
## Datasets
In this work, we processed two datasets for cfDNA methylation analysis: the source domain dataset, which included 2,801 DNA methylation samples across 91 classes (82 cancer types and 9 healthy control types), and the target domain dataset, which consisted of 470 cfDNA methylation samples across 10 classes (9 cancer types and healthy control). To ensure data integrity and accuracy, we excluded CpG sites with more than 30% missing methylation values per class, as well as samples with over 20% missing values. After rigorous filtering, a total of 6,585 common CpG sites were retained for further analysis. The raw data can be found in the corresponding datasets in the paper, and the processed data along with related intermediate results can be obtained by contacting the authors.
