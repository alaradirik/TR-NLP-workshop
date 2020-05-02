# Tr-NLP Workshop 
Açık Seminer 2020 - Turkish NLP Seminar and Workshop

This repo includes the notebooks and slides for the Turkish Natural Language workshop. The implemented modules are:
- Text preprocessing
- Named Entity Recognition with SpaCy
- Unsupervised text classification with K-Means


 ## Dataset
[TWNERTC](https://data.mendeley.com/datasets/cdcztymf4k/1) (Turkish Wikipedia Named-Entity Recognition and Text Categorization Dataset ) by Sahin, et al. is used for Named Entity Recognition. The TWNERTC dataset contains approximately 300K named entities in 77 domains with more than 1000 fine-grained entity types. A subset of the dataset (the astronomy domain) is provided in the repo and the full clean version of the dataset in json format can be downloaded [here](https://drive.google.com/file/d/1o0j4UcEBCehwJSG2SHOl_I-h8TTA6pdI/view). 

A small [Turkish news dataset](https://hakan.io/makine-ogrenmesi-turkce-haber-metinleri-veri-seti/) crawled from various news websiteds is used for text clustering. This dataset contains news in 5 categories (economy, arts, politics, sports, technology) with 100 samples per category. 

 ## Notebooks
Clone the repo and install the requirements before running the notebooks:
```
git clone https://github.com/alaradirik/TR-NLP-workshop.git

pip install -r requirements.txt
```
