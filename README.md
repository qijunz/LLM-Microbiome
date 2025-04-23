## Pilot analysis

Goal: LLM application on microbiome data. 

### Dataset

1. **Human cohort – MARS** 

- This dataset contains 294 individuals, 244 of them have amyloid status (a preclinical phenotype for Alzheimer's disease).
- Gut microbiome data was collected from shotgun metagenomic sequencing of fecal samples. Gut microbial features were characterized by the relative abundance of taxonomy rank (phylum, class, order, family, genus, and species).
- The microbiome data is in the format of 294 individuals x 819 microbial features, see `data/mars_taxa.csv`.
- Other host phenotypes and biomarkers are also available, see `data/mars_metadata.csv`.

2. **Mouse cohort – DO**

- (For NEXT step, validation etc.)

3. **Mouse cohort – HMDP**

- (For NEXT step, validation etc.)

### Prompt preparation

- In the pilot analysis, relative abundance of 134 bacteria species (i.e., the end-node/leaf of phylogenetic tree) were used, see `data/mars_species.csv`.
- Then these microbial feature values as well as the microbial feature names were convert into a single prompt for each individual, in the format of

> The gut microbiome of the individual shows a relative abundance of 0.00% for Akkermansia muciniphila, 0.11% for Alistipes communis, ..., 4.59% for UBA1394 sp900538575. All taxonomy names are from Genome Taxonomy Database (GTDB).


### LLM zero-shot settings
- The prompt text of each individual was tokenized, fit into LLM using AutoModel from transformers Python module.
- The last hidden layer was extracted (len=4,096), which were used as output embedding features for downstream analysis (e.g. classification task).
- LLM configurations are in Python script `analysis/llm_mars_zeroshot.py`.

### Downstream classification
- Currently, the pilot analysis used amyloid status for classification task. Other host phenotype/biomarkers can be used further (regression etc.)
- Random Forest Classifier, used 80% of data as training set and rest of 20% as testing set. ROC curves were visualized. See `analysis/llm_mars_rf.ipynb`. 


## Next steps

1. Downstream tasks
- Use cross-validation, get confusion matrix
- Test for other host phenotype/biomarkers
2. LLM encode
- Can get intermediate hidden layers
- LLM as encoders
3. LLM prompt
- Include microbial features from higher taxonomy rank.
- Include tree structure
- In-context learning, few-shot prediction
4. Test in other dataset (e.g., mouse cohorts)
