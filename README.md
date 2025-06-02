## Pilot analysis

Goal: LLM application on microbiome data. 

### Dataset

1. **Human cohort – MARS** 

- This dataset contains 294 individuals, 244 of them have amyloid status (a preclinical phenotype for Alzheimer's disease).
- Gut microbiome data was collected from shotgun metagenomic sequencing of fecal samples. Gut microbial features were characterized by the relative abundance of taxonomy rank (phylum, class, order, family, genus, and species).
- The microbiome data is in the format of 294 individuals x 601 microbial features, see `data/mars_taxa.csv`.
- Other host phenotypes and biomarkers are also available, see `data/mars_metadata.csv`.

2. **Mouse cohort – DO**

- This dataset contains 1177 individuals.
- Gut microbiome data was collected from 16S rRNA sequencing of fecal samples. Gut microbial features were characterized by the relative abundance of taxonomy rank (phylum, class, order, family, genus, and species).
- The microbiome data is in the format of 1177 individuals x 76 microbial features, see `do1200_taxa.csv`.
- Other host phenotypes and biomarkers are also available, see `data/do1200_metadata.csv`.


### Prompt preparation

- In the pilot analysis, relative abundance of bacteria taxa (i.e., the end-node/leaf of phylogenetic tree) were used.
- Then these microbial feature values as well as the microbial feature names were convert into a single prompt for each individual, in the format of

> The gut microbiome of the individual shows a relative abundance of 48.66105% for phylum Firmicutes_A, 0.77399% for phylum Proteobacteria, ..., 0.00000% for species Faecalimonas umbilicata. All taxonomy names are from Genome Taxonomy Database (GTDB).


### LLM zero-shot settings
- The prompt text of each individual was tokenized, fit into LLM using AutoModel from transformers Python module.
- The last hidden layer was extracted (len=4,096), which were used as output embedding features for downstream analysis (e.g. regression/classification tasks).
- LLM configurations are in Python script `microbiome_zeroshot.py`.
