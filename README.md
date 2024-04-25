
# VespaG: Expert-Guided Protein Language Models enable Accurate and Blazingly Fast Fitness Prediction

<img align="right" src="images/vespag.png" alt="image" height="20%" width="20%" />

**VespaG** is a blazingly fast single amino acid variant effect predictor, leveraging embeddings of the protein language model [ESM-2](https://github.com/facebookresearch/esm) ([Lin et al. 2022](https://www.science.org/doi/abs/10.1126/science.ade2574)) as input to a minimal deep learning model. 

To overcome the sparsity of experimental training data, we created a dataset of 39 million single amino acid variants from a subset of the Human proteome, which we then annotated using predictions from the multiple sequence alignment-based effect predictor [GEMME](http://www.lcqb.upmc.fr/GEMME/Home.html) ([Laine et al. 2019](https://doi.org/10.1093/molbev/msz179)) as a proxy for experimental scores. 

Assessed on the [ProteinGym](https://proteingym.org) ([Notin et al. 2023](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1)) benchmark, **VespaG** matches state-of-the-art methods while being several orders of magnitude faster, predicting the entire single-site mutational landscape for a human proteome in under a half hour on a consumer-grade laptop.

More details on **VespaG** can be found in the corresponding [preprint](https://www.biorxiv.org/).

### Quick Start - Running Inference with VespaG
1. Install necessary dependencies: `conda env create -f environment.yml`
2. Run `python -m vespag predict` with the following options:
- `--input/-i`: Path to FASTA-formatted file containing protein sequence(s) (**required**).
- `--output/-o`:Path for saving created CSV and/or H5 files. Defaults to `./output`. [default: None]
- `--embeddings/-e`: Path to pre-generated ESM2 (`esm2_t36_3B_UR50D`) input embeddings. Embeddings will be generated from scratch if no path is provided. **Please note that embedding generation on CPU is extremely slow and not recommended.** [default: None]
- `--mutation-file`: CSV file specifying specific mutations to score. If not provided, the whole single-site mutational landscape of all input proteins will be scored. [default: None]
- `--id-map`: CSV file mapping embedding IDs (first column) to FASTA IDs (second column) if they're different. Does not have to cover cases with identical IDs. [default: None]
- `--single-csv`: Whether to return one CSV file for all proteins instead of a single file for each protein [default: False]
- `--no-csv`: Whether no CSV output should be produced. [default: False]
- `--h5-output`: Whether a file containing predictions in HDF5 format should be created. [default: False]
- `--zero-idx`: Whether to enumerate protein sequences (both in- and output) starting at 0. [default: False]
```

Kindly note that data pre-processing, model training, and evaluation are currently not supported in the public GitHub repository.