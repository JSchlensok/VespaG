
# VespaG: Expert-Guided Protein Language Models enable Accurate and Blazingly Fast Fitness Prediction

<img align="right" src="images/vespag.png" alt="image" height="20%" width="20%" />

**VespaG** is a blazingly fast single amino acid variant effect predictor, leveraging embeddings of the protein language model [ESM-2](https://github.com/facebookresearch/esm) ([Lin et al. 2022](https://www.science.org/doi/abs/10.1126/science.ade2574)) as input to a minimal deep learning model. 

To overcome the sparsity of experimental training data, we created a dataset of 39 million single amino acid variants from a subset of the Human proteome, which we then annotated using predictions from the multiple sequence alignment-based effect predictor [GEMME](http://www.lcqb.upmc.fr/GEMME/Home.html) ([Laine et al. 2019](https://doi.org/10.1093/molbev/msz179)) as a proxy for experimental scores. 

Assessed on the [ProteinGym](https://proteingym.org) ([Notin et al. 2023](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1)) benchmark, **VespaG** matches state-of-the-art methods while being several orders of magnitude faster, predicting mutational landscapes for 20 thousand proteins in under an hour on a 32-core CPU. 

More details on **VespaG** can be found in the corresponding [preprint](https://www.biorxiv.org/).

### Quick Start - Running Inference with VespaG
1. Install necessary dependencies (f.e., with `conda env create -f environment.yml`)
2. Run `python -m src.vespag.runner.predict`. In short, this script needs an input fasta file, creates ESM-2 embeddings if none are provided, saves a csv output in folder `data/output` by default. You can use the following arguments:
    - `-h` for help
    - **required**: `-i INPUT` A path to a fasta-formatted text file containing protein sequence(s). 
    - optional: `-o OUTPUT` A path for saving the created CSV files. Default directory is `./data/output`.
    - optional: `-e EMBEDDINGS` A path to pre-generated ESM-2 input embeddings. If not provided, embeddings will be saved in `./data/output/esm2_embeddings.h5`
    - optional: `--h5_output H5_OUTPUT` Whether a file containing all predictions in HDF5 format should be saved.
    - optional: `--single_csv SINGLE_CSV` Whether to return one CSV file for all proteins instead of per-protein CSV files.
    - optional: `--no_csv NO_CSV` Whether no CSV output should be produced.
    - optional: `--zero_idx ZERO_IDX` Whether to enumerate the sequence starting at 0. Default is starting at 1. 