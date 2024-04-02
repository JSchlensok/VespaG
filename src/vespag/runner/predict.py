from typing import Optional
import h5py
from tqdm import tqdm
import torch
from pathlib import Path
import argparse, sys
from Bio import SeqIO

from src.vespag.runner.type_hinting import *
from src.vespag.runner.utils import *
from src.vespag.utils import get_device
from src.vespag.utils import load_model_from_config
from src.vespag.runner.embeddings import get_esm2_embeddings



# def load_model(config_key: str, params: dict, checkpoint_dir: Path, embedding_type: str) -> torch.nn.Module:
def load_model() -> torch.nn.Module:
    architecture = 'fnn'
    model_parameters = {'hidden_dims': [256], 'dropout_rate': 0.2}
    embedding_type = 'esm2'
    model = load_model_from_config(architecture, model_parameters, embedding_type)

    checkpoint_file = "./model_weights/state_dict_v2.pt"
    model.load_state_dict(torch.load(checkpoint_file))

    return model.eval()


def predict(
        fasta_file: Path,
        out_path: Optional[Path] = './data/output',
        embeddings_file: Optional[Path] = None,
        mutation_file: Optional[Path] = None,
        id_map: Optional[Path] = None,
        model: Architecture = "fnn",
        embedding_type: EmbeddingType = "esm2",
        single_csv: bool = False,
        no_csv: bool = False,
        h5_output: bool = False,
        one_based_mutations: Optional[bool] = False,
        zero_idx: Optional[bool] = False,
        # wandb_logdir: Path = Path("/mnt/project/schlensok/VESPA2/logs/vespa2")
) -> None:
    
    device = get_device()

    # load VespaG
    model = load_model()
    model = model.to(device, dtype=torch.float32)

    # compute embeddings
    if embeddings_file is None:
        embeddings = get_esm2_embeddings(fasta_file, out_path)
    # or load precomputed embeddings
    else:
        # load embeddings
        embeddings = {id: torch.tensor(emb[()], dtype=torch.float32, device=device) for id, emb in
                    tqdm(h5py.File(embeddings_file).items(), desc="Loading embeddings")}
    
    # TODO check with JS if needed
    # if id_map:
    #     for line in id_map.open("r").readlines():
    #         h5_id, fasta_id = line.split(",")
    #         embeddings[fasta_id] = embeddings[h5_id]
    #         del embeddings[h5_id]

    # if mutation file provided, run VespaG for specific mutations
    if mutation_file:
        mutations_per_protein = read_mutation_file(mutation_file, one_indexed=one_based_mutations)
    # no mutation file, run VespaG for entire mutational landscape
    elif fasta_file:
        # generate mutational landscape
        sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}
        mutations_per_protein = {
            protein_id: [
                SAV(i, wildtype_aa, other_aa)
                for i, wildtype_aa in enumerate(sequence)
                for other_aa in AMINO_ACIDS if other_aa != wildtype_aa]
            for protein_id, sequence in tqdm(sequences.items(), desc="Generating full mutational landscape")}


    # inference loop
    vespag_scores = {}
    scores_per_protein = {}
    for id, embedding in (pbar := tqdm(embeddings.items(), desc="Generating predictions")):
        pbar.set_postfix({"current_protein": id})
        # get VespaG predictions
        y = model(embedding)
        # set wildtype to 0
        if fasta_file:
            y = mask_non_mutations(y, sequences[id])
            
        # dict with mutation (fromAA,position,toAA) as key and score as item
        if zero_idx:
            scores_per_protein[id] = {
                mutation: compute_mutation_score(y, mutation)
                for mutation in mutations_per_protein[id]
            }
        else:
            scores_per_protein[id] = {
                f'{str(mutation)[0]}{int(str(mutation)[1])+1}{str(mutation)[2]}': compute_mutation_score(y, mutation)
                for mutation in mutations_per_protein[id]
            }
        # optional: prepare h5 file
        if h5_output:
            vespag_scores[id] = y.detach()

    # write h5 file
    if h5_output:
        with h5py.File(out_path + '/vespag_scores_all.h5', "w") as f:
            for id, vespag_prediction in tqdm(vespag_scores.items(), desc="Generating H5 output file"):
                f.create_dataset(id, data=vespag_prediction)

    # write csv output (default: single csv file for each mutational landscape of protein sequence)
    if not no_csv:
        if single_csv:
            # TODO verify that out_path is directory
            for protein_id, mutations in tqdm(scores_per_protein.items(), desc="Generating output files"):
                output_file = Path(out_path, protein_id + ".csv")
                with output_file.open("w+") as f:
                    f.writelines("Mutation, VespaG\n")
                    f.writelines([f"{str(sav)},{score}\n" for sav, score in mutations.items()])
        else:
            # TODO verify that out_path is CSV file
            output_file = Path(out_path, "vespag_scores_all.csv")
            with output_file.open("w+") as f:
                f.writelines("Mutation, VespaG\n")
                f.writelines([line for line in tqdm([
                    f"{protein_id}_{str(sav)},{score}\n"
                    for protein_id, mutations in scores_per_protein.items()
                    for sav, score in mutations.items()], desc="Generating output file")
                              ])


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""
    # Instantiate the parser
    parser = argparse.ArgumentParser(prog='predict', description=( 
            'predict.py generates VespaG variant effect score predictions'+
            ' for a given file containing sequence(s) in FASTA-format.') )
    # Required positional argument
    parser.add_argument( '-i', '--input', required=True, type=str,
                    help='A path to a fasta-formatted text file containing protein sequence(s).')
    # Optional positional argument
    parser.add_argument( '-o', '--output', required=False, type=str, 
                    help='A path for saving the created CSV files. Default directory is "./data/output".')
    # Optional positional argument
    parser.add_argument( '-e', '--embeddings', required=False, type=str, 
                    help='A path to pre-generated ESM-2 input embeddings. If not provided, embeddings will be saved in "./data/output/esm2_embeddings.h5"')
    # Optional positional argument
    parser.add_argument('--h5_output', required=False, type=str,
                    default=None,
                    help='Whether a file containing all predictions in HDF5 format should be saved.' )
    # Optional positional argument
    parser.add_argument('--single_csv', required=False, type=str,
                    default=None,
                    help='Whether to return one CSV file for all proteins instead of per-protein CSV files.' )
    # Optional positional argument
    parser.add_argument('--no_csv', required=False, type=str,
                    default=None,
                    help='Whether no CSV output should be produced.' )
    # Optional argument
    parser.add_argument('--zero_idx', type=bool, 
                    default=False,
                    help="Whether to enumerate the sequence starting at 0. Default is starting at 1.")
    return parser

def main():
    parser     = create_arg_parser()
    args       = parser.parse_args()

    parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    
    #required
    seq_path   = Path(args.input)
    #optional
    out_path   = Path(args.output) if args.output is not None else None
    emb_path   = Path(args.embeddings) if args.embeddings is not None else None
    h5_output = args.h5_output if args.h5_output is not None else None
    no_csv = args.no_csv if args.no_csv is not None else None
    single_csv = args.single_csv if args.single_csv is not None else None
    zero_idx = args.zero_idx if args.zero_idx is not None else None
    
    predict(fasta_file=seq_path, 
            out_path=out_path,
            embeddings_file=emb_path,
            h5_output=h5_output,
            no_csv=no_csv,
            single_csv=single_csv, 
            zero_idx=zero_idx,
            )

if __name__ == '__main__':
    main()
    # predict(embeddings_file= 'data/test/test.h5',
    #     output= 'data/test/test',
    #     fasta_file= 'data/test/test.fasta',
    #     model = "fnn",
    #     embedding_type = "esm2",
    #     single_csv= True,
    #     no_csv= False,
    #     )
    # predict(embeddings_file= 'data/test/proteingym_217_esm2.h5',
    #     output= 'data/test/pg217_1idx',
    #     fasta_file= 'data/test/proteingym_217.fasta',
    #     model = "fnn",
    #     embedding_type = "esm2",
    #     single_csv= True,
    #     no_csv= False,
    #     )