from typing import Optional

# from src.vespa2.runner.train import *
from src.vespag.runner.type_hinting import *
from src.vespag.runner.utils import *
from src.vespag.utils import get_device
from src.vespag.utils import load_model_from_config


import h5py
from tqdm import tqdm
import torch
from pathlib import Path
from Bio import SeqIO


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
        embeddings_file: Path,
        output: Path,
        fasta_file: Optional[Path] = None,
        mutation_file: Optional[Path] = None,
        id_map: Optional[Path] = None,
        model: Architecture = "fnn",
        embedding_type: EmbeddingType = "esm2",
        single_csv: bool = True,
        no_csv: bool = False,
        h5_output: Optional[Path] = None,
        one_based_mutations: Optional[bool] = False,
        idx: Optional[str] = '1idx',
        # wandb_logdir: Path = Path("/mnt/project/schlensok/VESPA2/logs/vespa2")
) -> None:
    if embeddings_file is None:
        # TODO generate embeddings
        pass

    device = get_device()
    # load embeddings
    embeddings = {id: torch.tensor(emb[()], dtype=torch.float32, device=device) for id, emb in
                  tqdm(h5py.File(embeddings_file).items(), desc="Loading embeddings")}

    if id_map:
        for line in id_map.open("r").readlines():
            h5_id, fasta_id = line.split(",")
            embeddings[fasta_id] = embeddings[h5_id]
            del embeddings[h5_id]

    if mutation_file:
        mutations_per_protein = read_mutation_file(mutation_file, one_indexed=one_based_mutations)
    elif fasta_file:
        # generate mutational landscape
        sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}
        mutations_per_protein = {
            protein_id: [
                SAV(i, wildtype_aa, other_aa)
                for i, wildtype_aa in enumerate(sequence)
                for other_aa in AMINO_ACIDS if other_aa != wildtype_aa]
            for protein_id, sequence in tqdm(sequences.items(), desc="Generating full mutational landscape")}

    # TODO load best model for given architecture & embedding type
    # if model in ["fnn", "cnn"]:
    #     model = ModelLoader(wandb_logdir).load_best(model.value.lower(), embedding_type.value.lower())
    # elif model == "mean":
    #     fnn = ModelLoader(wandb_logdir).load_best("fnn", embedding_type.value.lower())
    #     cnn = ModelLoader(wandb_logdir).load_best("cnn", embedding_type.value.lower())
    #     model = MeanModel(fnn, cnn)

    model = load_model()
    model = model.to(device, dtype=torch.float32)

    # inference loop
    vespag_scores = {}
    scores_per_protein = {}
    for id, embedding in (pbar := tqdm(embeddings.items(), desc="Generating predictions")):
        pbar.set_postfix({"current_protein": id})
        y = model(embedding)
        if fasta_file:
            y = mask_non_mutations(y, sequences[id])
        if h5_output:
            vespag_scores[id] = y.detach()
        # scores_per_protein[id] = {
        #     mutation: compute_mutation_score(y, mutation)
        #     for mutation in mutations_per_protein[id]
        # }
        scores_per_protein[id] = {
            f'{str(mutation)[0]}{int(str(mutation)[1])+1}{str(mutation)[2]}': compute_mutation_score(y, mutation)
            for mutation in mutations_per_protein[id]
        }

    # write output
    if not no_csv:
        if not single_csv:
            # TODO verify that output is CSV file
            with output.open("w+") as f:
                f.writelines([line for line in tqdm([
                    f"{protein_id}_{str(sav)},{score}\n"
                    for protein_id, mutations in scores_per_protein.items()
                    for sav, score in mutations.items()], desc="Generating output file")
                              ]),
        else:
            # TODO verify that output is directory
            for protein_id, mutations in tqdm(scores_per_protein.items(), desc="Generating output files"):
                output_file = Path(output, protein_id + ".csv")
                with output_file.open("w+") as f:
                    f.writelines("Mutation, VespaG\n")
                    f.writelines([f"{str(sav)},{score}\n" for sav, score in mutations.items()])

    if h5_output:
        with h5py.File(h5_output, "w") as f:
            for id, vespag_prediction in tqdm(vespag_scores.items(), desc="Generating H5 output file"):
                f.create_dataset(id, data=vespag_prediction)


if __name__ == '__main__':
    predict(embeddings_file= 'data/test/proteingym_217_esm2.h5',
        output= 'data/test/pg217_1idx',
        fasta_file= 'data/test/proteingym_217.fasta',
        model = "fnn",
        embedding_type = "esm2",
        single_csv= True,
        no_csv= False,
        )