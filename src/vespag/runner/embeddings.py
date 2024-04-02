import torch

from pathlib import Path
from typing import Optional

from Bio import SeqIO
import h5py
import re

from src.vespag.utils import get_device


def get_esm2_embeddings(
        fasta_path: Path, 
        out_path: Path,
        model_name: str= "facebook/esm2_t36_3B_UR50D", 
        maxlen: Optional[int]= 4096,
    )-> None:
    from transformers import AutoTokenizer, AutoModel

    print(f'Generating ESM-2 embeddings for {fasta_path}')

    device = get_device()

    ids = [str(record.id) for record in SeqIO.parse(fasta_path, "fasta")]
    seqs = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    inputs = tokenizer(seqs, return_tensors="pt", truncation=True, padding=True, max_length=maxlen).to(device) 
    with torch.no_grad():
        outputs = model(**inputs)
    
    # x = torch.tensor(outputs.last_hidden_state[:, 1:-1, :].squeeze(0), dtype=torch.float32)
    last_hidden_states = outputs.last_hidden_state[:, 1:-1, :].squeeze(0)
    x = last_hidden_states.detach().cpu()

    with h5py.File(out_path + '/esm2_embeddings.h5', "w") as hdf:
        for label, embedding in zip(ids, x):
            hdf.create_dataset(name=label, data=embedding)

    return {label: emb for label, emb in zip(ids, last_hidden_states)}

# TODO finish h5file creation (padding + length)
# https://github.com/agemagician/ProtTrans/blob/master/Embedding/prott5_embedder.py
def get_prott5_embeddings(
        fasta_path: Path, 
        out_path: Path,
        model_name: str= "Rostlab/prot_t5_xl_uniref50", 
        maxlen: Optional[int]= 4096,
    )-> None:
    from transformers import T5Tokenizer, T5EncoderModel

    device = get_device()

    ids = [str(record.id) for record in SeqIO.parse(fasta_path, "fasta")]
    seqs = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]
    # replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
    seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seqs]

    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False).to(device)
    model = T5EncoderModel.from_pretrained(model_name).to(device)

    # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
    model.full() if device=='cpu' else model.half()

    # tokenize sequences and pad up to the longest sequence in the batch
    seqs_tok = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest",return_tensors='pt').to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_rpr = model(
                seqs_tok.input_ids, 
                attention_mask=seqs_tok.attention_mask
                )

    # extract residue embeddings for the first ([0,:]) sequence in the batch and remove padded & special tokens, incl. prefix ([0,1:8]) 
    emb_0 = embedding_rpr.last_hidden_state[0,1:8] # shape (7 x 1024)
    # same for the second ([1,:]) sequence but taking into account different sequence lengths ([1,:6])
    emb_1 = embedding_rpr.last_hidden_state[1,1:6] # shape (5 x 1024)


if __name__ == '__main__':
    get_esm2_embeddings(
        fasta_path='/mnt/project/marquet/Variants/VespaG/VespaG/data/test/test.fasta', 
        out_path='/mnt/project/marquet/Variants/VespaG/VespaG/data/test',
        )
