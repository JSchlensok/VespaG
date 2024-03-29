from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

from pathlib import Path
from typing import Optional

from Bio import SeqIO
import h5py


def get_esm2_embeddings(
        fasta_path: Path, 
        out_path: Path,
        model_name: str= "facebook/esm2_t36_3B_UR50D", 
        maxlen: Optional[int]= 4096,
    )-> None:
    """    
    Returns:
    - numpy.ndarray: A matrix where each row is the embedding of a token in the text.
    """

    ids = [str(record.id) for record in SeqIO.parse(fasta_path, "fasta")]
    seqs = [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(seqs, return_tensors="pt", truncation=True, padding=True, max_length=maxlen) 
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_states = outputs.last_hidden_state[:, 1:-1, :].squeeze(0)
    x = last_hidden_states.detach()

    with h5py.File(out_path, "w") as hdf:
        for label, embedding in zip(ids, x):
            hdf.create_dataset(name=label, data=embedding)

    return {label: emb for label, emb in zip(ids, x)}


if __name__ == '__main__':
    get_esm2_embeddings(
        fasta_path='/mnt/project/marquet/Variants/VespaG/VespaG/data/test/test.fasta', 
        out_path='/mnt/project/marquet/Variants/VespaG/VespaG/data/test/test.h5',
        )
