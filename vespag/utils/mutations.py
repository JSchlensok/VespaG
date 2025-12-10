from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import polars as pl
import rich
import torch
from jaxtyping import Float

from .utils import AMINO_ACIDS, GEMME_ALPHABET, normalize_scores, transform_scores


@dataclass
class SAV:
    position: int
    from_aa: str
    to_aa: str
    one_indexed: bool = False

    @classmethod
    def from_sav_string(cls, sav_string: str, one_indexed: bool = False, offset: int = 0) -> SAV:
        from_aa, to_aa = sav_string[0], sav_string[-1]
        position = int(sav_string[1:-1]) - offset
        if one_indexed:
            position -= 1
        return SAV(position, from_aa, to_aa, one_indexed=one_indexed)

    def __str__(self) -> str:
        pos = self.position
        if self.one_indexed:
            pos += 1
        return f"{self.from_aa}{pos}{self.to_aa}"

    def __hash__(self):
        return hash(str(self))


@dataclass
class Mutation:
    savs: list[SAV]

    @classmethod
    def from_mutation_string(cls, mutation_string: str, one_indexed: bool = False, offset: int = 0) -> Mutation:
        return Mutation(
            [
                SAV.from_sav_string(sav_string, one_indexed=one_indexed, offset=offset)
                for sav_string in mutation_string.split(":")
            ]
        )

    def __str__(self) -> str:
        return ":".join([str(sav) for sav in self.savs])

    def __hash__(self):
        return hash(str(self))

    def __iter__(self):
        yield from self.savs


def mask_non_mutations(
    prediction: Float[torch.Tensor, "length 20"], wildtype_sequence
) -> Float[torch.Tensor, "length 20"]:
    """
    Simply set the predicted effect of the wildtype amino acid at each position (i.e. all non-mutations) to 0
    """
    prediction[
        torch.tensor([i for i in range(len(wildtype_sequence)) if wildtype_sequence[i] != 'X']),
        torch.tensor([GEMME_ALPHABET.index(aa) for aa in wildtype_sequence if aa != 'X']),
    ] = 0.0

    return prediction


def read_mutation_file(mutation_file: Path, one_indexed: bool = False) -> dict[str, list[SAV]]:
    mutations_per_protein = defaultdict(list)
    for row in pl.read_csv(mutation_file, has_header=False).iter_rows():
        mutations_per_protein[row[0]].append(Mutation.from_mutation_string(row[1], one_indexed))

    return mutations_per_protein


def compute_mutation_score(
    substitution_score_matrix: Float[np.typing.ArrayLike, "length 20"],
    mutation: Mutation | SAV,
    alphabet: str = GEMME_ALPHABET,
    transform: bool = True,
    embedding_type: EmbeddingType = "esm2",
    normalize: bool = False,
    pbar: rich.progress.Progress | None = None,
    progress_id: int | None = None,
) -> float:
    if pbar:
        pbar.advance(progress_id)

    if isinstance(mutation, Mutation):
        raw_scores = [substitution_score_matrix[sav.position][alphabet.index(sav.to_aa)].item() for sav in mutation]
    else:
        raw_scores = [substitution_score_matrix[mutation.position][alphabet.index(mutation.to_aa)].item()]

    if transform:
        raw_scores = transform_scores(raw_scores, embedding_type)

    score = sum(raw_scores)

    if normalize:
        score = normalize_scores(score)

    return score


def generate_sav_landscape(
    sequences: dict[str, str], zero_based_mutations: bool = False, tqdm: bool = True
) -> dict[str, list[SAV]]:
    """
    Generates all possible SAVs for the given protein sequences.

    Args:
        sequences: Protein ID - protein sequence dict.
        zero_based_mutations: Whether to return zero indices (e.g. A0M) or one indices (e.g. A1M).
        tqdm: If True, the generation is wrapped in a tqdm progress bar.

    Returns:
        A dictionary with sequence_id as key and SAVs as values.
    """
    if tqdm:
        from tqdm.rich import tqdm

        wrap_function = tqdm
    else:
        wrap_function = lambda d: d

    return {
        protein_id: [
            SAV(i, wildtype_aa, other_aa, not zero_based_mutations)
            for i, wildtype_aa in enumerate(sequence)
            for other_aa in AMINO_ACIDS
            if other_aa != wildtype_aa
        ]
        for protein_id, sequence in wrap_function(sequences.items())
    }
