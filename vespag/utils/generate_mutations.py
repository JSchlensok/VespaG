from typing import Dict, List

from .mutations import SAV
from .utils import AMINO_ACIDS


def generate_protein_mutations(sequences: Dict[str, str],
                               zero_based_mutations: bool = False,
                               tqdm: bool = True) -> Dict[str, List[SAV]]:
    """
    Generates all possible SAVs for the given protein sequences.

    Args:
        sequences: Dict with sequence_id to sequence.
        zero_based_mutations: If the SAV should be given with zero indexes (e.g. A0M) or one indexes (e.g. A1M).
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
