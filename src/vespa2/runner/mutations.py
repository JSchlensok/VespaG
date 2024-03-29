from __future__ import annotations
from dataclasses import dataclass

@dataclass
class SAV:
    position: int
    from_aa: str
    to_aa: str

    @classmethod
    def from_sav_string(cls, sav_string: str, one_indexed: bool=False, offset: int=0) -> SAV:
        from_aa, to_aa = sav_string[0], sav_string[-1]
        position = int(sav_string[1:-1]) - offset
        if one_indexed:
            position -= 1
        return SAV(position, from_aa, to_aa)

    def __str__(self) -> str:
        return f"{self.from_aa}{self.position}{self.to_aa}"

    def __hash__(self):
        return hash(str(self))

@dataclass
class Mutation:
    savs: list[SAV]

    @classmethod
    def from_mutation_string(cls, mutation_string: str, one_indexed: bool=False, offset: int=0) -> Mutation:
        return Mutation([SAV.from_sav_string(sav_string, one_indexed=one_indexed, offset=offset) for sav_string in mutation_string.split(':')])

    def __str__(self) -> str:
        return ':'.join([str(sav) for sav in self.savs])

    def __hash__(self):
        return hash(str(self))

    def __iter__(self):
        yield from self.savs
