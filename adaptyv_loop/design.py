"""Design generation and scoring — the front half of the loop.

The rest of this package decides *which* designs to buy. This module produces
them, so a campaign can run end to end: generate, score, allocate, submit,
learn.

Four generators are provided, each registered as a distinct **method** because
method is the unit the selection policy allocates across:

``scaffold_mutagenesis``
    Lead optimization. Mutates CDR positions of a known binder scaffold,
    sampling replacements from natural amino-acid frequencies.
``plm_guided``
    Masks CDR positions and samples replacements from ESM-2's predicted
    distribution, so proposals stay in-distribution for real protein space.
``germline_recombination``
    Grafts a randomized CDR3 of variable length onto a germline framework.
    Closest thing here to de novo, and correspondingly harder.
``random_control``
    Uniform-random CDR residues. A deliberate negative control — a campaign
    that cannot rank this last is not learning anything.

CDRs are located by conserved framework anchors (the Cys before CDR1, the
``W[VFIL]RQ`` motif after it, and the ``Y[YFHC]C ... WG.G`` bracket around
CDR3) rather than by fixed indices, so the generators work on any antibody or
nanobody scaffold. This is anchor-based approximation, not formal IMGT
numbering; it is adequate for proposing variants and is not a substitute for
proper numbering in a real pipeline.

Scoring uses ESM-2 pseudo-log-likelihood, computed locally. On Apple silicon
the 35M-parameter model loads in about five seconds and scores hundreds of
designs in well under a minute.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .selection import Candidate

log = logging.getLogger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: Background amino-acid frequencies in vertebrate proteins (UniProt-derived).
#: Sampling from these rather than uniformly keeps mutagenesis in a plausible
#: composition regime.
NATURAL_FREQ = {
    "A": 0.074, "C": 0.025, "D": 0.054, "E": 0.054, "F": 0.047,
    "G": 0.074, "H": 0.026, "I": 0.068, "K": 0.058, "L": 0.099,
    "M": 0.025, "N": 0.045, "P": 0.039, "Q": 0.034, "R": 0.052,
    "S": 0.057, "T": 0.051, "V": 0.073, "W": 0.013, "Y": 0.032,
}

#: Kyte-Doolittle hydropathy. Used only to flag aggregation-prone CDR3s.
HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
    "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
    "W": -0.9, "Y": -1.3,
}

# Conserved framework anchors bracketing the CDRs.
_CDR1 = re.compile(r"C(?P<cdr1>.{5,20}?)W[VFIL]RQ")
_CDR3 = re.compile(r"Y[YFHC]C(?P<cdr3>.{3,30}?)WG.G")

#: A real anti-EGFR nanobody (VHH) from the public competition, used as the
#: optimization scaffold.
VHH_SCAFFOLD = (
    "EVQLLESGGGVVKPGGSLRLSCAASGRTFSSYAMGWFRQAPGKGLEWVSAINWSSGSTYYADSVKG"
    "RFTISRDDAKNSLYLQMNSLRAEDTAVYYCAAGYQINSGNYNFKDYEYDYWGQGTLVTVSS"
)


@dataclass(frozen=True)
class CDRs:
    """Half-open ``[start, stop)`` spans of the located CDRs."""

    cdr1: tuple[int, int] | None
    cdr3: tuple[int, int] | None

    def positions(self) -> list[int]:
        out: list[int] = []
        for span in (self.cdr1, self.cdr3):
            if span:
                out.extend(range(*span))
        return out


def locate_cdrs(sequence: str) -> CDRs:
    """Locate CDR1 and CDR3 by conserved framework anchors.

    Returns empty spans for sequences with no antibody framework — de novo
    minibinders, for instance — so callers can skip them rather than mutate
    arbitrary positions.
    """
    m1, m3 = _CDR1.search(sequence), _CDR3.search(sequence)
    return CDRs(
        cdr1=(m1.start("cdr1"), m1.end("cdr1")) if m1 else None,
        cdr3=(m3.start("cdr3"), m3.end("cdr3")) if m3 else None,
    )


def _weighted_residue(rng: random.Random) -> str:
    return rng.choices(list(NATURAL_FREQ), weights=list(NATURAL_FREQ.values()))[0]


# ---- generators --------------------------------------------------------

Generator = Callable[[str, int, random.Random], list[str]]


def scaffold_mutagenesis(
    scaffold: str, n: int, rng: random.Random, *, n_mutations: int = 4
) -> list[str]:
    """Mutate a handful of CDR positions, sampling from natural frequencies."""
    cdrs = locate_cdrs(scaffold)
    positions = cdrs.positions()
    if not positions:
        return []
    out = []
    for _ in range(n):
        seq = list(scaffold)
        for pos in rng.sample(positions, min(n_mutations, len(positions))):
            seq[pos] = _weighted_residue(rng)
        out.append("".join(seq))
    return out


def germline_recombination(
    scaffold: str, n: int, rng: random.Random, *, length_range: tuple[int, int] = (8, 20)
) -> list[str]:
    """Graft a fully randomized CDR3 of variable length onto the framework."""
    cdrs = locate_cdrs(scaffold)
    if not cdrs.cdr3:
        return []
    start, stop = cdrs.cdr3
    out = []
    for _ in range(n):
        length = rng.randint(*length_range)
        cdr3 = "".join(_weighted_residue(rng) for _ in range(length))
        out.append(scaffold[:start] + cdr3 + scaffold[stop:])
    return out


def random_control(scaffold: str, n: int, rng: random.Random) -> list[str]:
    """Uniform-random CDR residues. The negative control."""
    cdrs = locate_cdrs(scaffold)
    positions = cdrs.positions()
    if not positions:
        return []
    out = []
    for _ in range(n):
        seq = list(scaffold)
        for pos in positions:
            seq[pos] = rng.choice(AMINO_ACIDS)
        out.append("".join(seq))
    return out


class ESM2:
    """Local ESM-2, used for both scoring and PLM-guided design.

    Loaded lazily so importing this module stays cheap for callers that only
    want the non-PLM generators.
    """

    def __init__(self, model_name: str = "esm2_t12_35M_UR50D") -> None:
        self.model_name = model_name
        self._model = None
        self._alphabet = None
        self._device = None

    def _ensure(self):
        if self._model is not None:
            return
        import esm
        import torch

        log.info("loading %s ...", self.model_name)
        self._model, self._alphabet = getattr(esm.pretrained, self.model_name)()
        self._model.eval()
        self._device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = self._model.to(self._device)

    def pseudo_log_likelihood(self, sequences: Sequence[str], batch_size: int = 16) -> list[float]:
        """Length-normalized pseudo-log-likelihood, one score per sequence.

        Uses the single-forward-pass ("wt-marginal") approximation: sum the log
        probability the model assigns to the residue actually present at each
        position. Length normalization keeps variable-length CDR3s comparable.
        """
        import torch

        self._ensure()
        converter = self._alphabet.get_batch_converter()
        scores: list[float] = []
        for i in range(0, len(sequences), batch_size):
            chunk = sequences[i : i + batch_size]
            _, _, tokens = converter([(str(j), s) for j, s in enumerate(chunk)])
            tokens = tokens.to(self._device)
            with torch.no_grad():
                logits = self._model(tokens)["logits"]
            logprobs = torch.log_softmax(logits, dim=-1)
            for row, seq in enumerate(chunk):
                # Offset by one for the prepended BOS token.
                idx = torch.arange(1, len(seq) + 1, device=self._device)
                actual = tokens[row, idx]
                total = logprobs[row, idx, actual].sum().item()
                scores.append(total / len(seq))
        return scores

    def guided_design(
        self,
        scaffold: str,
        n: int,
        rng: random.Random,
        *,
        n_mutations: int = 4,
        temperature: float = 1.0,
    ) -> list[str]:
        """Mask CDR positions and sample replacements from ESM-2's posterior."""
        import torch

        self._ensure()
        positions = locate_cdrs(scaffold).positions()
        if not positions:
            return []
        converter = self._alphabet.get_batch_converter()
        mask_idx = self._alphabet.mask_idx
        aa_tokens = {aa: self._alphabet.get_idx(aa) for aa in AMINO_ACIDS}
        token_ids = torch.tensor([aa_tokens[a] for a in AMINO_ACIDS], device=self._device)

        out = []
        for _ in range(n):
            seq = list(scaffold)
            for pos in rng.sample(positions, min(n_mutations, len(positions))):
                _, _, tokens = converter([("x", "".join(seq))])
                tokens = tokens.to(self._device)
                tokens[0, pos + 1] = mask_idx
                with torch.no_grad():
                    logits = self._model(tokens)["logits"][0, pos + 1]
                probs = torch.softmax(logits[token_ids] / temperature, dim=-1)
                choice = torch.multinomial(probs, 1).item()
                seq[pos] = AMINO_ACIDS[choice]
            out.append("".join(seq))
        return out


def cdr3_quality(sequence: str) -> float:
    """A crude developability proxy for the CDR3, in roughly [-1, 1].

    Penalizes strong net hydrophobicity (aggregation-prone) and extreme net
    charge, both of which track poor expression and non-specific binding. Not a
    prediction of affinity — just a filter on obviously bad loops.
    """
    cdrs = locate_cdrs(sequence)
    if not cdrs.cdr3:
        return 0.0
    loop = sequence[slice(*cdrs.cdr3)]
    if not loop:
        return 0.0
    hydro = sum(HYDROPATHY.get(a, 0.0) for a in loop) / len(loop)
    charge = sum((a in "KR") - (a in "DE") for a in loop) / len(loop)
    return -abs(hydro) / 4.5 - abs(charge) * 1.5 + 0.5


def stable_id(sequence: str, prefix: str = "d") -> str:
    """Content-addressed design id, so the same sequence keeps the same id."""
    return f"{prefix}-{hashlib.sha1(sequence.encode()).hexdigest()[:10]}"


def generate_library(
    n_per_method: int = 100,
    *,
    scaffold: str = VHH_SCAFFOLD,
    seed: int = 0,
    esm: ESM2 | None = None,
    include_plm: bool = True,
) -> list[Candidate]:
    """Build a design library across all generators, scored by ESM-2.

    Returns :class:`~adaptyv_loop.selection.Candidate` objects tagged with the
    generator that produced them, ready to hand to ``select_designs`` or a
    ``Campaign``.
    """
    rng = random.Random(seed)
    esm = esm or ESM2()

    batches: list[tuple[str, Iterable[str]]] = [
        ("scaffold_mutagenesis", scaffold_mutagenesis(scaffold, n_per_method, rng)),
        ("germline_recombination", germline_recombination(scaffold, n_per_method, rng)),
        ("random_control", random_control(scaffold, n_per_method, rng)),
    ]
    if include_plm:
        batches.append(("plm_guided", esm.guided_design(scaffold, n_per_method, rng)))

    seen: set[str] = set()
    methods: list[str] = []
    sequences: list[str] = []
    for method, seqs in batches:
        for seq in seqs:
            if seq in seen:
                continue  # a duplicate would be paid for twice for one measurement
            seen.add(seq)
            methods.append(method)
            sequences.append(seq)

    log.info("scoring %d designs with ESM-2 ...", len(sequences))
    scores = esm.pseudo_log_likelihood(sequences)

    return [
        Candidate(
            id=stable_id(seq),
            sequence=seq,
            method=method,
            score=score,
            metadata={"cdr3_quality": round(cdr3_quality(seq), 4)},
        )
        for method, seq, score in zip(methods, sequences, scores)
    ]
