"""
oracle.py — Pluggable oracle abstraction for pERbacco.

An oracle receives a *batch* of record IDs and returns, for each pair within
that batch, whether the two records refer to the same real-world entity.  This
matches the "batched oracle" model described in the paper.

Two concrete implementations are provided:

  GroundTruthOracle  — answers using pre-annotated ground-truth labels.
                       This replicates the original behaviour of the code and
                       is the default used in the paper's experiments.

  LLMOracle          — sends the full batch to an OpenAI chat model in a
                       single prompt, exploiting cross-record context.  The
                       model returns a JSON list of matching pairs.

Usage
-----
from oracle import GroundTruthOracle, LLMOracle

# Ground-truth (default)
oracle = GroundTruthOracle(dict_ground_truth)

# LLM
oracle = LLMOracle(
    model="gpt-4o-mini",
    api_key="sk-...",
    record_data={0: {"title": "...", "author": "..."}, ...},
)

decisions = oracle.query_batch([0, 1, 2, 3])
# decisions: {(0, 1): True, (0, 2): False, ...}
"""

from __future__ import annotations

import itertools
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseOracle(ABC):
    """Abstract base class for ER oracles.

    Subclasses must implement :meth:`query_batch`, which decides whether each
    pair of records in a given batch refer to the same real-world entity.
    """

    @abstractmethod
    def query_batch(self, batch: List[int]) -> Dict[Tuple[int, int], bool]:
        """Decide, for every pair in *batch*, whether the two records match.

        Args:
            batch: Sequence of record / entity IDs forming the current batch.

        Returns:
            A dictionary mapping ``(u, v)`` → ``bool`` where ``u < v``.
            ``True`` means the two records refer to the same real-world entity.
            Every pair in ``itertools.combinations(batch, 2)`` must be present.
        """


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth oracle
# ─────────────────────────────────────────────────────────────────────────────

class GroundTruthOracle(BaseOracle):
    """Oracle that answers queries using pre-annotated ground-truth data.

    This is the original oracle used in the paper's experiments and preserves
    the exact semantics of the ``if u in self.dict_ground_truth[v]`` check
    that existed in ``class_entity.query()``.

    Args:
        dict_ground_truth: Mapping ``record_id → set of matching record IDs``.
                           Produced by ``class_entity.create_dict_ground_truth()``.
    """

    def __init__(self, dict_ground_truth: Dict[int, Set[int]]) -> None:
        self.dict_ground_truth = dict_ground_truth

    def query_batch(self, batch: List[int]) -> Dict[Tuple[int, int], bool]:
        results: Dict[Tuple[int, int], bool] = {}
        for x, y in itertools.combinations(batch, 2):
            u, v = (x, y) if x < y else (y, x)
            results[(u, v)] = u in self.dict_ground_truth.get(v, set())
        return results


# ─────────────────────────────────────────────────────────────────────────────
# LLM oracle
# ─────────────────────────────────────────────────────────────────────────────

class LLMOracle(BaseOracle):
    """Oracle that queries an OpenAI chat model to decide whether records match.

    The entire batch of records is sent in a single prompt so the model can
    exploit cross-record context — matching the "batched oracle" spirit of the
    paper.  The model is expected to return a JSON array of ``[id1, id2]`` pairs
    representing duplicates.

    Args:
        model:        OpenAI model name, e.g. ``"gpt-4o-mini"`` or ``"gpt-4o"``.
        api_key:      OpenAI API key.  If ``None``, the client will fall back to
                      the ``OPENAI_API_KEY`` environment variable.
        record_data:  Mapping ``record_id → {attribute: value, …}`` used to
                      build the prompt.  Keyed by the same integer IDs used in
                      the similarity graph.
        max_retries:  How many times to retry on transient API errors.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str],
        record_data: Dict[int, dict],
        max_retries: int = 3,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for LLMOracle.  "
                "Install it with:  pip install openai"
            ) from exc

        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.record_data = record_data
        self.max_retries = max_retries

    # ── Prompt construction ───────────────────────────────────────────────── #

    def _format_record(self, record_id: int) -> str:
        record = self.record_data.get(record_id, {})
        attrs = ", ".join(f"{k}: {v}" for k, v in record.items())
        return f"[{record_id}] {attrs}"

    def _build_prompt(self, batch: List[int]) -> str:
        """Build the prompt sent to the LLM.

        NOTE: This is a minimal placeholder prompt.  For production use, tailor
        the prompt to your dataset's schema and add few-shot examples.
        """
        records_block = "\n".join(self._format_record(r) for r in batch)
        prompt = (
            "You are an expert in entity resolution.\n"
            "Below is a list of records.  Your task is to identify which pairs of "
            "records refer to the same real-world entity.\n\n"
            f"Records:\n{records_block}\n\n"
            "Return ONLY a JSON array of [id1, id2] pairs where the two records are "
            "duplicates of each other.  If there are no duplicates, return an empty "
            "array [].  Do not include any explanation — JSON only.\n\n"
            "Answer:"
        )
        return prompt

    # ── Response parsing ──────────────────────────────────────────────────── #

    def _parse_response(
        self, text: str, batch: List[int]
    ) -> Dict[Tuple[int, int], bool]:
        """Parse the LLM's JSON response into a match-decision dictionary."""
        # Initialise all pairs as non-matches
        results: Dict[Tuple[int, int], bool] = {
            (min(x, y), max(x, y)): False
            for x, y in itertools.combinations(batch, 2)
        }

        # Extract the first JSON array from the response text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            print(f"[LLMOracle] Could not find a JSON array in response: {text[:200]!r}")
            return results

        try:
            pairs = json.loads(match.group())
            for pair in pairs:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    u, v = int(pair[0]), int(pair[1])
                    key = (min(u, v), max(u, v))
                    if key in results:
                        results[key] = True
                    else:
                        print(
                            f"[LLMOracle] Pair {key} returned by LLM is not in the "
                            "current batch — ignored."
                        )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"[LLMOracle] Failed to parse response: {exc}\nRaw: {text[:300]!r}")

        return results

    # ── Core query method ─────────────────────────────────────────────────── #

    def query_batch(self, batch: List[int]) -> Dict[Tuple[int, int], bool]:
        """Send the batch to the LLM and return match decisions for all pairs."""
        prompt = self._build_prompt(batch)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                return self._parse_response(
                    response.choices[0].message.content, batch
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[LLMOracle] API error (attempt {attempt}/{self.max_retries}): {exc}"
                )
                if attempt == self.max_retries:
                    # Fail-safe: treat all pairs in the batch as non-matches
                    print("[LLMOracle] All retries exhausted — returning all non-matches.")
                    return {
                        (min(x, y), max(x, y)): False
                        for x, y in itertools.combinations(batch, 2)
                    }


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def create_oracle(oracle_type: str, **kwargs) -> BaseOracle:
    """Instantiate the appropriate oracle.

    Args:
        oracle_type: ``"ground_truth"`` or ``"llm"``.
        **kwargs:    Forwarded to the oracle's ``__init__``.

    Returns:
        An instance of :class:`BaseOracle`.
    """
    if oracle_type == "ground_truth":
        return GroundTruthOracle(**kwargs)
    if oracle_type == "llm":
        return LLMOracle(**kwargs)
    raise ValueError(
        f"Unknown oracle type: {oracle_type!r}.  "
        "Valid choices are 'ground_truth' and 'llm'."
    )
