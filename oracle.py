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
from pathlib import Path
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

    The entire batch of records is sent in a single prompt.  The model is asked
    to return a *clustering* of the records — an array of groups where each group
    contains IDs that refer to the same real-world entity.  This is more natural
    than asking for pairs and guarantees that the model's output is transitively
    consistent within each group (if A and B are in the same group, and B and C
    are in the same group, A and C are implicitly matched too).

    The clustering is then converted internally to a ``{(u, v): bool}`` dict so
    the rest of the codebase does not need to change.

    The prompt template is loaded from ``config/prompts/llm_oracle.txt`` (relative
    to this file).  Edit that file to customise the prompt without touching code.

    Args:
        model:        OpenAI model name, e.g. ``"gpt-4o-mini"`` or ``"gpt-4o"``.
        api_key:      OpenAI API key.  If ``None``, the client will fall back to
                      the ``OPENAI_API_KEY`` environment variable.
        record_data:  Mapping ``record_id → {attribute: value, …}`` used to
                      build the prompt.  Keyed by the same integer IDs used in
                      the similarity graph.
        max_retries:  How many times to retry on transient API errors.
    """

    # Default template path, relative to this source file
    _DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "config" / "llm_oracle.yaml"

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
        self._prompt_template = self._load_prompt_template()

    # ── Prompt construction ───────────────────────────────────────────────── #

    @classmethod
    def _load_prompt_template(cls) -> str:
        """Load the prompt template from ``config/prompts/llm_oracle.yaml``.

        The YAML file is expected to have a single ``template`` key whose value
        is the prompt string (with a ``{records}`` placeholder).  Falls back to
        an inline copy of the same text if the file is missing or pyyaml is not
        installed.
        """
        if cls._DEFAULT_TEMPLATE_PATH.exists():
            try:
                import yaml
                data = yaml.safe_load(cls._DEFAULT_TEMPLATE_PATH.read_text())
                return data["template"]
            except Exception as exc:
                print(f"[LLMOracle] Could not load prompt template: {exc} — using inline fallback.")

        print(
            f"[LLMOracle] Prompt template not found at "
            f"{cls._DEFAULT_TEMPLATE_PATH} — using inline fallback."
        )
        return (
            "You are an expert in entity resolution.\n\n"
            "Your task is to group the following records by the real-world entity "
            "they describe.\n"
            "Records that refer to the same real-world entity belong in the same group.\n"
            "Records that do not match any other record form a group of one.\n"
            "Every record must appear in exactly one group.\n\n"
            "Records:\n{records}\n\n"
            "Return ONLY a JSON array of groups, where each group is an array of "
            "record IDs.\n"
            "Do not include any explanation or commentary — JSON only.\n\n"
            "For example, if records 1, 3 and 7 refer to the same entity, record 2 "
            "is unique,\nand records 4 and 5 refer to the same entity, return:\n\n"
            "[[1, 3, 7], [2], [4, 5]]\n\n"
            "Answer:"
        )

    def _format_record(self, record_id: int) -> str:
        record = self.record_data.get(record_id, {})
        attrs = ", ".join(f"{k}: {v}" for k, v in record.items())
        return f"[{record_id}] {attrs}"

    def _build_prompt(self, batch: List[int]) -> str:
        """Fill the prompt template with the formatted records for this batch."""
        records_block = "\n".join(self._format_record(r) for r in batch)
        return self._prompt_template.replace("{records}", records_block)

    # ── Response parsing ──────────────────────────────────────────────────── #

    def _parse_response(
        self, text: str, batch: List[int]
    ) -> Dict[Tuple[int, int], bool]:
        """Parse a cluster-based JSON response into a match-decision dictionary.

        The LLM returns an array of groups, e.g. ``[[1, 3], [2], [4, 5]]``.
        All pairs *within* the same group are marked True; all pairs *across*
        groups are marked False.  Records omitted from the response (the LLM
        forgot to list a singleton) are treated as unmatched.
        """
        # Initialise all pairs as non-matches
        results: Dict[Tuple[int, int], bool] = {
            (min(x, y), max(x, y)): False
            for x, y in itertools.combinations(batch, 2)
        }

        # Extract the outermost JSON array from the response
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            print(f"[LLMOracle] Could not find a JSON array in response: {text[:200]!r}")
            return results

        batch_set = set(batch)
        try:
            clusters = json.loads(match.group())
            for cluster in clusters:
                if not isinstance(cluster, list):
                    continue
                # Keep only IDs that actually belong to the current batch
                valid_ids = [int(rid) for rid in cluster if int(rid) in batch_set]
                # Every pair within this cluster is a match
                for u, v in itertools.combinations(valid_ids, 2):
                    key = (min(u, v), max(u, v))
                    if key in results:
                        results[key] = True
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
