"""doc_to_text variant that prepends an instruction once, before all examples.

The few-shot examples are a fixed, ordered list.  Prepending only to the
first few-shot document makes the instruction appear exactly once at the
start of the conversation — mirroring the behaviour of lm_eval's
`description` field — rather than repeating before every user turn.

Set BOBENCH_INSTRUCTION before calling lm_eval to inject the instruction.
"""
import os

from lm_eval.tasks.minerva_math.utils import doc_to_text as _base_doc_to_text

# First problem in list_fewshot_samples() — used to detect the first turn.
_FIRST_FEWSHOT_PREFIX = "Find the domain of the expression"


def doc_to_text(doc: dict) -> str:
    instruction = os.environ.get("BOBENCH_INSTRUCTION", "")
    base = _base_doc_to_text(doc)
    if instruction and doc.get("problem", "").startswith(_FIRST_FEWSHOT_PREFIX):
        return instruction + "\n\n" + base
    return base
