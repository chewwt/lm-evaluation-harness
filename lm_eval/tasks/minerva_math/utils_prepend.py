"""doc_to_text variant that prepends an instruction from the environment.

Set BOBENCH_INSTRUCTION before calling lm_eval to inject the instruction
in front of each problem (user-turn) rather than as a system prompt.
"""
import os

from lm_eval.tasks.minerva_math.utils import doc_to_text as _base_doc_to_text


def doc_to_text(doc: dict) -> str:
    instruction = os.environ.get("BOBENCH_INSTRUCTION", "")
    if instruction:
        return instruction + "\n\n" + _base_doc_to_text(doc)
    return _base_doc_to_text(doc)
