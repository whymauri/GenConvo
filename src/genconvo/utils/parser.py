"""
Parser for extracting Q&A pairs from GenConvoBench pipeline results.
"""

from typing import Dict, List, Any, Tuple, Iterable, Mapping, Optional
from dataclasses import dataclass, fields
from datetime import datetime
import hashlib
from .schemas import ParseContext


@dataclass
class QAPair:
    """A question-answer pair with metadata."""
    run_id: str
    prompt_type: str
    question: str
    answer: str
    model: str
    temperature: float
    filename: str
    document_hash: str
    layer_index: int
    timestamp: str


def _keys_with_suffix(items: Iterable[str], suffix: str) -> List[str]:
    return [k for k in items if k.endswith(suffix)]

def _first_str_value_by_suffix(mapping: Mapping[str, Any], suffix: str) -> str:
    return next((v for k, v in mapping.items() if k.endswith(suffix) and isinstance(v, str)), "")


def parse_results(
    results: Tuple[Dict[str, Any], List[str]],
    context: ParseContext,
) -> List[QAPair]:
    """
    Parse verdict pipeline results into Q&A pairs.
    
    Args:
        results: A tuple of (results_dict, leaf_prefixes) from the pipeline
        context: Immutable parse context with metadata
    
    Returns:
        List of QAPair objects
    """
    qa_pairs = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find document and questions once
    results_dict, leaf_prefixes = results

    document = _first_str_value_by_suffix(results_dict, "_document")
    questions_key = _keys_with_suffix(results_dict.keys(), "_questions")[0]
    answer_keys: List[str] = _keys_with_suffix(leaf_prefixes, "_answer")

    document_hash = hashlib.md5((document or "").encode()).hexdigest()
    questions = results_dict.get(questions_key, [])

    for i, a_key in enumerate(answer_keys):
        answer_text = results_dict.get(a_key, "")
        question_text = questions[i]

        qa_pairs.append(
            QAPair(
                run_id=run_id,
                prompt_type=context.prompt_type,
                question=question_text,
                answer=answer_text,
                model=context.model,
                temperature=context.temperature,
                filename=context.filename,
                document_hash=document_hash,
                layer_index=i,
                timestamp=datetime.now().isoformat(),
            )
        )
    
    return qa_pairs


def qa_pairs_to_dataset(qa_pairs: List[QAPair]):
    """Convert Q&A pairs to HuggingFace dataset format (dict of column lists).

    Automatically stays in sync with QAPair fields.
    """
    return {f.name: [getattr(pair, f.name) for pair in qa_pairs] for f in fields(QAPair)}