from typing import List

from verdict.schema import Schema

from .base import BaseCachedUnit


class AnswerUnit(BaseCachedUnit):
    """Generate one answer with document cached in system message."""

    class InputSchema(Schema):
        document: str
        questions: List[str]

    class ResponseSchema(Schema):
        answer: str

    def __init__(self):
        super().__init__()

    def populate_prompt_message(self, input_data, logger):
        """Use BaseCachedUnit to wrap with cache control after building messages."""
        # Use the instance index assigned by Layer via idx(), defaulting to 0
        idx = int(getattr(self, "index", 0))

        question_text = input_data.questions[idx]
        self._system_text = input_data.document
        self._user_text = (
            "Answer this question based on the document above:\n\n"
            f"{question_text}\n\n"
            "You may need to think about the question before answering."
            " But once done, provide a direct and concise answer."
        )
        # Build prompt immediately without additional diagnostics/delays
        return super().populate_prompt_message(input_data, logger)

    # Layer(...) calls idx(i+1) on repeated nodes. Capture it once to avoid parsing prefixes.
    def idx(self, value: int) -> int:  # type: ignore[override]
        self.index = max(0, value - 1)
        return value

    def build_system(self, input_data) -> str:
        return getattr(self, "_system_text", "")

    def build_user(self, input_data) -> str:
        return getattr(self, "_user_text", "")
