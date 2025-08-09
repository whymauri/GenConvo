from typing import List

from verdict.schema import Schema

from ..utils.schemas import DocumentInput
from .base import BaseCachedUnit


class QuestionsUnit(BaseCachedUnit):
    """Generate N questions in one call with document cached in system message."""

    class InputSchema(DocumentInput):
        pass

    class ResponseSchema(Schema):
        questions: List[str]

    class OutputSchema(Schema):
        document: str
        questions: List[str]

    def __init__(self, prompt_template: str, num_questions: int):
        super().__init__()
        self.prompt_template = prompt_template
        self.num_questions = num_questions

    # Use BaseCachedUnit's populate_prompt_message which calls these hooks
    def build_system(self, input_data: DocumentInput) -> str:
        # Use original pipeline source document bytes if available
        source = getattr(self, "source_input", None)
        source_document = getattr(source, "document", None)
        return source_document if isinstance(source_document, str) else input_data.document

    def build_user(self, input_data: DocumentInput) -> str:
        return (
            f"{self.prompt_template}\n\n"
            f"Generate exactly {self.num_questions} unique and diverse questions based on the document above."
        )

    def populate_prompt_message(self, input_data: DocumentInput, logger):
        return super().populate_prompt_message(input_data, logger)

    def process(
        self, input_data: DocumentInput, response: "QuestionsUnit.ResponseSchema"
    ) -> "QuestionsUnit.OutputSchema":
        return self.OutputSchema(document=input_data.document, questions=response.questions)