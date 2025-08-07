from typing import List, Dict

from verdict import Unit
from verdict.schema import Schema
from verdict.prompt import PromptMessage

from ..utils.schemas import DocumentInput


class QuestionGeneration(Unit):
    """Generate one question with document cached in system message."""

    class InputSchema(DocumentInput):
        pass

    class ResponseSchema(Schema):
        question: str

    def __init__(self, prompt_template: str):
        super().__init__()
        self.prompt_template = prompt_template

    def prompt(self, input_data: DocumentInput) -> str:
        return f"""@system
            {input_data.document}

            @user
            {self.prompt_template}

            Generate exactly one unique question based on the document above.
        """

    def populate_prompt_message(
        self, input_data: DocumentInput, logger
    ) -> PromptMessage:
        """Override to add cache control to document."""
        # Create custom message with cache control
        prompt_template = self.prompt_template

        class CachedPromptMessage(PromptMessage):
            def to_messages(self, add_nonce: bool = False) -> List[Dict]:
                return [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": input_data.document,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": f"""{prompt_template}

                        Generate exactly one unique question based on the document above.
                        """,
                    },
                ]

        return CachedPromptMessage(
            system=input_data.document,
            user=f"{self.prompt_template}\n\nGenerate exactly one unique question based on the document above.",
            input_schema=input_data,
        )
