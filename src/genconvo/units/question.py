from verdict import Unit
from verdict.prompt import PromptMessage
from verdict.schema import Schema
from typing import List, Dict

from ..utils.schemas import DocumentInput


class QuestionGeneration(Unit):
    """Generate one question with document cached in system message."""

    class InputSchema(DocumentInput):
        pass

    class ResponseSchema(Schema):
        question: str

    class OutputSchema(Schema):
        question: str
        document: str

    def __init__(self, prompt_template: str):
        super().__init__()
        self.prompt_template = prompt_template
        # Use standard Verdict prompt pattern
        self.prompt(
            f"""@system
{{input.document}}

@user
{prompt_template}

Generate exactly one unique question based on the document above.
"""
        )

    def populate_prompt_message(self, input_data: DocumentInput, logger):
        """Override to add cache control to document."""
        # Create custom message with cache control
        prompt_template = self.prompt_template

        class CachedPromptMessage:
            def __init__(self, system, user, input_schema):
                self.system = system
                self.user = user
                self.input_schema = input_schema

            def to_messages(self, add_nonce: bool = False) -> List[Dict]:
                return [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": input_data.document,
                                # "cache_control": {"type": "ephemeral"},
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

    def process(
        self, input_data: DocumentInput, response: "QuestionGeneration.ResponseSchema"
    ) -> "QuestionGeneration.OutputSchema":
        """Include document in response for downstream units."""
        return self.OutputSchema(
            question=response.question, document=input_data.document
        )
