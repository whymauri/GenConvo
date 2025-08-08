from verdict import Unit
from verdict.prompt import PromptMessage
from verdict.schema import Schema
from typing import List, Dict


class AnswerGeneration(Unit):
    """Generate one answer with document cached in system message."""

    class InputSchema(Schema):
        question: str
        document: str

    class ResponseSchema(Schema):
        answer: str

    def __init__(self):
        super().__init__()
        # Remove CoT entirely for direct answer generation
        self.prompt(
            """@system
{input.document}

@user
Answer this question based on the document above:

{input.question}

Provide a direct, concise answer.
"""
        )

    def populate_prompt_message(self, input_data, logger):
        """Override to add cache control to document."""

        # Create custom message with cache control
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
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": f"""Answer this question based on the document above:

{input_data.question}

Provide a direct, concise answer.""",
                    },
                ]

        return CachedPromptMessage(
            system=input_data.document,
            user=f"Answer this question based on the document above:\n\n{input_data.question}\n\nProvide a direct, concise answer.",
            input_schema=input_data,
        )
