import random
from typing import List, Dict

from verdict import Unit
from verdict.schema import Schema
from verdict.prompt import PromptMessage

from ..prompts.answers import ANSWER_PROMPT


class AnswerGeneration(Unit):
    """Generate one answer with document cached in system message."""

    def __init__(self, cot_instructions: List[str]):
        super().__init__()
        self.cot_instructions = cot_instructions

    class InputSchema(Schema):
        question: str
        document: str

    class ResponseSchema(Schema):
        answer: str

    def prompt(self, input_data: "AnswerGeneration.InputSchema") -> str:
        cot_instruction = random.choice(self.cot_instructions)
        return f"""@system
            {input_data.document}

            @user
            {cot_instruction}

            {ANSWER_PROMPT.format(question=input_data.question)}
        """

    def populate_prompt_message(
        self, input_data: "AnswerGeneration.InputSchema", logger
    ) -> PromptMessage:
        """Override to add cache control to document."""
        cot_instruction = random.choice(self.cot_instructions)

        # Create custom message with cache control
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
                        "content": f"""{cot_instruction}

                        {ANSWER_PROMPT.format(question=input_data.question)}
                        """,
                    },
                ]

        return CachedPromptMessage(
            system=input_data.document,
            user=f"{cot_instruction}\n\n{ANSWER_PROMPT.format(question=input_data.question)}",
            input_schema=input_data,
        )
