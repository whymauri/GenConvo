import random
from typing import List, Dict

from verdict import Unit
from verdict.schema import Schema
from verdict.prompt import PromptMessage

from ..prompts.answers import ANSWER_PROMPT


class QuestionToAnswerTransform(Unit):
    """Transform questions into answer generation inputs with random CoT."""

    class InputSchema(Schema):
        questions: List[str]
        document: str

    class ResponseSchema(Schema):
        answer_inputs: List[Dict[str, str]]

    def __init__(self, cot_instructions: List[str]):
        super().__init__()
        self.cot_instructions = cot_instructions

    def prompt(self, input_data: "QuestionToAnswerTransform.InputSchema") -> str:
        # This is a pure data transformation - no LLM call needed
        return ""

    def execute(
        self,
        input_data: "QuestionToAnswerTransform.InputSchema",
        execution_context=None,
    ):
        """Transform questions into answer generation inputs."""
        answer_inputs = []
        for question in input_data.questions:
            cot_instruction = random.choice(self.cot_instructions)
            answer_inputs.append(
                {
                    "question": question,
                    "document": input_data.document,
                    "cot_instruction": cot_instruction,
                }
            )
        return self.ResponseSchema(answer_inputs=answer_inputs)

class AnswerGeneration(Unit):
    """Generate one answer with document cached in system message."""

    class InputSchema(Schema):
        question: str
        document: str
        cot_instruction: str

    class ResponseSchema(Schema):
        answer: str

    def prompt(self, input_data: "AnswerGeneration.InputSchema") -> str:
        return f"""@system
{input_data.document}

@user
{input_data.cot_instruction}

{ANSWER_PROMPT.format(question=input_data.question)}
"""

    def populate_prompt_message(
        self, input_data: "AnswerGeneration.InputSchema", logger
    ) -> PromptMessage:
        """Override to add cache control to document."""

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
                        "content": f"""{input_data.cot_instruction}

{ANSWER_PROMPT.format(question=input_data.question)}""",
                    },
                ]

        return CachedPromptMessage(
            system=input_data.document,
            user=f"{input_data.cot_instruction}\n\n{ANSWER_PROMPT.format(question=input_data.question)}",
            input_schema=input_data,
        )