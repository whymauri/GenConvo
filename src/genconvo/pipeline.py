"""
Verdict pipeline for the GenConvoBench dataset. https://arxiv.org/pdf/2506.06266

Simple pipeline that processes one question prompt type and generates N questions + N answers.
Each question/answer is generated with document cached in system message.
"""

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from verdict import Pipeline, Unit, Layer
from verdict.schema import Schema
from verdict.prompt import PromptMessage

from .prompts.questions import GEN_CONVO_PROMPT_REGISTRY
from .prompts.answers import ANSWER_PROMPT
from .prompts.cot import COT_INSTRUCTIONS


@dataclass
class Context:
    name: str
    path: str
    document: str

    def load_document(self) -> str:
        """Load document from pickle file."""
        if self.document is None:
            with open(self.path, "rb") as f:
                self.document = pickle.load(f)
        return self.document


class DocumentInput(Schema):
    document: str


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

Generate exactly one unique question based on the document above.""",
                    },
                ]

        return CachedPromptMessage(
            system=input_data.document,
            user=f"{self.prompt_template}\n\nGenerate exactly one unique question based on the document above.",
            input_schema=input_data,
        )


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


class VerdictPipeline:
    """Pipeline: Document -> N Questions -> N Answers with caching."""

    def __init__(
        self,
        context: Context,
        prompt_type: str,
        num_questions: int = 16,
        model_name: str = "claude-3-5-sonnet-20241022",
        max_workers: int = 8,
        temperature: float = 0.7,
    ):
        self.context = context
        self.prompt_type = prompt_type
        self.num_questions = num_questions
        self.model_name = model_name
        self.max_workers = max_workers
        self.temperature = temperature
        self.prompt_template = GEN_CONVO_PROMPT_REGISTRY[prompt_type]
        self.cot_instructions = COT_INSTRUCTIONS

    def create_pipeline(self) -> Pipeline:
        """Create complete pipeline: questions -> transform -> answers."""

        # Layer 1: Generate N questions in parallel
        question_layer = Layer(
            QuestionGeneration(self.prompt_template), self.num_questions
        )

        # Layer 2: Transform questions to answer inputs
        transform_unit = QuestionToAnswerTransform(self.cot_instructions)

        # Layer 3: Generate N answers in parallel
        answer_unit = AnswerGeneration()

        pipeline = (
            Pipeline(name=f"GenConvoBench-{self.prompt_type}")
            >> question_layer
            >> transform_unit
            >> answer_unit
        )

        # Configure model with inference parameters
        pipeline = pipeline.via(
            self.model_name,
            retries=3,
            temperature=self.temperature,
            max_tokens=2000,
            timeout=120,
        )

        return pipeline

    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        document = self.context.load_document()
        input_data = DocumentInput(document=document)
        pipeline = self.create_pipeline()

        # Run pipeline - should return structured question-answer pairs
        results = pipeline.run(
            input_data=input_data,  # type: ignore
            max_workers=self.max_workers,  # type: ignore
            display=True,  # type: ignore
            graceful=True,  # type: ignore
        )

        return {
            "context": {
                "name": self.context.name,
                "path": self.context.path,
                "model": self.model_name,
                "temperature": self.temperature,
                "prompt_type": self.prompt_type,
            },
            "results": results,
            "total_questions": self.num_questions,
        }

    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    def __call__(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run pipeline and optionally save results."""
        results = self.run()

        if output_path:
            self.save_results(results, output_path)

        return results
