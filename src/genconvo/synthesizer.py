"""
Verdict pipeline for the GenConvoBench dataset. https://arxiv.org/pdf/2506.06266

Simple pipeline that processes one question prompt type and generates N questions + N answers.
Each question/answer is generated with document cached in system message.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from verdict import Pipeline, Layer
from verdict.model import ProviderModel

from .prompts.questions import GEN_CONVO_PROMPT_REGISTRY
from .units.question import QuestionGeneration
from .units.answer import AnswerGeneration
from .utils.schemas import DocumentInput


class GenConvoSynthesizer:
    """Pipeline: Document -> N Questions -> N Answers with caching."""

    def __init__(
        self,
        dataset_directory: str,
        filename: str,
        prompt_type: str,
        num_questions: int = 16,
        model_name: str = "claude-3-7",
        max_workers: int = 8,
        temperature: float = 0.7,
    ):
        self.dataset_directory = Path(dataset_directory)
        self.filename = filename
        self.prompt_type = prompt_type
        self.num_questions = num_questions
        self.model_name = model_name
        self.max_workers = max_workers
        self.temperature = temperature
        self.prompt_template = GEN_CONVO_PROMPT_REGISTRY[prompt_type]

        self._document: Optional[str] = None

    def _load_document(self) -> str:
        """Load document from markdown file."""
        if self._document is None:
            file_path = self.dataset_directory / self.filename
            with open(file_path, "r", encoding="utf-8") as f:
                self._document = f.read()
        return self._document[-10000:]

    def create_pipeline(self) -> Pipeline:
        """Create complete pipeline: questions -> transform -> answers."""

        question = QuestionGeneration(self.prompt_template)
        answer = AnswerGeneration()

        self_study = Layer(
            question >> answer,
            inner="none",
            outer="dense",
            repeat=self.num_questions,
        )

        pipeline = Pipeline(name=f"GenConvoBench-{self.prompt_type}") >> self_study
        model = ProviderModel(self.model_name, use_nonce=False)
        return pipeline.via(
            model,  # type: ignore
            temperature=self.temperature,
        )

    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline."""
        document = self._load_document()
        input_data = DocumentInput(document=document)
        pipeline = self.create_pipeline()

        # Run pipeline - should return structured question-answer pairs
        results = pipeline.run(
            input_data=input_data,  # type: ignore
            max_workers=self.max_workers,  # type: ignore
            display=True,  # type: ignore
        )

        return {
            "context": {
                "filename": self.filename,
                "dataset_directory": str(self.dataset_directory),
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
