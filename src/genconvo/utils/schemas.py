import json
from dataclasses import dataclass
from typing import Any, Mapping, Dict

from verdict.schema import Schema


class DocumentInput(Schema):
    document: str


@dataclass(frozen=True)
class ParseContext:
    """Immutable parsing context for converting pipeline results to Q&A pairs."""
    filename: str
    dataset_directory: str
    model: str
    temperature: float
    prompt_type: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ParseContext":
        """Create a ParseContext from a mapping, raising if required keys are missing."""
        return cls(
            filename=str(mapping["filename"]),
            dataset_directory=str(mapping["dataset_directory"]),
            model=str(mapping["model"]),
            temperature=float(mapping["temperature"]),
            prompt_type=str(mapping["prompt_type"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "dataset_directory": self.dataset_directory,
            "model": self.model,
            "temperature": self.temperature,
            "prompt_type": self.prompt_type,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
