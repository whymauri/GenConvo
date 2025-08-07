import pickle

from dataclasses import dataclass
from verdict.schema import Schema

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