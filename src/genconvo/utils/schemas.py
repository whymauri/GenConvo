import pickle
from pathlib import Path
from typing import Optional

from dataclasses import dataclass
from verdict.schema import Schema


class DocumentInput(Schema):
    document: str
