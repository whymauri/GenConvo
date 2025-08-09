from abc import ABC, abstractmethod
from typing import Any

from verdict import Unit
from verdict.schema import Schema

from ..utils.cached_prompt import CachedPromptMessage


class BaseCachedUnit(Unit, ABC):
    """Unit base that centralizes prompt construction and Anthropic prompt caching.

    Subclasses implement build_system/build_user; self.prompt is a minimal stub.
    """

    def __init__(self) -> None:
        super().__init__()
        # Minimal stub to satisfy Verdict's requirement that a prompt exists.
        # Real prompt is built in populate_prompt_message.
        self.prompt("stub")

    # Satisfy UnitRegistry requirements with minimal schemas
    class ResponseSchema(Schema):
        pass

    class OutputSchema(ResponseSchema):
        pass

    @abstractmethod
    def build_system(self, input_data: Any) -> str:
        """Return system message text (usually the document)."""

    @abstractmethod
    def build_user(self, input_data: Any) -> str:
        """Return user message text for this unit instance."""

    def populate_prompt_message(self, input_data, logger):  # type: ignore[override]
        system_text = self.build_system(input_data)
        user_text = self.build_user(input_data)
        return CachedPromptMessage(system=system_text, user=user_text, input_schema=input_data)

