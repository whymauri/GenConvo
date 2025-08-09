from typing import List, Dict, Any


class CachedPromptMessage:
    """Simple prompt message wrapper that adds Anthropic prompt caching to system text."""

    def __init__(self, system: str, user: str, input_schema: Any):
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
                        "text": self.system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": self.user,
            },
        ]

