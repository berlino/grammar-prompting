from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass(frozen=True)
class LLMResponse:
    """A single response from a LargeLanguageModel."""
    prompt_text: str
    response_text: str
    prompt_info: Dict
    other_info: Dict