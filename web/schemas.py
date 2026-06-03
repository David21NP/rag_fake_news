from typing import Literal, Protocol, TypedDict

from pydantic import BaseModel, Field


class Metrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1: float


class ModelResponse(BaseModel):
    """
    {
        "label": "fake" | "real",
        "confidence": 0.0–1.0,
        "reasoning": "Brief explanation referencing the retrieved articles."
    }
    """

    label: Literal["fake", "real"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str


class GeneratedResponse(BaseModel):
    thinking: str | None
    answer: ModelResponse


class Embedder(Protocol):
    def __call__(
        self,
        *,
        text: str,
        label: int,
        title: str | None = None,
    ) -> None: ...


class Generator(Protocol):
    def __call__(
        *,
        text: str,
        title: str | None = None,
        top_k: int | None = None,
        think: bool = True,
    ) -> GeneratedResponse: ...
