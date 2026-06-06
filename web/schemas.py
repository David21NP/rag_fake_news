from typing import Literal, TypedDict

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
    time_elapsed: str
    time_elapsed_llm: str
