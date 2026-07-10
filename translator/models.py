from pydantic import BaseModel


class AnalysisResult(BaseModel):
    domain: str
    formality_level: str
    has_honorifics: bool
    cultural_notes: list[str]
    implicit_subjects: list[str]


class ReviewResult(BaseModel):
    accuracy_score: int
    naturalness_score: int
    issues: list[str]
    suggestions: list[str]


class DriftResult(BaseModel):
    has_drift: bool
    drift_notes: list[str]


class FinalOutput(BaseModel):
    source_text: str
    english_text: str
    translator_notes: list[str]
    analysis: AnalysisResult


def thinking_kwargs(model: str) -> dict:
    """Adaptive thinking where the model supports it, nothing where it doesn't.

    Haiku 4.5 rejects `thinking: {"type": "adaptive"}` with a 400 — sending it
    unconditionally used to kill live sessions when Haiku was selected.
    """
    return {} if "haiku" in model else {"thinking": {"type": "adaptive"}}
