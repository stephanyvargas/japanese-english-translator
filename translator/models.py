from pydantic import BaseModel


class AnalysisResult(BaseModel):
    domain: str
    formality_level: str
    has_keigo: bool
    cultural_notes: list[str]
    implicit_subjects: list[str]


class ReviewResult(BaseModel):
    accuracy_score: int
    naturalness_score: int
    issues: list[str]
    suggestions: list[str]


class FinalOutput(BaseModel):
    japanese_source: str
    english_text: str
    translator_notes: list[str]
    analysis: AnalysisResult
