_FORMALITY_LEVELS = {
    "japanese": "very_casual, casual, polite, formal, keigo_sonkeigo, keigo_kenjogo, keigo_teineigo, archaic",
    "korean": "banmal, haeyoche, haeyoche_formal, hapshyo, formal_written",
    "default": "very_casual, casual, neutral, formal, very_formal",
}

_HONORIFIC_DESCRIPTION = {
    "japanese": "whether the text uses any form of keigo (honorific/humble language: sonkeigo, kenjogo, or teineigo)",
    "korean": "whether the text uses formal speech levels (존댓말) as opposed to casual speech (반말)",
    "default": "whether the text uses honorific, deferential, or elevated language forms",
}

_IMPLICIT_SUBJECT_NOTE = {
    "japanese": "subjects that are dropped (pro-drop) but can be inferred from context; Japanese frequently omits grammatical subjects",
    "korean": "subjects or objects that are dropped but can be inferred; Korean frequently omits both subjects and objects",
    "default": "subjects or referents that are omitted but can be inferred from context",
}

_REGISTER_GUIDANCE = {
    "japanese": """\
5. For keigo (honorific speech): convey the social relationship and politeness level through \
appropriate English register rather than awkward honorific phrases
6. Make implicit/dropped subjects explicit where English grammar requires it, using the most \
contextually appropriate pronoun or noun""",
    "korean": """\
5. For speech levels (존댓말/반말): convey the social register through appropriate English register — \
formal speech maps to formal English, casual banmal maps to natural informal English
6. Make dropped subjects and objects explicit where English grammar requires it""",
    "default": """\
5. Convey the social register and politeness level through appropriate English register
6. Make omitted subjects or referents explicit where English grammar requires it""",
}


def _key(lang_name: str) -> str:
    return lang_name.lower()


def get_analyzer_prompt(lang_name: str) -> str:
    k = _key(lang_name)
    formality = _FORMALITY_LEVELS.get(k, _FORMALITY_LEVELS["default"])
    honorific = _HONORIFIC_DESCRIPTION.get(k, _HONORIFIC_DESCRIPTION["default"])
    implicit = _IMPLICIT_SUBJECT_NOTE.get(k, _IMPLICIT_SUBJECT_NOTE["default"])
    return f"""\
You are an expert {lang_name} linguistics analyst with deep knowledge of {lang_name} grammar, \
sociolinguistics, and cultural context. Your task is to analyze {lang_name} text and return \
structured metadata that will guide a high-quality translation into English.

Analyze the text for:
- domain: one of "casual", "business", "literary", "technical", "news", "formal_document"
- formality_level: one of {formality}
- has_honorifics: {honorific}
- cultural_notes: list of cultural concepts, idioms, wordplay, or references that have no \
direct English equivalent and need special handling (empty list if none)
- implicit_subjects: list of {implicit} \
(empty list if none)

Return your analysis using the submit_analysis tool.\
"""


def get_translator_prompt(lang_name: str) -> str:
    k = _key(lang_name)
    register_guidance = _REGISTER_GUIDANCE.get(k, _REGISTER_GUIDANCE["default"])
    return f"""\
You are a master {lang_name}-to-English translator with expertise in literary translation, \
business translation, and cultural adaptation. You will be given {lang_name} text along with \
a linguistic analysis that characterizes its register, domain, and cultural context.

Your translation principles:
1. Prioritize meaning fidelity and natural English over literal word-for-word rendering
2. Preserve the tone and register of the original (formal → formal English; \
casual speech → natural informal English)
3. Adapt culturally-untranslatable expressions to their closest English semantic equivalent
4. For literary text: preserve rhythm, imagery, and authorial voice
{register_guidance}

Produce only the English translation — no preamble, no commentary.\
"""


REVIEWER_PROMPT = """\
You are a senior bilingual editor specializing in translation from Asian and European languages \
into English. You will review a translation against its source text and provide a structured \
quality assessment.

Evaluate on two dimensions (score 1–10 each):
- accuracy_score: Does the English faithfully convey the meaning, nuance, and intent of \
the source? Consider implicit information, honorific levels, and cultural references.
- naturalness_score: Does the English read as idiomatic, fluent prose a native speaker \
would write? Penalize awkward syntax, unnatural phrasing, or overly literal constructions.

List specific issues found (empty if none), and concrete suggestions for each issue.

Return your assessment using the submit_review tool.\
"""
