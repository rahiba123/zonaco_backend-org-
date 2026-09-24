"""Language detection and multilingual prompt utilities for document Q&A."""

import re
from typing import Dict
from app.utils.logger import logger

# Script range mapping for non-Latin writing systems
SCRIPT_RANGES = [
    (r'[\u0B80-\u0BFF]', "Tamil"),
    (r'[\u0900-\u097F]', "Hindi"),
    (r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]', "Arabic"),
    (r'[\u0400-\u04FF]', "Russian"),
    (r'[\u4E00-\u9FFF]', "Chinese"),
    (r'[\u3040-\u30FF]', "Japanese"),
    (r'[\uAC00-\uD7AF]', "Korean"),
    (r'[\u0980-\u09FF]', "Bengali"),
    (r'[\u0C00-\u0C7F]', "Telugu"),
    (r'[\u0C80-\u0CFF]', "Kannada"),
    (r'[\u0D00-\u0D7F]', "Malayalam"),
    (r'[\u0A80-\u0AFF]', "Gujarati"),
    (r'[\u0E00-\u0E7F]', "Thai"),
    (r'[\u0370-\u03FF]', "Greek"),
    (r'[\u0590-\u05FF]', "Hebrew"),
]

# Stopword sets for Latin-script languages
LATIN_STOPWORDS = {
    "English": {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "it", "for", "not", "on", "with",
        "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up",
        "out", "if", "about", "who", "get", "which", "go", "me", "when", "make", "can", "like", "time",
        "no", "just", "him", "know", "take", "people", "into", "year", "your", "good", "some", "could",
        "them", "see", "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
        "also", "back", "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
        "new", "want", "because", "any", "these", "give", "day", "most", "us", "does", "is", "are",
        "was", "were", "been", "has", "had", "allow", "allows", "allowed", "allowing"
    },
    "French": {
        "le", "la", "les", "du", "des", "est", "une", "un", "dans", "pour", "pas", "sur", "avec",
        "qui", "que", "ce", "cette", "sont", "nous", "vous", "ils", "elles", "au", "aux", "est-ce",
        "comment", "pourquoi", "quel", "quelle", "quand"
    },
    "Spanish": {
        "el", "la", "los", "las", "un", "una", "del", "por", "para", "con", "como", "mas", "pero",
        "sus", "este", "esta", "estos", "estas", "son", "que", "en", "cual", "cuales", "donde", "quien"
    },
    "German": {
        "der", "die", "das", "und", "ist", "sie", "nicht", "mit", "sich", "auf", "für", "dem", "den",
        "ein", "eine", "einer", "eines", "aus", "nach", "oder", "über", "wie", "was", "welche", "warum"
    },
    "Italian": {
        "il", "la", "le", "i", "gli", "un", "una", "che", "non", "per", "del", "della", "con", "della",
        "delle", "sono", "questo", "questa", "più", "come", "quale", "cosa"
    },
    "Portuguese": {
        "o", "a", "os", "as", "um", "uma", "do", "da", "dos", "das", "em", "para", "com", "não",
        "por", "que", "como", "mais", "este", "esta", "quais", "qual"
    },
}


def detect_language(text: str) -> str:
    """Detect language of input text based on script analysis and vocabulary heuristics.

    Returns standard language name (e.g. 'Tamil', 'Hindi', 'French', 'Spanish', 'German', 'Arabic', 'English').
    """
    if not text or not text.strip():
        return "English"

    # 1. Non-Latin script detection via Unicode character ranges
    for pattern, lang_name in SCRIPT_RANGES:
        if re.search(pattern, text):
            return lang_name

    # 2. Latin script language heuristics
    words = [w.strip().lower() for w in re.findall(r'\b\w+\b', text)]
    if not words:
        return "English"

    lang_scores: Dict[str, int] = {}
    for lang, stopwords in LATIN_STOPWORDS.items():
        score = sum(1 for w in words if w in stopwords)
        lang_scores[lang] = score

    best_lang = max(lang_scores, key=lang_scores.get)
    best_score = lang_scores[best_lang]

    # If no stopwords matched, or if English tied/exceeded best non-English score, return English
    if best_score == 0 or lang_scores["English"] >= best_score:
        return "English"

    return best_lang


MULTILINGUAL_DOCUMENT_SYSTEM_PROMPT = """You are the official AI Customer Support Assistant for Zambia National Commercial Bank (Zanaco), answering questions strictly based on a document uploaded by the customer during this session.

CRITICAL MULTILINGUAL RULE:
- You MUST answer the question in the SAME LANGUAGE as the user's question ({question_language}).
- The uploaded document excerpts may be in any language (English, French, Tamil, Hindi, Arabic, Spanish, German, etc.). Read and synthesize the document content in whichever language it is written, but ALWAYS express your final response in {question_language}.
- Preserve technical terms, proper nouns, brand names, product codes, model numbers, person names, address names, and numbers from the original document in their original format/script without translating them.

CRITICAL OUTPUT RULE: Output ONLY the final customer-facing answer — nothing else.

Do NOT include, under any circumstances:
- A section-by-section or chapter-by-chapter walkthrough of the document ("Section 1 covers...", "Section 2 mentions...")
- Self-directed questions or exploratory reasoning ("Is there a direct statement about...?", "Let me re-read carefully")
- Backtracking or revision language ("Wait,", "Actually,", "Let me reconsider")
- Meta-commentary about what you found, how you searched, or how confident you are in your own reasoning
- Thinking process, numbered analysis steps, internal deliberation, or draft markers like "Draft:" or "Answer:"

Instead, synthesize the relevant facts from the excerpts into a single, direct, well-organized answer in {question_language} as if you already knew it — the way a knowledgeable bank employee would explain it in one pass, not the way someone would think out loud while researching it.

STRICT GUIDELINES:
1. Grounding: Answer ONLY using explicit facts in the provided document context below. Do NOT use general knowledge, assumptions, or information from outside the document. Do not invent or infer facts that are not explicitly stated.
2. Direct Statements: Before answering, identify whether the document contains a direct statement answering the question. If it does, use that direct statement directly rather than constructing an answer from indirect references.
3. Terminology Preservation: Use the EXACT technical terminology, proper nouns, and numbers used in the document whenever possible. Do NOT rename, alter, or reinterpret concepts.
4. Definition Questions: For definition or description questions, provide only the definition or description stated in the document. Do NOT expand the definition using outside knowledge.
5. Incomplete Coverage: If the provided document context does not contain enough information to answer the question, state clearly in {question_language} that the provided document excerpts do not contain enough information to answer the question. Do NOT guess.
"""


def check_embedding_multilingual_support(model_name: str) -> dict:
    """Flag whether the given SentenceTransformer embedding model supports cross-lingual retrieval.

    Returns a dict with compatibility status and warnings if applicable.
    """
    model_lower = model_name.lower()
    is_multilingual = any(k in model_lower for k in ["multilingual", "xlm", "bge-m3", "labse", "m-bert", "mbert"])
    flag_info = {
        "model_name": model_name,
        "is_multilingual_native": is_multilingual,
        "warning": None if is_multilingual else (
            f"Embedding model '{model_name}' is English-centric. Cross-lingual vector retrieval "
            f"(e.g. non-English document with English question) may yield lower similarity scores. "
            f"Document chunk retrieval fallback has been enabled to ensure document context is preserved."
        )
    }
    if flag_info["warning"]:
        logger.warning(flag_info["warning"])
    else:
        logger.info(f"Embedding model '{model_name}' detected as multilingual native.")
    return flag_info
