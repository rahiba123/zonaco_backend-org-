"""RAG service coordinating scoped vector retrieval, threshold evaluation, and OpenRouter LLM generation."""

from dataclasses import dataclass, field
import re
import time
from typing import List, Optional
import httpx
from app.config import get_settings
from app.services.document_store import DocumentSearchResult, DocumentVectorStoreService, get_document_vector_store
from app.services.vector_store import SearchResult, VectorStoreService, get_vector_store
from app.utils.exceptions import LLMServiceException
from app.utils.logger import logger, log_chat_interaction

# ---------------------------------------------------------------------------
# LLM filler-phrase & thinking-block stripping
#
# Free-tier and reasoning models often pollute answers with:
#   (a) Filler openers: "Based on the context provided, ..."
#   (b) Full thinking blocks: "Here's a thinking process:\n1. Analyze..."
#       or <think>...</think> XML tags.
#
# We strip both layers so users always receive a clean, direct reply.
# ---------------------------------------------------------------------------

# --- (a) Filler opener patterns ---
_FILLER_PATTERNS = [
    # Covers: "Based on the context", "Based on the provided document excerpts", etc.
    r"^Based on (?:the |this |these |our )?(?:provided |above |following |available |given |uploaded )?(?:context|FAQ(?: context)?|document(?: excerpts?)?|information|excerpt|excerpts|records)[,.]?\s*",
    r"^According to (?:the (?:FAQ|context|document|provided context|excerpt)|our (?:FAQ|records))[,.]?\s*",
    r"^From the (?:FAQ|context|document|provided (?:context|information))[,.]?\s*",
    r"^Using the (?:FAQ|context|provided (?:context|information|excerpt))[,.]?\s*",
    r"^As (?:per|stated in) the (?:FAQ|context|document|provided (?:context|information))[,.]?\s*",
    r"^The (?:FAQ|context|document|provided information) (?:states?|indicates?|mentions?|says?) that\s*",
    r"^(?:I can see that|Looking at the (?:context|document|FAQ),?)\s*",
]
_FILLER_RE = re.compile("|".join(_FILLER_PATTERNS), re.IGNORECASE)

# --- (b) Thinking-block detection patterns ---
# Matches the full thinking preamble that reasoning models output before the answer.
_THINKING_BLOCK_PATTERNS = [
    # <think>...</think> XML tags (some models use these)
    re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE),
    # "Here's a thinking process:" / "Here is my thinking:" preambles followed by numbered steps
    re.compile(
        r"^Here(?:'s| is)(?: a| my)? (?:thinking process|thought process|analysis|reasoning)[:\.].*?(?=\n\n|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
    # "Let me think/analyze/break down/walk through this:" openers
    re.compile(
        r"^Let me (?:think|analyze|break (?:this )?down|walk (?:through )?this|consider|work through this)[:\.,]?\s*\n?",
        re.IGNORECASE,
    ),
]

# Detects whether the ENTIRE response looks like a reasoning dump
# (numbered step headings like "1.  **Analyze User Input:**")
_THINKING_STEP_RE = re.compile(
    r"^\s*\d+\.\s+\*{0,2}(?:Analyze|Determine|Consider|Review|Evaluate|Think|Plan|Step|Understand)",
    re.IGNORECASE | re.MULTILINE,
)


# Regex that detects "Draft:", "Answer:", "Final Answer:", "Response:" markers
# — the model writes reasoning then uses one of these to signal the actual answer.
_DRAFT_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:Draft|Answer|Final Answer|Final Response|Response|My (?:Answer|Response))\s*:\s*",
    re.IGNORECASE,
)

# User-Uploaded Document Summary & Explicit Intent Detection
_SUMMARY_QUERY_PATTERNS = [
    r"\bsummary\b",
    r"\bsummarize\b",
    r"\bsummarise\b",
    r"\boverview\b",
    r"\bmain points?\b",
    r"\bkey points?\b",
    r"\bgist\b",
    r"\bsynopsis\b",
    r"\bbrief\b",
    r"\bwhat is (?:this|the|my|uploaded)?\s*(?:pdf|docx|txt|document|file)\s*about\b",
    r"\bwhat (?:does|is) (?:this|the|my|uploaded)?\s*(?:pdf|docx|txt|document|file)\s*(?:cover|contain|say|about)\b",
    r"\bexplain (?:this|the|my|uploaded)?\s*(?:pdf|docx|txt|document|file)\b",
    r"\babout (?:this|the|my|uploaded)?\s*(?:pdf|docx|txt|document|file)\b",
    r"\bexplain (?:the )?summary\b",
]
_SUMMARY_QUERY_RE = re.compile("|".join(_SUMMARY_QUERY_PATTERNS), re.IGNORECASE)

_EXPLICIT_DOC_PATTERNS = [
    r"\bthis document\b",
    r"\bthe document\b",
    r"\buploaded document\b",
    r"\bmy document\b",
    r"\bthis file\b",
    r"\bthe file\b",
    r"\buploaded file\b",
    r"\bmy file\b",
    r"\bthis pdf\b",
    r"\bthe pdf\b",
    r"\buploaded pdf\b",
    r"\bin this document\b",
    r"\bfrom this document\b",
    r"\baccording to (?:this|the) document\b",
]
_EXPLICIT_DOC_RE = re.compile("|".join(_EXPLICIT_DOC_PATTERNS), re.IGNORECASE)

# Paragraph-level patterns that identify reasoning / meta-commentary lines.
# If a paragraph matches ANY of these it is dropped from the final answer.
_REASONING_PARA_PATTERNS = [
    re.compile(r"^\d+\.\s+\*{0,2}(?:Scan|Analyze|Determine|Consider|Review|Evaluate|Plan|Step|Understand|Identify|Look|Check)", re.IGNORECASE),
    re.compile(r"^(?:Here(?:'s| is)|Let(?:'s| me| us))", re.IGNORECASE),
    re.compile(r"^I(?:'ll| will| need to| should| must| can| am going to)", re.IGNORECASE),
    re.compile(r"^(?:Key point|Key info|Key parts|Note:|Note that|Note -)", re.IGNORECASE),
    re.compile(r"^(?:All|The) (?:document|context|FAQ|excerpt|section)s? (?:consistently |always |clearly )?(?:say|state|indicate|mention|show|discuss|talk)", re.IGNORECASE),
    re.compile(r"^(?:Let'?s? (?:craft|structure|write|draft|build|form|create|compose|think about|consider|re-read) the (?:answer|response|reply|document|section))", re.IGNORECASE),
    re.compile(r"^(?:Draft|Planning|Outline|Summary of context|Key points across)", re.IGNORECASE),
    re.compile(r"^(?:- User query:|- Selected Category:|- I need to)", re.IGNORECASE),
    re.compile(r"^(?:Okay|Alright|Sure|Right),?\s+(?:so\s+)?(?:let|I|the)", re.IGNORECASE),
    re.compile(r"^(?:-?\s*(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|\d+(?:st|nd|rd|th)?)\s+(?:section|part|chapter|excerpt)|-?\s*Section \d+[:\s]|Section \d+ covers)", re.IGNORECASE),
    re.compile(r"^(?:Excerpt|Chapter) \d+[:\s]", re.IGNORECASE),
    re.compile(r"^(?:So from the excerpts|So from the document|The question is:|Based strictly on the document excerpts|The answer should reflect|Check against constraints:)", re.IGNORECASE),
    re.compile(r"^(?:Only final customer-facing answer:|No thinking process/reasoning:)", re.IGNORECASE),
    re.compile(r"^(?:Constraint check|Check against|Verification of constraints):?", re.IGNORECASE),
    re.compile(r"^(?:Wait,|Actually,|Let me re-read|Is there a direct statement|I don't see an explicit list|This suggests that|Reading the first section|Also: \"|The question: \"|The question\s*:|But I need to be)", re.IGNORECASE),
]


def _is_reasoning_paragraph(para: str) -> bool:
    """Return True if the paragraph looks like internal reasoning, not a customer answer."""
    for pat in _REASONING_PARA_PATTERNS:
        if pat.match(para):
            return True
    return False


def _extract_answer_from_thinking_block(text: str) -> str:
    """Pull the customer-facing answer out of a thinking dump.

    Strategy (in priority order):
    1. If model used a 'Draft:' / 'Answer:' marker, take everything after it.
    2. Otherwise filter paragraphs by removing all reasoning-looking ones.
    3. Fallback: return the last paragraph.
    """
    # Strategy 1 — explicit answer marker
    marker_match = _DRAFT_MARKER_RE.search(text)
    if marker_match:
        after_marker = text[marker_match.end():].strip()
        if after_marker:
            return after_marker

    # Strategy 2 — paragraph filtering
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return text

    answer_paragraphs = [
        p for p in paragraphs
        if not _is_reasoning_paragraph(p)
        and not _THINKING_STEP_RE.match(p)
        and len(p) > 30  # skip short transition fragments
    ]

    if answer_paragraphs:
        return "\n\n".join(answer_paragraphs)

    # Strategy 3 — fallback: last paragraph
    return paragraphs[-1]


def _strip_thinking_block(text: str) -> str:
    """Remove thinking/reasoning blocks that LLMs output before the real answer."""
    result = text.strip()

    # Step 1 — strip <think>...</think> XML tags
    for pattern in _THINKING_BLOCK_PATTERNS[:1]:
        result = pattern.sub("", result).strip()

    # Step 2 — check if response looks like a thinking dump and extract answer
    is_thinking_dump = (
        _THINKING_STEP_RE.search(result) is not None
        or _DRAFT_MARKER_RE.search(result) is not None
        or bool(re.match(r"^Here(?:'s| is)(?: a| my)? (?:thinking|thought|analysis|reasoning)", result, re.IGNORECASE))
        or bool(re.match(r"^All (?:document|context|FAQ)s? (?:consistently )?(?:say|state|indicate)", result, re.IGNORECASE))
        or bool(re.match(r"^Let'?s? (?:craft|structure|write|draft|think about)", result, re.IGNORECASE))
        or bool(re.search(r"\nDraft\s*:", result, re.IGNORECASE))
        or bool(re.search(r"Excerpt\s+\d+\s*:", result, re.IGNORECASE))
        or bool(re.search(r"(?:First|Second|Third|Fourth|Fifth|Sixth)\s+section\s*:", result, re.IGNORECASE))
        or bool(re.search(r"Section\s+\d+\s*:", result, re.IGNORECASE))
        or bool(re.search(r"Check against constraints:", result, re.IGNORECASE))
        or bool(re.search(r"Wait,\s+let me", result, re.IGNORECASE))
        or bool(re.search(r"Is there a direct statement", result, re.IGNORECASE))
        or bool(re.search(r"Key parts about", result, re.IGNORECASE))
        or bool(re.search(r"So from the document", result, re.IGNORECASE))
        or bool(re.search(r"But I need to be very careful", result, re.IGNORECASE))
    )

    if is_thinking_dump:
        result = _extract_answer_from_thinking_block(result)

    return result.strip()


def _strip_llm_filler(text: str) -> str:
    """Remove thinking blocks and filler openers from an LLM response.

    Applied after every LLM call so users always receive a direct, clean answer.
    Pass 1: Remove thinking/reasoning blocks (reasoning model leakage).
    Pass 2: Remove filler opener phrases (up to 3 iterations).
    Pass 3: Strip lingering Excerpt/Section citations, echo questions, or internal meta headers.
    """
    # Pass 1 — thinking block removal
    cleaned = _strip_thinking_block(text)

    # Pass 2 — filler opener removal (up to 3 stacked openers)
    for _ in range(3):
        new = _FILLER_RE.sub("", cleaned, count=1).strip()
        if new == cleaned:
            break
        if new:
            new = new[0].upper() + new[1:]
        cleaned = new

    # Pass 3 — Strip lingering Excerpt X / Section X citations, echo questions, or meta prefixes
    cleaned = re.sub(r"\b(?:According to|In|From|As stated in)?\s*(?:Excerpt|Section)\s+\d+[:\.,]?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^(?:What|How|Which|Can|Does|Is|Are)\b.*?\?\s*(?:Based on (?:the |this )?doc(?:ument)?:\s*)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^Based on (?:the |this )?(?:doc|document|excerpts?)[:\.,]?\s*", "", cleaned, flags=re.IGNORECASE).strip()

    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned




# Strict banking system prompt guardrail (used for FAQ-grounded answers)
BANKING_SYSTEM_PROMPT = """You are the official AI Customer Support Assistant for Zambia National Commercial Bank (Zanaco).
Your primary duty is to provide helpful, accurate, and professional answers to customer inquiries based ONLY on the provided FAQ context below.

CRITICAL OUTPUT RULE: Output ONLY the final customer-facing answer. Do NOT include any thinking process, reasoning steps, numbered analysis, internal deliberation, self-reflection, or meta-commentary. Do NOT write things like "Here's a thinking process", "Let me analyze", "Step 1:", "Analyze User Input:", or any similar internal reasoning. Start your response directly with the answer.

STRICT GUIDELINES:
1. Grounding: Answer the user's question ONLY using the factual information given in the FAQ Context.
2. No Hallucinations: NEVER invent bank policies, fee amounts, interest rates, contact details, or banking procedures not mentioned in the context.
3. Tone: Professional, courteous, clear, and reassuring.
4. Partial / Incomplete Coverage: If the FAQ context does not contain enough information to fully answer the user's query, state clearly that the FAQ does not have full details on that topic, and politely suggest connecting to a live agent.
5. Links / URLs: Do NOT invent or make up URLs. The system will deliver official links separately to the customer.
6. Brevity: Be concise and directly address the customer's question. Use bullet points where appropriate for step-by-step instructions.
"""

# Used when answering from a document the user uploaded in this session.
DOCUMENT_SYSTEM_PROMPT = """You are the official AI Customer Support Assistant for Zambia National Commercial Bank (Zanaco), answering questions strictly based on a document uploaded by the customer during this session.

CRITICAL OUTPUT RULE: Output ONLY the final customer-facing answer — nothing else.

Do NOT include, under any circumstances:
- A section-by-section or chapter-by-chapter walkthrough of the document ("Section 1 covers...", "Section 2 mentions...")
- Self-directed questions or exploratory reasoning ("Is there a direct statement about...?", "Let me re-read carefully")
- Backtracking or revision language ("Wait,", "Actually,", "Let me reconsider")
- Meta-commentary about what you found, how you searched, or how confident you are in your own reasoning
- Thinking process, numbered analysis steps, internal deliberation, or draft markers like "Draft:" or "Answer:"

Instead, synthesize the relevant facts from the excerpts into a single, direct, well-organized answer as if you already knew it — the way a knowledgeable bank employee would explain it in one pass, not the way someone would think out loud while researching it.

STRICT GUIDELINES:
1. Grounding: Answer ONLY using explicit facts in the provided document context below. Do NOT use general knowledge, assumptions, or information from outside the document. Do not invent or infer facts that are not explicitly stated.
2. Direct Statements: Before answering, identify whether the document contains a direct statement answering the question. If it does, use that direct statement directly rather than constructing an answer from indirect references.
3. Terminology Preservation: Use the EXACT terminology used in the document whenever possible. Do NOT rename, alter, or reinterpret concepts.
4. Definition Questions: For definition or description questions, provide only the definition or description stated in the document. Do NOT expand the definition using outside knowledge.
5. Incomplete Coverage: If the provided document context does not contain enough information to answer the question, say so plainly:
   "The provided document excerpts do not contain enough information to answer this question."
   Do NOT guess.
"""

# Used only when GENERAL_LLM_FALLBACK_ENABLED=True and neither the document nor the FAQ matched.
GENERAL_KNOWLEDGE_SYSTEM_PROMPT = """You are a helpful general-purpose assistant chatting on Zanaco's support channel.
The customer's question did not match Zanaco's official FAQ knowledge base or their uploaded document.

CRITICAL OUTPUT RULE: Output ONLY the final customer-facing answer. Do NOT include any thinking process, reasoning steps, numbered analysis, internal deliberation, or meta-commentary. Do NOT write things like "Here's a thinking process", "Let me analyze", "Step 1:", or any similar internal reasoning. Start your response directly with the answer.

STRICT GUIDELINES:
1. Answer using your own general knowledge, as helpfully and accurately as you can.
2. Clearly state up front that this answer is NOT official Zanaco information and should be verified with the bank directly.
3. NEVER state specific Zanaco fees, rates, account terms, or policies as fact — you don't have grounded data on those.
   If asked about Zanaco-specific details, say you don't have official confirmation and suggest contacting the bank.
4. Tone: Professional, courteous, and clear.
"""



@dataclass
class RAGResponse:
    """Standardized response from RAG generation pipeline."""
    answer: str
    matched_faq_intent: Optional[str]
    links: List[str]
    confidence_score: float
    should_offer_escalation: bool
    answer_source: str = "faq"  # "faq" | "user_document" | "general_llm"
    retrieved_sources: List[SearchResult] = field(default_factory=list)


class RAGService:
    """Coordinates retrieval from ChromaDB, confidence threshold gating, and OpenRouter generation."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreService] = None,
        document_vector_store: Optional[DocumentVectorStoreService] = None,
    ):
        self.settings = get_settings()
        self.vector_store = vector_store or get_vector_store()
        self.document_vector_store = document_vector_store or get_document_vector_store()

    async def answer_question(
        self,
        session_id: str,
        question: str,
        category: Optional[str] = None
    ) -> RAGResponse:
        """Execute full routing workflow:

        1. If this session has an uploaded document AND the question is a good match for it,
           answer strictly from that document.
        2. Otherwise, fall back to the existing FAQ knowledge-base pipeline.
        3. If FAQ also doesn't match and GENERAL_LLM_FALLBACK_ENABLED is on, answer from the
           LLM's general knowledge (clearly labeled unofficial). Otherwise offer live-agent escalation.
        """
        doc_was_checked = False

        # Step 0: Try the session's uploaded document first, if one exists.
        if self.document_vector_store.has_documents(session_id):
            doc_response = await self._try_answer_from_document(session_id, question)
            if doc_response is not None:
                return doc_response
            # Document was checked but didn't match — remember this for the fallback message.
            doc_was_checked = True

        return await self._answer_from_faq(session_id, question, category, doc_was_checked=doc_was_checked)


    async def _try_answer_from_document(self, session_id: str, question: str) -> Optional[RAGResponse]:
        """Attempt to answer from the session's uploaded document. Returns None if not a confident match."""
        start_time = time.perf_counter()

        is_summary_query = bool(_SUMMARY_QUERY_RE.search(question))
        is_explicit_doc_query = bool(_EXPLICIT_DOC_RE.search(question))

        doc_results: List[DocumentSearchResult] = self.document_vector_store.query(
            session_id=session_id,
            query_text=question,
            n_results=self.settings.USER_DOC_TOP_K
        )

        top_doc = doc_results[0] if doc_results else None
        top_score = top_doc.similarity_score if top_doc else 0.0

        # Handle summary/overview queries or explicit document references or standard high-similarity matches
        if is_summary_query or is_explicit_doc_query or (top_doc and top_score >= self.settings.USER_DOC_SIMILARITY_THRESHOLD):
            # For summary queries or explicit document queries with low vector scores, fetch sequential chunks
            if is_summary_query or (is_explicit_doc_query and top_score < self.settings.USER_DOC_SIMILARITY_THRESHOLD):
                all_chunks = self.document_vector_store.get_all_chunks(session_id=session_id, limit=30)
                if all_chunks:
                    doc_results = all_chunks
                    top_score = max(top_score, 0.90)

            if not doc_results:
                logger.info(f"Session {session_id}: no document chunks found for session.")
                return None

            context_blocks = []
            for idx, res in enumerate(doc_results, 1):
                context_blocks.append(f"--- Document Section ---\n{res.text}")
            context_str = "\n\n".join(context_blocks)

            if is_summary_query:
                user_prompt = (
                    f"Customer Question: {question}\n\n"
                    f"--- Document Excerpts ---\n"
                    f"{context_str}\n"
                    f"--- End of Document Excerpts ---\n\n"
                    f"Provide a clear, detailed, and direct summary of the uploaded document based strictly on the excerpts above."
                )
            else:
                user_prompt = (
                    f"Customer Question: {question}\n\n"
                    f"--- Document Excerpts ---\n"
                    f"{context_str}\n"
                    f"--- End of Document Excerpts ---\n\n"
                    f"Answer the customer's question using only the excerpts above."
                )

            generated_answer = await self._call_openrouter(user_prompt, system_prompt=DOCUMENT_SYSTEM_PROMPT)

            latency_ms = (time.perf_counter() - start_time) * 1000
            log_chat_interaction(
                session_id=session_id,
                question=question,
                category=None,
                confidence_score=top_score,
                matched_intent=None,
                should_offer_escalation=False,
                latency_ms=latency_ms
            )

            return RAGResponse(
                answer=generated_answer,
                matched_faq_intent=None,
                links=[],
                confidence_score=top_score,
                should_offer_escalation=False,
                answer_source="user_document",
                retrieved_sources=[]
            )

        logger.info(
            f"Session {session_id}: uploaded document top score {top_score:.3f} below "
            f"threshold {self.settings.USER_DOC_SIMILARITY_THRESHOLD}; falling back to FAQ."
        )
        return None

    async def _answer_from_faq(
        self,
        session_id: str,
        question: str,
        category: Optional[str] = None,
        doc_was_checked: bool = False
    ) -> RAGResponse:
        """Original FAQ-grounded RAG pipeline, with optional general-LLM fallback.

        Args:
            doc_was_checked: True when an uploaded document was queried but scored below
                             threshold, so the fallback message can tell the user both
                             sources were checked.
        """
        start_time = time.perf_counter()

        # Step 1: Scoped Vector Retrieval
        search_results = self.vector_store.query(
            query_text=question,
            category=category,
            n_results=self.settings.TOP_K
        )

        top_match: Optional[SearchResult] = search_results[0] if search_results else None
        top_score = top_match.similarity_score if top_match else 0.0
        top_intent = top_match.faq_intent if top_match else None

        # Gather deduplicated links from top matches
        collected_links: List[str] = []
        for res in search_results:
            for link in res.links:
                if link and link not in collected_links:
                    collected_links.append(link)

        # Step 2: Confidence Threshold Evaluation
        if top_score < self.settings.SIMILARITY_THRESHOLD or not top_match:
            latency_ms = (time.perf_counter() - start_time) * 1000

            if self.settings.GENERAL_LLM_FALLBACK_ENABLED:
                general_answer = await self._call_openrouter(
                    f"Customer Question: {question}",
                    system_prompt=GENERAL_KNOWLEDGE_SYSTEM_PROMPT
                )
                log_chat_interaction(
                    session_id=session_id,
                    question=question,
                    category=category,
                    confidence_score=top_score,
                    matched_intent=top_intent,
                    should_offer_escalation=False,
                    latency_ms=latency_ms
                )
                return RAGResponse(
                    answer=general_answer,
                    matched_faq_intent=top_intent,
                    links=collected_links,
                    confidence_score=top_score,
                    should_offer_escalation=False,
                    answer_source="general_llm",
                    retrieved_sources=search_results
                )

            # Build a context-aware fallback message
            if doc_was_checked:
                fallback_answer = (
                    "I checked both your uploaded document and our official FAQ knowledge base, "
                    "but I couldn't find a relevant answer to your question in either source. "
                    "Would you like to connect with a Zanaco customer support agent who can assist you further?"
                )
            else:
                fallback_answer = (
                    "I'm sorry, I couldn't find an exact answer to your inquiry in our official FAQ knowledge base. "
                    "Would you like to connect with a Zanaco customer support agent for further assistance?"
                )
            log_chat_interaction(
                session_id=session_id,
                question=question,
                category=category,
                confidence_score=top_score,
                matched_intent=top_intent,
                should_offer_escalation=True,
                latency_ms=latency_ms
            )
            return RAGResponse(
                answer=fallback_answer,
                matched_faq_intent=top_intent,
                links=collected_links,
                confidence_score=top_score,
                should_offer_escalation=True,
                answer_source="faq",
                retrieved_sources=search_results
            )



        # Step 3: Construct Context from Retrieved Documents
        context_blocks = []
        for idx, res in enumerate(search_results, 1):
            block = (
                f"[Document {idx} - Category: {res.category} | Intent: {res.faq_intent}]\n"
                f"Question: {res.question}\n"
                f"Response: {res.response}"
            )
            if res.details:
                block += f"\nDetails: {res.details}"
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)

        user_prompt = (
            f"Customer Inquiry: {question}\n\n"
            f"Selected Category: {category or 'General'}\n\n"
            f"--- FAQ Context ---\n"
            f"{context_str}\n"
            f"--- End of FAQ Context ---\n\n"
            f"Please provide a direct, helpful, and accurate response based strictly on the above context."
        )

        # Step 4: Call OpenRouter LLM
        generated_answer = await self._call_openrouter(user_prompt, system_prompt=BANKING_SYSTEM_PROMPT)

        latency_ms = (time.perf_counter() - start_time) * 1000
        log_chat_interaction(
            session_id=session_id,
            question=question,
            category=category,
            confidence_score=top_score,
            matched_intent=top_intent,
            should_offer_escalation=False,
            latency_ms=latency_ms
        )

        return RAGResponse(
            answer=generated_answer,
            matched_faq_intent=top_intent,
            links=collected_links,
            confidence_score=top_score,
            should_offer_escalation=False,
            answer_source="faq",
            retrieved_sources=search_results
        )

    async def _call_openrouter(self, user_content: str, system_prompt: str = BANKING_SYSTEM_PROMPT) -> str:
        """Execute async HTTP POST request to OpenRouter chat completions with candidate fallbacks."""
        api_key = self.settings.OPENROUTER_API_KEY
        if not api_key:
            logger.error("OPENROUTER_API_KEY is not configured.")
            raise LLMServiceException("OpenRouter API key is missing. Please configure OPENROUTER_API_KEY in .env.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://zanaco.co.zm",
            "X-Title": "Zanaco FAQ Chatbot Backend",
        }

        # Candidate models ordered by priority.
        # Primary model comes from settings (OPENROUTER_MODEL env var).
        # Fallbacks are tried in order if the primary fails.
        candidate_models = [self.settings.OPENROUTER_MODEL]
        fallback_models = [
            "nvidia/nemotron-3.5-lightning:free",
            "inclusionai/ling-3.0-flash-fin:free",
            "minimax/minimax-m2.7:free",
            "liquid/lfm-2.5-2.6b:free",
            "minimax/minimax-m3:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "z-ai/glm-5.2:free",
        ]
        for fb in fallback_models:
            if fb not in candidate_models:
                candidate_models.append(fb)

        url = f"{self.settings.OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
        last_error = None

        async with httpx.AsyncClient(timeout=30.0) as client:
            for model_id in candidate_models:
                payload = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1024,
                }
                logger.info(f"Calling OpenRouter model '{model_id}'...")

                try:
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        choices = data.get("choices", [])
                        if choices:
                            message = choices[0].get("message")
                            # Some models return message=None when content is null
                            # (e.g. finish_reason=tool_calls). Treat as empty.
                            content = (message or {}).get("content") or ""
                            answer_text = content.strip()
                            if answer_text:
                                return _strip_llm_filler(answer_text)
                            logger.warning(
                                f"OpenRouter model '{model_id}' returned empty content; trying next model."
                            )
                            last_error = f"Model '{model_id}' returned empty content"
                            continue

                    logger.warning(
                        f"OpenRouter model '{model_id}' failed with status {response.status_code}: {response.text}"
                    )
                    last_error = f"Status {response.status_code}: {response.text}"

                except httpx.TimeoutException:
                    logger.warning(f"OpenRouter request to '{model_id}' timed out.")
                    last_error = f"Model '{model_id}' timed out"
                except httpx.RequestError as exc:
                    logger.warning(f"Network error with model '{model_id}': {exc}")
                    last_error = str(exc)

            # Dynamic Discovery Fallback: Fetch currently active :free models from OpenRouter API if static list fails
            try:
                logger.info("Attempting dynamic OpenRouter model discovery...")
                models_url = f"{self.settings.OPENROUTER_BASE_URL.rstrip('/')}/models"
                models_res = await client.get(models_url, headers=headers)
                if models_res.status_code == 200:
                    data = models_res.json()
                    dynamic_models = [
                        m["id"] for m in data.get("data", [])
                        if isinstance(m, dict) and ":free" in m.get("id", "") and m.get("id") not in candidate_models
                    ]
                    for model_id in dynamic_models[:5]:
                        logger.info(f"Calling dynamically discovered OpenRouter model '{model_id}'...")
                        payload = {
                            "model": model_id,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_content}
                            ],
                            "temperature": 0.2,
                            "max_tokens": 1024,
                        }
                        try:
                            response = await client.post(url, headers=headers, json=payload)
                            if response.status_code == 200:
                                data = response.json()
                                choices = data.get("choices", [])
                                if choices:
                                    content = (choices[0].get("message") or {}).get("content") or ""
                                    if content.strip():
                                        return _strip_llm_filler(content.strip())
                        except Exception:
                            continue
            except Exception as exc:
                logger.warning(f"Dynamic OpenRouter model discovery failed: {exc}")

        logger.error(f"All candidate OpenRouter models failed. Last error: {last_error}")
        raise LLMServiceException(f"Failed to generate answer from LLM provider: {last_error}")


# Singleton RAG service instance
rag_service = RAGService()


def get_rag_service() -> RAGService:
    """Dependency injection provider for RAGService."""
    return rag_service