"""RAG service coordinating scoped vector retrieval, threshold evaluation, and OpenRouter LLM generation."""

from dataclasses import dataclass, field
import time
from typing import List, Optional
import httpx
from app.config import get_settings
from app.services.vector_store import SearchResult, VectorStoreService, get_vector_store
from app.utils.exceptions import LLMServiceException
from app.utils.logger import logger, log_chat_interaction

# Strict banking system prompt guardrail
BANKING_SYSTEM_PROMPT = """You are the official AI Customer Support Assistant for Zambia National Commercial Bank (Zanaco).
Your primary duty is to provide helpful, accurate, and professional answers to customer inquiries based ONLY on the provided FAQ context below.

STRICT GUIDELINES:
1. Grounding: Answer the user's question ONLY using the factual information given in the FAQ Context.
2. No Hallucinations: NEVER invent bank policies, fee amounts, interest rates, contact details, or banking procedures not mentioned in the context.
3. Tone: Professional, courteous, clear, and reassuring.
4. Partial / Incomplete Coverage: If the FAQ context does not contain enough information to fully answer the user's query, state clearly that the FAQ does not have full details on that topic, and politely suggest connecting to a live agent.
5. Links / URLs: Do NOT invent or make up URLs. The system will deliver official links separately to the customer.
6. Brevity: Be concise and directly address the customer's question. Use bullet points where appropriate for step-by-step instructions.
"""


@dataclass
class RAGResponse:
    """Standardized response from RAG generation pipeline."""
    answer: str
    matched_faq_intent: Optional[str]
    links: List[str]
    confidence_score: float
    should_offer_escalation: bool
    retrieved_sources: List[SearchResult] = field(default_factory=list)


class RAGService:
    """Coordinates retrieval from ChromaDB, confidence threshold gating, and OpenRouter generation."""

    def __init__(self, vector_store: Optional[VectorStoreService] = None):
        self.settings = get_settings()
        self.vector_store = vector_store or get_vector_store()

    async def answer_question(
        self,
        session_id: str,
        question: str,
        category: Optional[str] = None
    ) -> RAGResponse:
        """Execute full RAG workflow: retrieval -> confidence check -> LLM call."""
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
        generated_answer = await self._call_openrouter(user_prompt)

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
            retrieved_sources=search_results
        )

    async def _call_openrouter(self, user_content: str) -> str:
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

        # Candidate models ordered by priority
        candidate_models = [self.settings.OPENROUTER_MODEL]
        fallback_models = [
            "nvidia/nemotron-nano-9b-v2:free",
            "google/gemma-4-26b-a4b-it:free",
            "nvidia/nemotron-3.5-lightning:free",
            "openai/gpt-oss-20b:free",
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
                        {"role": "system", "content": BANKING_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 600,
                }
                logger.info(f"Calling OpenRouter model '{model_id}'...")

                try:
                    response = await client.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        choices = data.get("choices", [])
                        if choices and "message" in choices[0]:
                            answer_text = choices[0]["message"].get("content", "").strip()
                            return answer_text

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

        logger.error(f"All candidate OpenRouter models failed. Last error: {last_error}")
        raise LLMServiceException(f"Failed to generate answer from LLM provider: {last_error}")


# Singleton RAG service instance
rag_service = RAGService()


def get_rag_service() -> RAGService:
    """Dependency injection provider for RAGService."""
    return rag_service
