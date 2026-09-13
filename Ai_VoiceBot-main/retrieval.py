from typing import List, Dict, Any
import ollama
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBED_MODEL, CHAT_MODEL
from embeddings import embed_text
from handover import ConversationHistory

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# These are checked BEFORE LLM - LLM never sees these queries
IMMEDIATE_HANDOVER_PHRASES = [
    "speak to agent", "talk to agent", "talk to human",
    "speak to human", "human agent", "real person",
    "want an agent", "want a human", "need an agent",
    "connect me to", "transfer me", "escalate",
    "representative", "live agent", "live person",
    "cancel account", "refund", "complaint", "dispute",
    "charged wrong", "charged twice", "not helpful",
    "frustrated", "angry", "useless", "terrible",
    "i want to talk", "talk to a", "speak to a",
    "agent", "human", "person"
]


def check_immediate_handover(text: str) -> bool:
    """Check BEFORE LLM - pure keyword match, 100% reliable"""
    text_lower = text.lower()
    for phrase in IMMEDIATE_HANDOVER_PHRASES:
        if phrase in text_lower:
            print(f"[DEBUG] Handover triggered by phrase: '{phrase}'")
            return True
    return False


def search_faqs(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    q_emb = embed_text(query)
    res = index.query(vector=q_emb, top_k=top_k, include_metadata=True)
    return res.get("matches", [])


def build_context(matches: List[Dict[str, Any]]) -> str:
    parts = []
    for m in matches:
        md = m["metadata"]
        txt = md.get("text", "")
        src = md.get("source", "")
        parts.append(f"{txt}\n(Source: {src})")
    return "\n\n---\n\n".join(parts)


def answer_billing_question(user_query: str, history: ConversationHistory) -> dict:
    print(f"[DEBUG] Received query: '{user_query}'")

    # STEP 1: Check handover IMMEDIATELY - never reaches LLM
    if check_immediate_handover(user_query):
        print("[DEBUG] Handover check triggered!")
        return {
            "type": "handover_check",
            "message": None
        }

    print("[DEBUG] No handover trigger, searching FAQs...")

    # STEP 2: Search FAQs
    matches = search_faqs(user_query, top_k=5)

    # STEP 3: Low confidence → handover
    if not matches or matches[0]["score"] < 0.5:
        print("[DEBUG] Low confidence score, triggering handover")
        return {
            "type": "handover_check",
            "message": "I'm not confident I can answer that."
        }

    print(f"[DEBUG] Found {len(matches)} matches, sending to LLM...")

    # STEP 4: Only NOW send to LLM with strict prompt
    context = build_context(matches)
    system_prompt = (
        "You are a T-Mobile billing assistant. "
        "Answer in 2-3 sentences ONLY using the context below. "
        "Do NOT write code. Do NOT make lists. Give a direct short answer only.\n\n"
        f"Context:\n{context}"
    )

    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        options={"num_predict": 100}
    )

    return {
        "type": "bot",
        "message": resp["message"]["content"]
    }