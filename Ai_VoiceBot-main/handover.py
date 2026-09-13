from typing import List, Dict

HANDOVER_TRIGGERS = [
    "speak to agent", "speak to human", "talk to human",
    "talk to agent", "real person", "human agent",
    "representative", "escalate", "complaint", "refund",
    "dispute", "charged twice", "cancel account",
    "connect me", "want a human", "need human",
    "not helpful", "frustrated", "want to speak",
    "human please", "agent please"
]

FRUSTRATION_WORDS = [
    "angry", "frustrated", "annoyed", "terrible",
    "worst", "useless", "stupid", "ridiculous"
]

class ConversationHistory:
    def __init__(self):
        self.turns: List[Dict] = []
        self.handover_requested = False

    def add(self, role: str, message: str):
        self.turns.append({"role": role, "content": message})

    def needs_handover(self, text: str) -> bool:
        text_lower = text.lower()
        if any(t in text_lower for t in HANDOVER_TRIGGERS):
            return True
        if any(f in text_lower for f in FRUSTRATION_WORDS):
            return True
        return False

    def format_history(self) -> str:
        lines = ["\n===== CONVERSATION HISTORY ====="]
        for turn in self.turns:
            role = "User" if turn["role"] == "user" else "Bot"
            lines.append(f"{role}: {turn['content']}")
        lines.append("================================\n")
        return "\n".join(lines)

    def clear(self):
        self.turns = []
        self.handover_requested = False