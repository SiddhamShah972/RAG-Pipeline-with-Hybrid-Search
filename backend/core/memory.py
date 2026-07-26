import uuid
import time
from typing import List, Dict, Optional
from collections import OrderedDict

class ConversationMemory:
    """In-memory conversation store with LRU eviction. No external DB needed."""
    
    _instance = None
    MAX_SESSIONS = 100
    MAX_TURNS_PER_SESSION = 50
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.sessions: OrderedDict[str, List[Dict]] = OrderedDict()
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        self._evict_if_needed()
        return session_id
    
    def add_turn(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        # Trim old turns
        if len(self.sessions[session_id]) > self.MAX_TURNS_PER_SESSION:
            self.sessions[session_id] = self.sessions[session_id][-self.MAX_TURNS_PER_SESSION:]
    
    def get_history(self, session_id: str, last_n: int = 6) -> List[Dict]:
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id][-last_n:]
    
    def contextualize_query(self, session_id: str, query: str) -> str:
        """Rewrites the user query using chat history for standalone meaning."""
        history = self.get_history(session_id, last_n=4)
        if not history:
            return query
        
        # Build history string for the LLM to contextualize
        history_str = "\n".join([
            f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content'][:200]}"
            for h in history
        ])
        return query, history_str  # Pass both to the generator
    
    def _evict_if_needed(self):
        while len(self.sessions) > self.MAX_SESSIONS:
            self.sessions.popitem(last=False)  # Remove oldest
