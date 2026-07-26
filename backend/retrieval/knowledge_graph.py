import networkx as nx
import pickle
import os
import google.generativeai as genai
from backend.core.config import settings
from typing import List, Dict, Any, Tuple
import json
import re
import time
import structlog

logger = structlog.get_logger()

KG_PATH = "data/knowledge_graph.pkl"

class KnowledgeGraph:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._load()
    
    def extract_and_add(self, chunks: List[Dict[str, Any]]):
        """Extract entity-relationship triples from chunks via Gemini."""
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(settings.GEMINI_MODEL)
        
        for chunk in chunks:
            try:
                time.sleep(0.5)  # Rate limiting
                prompt = f"""Extract all entity-relationship-entity triples from this text.
Return ONLY a JSON array of triples like:
[["Entity A", "relationship", "Entity B"], ...]

Rules:
- Entities should be proper nouns, key concepts, or technical terms
- Relationships should be short verb phrases (e.g., "is part of", "causes", "uses")
- Extract at most 5 triples per chunk
- If no clear triples exist, return []

Text:
{chunk['text'][:1500]}"""
                
                response = model.generate_content(prompt)
                content = response.text.strip()
                
                # Parse JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                triples = json.loads(content)
                
                for subj, rel, obj in triples:
                    self.graph.add_node(subj, type="entity")
                    self.graph.add_node(obj, type="entity")
                    self.graph.add_edge(subj, obj, 
                                       relation=rel, 
                                       source=chunk.get("metadata", {}).get("source", "unknown"),
                                       chunk_text=chunk["text"][:500])
                    
            except Exception as e:
                logger.warning("Triple extraction failed", error=str(e))
                continue
        
        self._save()
        logger.info("Knowledge graph updated",
                    nodes=self.graph.number_of_nodes(),
                    edges=self.graph.number_of_edges())
    
    def query_graph(self, query_entities: List[str], max_hops: int = 2) -> List[Dict]:
        """Find related entities and their connecting edges."""
        results = []
        visited = set()
        
        for entity in query_entities:
            # Fuzzy match entity names in the graph
            matched = [n for n in self.graph.nodes() 
                       if entity.lower() in n.lower() or n.lower() in entity.lower()]
            
            for node in matched:
                self._traverse(node, max_hops, visited, results)
        
        return results[:20]  # Cap results
    
    def _traverse(self, node, hops_left, visited, results):
        if hops_left <= 0 or node in visited:
            return
        visited.add(node)
        
        for neighbor in self.graph.neighbors(node):
            edge_data = self.graph[node][neighbor]
            results.append({
                "subject": node,
                "relation": edge_data.get("relation", "related_to"),
                "object": neighbor,
                "source": edge_data.get("source", "unknown"),
                "context": edge_data.get("chunk_text", "")
            })
            self._traverse(neighbor, hops_left - 1, visited, results)
    
    def get_stats(self) -> Dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges()
        }
    
    def _save(self):
        os.makedirs(os.path.dirname(KG_PATH), exist_ok=True)
        with open(KG_PATH, 'wb') as f:
            pickle.dump(self.graph, f)
    
    def _load(self):
        if os.path.exists(KG_PATH):
            with open(KG_PATH, 'rb') as f:
                self.graph = pickle.load(f)
