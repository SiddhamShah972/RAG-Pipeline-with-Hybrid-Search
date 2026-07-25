import json
import sys
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.chat_models.vertexai.ChatVertexAI'] = MagicMock()

import os
import time
from pathlib import Path
from datasets import Dataset
import pandas as pd

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import requests

from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

API_URL = "http://localhost:8080"

def evaluate_pipeline(dataset_path: str, top_k: int = 5):
    with open(dataset_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)
        
    print(f"Loaded {len(qa_pairs)} questions from {dataset_path}")
    
    questions = []
    expected_answers = []
    actual_answers = []
    retrieved_contexts = []
    
    for item in qa_pairs:
        query = item["question"]
        expected_ans = item["expected_answer"]
        
        # Call the local query API
        payload = {"query": query, "top_k": top_k}
        try:
            resp = requests.post(f"{API_URL}/query", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            actual_ans = data["answer"]
            contexts = [c["text"] for c in data.get("chunks", [])]
            
            questions.append(query)
            expected_answers.append(expected_ans)
            actual_answers.append(actual_ans)
            retrieved_contexts.append(contexts)
            
        except Exception as e:
            print(f"Failed to query '{query}': {e}")
            continue
            
    if not questions:
        print("No valid queries completed.")
        return None
        
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": actual_answers,
        "contexts": retrieved_contexts,
        "ground_truth": expected_answers
    })
    
    # Initialize Ollama models for Ragas
    from langchain_ollama import ChatOllama
    from langchain_ollama import OllamaEmbeddings
    llm = ChatOllama(model="phi3:mini", base_url="http://ollama:11434")
    embeddings = OllamaEmbeddings(model="phi3:mini", base_url="http://ollama:11434")
    
    # Wrap models for RAGAS 0.2 compatibility
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(embeddings)
    
    print("Running RAGAS evaluation...")
    
    # Explicitly set the LLM and Embeddings for each metric
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_emb
    context_precision.llm = ragas_llm
    context_recall.llm = ragas_llm
    
    from ragas.run_config import RunConfig
    run_config = RunConfig(max_workers=1)
    
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_config
    )
    
    df = result.to_pandas()
    metrics_dict = df.to_dict(orient="records")
    
    # Calculate mean for each metric (skip non-numeric columns like question, context, answer)
    metric_cols = [c for c in df.columns if c not in ["question", "answer", "contexts", "ground_truth"]]
    summary_dict = {col: df[col].mean() for col in metric_cols}
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "top_k": top_k,
        "metrics_summary": summary_dict,
        "details": metrics_dict
    }
    
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Evaluation complete. Results saved to {out_file}")
    for k, v in summary["metrics_summary"].items():
        print(f"{k}: {v:.4f}")
        
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="eval/eval_dataset.json")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    
    evaluate_pipeline(args.dataset, args.top_k)
