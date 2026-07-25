import json
import os
from pathlib import Path
from backend.ingestion.loaders import load_document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# We need GEMINI_API_KEY to be set
if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not set in environment")

# Note: RAGAS defaults to OpenAI, but we'll use Gemini
# Ensure GOOGLE_API_KEY is available for langchain if needed
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

def generate_synthetic_dataset(pdf_path: str, output_path: str, num_questions: int = 15):
    print(f"Loading document from {pdf_path}...")
    text = load_document(pdf_path, "application/pdf")
    
    # Simple chunking for generation
    # Split text into large enough blocks
    block_size = 2000
    blocks = [text[i:i+block_size] for i in range(0, len(text), block_size)]
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    prompt = PromptTemplate.from_template(
        """You are a helpful assistant that generates evaluation questions and expected answers based on a given context.
        Generate exactly 1 question and its corresponding expected answer from the context provided.
        The question must be clearly answered by the context, but it should not be trivial.
        
        Respond ONLY with a JSON object in this format:
        {{
            "question": "The question here",
            "expected_answer": "The expected answer here"
        }}
        
        Context:
        {context}
        """
    )
    
    chain = prompt | llm
    
    dataset = []
    
    print(f"Generating {min(num_questions, len(blocks))} QA pairs...")
    for i, block in enumerate(blocks[:num_questions]):
        if len(block.strip()) < 200:
            continue
            
        try:
            print(f"Processing block {i+1}...")
            response = chain.invoke({"context": block})
            # Clean up response if markdown formatting is present
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
                
            qa_pair = json.loads(content)
            
            dataset.append({
                "question": qa_pair["question"],
                "expected_answer": qa_pair["expected_answer"]
                # We don't store context here because RAGAS evaluates the retrieved context
            })
        except Exception as e:
            print(f"Failed to generate for block {i+1}: {e}")
            
    print(f"Generated {len(dataset)} items.")
    
    # Save dataset
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    pdf_file = "rag_test.pdf"
    out_file = "eval/eval_dataset.json"
    generate_synthetic_dataset(pdf_file, out_file)
