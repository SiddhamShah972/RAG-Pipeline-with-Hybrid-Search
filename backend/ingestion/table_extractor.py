import pdfplumber
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger()

def extract_tables_from_pdf(file_path: str, filename: str) -> List[Dict[str, Any]]:
    """
    Extracts tables from a PDF and converts them to markdown format.
    Each table becomes a separate retrievable chunk.
    """
    table_chunks = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                
                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue  # Skip empty or header-only tables
                    
                    # Convert to markdown table
                    md_table = _table_to_markdown(table)
                    
                    if len(md_table.strip()) < 20:
                        continue  # Skip tiny tables
                    
                    table_chunks.append({
                        "text": f"[TABLE from {filename}, Page {page_num + 1}]\n{md_table}",
                        "metadata": {
                            "source": filename,
                            "page_number": page_num + 1,
                            "chunk_type": "table",
                            "table_index": table_idx
                        }
                    })
    except Exception as e:
        logger.warning("Table extraction failed", filename=filename, error=str(e))
    
    logger.info("Table extraction complete",
                filename=filename,
                tables_found=len(table_chunks))
    
    return table_chunks

def _table_to_markdown(table: list) -> str:
    """Converts a list-of-lists table to a markdown table string."""
    if not table:
        return ""
    
    # Clean None values
    cleaned = []
    for row in table:
        cleaned.append([str(cell) if cell else "" for cell in row])
    
    if not cleaned or not cleaned[0]:
        return ""
        
    # Build markdown
    header = "| " + " | ".join(cleaned[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(cleaned[0])) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in cleaned[1:]
    ]
    
    return "\n".join([header, separator] + rows)
