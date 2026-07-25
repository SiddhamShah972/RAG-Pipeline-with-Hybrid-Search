import os
from bs4 import BeautifulSoup
from docx import Document


def load_document(file_path: str, mime_type: str) -> str:
    """
    Loads text from a file based on its mime_type/extension.
    Returns the extracted text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.split('.')[-1].lower()

    if ext == 'txt' or mime_type == 'text/plain':
        return _load_txt(file_path)
    elif ext == 'pdf' or mime_type == 'application/pdf':
        return _load_pdf(file_path)
    elif ext in ['htm', 'html'] or mime_type == 'text/html':
        return _load_html(file_path)
    elif ext == 'docx' or mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return _load_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def _load_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def _load_pdf(file_path: str) -> str:
    import fitz # PyMuPDF
    doc = fitz.open(file_path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)

def _load_html(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n')

def _load_docx(file_path: str) -> str:
    doc = Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])
