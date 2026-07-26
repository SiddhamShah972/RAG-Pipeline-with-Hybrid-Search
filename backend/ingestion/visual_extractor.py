import fitz  # PyMuPDF
import google.generativeai as genai
from backend.core.config import settings
from typing import List, Dict, Any
import base64
import os
import time
import structlog

logger = structlog.get_logger()

# Rate limiting for Gemini free tier
_last_call_time = 0
MIN_CALL_INTERVAL = 1.0  # 1 second between calls

def _rate_limit():
    """Respect Gemini free tier: ~15 RPM for vision."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - elapsed)
    _last_call_time = time.time()

def extract_images_from_pdf(file_path: str, min_size_kb: int = 5) -> List[Dict[str, Any]]:
    """
    Extracts images from a PDF, filters out tiny icons,
    and returns image bytes + page number.
    """
    doc = fitz.open(file_path)
    images = []
    
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Skip tiny images (icons, bullets, etc.)
            if len(image_bytes) < min_size_kb * 1024:
                continue
            
            images.append({
                "image_bytes": image_bytes,
                "page_number": page_num + 1,
                "ext": base_image["ext"],
                "index": img_index
            })
    
    doc.close()
    return images

def describe_image_with_gemini(image_bytes: bytes, ext: str, page_num: int) -> str:
    """
    Sends an image to Gemini Vision and gets a text description.
    Uses the free-tier multimodal endpoint.
    """
    _rate_limit()
    
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(settings.GEMINI_MODEL)
    
    # Gemini accepts raw image bytes
    import PIL.Image
    import io
    image = PIL.Image.open(io.BytesIO(image_bytes))
    
    prompt = """Describe this image/diagram/chart in detail for a search index.
Include:
- What type of visual this is (chart, diagram, photo, table, flowchart, etc.)
- All text/labels visible in the image
- Key data points, relationships, or information conveyed
- The overall meaning or purpose of this visual

Be thorough and factual. This description will be used for search retrieval."""
    
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        logger.warning("Gemini vision failed", page=page_num, error=str(e))
        return ""

def extract_visual_chunks(file_path: str, filename: str) -> List[Dict[str, Any]]:
    """
    Full pipeline: extract images → describe via Gemini → return as chunks.
    Returns list of {"text": description, "metadata": {...}}
    """
    images = extract_images_from_pdf(file_path)
    visual_chunks = []
    
    for img in images:
        description = describe_image_with_gemini(
            img["image_bytes"], img["ext"], img["page_number"]
        )
        
        if not description.strip():
            continue
        
        # Save image to data/images/ for potential UI display later
        img_dir = "data/images"
        os.makedirs(img_dir, exist_ok=True)
        img_filename = f"{filename}_p{img['page_number']}_i{img['index']}.{img['ext']}"
        img_path = os.path.join(img_dir, img_filename)
        with open(img_path, "wb") as f:
            f.write(img["image_bytes"])
        
        visual_chunks.append({
            "text": f"[FIGURE from {filename}, Page {img['page_number']}]\n{description}",
            "metadata": {
                "source": filename,
                "page_number": img["page_number"],
                "chunk_type": "visual",
                "image_path": img_path
            }
        })
    
    logger.info("Visual extraction complete",
                filename=filename,
                images_found=len(images),
                chunks_created=len(visual_chunks))
    
    return visual_chunks
