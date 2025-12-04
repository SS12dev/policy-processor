import io
import base64
import logging
from typing import List, Optional
from PIL import Image

logger = logging.getLogger(__name__)

def convert_pdf_to_images(
    pdf_bytes: bytes,
    poppler_path: Optional[str] = None,
    dpi: int = 200,
    fmt: str = 'JPEG',
    high_compression: bool = False
) -> List[str]:
    """
    Convert PDF pages to base64-encoded images.

    Args:
        pdf_bytes: Raw PDF file bytes
        poppler_path: Path to Poppler binaries (Windows)
        dpi: Image resolution
        fmt: Image format (JPEG or PNG)
        high_compression: Use aggressive compression for large documents

    Returns:
        List of base64-encoded images
    """
    try:
        from pdf2image import convert_from_bytes

        kwargs = {'dpi': dpi, 'fmt': fmt}
        if poppler_path:
            kwargs['poppler_path'] = poppler_path

        images = convert_from_bytes(pdf_bytes, **kwargs)
        base64_images = []

        for i, img in enumerate(images):
            buffered = io.BytesIO()
            # Use aggressive compression for scanned documents to reduce 413 errors
            if high_compression:
                # Reduce image size and quality for large payloads
                img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
                quality = 60
            else:
                quality = 85
                
            img.save(buffered, format=fmt, quality=quality)
            img_bytes = buffered.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            base64_images.append(img_b64)
            
            if high_compression:
                logger.info(f"Image {i+1}: Compressed to {len(img_b64)} chars (quality={quality})")

        logger.info(f"Converted {len(images)} pages to {fmt} images{' (high compression)' if high_compression else ''}")
        return base64_images

    except ImportError:
        logger.error("pdf2image not installed. Install with: pip install pdf2image")
        raise
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        raise

def batch_pages(
    total_pages: int,
    batch_size: int = 3
) -> List[List[int]]:
    """
    Create page number batches for processing.

    Args:
        total_pages: Total number of pages
        batch_size: Pages per batch

    Returns:
        List of page number lists
    """
    batches = []
    for i in range(0, total_pages, batch_size):
        batch = list(range(i, min(i + batch_size, total_pages)))
        batches.append(batch)

    logger.info(f"Created {len(batches)} batches for {total_pages} pages (batch_size={batch_size})")
    return batches