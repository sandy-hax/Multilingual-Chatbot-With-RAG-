"""
ocr_service.py
==============
Lightweight OCR and document ingestion service.
Extracts clean text from PDFs, TXT/MD files, DOCX, and images (PNG, JPG, WEBP).
Uses PyPDF / pdfplumber for digital PDFs and EasyOCR / PIL for image OCR.
"""

import os
import io
from PIL import Image

# Global EasyOCR reader instance initialized lazily
_easyocr_reader = None

def get_easyocr_reader(languages=None):
    """Lazy initializer for EasyOCR reader to save startup memory."""
    global _easyocr_reader
    if languages is None:
        languages = ['en', 'ta', 'hi']
    if _easyocr_reader is None:
        try:
            import easyocr
            # Use GPU if available, else CPU
            import torch
            use_gpu = torch.cuda.is_available()
            _easyocr_reader = easyocr.Reader(languages, gpu=use_gpu)
            print(f"ℹ️ EasyOCR reader initialized (languages={languages}, gpu={use_gpu})")
        except Exception as e:
            print(f"⚠️ EasyOCR initialization warning: {e}. Will attempt fallback extraction.")
            _easyocr_reader = None
    return _easyocr_reader


class OCRService:
    """
    Service to process uploaded files (PDFs, images, text) and extract readable text.
    """

    @staticmethod
    def extract_text_from_file(file_path: str) -> str:
        """Extract text based on file extension."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.txt', '.md', '.csv', '.json', '.log']:
            return OCRService._extract_from_txt(file_path)
        elif ext == '.pdf':
            return OCRService._extract_from_pdf(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']:
            return OCRService._extract_from_image(file_path)
        elif ext == '.docx':
            return OCRService._extract_from_docx(file_path)
        else:
            # Fallback: try reading as plain text
            try:
                return OCRService._extract_from_txt(file_path)
            except Exception:
                raise ValueError(f"Unsupported file format: {ext}")

    @staticmethod
    def _extract_from_txt(file_path: str) -> str:
        """Read plain text files with UTF-8 encoding (handling errors)."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()

    @staticmethod
    def _extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF, running OCR on scanned/image pages."""
        extracted_pages = []

        # 1. Try PyPDF or pdfplumber
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    extracted_pages.append(f"--- Page {i+1} ---\n" + page_text.strip())
                else:
                    # Page seems scanned or image-only, render image if pdf2image available
                    ocr_text = OCRService._ocr_pdf_page(file_path, page_num=i)
                    if ocr_text:
                        extracted_pages.append(f"--- Page {i+1} (OCR) ---\n" + ocr_text)
        except Exception as e:
            print(f"⚠️ PyPDF extraction failed ({e}); attempting fallback OCR extraction.")

        full_text = "\n\n".join(extracted_pages).strip()
        if full_text:
            return full_text

        # 2. Fallback: pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        extracted_pages.append(f"--- Page {i+1} ---\n" + text.strip())
            full_text = "\n\n".join(extracted_pages).strip()
            if full_text:
                return full_text
        except Exception as e:
            print(f"⚠️ pdfplumber failed: {e}")

        return "No extractable text found in PDF."

    @staticmethod
    def _ocr_pdf_page(pdf_path: str, page_num: int) -> str:
        """Extract image from PDF page and perform OCR."""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
            if images:
                return OCRService._extract_from_pil_image(images[0])
        except Exception as e:
            print(f"⚠️ PDF image render failed for page {page_num}: {e}")
        return ""

    @staticmethod
    def _extract_from_image(file_path: str) -> str:
        """Extract text from an image file using EasyOCR."""
        try:
            image = Image.open(file_path)
            return OCRService._extract_from_pil_image(image)
        except Exception as e:
            print(f"⚠️ Image open failed: {e}")
            return ""

    @staticmethod
    def _extract_from_pil_image(image: Image.Image) -> str:
        """OCR PIL Image using EasyOCR or pytesseract fallback."""
        # Try EasyOCR
        reader = get_easyocr_reader()
        if reader is not None:
            try:
                # Convert image to bytes or numpy array
                import numpy as np
                img_np = np.array(image.convert("RGB"))
                results = reader.readtext(img_np, detail=0)
                extracted = " ".join(results).strip()
                if extracted:
                    return extracted
            except Exception as e:
                print(f"⚠️ EasyOCR execution failed: {e}")

        # Fallback to pytesseract if installed
        try:
            import pytesseract
            return pytesseract.image_to_string(image).strip()
        except Exception:
            pass

        return ""

    @staticmethod
    def _extract_from_docx(file_path: str) -> str:
        """Extract text from Word .docx file."""
        try:
            import docx
            doc = docx.Document(file_path)
            full_text = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n".join(full_text)
        except Exception as e:
            print(f"⚠️ docx extraction failed: {e}")
            return ""


ocr_service = OCRService()
