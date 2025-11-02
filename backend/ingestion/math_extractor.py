"""Extract math images from PDFs and run pix2tex (if available) to get LaTeX."""
import os
import tempfile
import subprocess
from typing import Dict, List
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


def extract_images_from_pdf(path: str) -> Dict[int, List[str]]:
    """Return mapping page_number -> list of image temp file paths extracted from the PDF."""
    pages = {}
    if fitz is None:
        return pages
    doc = fitz.open(path)
    for i in range(len(doc)):
        page = doc[i]
        img_list = page.get_images(full=True)
        paths = []
        for img_index, img in enumerate(img_list):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n >= 5:  # CMYK
                pix = fitz.Pixmap(fitz.csRGB, pix)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            pix.save(tmp.name)
            tmp.close()
            paths.append(tmp.name)
            pix = None
        if paths:
            pages[i+1] = paths
    return pages


def run_pix2tex_on_image(image_path: str) -> str:
    """Run pix2tex CLI on image and return LaTeX output if available."""
    # Try CLI `pix2tex` if installed
    try:
        # pix2tex CLI prints latex to stdout by default
        out = subprocess.check_output(["pix2tex", image_path], stderr=subprocess.DEVNULL, timeout=30)
        return out.decode('utf-8').strip()
    except Exception:
        return ""


def extract_math_from_pdf(path: str) -> Dict[int, List[str]]:
    """Return mapping page_number -> list of LaTeX strings extracted from image-based math."""
    res = {}
    pages = extract_images_from_pdf(path)
    for pnum, imgs in pages.items():
        latexes = []
        for img in imgs:
            latex = run_pix2tex_on_image(img)
            if latex:
                latexes.append(latex)
        if latexes:
            res[pnum] = latexes
    return res


