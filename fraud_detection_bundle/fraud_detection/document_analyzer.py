"""Document analysis — pages -> images -> Claude vision + OCR.

Supports raster docs (PNG/JPG of statements, IDs, contracts) directly and
PDFs via pypdfium2 (preferred) or pdf2image. OCR is optional through
pytesseract; without it, the analyzer relies on Claude's native OCR.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fraud_detection.image_forensics import ImageForensics, ImageForensicsResult
from fraud_detection.utils import clamp01, risk_label


@dataclass
class DocumentFinding:
    path: str
    page_count: int
    page_paths: list[str]
    page_forensics: list[ImageForensicsResult]
    ocr_text: str | None
    text_anomalies: list[str]
    overall_score: float
    risk: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["page_forensics"] = [r.to_dict() for r in self.page_forensics]
        return d


class DocumentAnalyzer:
    def __init__(self, image_forensics: ImageForensics | None = None) -> None:
        self.image_forensics = image_forensics or ImageForensics()

    # ---------------------------------------------------- pdf -> pages
    def render_pages(self, path: str | Path, out_dir: str | Path) -> list[Path]:
        p = Path(path)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() != ".pdf":
            return [p]

        try:
            import pypdfium2 as pdfium  # type: ignore

            pdf = pdfium.PdfDocument(str(p))
            pages = []
            for i, page in enumerate(pdf):
                bitmap = page.render(scale=2.0)
                pil = bitmap.to_pil()
                dst = out / f"{p.stem}_page_{i + 1}.png"
                pil.save(dst, format="PNG")
                pages.append(dst)
            return pages
        except ImportError:
            pass

        try:
            from pdf2image import convert_from_path  # type: ignore

            images = convert_from_path(str(p), dpi=200)
            pages = []
            for i, im in enumerate(images):
                dst = out / f"{p.stem}_page_{i + 1}.png"
                im.save(dst, format="PNG")
                pages.append(dst)
            return pages
        except ImportError as e:
            raise RuntimeError(
                "PDF rendering requires `pypdfium2` or `pdf2image`. "
                "Install with: pip install pypdfium2"
            ) from e

    # ------------------------------------------------------------ OCR
    def ocr(self, path: str | Path) -> str | None:
        try:
            import pytesseract  # type: ignore
            from PIL import Image
        except ImportError:
            return None
        try:
            return pytesseract.image_to_string(Image.open(path))
        except Exception:  # noqa: BLE001
            return None

    # ----------------------------------------------- text-level checks
    def text_anomalies(self, text: str) -> list[str]:
        flags: list[str] = []
        if not text:
            return flags

        # Currency / amount inconsistencies (common in forged statements):
        amounts = [
            float(m.group(1).replace(",", ""))
            for m in re.finditer(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)", text)
            if m.group(1)
        ]
        if amounts and max(amounts) > 10 * (sum(amounts) / len(amounts) + 1e-6):
            flags.append("amount_outlier_present")

        # Suspicious font-tell glyphs that often appear after PDF stitching:
        if re.search(r"[�]", text):
            flags.append("replacement_glyph_present")
        if re.search(r"\bI\s*\d{3}\b", text):  # mistaken "I" for "1"
            flags.append("digit_letter_confusion")

        # Date format inconsistencies in the same doc:
        date_formats = set()
        for pat, label in (
            (r"\b\d{4}-\d{2}-\d{2}\b", "iso"),
            (r"\b\d{2}/\d{2}/\d{4}\b", "slash"),
            (r"\b\d{2}-\d{2}-\d{4}\b", "dash"),
            (r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b", "verbose"),
        ):
            if re.search(pat, text):
                date_formats.add(label)
        if len(date_formats) >= 2:
            flags.append(f"mixed_date_formats:{','.join(sorted(date_formats))}")

        # Serial / reference number gaps:
        refs = re.findall(r"\b(?:Ref|Reference|Txn|Transaction)[\s#:]*([A-Z0-9-]{4,})\b", text)
        if refs and len(set(refs)) < len(refs) / 2:
            flags.append("duplicate_reference_numbers")

        return flags

    # ------------------------------------------------------ orchestrate
    def analyze(
        self, path: str | Path, work_dir: str | Path = "/tmp/fraud_pages"
    ) -> DocumentFinding:
        pages = self.render_pages(path, work_dir)
        page_results = [self.image_forensics.analyze(pp) for pp in pages]

        text_chunks: list[str] = []
        for pp in pages:
            t = self.ocr(pp)
            if t:
                text_chunks.append(t)
        full_text = "\n\n".join(text_chunks) if text_chunks else None
        text_flags = self.text_anomalies(full_text or "")

        forensic_max = max((r.overall_score for r in page_results), default=0.0)
        overall = clamp01(0.7 * forensic_max + 0.3 * min(1.0, len(text_flags) / 3.0))

        notes: list[str] = []
        if forensic_max > 0.4:
            notes.append("At least one page shows elevated image-forensic risk.")
        for f in text_flags:
            notes.append(f"Text anomaly: {f}")

        return DocumentFinding(
            path=str(path),
            page_count=len(pages),
            page_paths=[str(p) for p in pages],
            page_forensics=page_results,
            ocr_text=full_text,
            text_anomalies=text_flags,
            overall_score=round(overall, 4),
            risk=risk_label(overall),
            notes=notes,
        )
