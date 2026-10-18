"""Check the rendered/text-layer attack with the exact homework PDF loader."""

import hashlib
import json
import sys
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hw2 import cv_text


def main():
    attack = ROOT / "task2" / "adversarial_cv.pdf"
    control = ROOT / "artifacts" / "control_cv.pdf"
    attack_text, control_text = cv_text(attack), cv_text(control)
    assert attack_text == control_text, "Loader must receive identical text"
    assert "BSc in Finance" in attack_text
    assert "MSc" not in attack_text
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    (output / "adversarial_extracted.txt").write_text(attack_text, encoding="utf-8")
    with fitz.open(attack) as pdf, fitz.open(control) as baseline:
        assert len(pdf) == 1
        rendered = pdf[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        rendered.save(str(output / "adversarial_preview.png"))
        assert rendered.samples != baseline[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).samples
        # Verify that only the degree's rectangle changes visually.
        outside = [fitz.Rect(0, 0, 596, 490), fitz.Rect(0, 520, 596, 842)]
        for clip in outside:
            assert pdf[0].get_pixmap(clip=clip).samples == baseline[0].get_pixmap(clip=clip).samples
    result = {
        "loader": "hw2.cv_text / MarkItDown(enable_plugins=False)",
        "pages": 1, "extracted_text_identical_to_control": True,
        "extracted_degree": "BSc in Finance", "rendered_degree": "MSc in Finance",
        "pdf_sha256": hashlib.sha256(attack.read_bytes()).hexdigest(),
        "text_sha256": hashlib.sha256(attack_text.encode()).hexdigest(),
    }
    (output / "attack_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
