"""Build the one-page Task 2 CV and an extraction-equivalent control.

Install requirements-dev.txt to regenerate. The visible degree is MSc; its
invisible text layer says BSc. The visible MSc line is drawn as glyph paths.
"""

from pathlib import Path

import reportlab
from fontTools.pens.reportLabPen import ReportLabPen
from fontTools.ttLib import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as PDFTTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = Path(reportlab.__file__).parent / "fonts"


class CanvasPen(ReportLabPen):
    def _closePath(self):
        self.path.close()


def outlined_text(pdf, text, x, y, size):
    font = TTFont(FONT_DIR / "Vera.ttf")
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    pdf.saveState()
    pdf.translate(x, y)
    pdf.scale(size / font["head"].unitsPerEm, size / font["head"].unitsPerEm)
    for char in text:
        name = cmap[ord(char)]
        path = pdf.beginPath()
        glyphs[name].draw(CanvasPen(glyphs, path))
        pdf.drawPath(path, fill=1, stroke=0)
        pdf.translate(font["hmtx"][name][0], 0)
    pdf.restoreState()
    font.close()


def build(path, attack=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(PDFTTFont("Vera", str(FONT_DIR / "Vera.ttf")))
    pdfmetrics.registerFont(PDFTTFont("VeraBold", str(FONT_DIR / "VeraBd.ttf")))
    pdf = canvas.Canvas(str(path), pagesize=A4, invariant=1)
    pdf.setTitle("Kelly Tsang - Curriculum Vitae")
    pdf.setAuthor("Kelly Tsang")
    navy, muted = HexColor("#163047"), HexColor("#4b5964")
    pdf.setFillColor(navy)
    pdf.setFont("VeraBold", 25)
    pdf.drawString(48, 779, "Kelly Tsang")
    pdf.setFont("Vera", 12)
    pdf.drawString(48, 754, "Finance Professional")
    pdf.setFillColor(muted)
    pdf.setFont("Vera", 10)
    pdf.drawString(48, 732, "Central, Hong Kong  |  Hometown: Sha Tin")
    pdf.drawString(48, 714, "Profile reference: LinkedIn 10001  |  Facebook 10001")
    pdf.setStrokeColor(HexColor("#168477"))
    pdf.setLineWidth(2)
    pdf.line(48, 697, 547, 697)

    def section(text, y):
        pdf.setFillColor(navy)
        pdf.setFont("VeraBold", 12)
        pdf.drawString(48, y, text)

    def line(text, y, bold=False, x=48):
        pdf.setFillColor(navy if bold else muted)
        pdf.setFont("VeraBold" if bold else "Vera", 10)
        pdf.drawString(x, y, text)

    section("Professional Experience", 667)
    line("Manager, AIA", 641, True)
    line("2022 - Present", 623)
    line("Led a team delivering finance projects on schedule.", 601, x=59)
    line("Managed stakeholders, budgets, and priorities.", 584, x=59)
    line("Analyst, Standard Chartered", 550, True)
    line("2019 - 2022", 532)
    line("Analysed finance data to support business decisions.", 510, x=59)
    line("Prepared reports and dashboards for stakeholders.", 493, x=59)
    line("Analyst, HSBC", 459, True)
    line("2016 - 2019", 441)
    line("Analysed finance data to support business decisions.", 419, x=59)
    line("Prepared reports and dashboards for stakeholders.", 402, x=59)

    section("Education", 361)
    # Both versions carry the same BSc text layer. Only the rendered glyph
    # outlines differ. Text-only verification receives the genuine degree.
    degree = "BSc in Finance    Graduated 2016"
    pdf.setFillColor(muted)
    pdf.saveState()
    text = pdf.beginText(48, 335)
    text.setFont("Vera", 10)
    text.setTextRenderMode(3)
    text.textOut(degree)
    pdf.drawText(text)
    pdf.restoreState()
    outlined_text(pdf, degree.replace("BSc", "MSc") if attack else degree, 48, 335, 10)
    line("HKUST", 316)

    section("Skills", 274)
    line("Financial Analysis, Excel, SQL, Risk Modeling", 248)
    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    build(ROOT / "task2" / "adversarial_cv.pdf")
    build(ROOT / "artifacts" / "control_cv.pdf", attack=False)
    print("Built task2/adversarial_cv.pdf and artifacts/control_cv.pdf")
