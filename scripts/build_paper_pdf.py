"""Render PAPER.md to PAPER.pdf for SSRN / arXiv submission.

    python scripts/build_paper_pdf.py [SOURCE.md [TARGET.pdf]]

The PDF is a build artifact of the manuscript, generated rather than hand-edited,
so it can never drift from PAPER.md - the same discipline the test suite applies
to the paper's figures.

Before rendering, every non-ASCII character in the manuscript is checked against
the embedded font. A missing glyph renders as a solid box, silently; this script
fails loudly instead.

Display equations (```latex blocks) are typeset with matplotlib's mathtext and
figures (``![caption](path)``, path relative to the repository root) are placed
at text width. Times New Roman / Consolas are used on Windows; elsewhere the
Times-like FreeSerif and Liberation Mono are used instead.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "PAPER.md"
TARGET = ROOT / "PAPER.pdf"
FONT_DIR = Path("C:/Windows/Fonts")
FALLBACK_DIR = Path("/usr/share/fonts/truetype")
EQUATION_DIR = ROOT / "reports" / "_equations"

SERIF, MONO = "PaperSerif", "PaperMono"


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def register_fonts() -> TTFont:
    faces = {
        SERIF: "times.ttf",
        f"{SERIF}-Bold": "timesbd.ttf",
        f"{SERIF}-Italic": "timesi.ttf",
        f"{SERIF}-BoldItalic": "timesbi.ttf",
        MONO: "consola.ttf",
        f"{MONO}-Bold": "consolab.ttf",
    }
    # FreeSerif rather than Liberation Serif: both are Times-like, but only
    # FreeSerif carries every glyph the manuscript uses (e.g. U+207B in 10⁻¹⁴)
    fallback = {
        "times.ttf": "freefont/FreeSerif.ttf",
        "timesbd.ttf": "freefont/FreeSerifBold.ttf",
        "timesi.ttf": "freefont/FreeSerifItalic.ttf",
        "timesbi.ttf": "freefont/FreeSerifBoldItalic.ttf",
        "consola.ttf": "liberation/LiberationMono-Regular.ttf",
        "consolab.ttf": "liberation/LiberationMono-Bold.ttf",
    }
    for name, file in faces.items():
        path = FONT_DIR / file
        if not path.exists():
            path = FALLBACK_DIR / fallback[file]
        if not path.exists():
            sys.exit(f"missing font file: {FONT_DIR / file} (and no fallback {path})")
        pdfmetrics.registerFont(TTFont(name, str(path)))

    pdfmetrics.registerFontFamily(
        SERIF, normal=SERIF, bold=f"{SERIF}-Bold",
        italic=f"{SERIF}-Italic", boldItalic=f"{SERIF}-BoldItalic",
    )
    pdfmetrics.registerFontFamily(MONO, normal=MONO, bold=f"{MONO}-Bold",
                                  italic=MONO, boldItalic=f"{MONO}-Bold")
    return pdfmetrics.getFont(SERIF)


def audit_glyphs(text: str, font: TTFont) -> None:
    """Fail if any character in the manuscript has no glyph in the body font."""
    cmap = font.face.charToGlyph
    missing = sorted({c for c in text if ord(c) > 127 and ord(c) not in cmap})
    if missing:
        listing = ", ".join(f"{c!r} U+{ord(c):04X}" for c in missing)
        sys.exit(f"glyphs missing from body font (would render as boxes): {listing}")


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

def styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle(
        "body", fontName=SERIF, fontSize=10.5, leading=14.2,
        alignment=TA_JUSTIFY, spaceAfter=6.5,
    )
    return {
        "title": ParagraphStyle("title", parent=body, fontName=f"{SERIF}-Bold",
                                fontSize=17, leading=21, alignment=TA_CENTER,
                                spaceAfter=8),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontSize=11.5,
                                   leading=15, alignment=TA_CENTER, spaceAfter=4),
        "front": ParagraphStyle("front", parent=body, fontSize=10, leading=13,
                                alignment=TA_CENTER, spaceAfter=3),
        "h2": ParagraphStyle("h2", parent=body, fontName=f"{SERIF}-Bold",
                             fontSize=13, leading=16.5, alignment=TA_LEFT,
                             spaceBefore=12, spaceAfter=5),
        "h3": ParagraphStyle("h3", parent=body, fontName=f"{SERIF}-BoldItalic",
                             fontSize=11, leading=14, alignment=TA_LEFT,
                             spaceBefore=8, spaceAfter=3),
        "body": body,
        "quote": ParagraphStyle("quote", parent=body, leftIndent=24,
                                rightIndent=24, alignment=TA_CENTER,
                                spaceBefore=4, spaceAfter=8),
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=16,
                                 bulletIndent=4, spaceAfter=3),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8, leading=9.8,
                               alignment=TA_LEFT, spaceAfter=0),
        "cellhead": ParagraphStyle("cellhead", parent=body, fontName=f"{SERIF}-Bold",
                                   fontSize=8, leading=9.8, alignment=TA_LEFT,
                                   spaceAfter=0),
        "code": ParagraphStyle("code", fontName=MONO, fontSize=8.5, leading=10.8,
                               leftIndent=10, spaceBefore=3, spaceAfter=7,
                               backColor=colors.HexColor("#F3F4F6")),
    }


# ---------------------------------------------------------------------------
# Inline markdown -> reportlab markup
# ---------------------------------------------------------------------------

PIPE = "\u0000PIPE\u0000"


# Letters only, so neither escape() nor the emphasis rules can touch it.
STAR = "LITERALASTERISKTOKEN"


def inline(text: str) -> str:
    """Convert inline markdown to reportlab paragraph markup.

    Code spans are extracted first so their contents are never read as emphasis,
    then everything is XML-escaped before any tags are introduced.
    """
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(m.group(1))
        return f"\u0001{len(spans) - 1}\u0001"

    text = re.sub(r"`([^`]+)`", stash, text)
    # "Δ*" is bootstrap-replicate notation, not emphasis - the only literal
    # asterisk in the manuscript. Unprotected, it closes a surrounding italic
    # span early and leaks a stray "*".
    text = text.replace("Δ*", "Δ" + STAR)
    text = escape(text.replace("\\|", PIPE))

    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<b><i>\1</i></b>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Emphasis boundaries are ASCII-only. Subscripts such as "₁" are Unicode
    # word characters, so a \w boundary never closed "*S*₁" and every italic
    # symbol followed by a subscript rendered as literal asterisks.
    text = re.sub(
        r"(?<![A-Za-z0-9*])\*(?!\s)(.+?)(?<!\s)\*(?![A-Za-z0-9*])",
        r"<i>\1</i>", text,
    )

    def unstash(m: re.Match) -> str:
        code = escape(spans[int(m.group(1))])
        return f'<font name="{MONO}" size="8.8">{code}</font>'

    text = re.sub(r"\u0001(\d+)\u0001", unstash, text)
    return text.replace(PIPE, "|").replace(STAR, "*")


def split_row(line: str) -> list[str]:
    line = line.strip().replace("\\|", PIPE)
    cells = [c.strip().replace(PIPE, "\\|") for c in line.strip("|").split("|")]
    return cells


def is_separator(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells if c.strip())


# ---------------------------------------------------------------------------
# Block parser
# ---------------------------------------------------------------------------

def _plain(cell: str) -> str:
    return re.sub(r"[*`]", "", cell).replace("\\|", "|")


def column_widths(rows: list[list[str]], ncols: int, width: float) -> list[float]:
    """Size columns from rendered text width rather than character counts.

    Character counts let the widest text column starve short numeric ones, so
    "Sharpe" wrapped to "Sharp/e" and "0.999" to "0.99/9". Numeric cells
    keep their natural width; only columns wider than the table can afford
    are shrunk, and never below their longest single word.
    """
    pad, size = 7.0, 8
    natural, floor = [], []
    for i in range(ncols):
        col = [_plain(r[i]) if i < len(r) else "" for r in rows]
        head_words = col[0].split() or [""]
        head = max(stringWidth(w, f"{SERIF}-Bold", size) for w in head_words)
        cells = max((stringWidth(c, SERIF, size) for c in col[1:]), default=0)
        words = max((stringWidth(w, SERIF, size) for c in col[1:] for w in c.split()),
                    default=0)
        natural.append(max(head, cells) + pad)
        floor.append(max(head, words) + pad)

    total = sum(natural)
    if total <= width:
        # spare room goes to the text-heavy columns, proportionally
        extra = width - total
        return [n + extra * n / total for n in natural]

    # shrink only what exceeds its floor, proportionally to the excess
    slack = [n - f for n, f in zip(natural, floor)]
    need = total - width
    if sum(slack) <= 0:
        return [width * n / total for n in natural]
    cut = min(1.0, need / sum(slack))
    out = [n - s * cut for n, s in zip(natural, slack)]
    if sum(out) > width:          # floors alone exceed the page: scale uniformly
        out = [w * width / sum(out) for w in out]
    return out


def build_table(rows: list[list[str]], st: dict, width: float) -> Table:
    header, body = rows[0], rows[1:]
    ncols = len(header)
    data = [[Paragraph(inline(c), st["cellhead"]) for c in header]]
    for r in body:
        r = (r + [""] * ncols)[:ncols]
        data.append([Paragraph(inline(c), st["cell"]) for c in r])

    widths = column_widths(rows, ncols, width)

    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
    ]))
    return t


def equation(tex: str, index: int, width: float) -> Image:
    """Typeset one display equation with matplotlib mathtext, as a centred image.

    mathtext lacks a few LaTeX macros; they are mapped to equivalents that render
    identically rather than being rewritten in the manuscript.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    expr = tex.strip()
    for a, b in ((r"\operatorname", r"\mathrm"), (r"\tfrac", r"\frac"),
                 (r"\text", r"\mathrm")):
        expr = expr.replace(a, b)
    # \mathrm{release date}: mathtext drops spaces inside \mathrm
    expr = re.sub(r"\\mathrm\{([^}]*)\}",
                  lambda m: r"\mathrm{" + m.group(1).replace(" ", r"\ ") + "}", expr)

    EQUATION_DIR.mkdir(parents=True, exist_ok=True)
    out = EQUATION_DIR / f"eq{index}.png"
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${expr}$", fontsize=11)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)

    img = Image(str(out))
    w, h = img.imageWidth * 72 / 300, img.imageHeight * 72 / 300
    scale = min(1.0, width / w)
    img.drawWidth, img.drawHeight = w * scale, h * scale
    img.hAlign = "CENTER"
    return img


def figure(path: str, width: float) -> Image:
    """A figure at text width (or its natural width, if narrower)."""
    src = (ROOT / path) if not Path(path).is_absolute() else Path(path)
    if not src.exists():
        sys.exit(f"figure not found: {src}")
    img = Image(str(src))
    w, h = img.imageWidth, img.imageHeight
    draw_w = min(width, w * 72 / 200)          # figures are saved at 200 dpi
    img.drawWidth, img.drawHeight = draw_w, draw_w * h / w
    img.hAlign = "CENTER"
    return img


def parse(markdown: str, st: dict, width: float) -> list:
    lines = markdown.splitlines()
    story: list = []
    para: list[str] = []
    front_matter = True   # everything before the first "## " is the title block
    i = 0

    def flush() -> None:
        if para:
            if front_matter:
                # title-block lines are distinct units (author, affiliation)
                # and must not be reflowed into one line
                markup = "<br/>".join(inline(s.strip()) for s in para)
                story.append(Paragraph(markup, st["front"]))
            else:
                text = " ".join(s.strip() for s in para)
                story.append(Paragraph(inline(text), st["body"]))
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```latex"):
            flush()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            story.append(Spacer(1, 2))
            story.append(equation("\n".join(block), len(story), width))
            story.append(Spacer(1, 6))
            i += 1
            continue

        if m := re.fullmatch(r"!\[[^\]]*\]\(([^)]+)\)", stripped):
            flush()
            story.append(Spacer(1, 4))
            story.append(figure(m.group(1), width))
            story.append(Spacer(1, 4))
            i += 1
            continue

        if stripped.startswith("```"):
            flush()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            # shrink a code block whose longest line would overrun the text width
            code_style = st["code"]
            avail = width - code_style.leftIndent - 4
            longest = max((stringWidth(b, MONO, code_style.fontSize) for b in block),
                          default=0)
            if longest > avail:
                size = code_style.fontSize * avail / longest
                code_style = ParagraphStyle("code_fit", parent=code_style,
                                            fontSize=size, leading=size * 1.27)
            story.append(Preformatted("\n".join(block), code_style))
            i += 1
            continue

        if stripped.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = split_row(lines[i])
                if not is_separator(cells):
                    rows.append(cells)
                i += 1
            if rows:
                story.append(Spacer(1, 3))
                story.append(build_table(rows, st, width))
                story.append(Spacer(1, 8))
            continue

        if stripped.startswith("# "):
            flush()
            story.append(Paragraph(inline(stripped[2:]), st["title"]))
        elif stripped.startswith("## "):
            flush()
            front_matter = False
            story.append(Paragraph(inline(stripped[3:]), st["h2"]))
        elif stripped.startswith("### "):
            flush()
            story.append(Paragraph(inline(stripped[4:]), st["h3"]))
        elif stripped == "---":
            flush()
            story.append(HRFlowable(width="100%", thickness=0.4,
                                    color=colors.HexColor("#9CA3AF"),
                                    spaceBefore=5, spaceAfter=7))
        elif stripped.startswith("> "):
            flush()
            quote = [stripped[2:]]
            while i + 1 < len(lines) and lines[i + 1].strip().startswith("> "):
                i += 1
                quote.append(lines[i].strip()[2:])
            story.append(Paragraph(inline(" ".join(quote)), st["quote"]))
        elif re.match(r"^[-*] ", stripped):
            flush()
            story.append(Paragraph(inline(stripped[2:]), st["bullet"],
                                   bulletText="\u2022"))
        elif m := re.match(r"^(\d+)\. (.*)", stripped):
            flush()
            story.append(Paragraph(inline(m.group(2)), st["bullet"],
                                   bulletText=f"{m.group(1)}."))
        elif not stripped:
            flush()
        else:
            if front_matter and not para and stripped.startswith("**") \
                    and stripped.endswith("**") and story and len(story) == 1:
                story.append(Paragraph(inline(stripped), st["subtitle"]))
            else:
                para.append(line)
        i += 1

    flush()

    # keep each heading with the block that follows it
    kept: list = []
    j = 0
    while j < len(story):
        f = story[j]
        if isinstance(f, Paragraph) and f.style.name in ("h2", "h3") and j + 1 < len(story):
            kept.append(KeepTogether([f, story[j + 1]]))
            j += 2
        else:
            kept.append(f)
            j += 1
    return kept


# ---------------------------------------------------------------------------
# Page furniture and build
# ---------------------------------------------------------------------------

def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(SERIF, 8.5)
    canvas.setFillColor(colors.HexColor("#4B5563"))
    canvas.drawString(doc.leftMargin, 0.55 * inch,
                      "Gao, Underpowered by Construction (working paper)")
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.55 * inch, str(doc.page))
    canvas.restoreState()


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else SOURCE
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else TARGET
    text = source.read_text(encoding="utf-8")
    title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")),
                 "Underpowered by Construction")
    font = register_fonts()
    audit_glyphs(text, font)

    margin = 0.95 * inch
    doc = SimpleDocTemplate(
        str(target), pagesize=letter,
        leftMargin=margin, rightMargin=margin,
        topMargin=0.85 * inch, bottomMargin=0.9 * inch,
        title=title,
        author="Ethan Gao",
        subject="Statistical power of backtest comparisons; macroeconomic regime models",
        keywords="regime switching, hidden Markov models, backtesting, statistical "
                 "power, Sharpe ratio inference, look-ahead bias",
        creator="scripts/build_paper_pdf.py",
    )
    story = parse(text, styles(), letter[0] - 2 * margin)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {target} ({target.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
