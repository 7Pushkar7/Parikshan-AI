"""Report -> PDF with ReportLab (pure Python, no system dependencies).

Design choices:
- ONE font family (Helvetica) everywhere, a fixed type scale, comfortable line spacing.
- Citations like [PMID:123, NCT0123] become numbered superscripts [1,2] with a numbered
  reference list at the end, like a journal paper. Unverified citations get a "?" in red.
- Markdown tables become real tables with wrapped cells, a shaded header row and zebra rows.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

INK = colors.HexColor("#14212B")
BODY = colors.HexColor("#25323D")
MUTED = colors.HexColor("#5C6B77")
ACCENT = colors.HexColor("#3346C7")
ACCENT_SOFT = colors.HexColor("#E8EBFB")
LINE = colors.HexColor("#D5DEE4")
ZEBRA = colors.HexColor("#F5F7F9")
WARN = colors.HexColor("#B3261E")
SAFETY = colors.HexColor("#A15C07")

CITE_RE = re.compile(r"\[((?:\s*(?:NCT\d{8}|PMID:\s*\d+|FAERS:[\w \-]+|LABEL:[\w \-]+)\s*[,;]?)+)\]", re.I)
ID_RE = re.compile(r"NCT\d{8}|PMID:\s*\d+|FAERS:[\w \-]+|LABEL:[\w \-]+", re.I)

_REPLACE = {"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
            "•": "-", "≤": "<=", "≥": ">=", "→": "->", "←": "<-", "↑": "up ",
            "↓": "down ", "α": "alpha", "β": "beta", "κ": "kappa", "μ": "u",
            "…": "...", " ": " ", "−": "-", " ": " ", " ": " ", "≈": "~"}


def _clean(text: str) -> str:
    for k, v in _REPLACE.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def norm_id(raw: str) -> str:
    raw = re.sub(r"\s+", "", raw) if raw.upper().startswith(("PMID", "NCT")) else raw.strip()
    if raw.upper().startswith(("NCT", "PMID")):
        return raw.upper()
    prefix, rest = raw.split(":", 1)
    return f"{prefix.upper()}:{rest.strip().lower()}"


# ------------------------------------------------------------------ styles

def _styles() -> dict[str, ParagraphStyle]:
    base = dict(fontName="Helvetica", textColor=BODY, alignment=TA_LEFT)
    return {
        "title": ParagraphStyle("title", **{**base, "fontName": "Helvetica-Bold", "textColor": INK},
                                fontSize=19, leading=24, spaceAfter=6),
        "meta": ParagraphStyle("meta", **{**base, "textColor": MUTED}, fontSize=8.5, leading=12, spaceAfter=2),
        "h2": ParagraphStyle("h2", **{**base, "fontName": "Helvetica-Bold", "textColor": ACCENT},
                             fontSize=12.5, leading=16, spaceBefore=14, spaceAfter=6, keepWithNext=1),
        "h3": ParagraphStyle("h3", **{**base, "fontName": "Helvetica-Bold", "textColor": INK},
                             fontSize=10.5, leading=14, spaceBefore=8, spaceAfter=3, keepWithNext=1),
        "body": ParagraphStyle("body", **base, fontSize=9.6, leading=14.2, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", **base, fontSize=9.6, leading=14.2),
        "cell": ParagraphStyle("cell", **base, fontSize=8.3, leading=11),
        "cellhead": ParagraphStyle("cellhead", **{**base, "fontName": "Helvetica-Bold", "textColor": INK},
                                   fontSize=8.3, leading=11),
        "quote": ParagraphStyle("quote", **{**base, "textColor": MUTED}, fontSize=9, leading=13,
                                leftIndent=8, spaceAfter=6),
        "ref": ParagraphStyle("ref", **base, fontSize=8.3, leading=11.5, leftIndent=18, firstLineIndent=-18,
                              spaceAfter=3),
    }


# ------------------------------------------------------------------ inline markdown -> ReportLab markup

class Citer:
    """Assigns numbers to citation IDs in order of first appearance."""

    def __init__(self, unverified: set[str]):
        self.order: list[str] = []
        self.unverified = {u.lower() for u in unverified}

    def num(self, cid: str) -> int:
        if cid not in self.order:
            self.order.append(cid)
        return self.order.index(cid) + 1

    def render(self, inner: str, sup: bool = True) -> str:
        parts = []
        for raw in ID_RE.findall(inner):
            cid = norm_id(raw)
            n = self.num(cid)
            if cid.lower() in self.unverified:
                parts.append(f'<font color="{WARN.hexval()}">{n}?</font>')
            else:
                parts.append(str(n))
        if not sup:
            return f'<font color="{ACCENT.hexval()}">[{",".join(parts)}]</font>'
        return f'<super><font size="6.5" color="{ACCENT.hexval()}">[{",".join(parts)}]</font></super>'


def inline(text: str, citer: Citer, sup: bool = True) -> str:
    text = _clean(text)
    cites: list[str] = []

    def stash(m):  # protect citations before escaping
        cites.append(citer.render(m.group(1), sup))
        return f"\x00{len(cites) - 1}\x00"

    text = CITE_RE.sub(stash, text)
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#3346C7">\1</link>', text)
    if sup:
        text = re.sub(r"\s+(\x00\d+\x00)", r"\1", text)  # no space before a superscript
    return re.sub(r"\x00(\d+)\x00", lambda m: cites[int(m.group(1))], text)


# ------------------------------------------------------------------ blocks

def _table(rows: list[list[str]], st: dict, citer: Citer, width: float) -> Table:
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    # Column widths: every column is at least as wide as its longest word (so words never break
    # into letters); the remaining space goes to the columns with the most text.
    plain = [[re.sub(r"\*\*|`", "", CITE_RE.sub("[00]", _clean(c))) for c in r] for r in rows]
    pad = 11

    def w(text, bold=False):
        return stringWidth(text, "Helvetica-Bold" if bold else "Helvetica", 8.3)

    min_w = [max(max((w(word, i == 0) for i, r in enumerate(plain) for word in r[c].split()), default=0) + pad,
                 11 * mm) for c in range(ncol)]
    want = [min(max(w(r[c], i == 0) for i, r in enumerate(plain)) + pad, width * 0.5) for c in range(ncol)]
    want = [max(a, b) for a, b in zip(want, min_w)]
    if sum(want) <= width:
        extra = width - sum(want)
        widths = [x + extra * x / sum(want) for x in want]
    else:
        room = max(width - sum(min_w), 0)
        flex = [x - m for x, m in zip(want, min_w)]
        widths = [m + (room * f / sum(flex) if sum(flex) else 0) for m, f in zip(min_w, flex)]
        if sum(widths) > width:  # extreme case: shrink everything proportionally
            widths = [x * width / sum(widths) for x in widths]
    data = [[Paragraph(inline(c, citer, sup=False), st["cellhead" if i == 0 else "cell"]) for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT_SOFT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    style += [("BACKGROUND", (0, i), (-1, i), ZEBRA) for i in range(2, len(data), 2)]
    t.setStyle(TableStyle(style))
    return t


def _bullets(items: list[str], st: dict, citer: Citer, ordered: bool) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(inline(i, citer), st["bullet"]), leftIndent=14, bottomPadding=3) for i in items],
        bulletType="1" if ordered else "bullet", start=1 if ordered else "\u2022",
        bulletFontName="Helvetica", bulletFontSize=9.6, bulletColor=ACCENT, leftIndent=14,
        spaceAfter=6)


def markdown_blocks(md_text: str, st: dict, citer: Citer, width: float) -> tuple[list, str | None]:
    """Turns the report Markdown into ReportLab flowables. Returns (flowables, title)."""
    out: list = []
    title = None
    lines = md_text.replace("\r", "").split("\n")
    i = 0
    para: list[str] = []

    def flush():
        if para:
            out.append(Paragraph(inline(" ".join(para), citer), st["body"]))
            para.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        s = line.strip()
        if not s or re.fullmatch(r"[-*_]{3,}", s):
            flush(); i += 1; continue
        if s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            text = s.lstrip("#").strip()
            if level == 1 and title is None:
                title = text
            elif level <= 2:
                out.append(Paragraph(inline(text, citer), st["h2"]))
            else:
                out.append(Paragraph(inline(text, citer), st["h3"]))
            i += 1; continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            if rows:
                out.append(Spacer(1, 4))
                out.append(_table(rows, st, citer, width))
                out.append(Spacer(1, 8))
            continue
        m = re.match(r"^\s*([-*+]|\d+[.)])\s+(.*)", line)
        if m:
            flush()
            ordered = m.group(1)[0].isdigit()
            items = []
            while i < len(lines):
                mm_ = re.match(r"^\s*([-*+]|\d+[.)])\s+(.*)", lines[i])
                if mm_:
                    items.append(mm_.group(2)); i += 1
                elif lines[i].strip() and lines[i].startswith((" ", "\t")) and items:
                    items[-1] += " " + lines[i].strip(); i += 1
                else:
                    break
            out.append(_bullets(items, st, citer, ordered))
            continue
        if s.startswith(">"):
            flush()
            out.append(Paragraph(inline(s.lstrip("> "), citer), st["quote"]))
            i += 1; continue
        para.append(s)
        i += 1
    flush()
    return out, title


# ------------------------------------------------------------------ page decoration

def _decorate(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(ACCENT)
    canvas.drawString(doc.leftMargin, h - 12 * mm, "PARIKSHAN AI")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin + 21 * mm, h - 12 * mm, "Clinical trial & drug safety intelligence")
    canvas.drawRightString(w - doc.rightMargin, h - 12 * mm, doc.report_date)
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, h - 14 * mm, w - doc.rightMargin, h - 14 * mm)
    canvas.line(doc.leftMargin, 13 * mm, w - doc.rightMargin, 13 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(doc.leftMargin, 9 * mm, "For research use only. Not medical advice. Verify against primary sources.")
    canvas.drawRightString(w - doc.rightMargin, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


TYPE_LABEL = {"PMID": "PubMed", "NCT": "ClinicalTrials.gov", "FAERS": "FDA FAERS", "LABEL": "FDA label"}


def build_report_pdf(body_md: str, out_path: Path, *, citations: list[dict], unverified: list[str],
                     model: str = "", fallback_title: str = "Evidence report") -> Path:
    """body_md = the LLM's Markdown report (without the Sources section).
    citations = [{"id", "title", "url"}] for verified IDs."""
    st = _styles()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=20 * mm, bottomMargin=18 * mm, title=fallback_title,
                            author="Parikshan AI")
    doc.report_date = datetime.now().strftime("%d %b %Y")
    width = A4[0] - doc.leftMargin - doc.rightMargin

    citer = Citer(set(unverified))
    blocks, title = markdown_blocks(body_md, st, citer, width)
    title = title or fallback_title
    doc.title = _clean(title)

    meta_bits = [f"Generated {doc.report_date}"]
    if model:
        meta_bits.append(f"Model: {model}")
    meta_bits.append(f"{len(citations)} verified sources")
    story = [Paragraph(escape(_clean(title)), st["title"]),
             Paragraph(escape("   |   ".join(meta_bits)), st["meta"])]
    accent_bar = Table([[""]], colWidths=[28 * mm], rowHeights=[2.2])
    accent_bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT)]))
    accent_bar.hAlign = "LEFT"
    story += [Spacer(1, 4), accent_bar, Spacer(1, 6)]
    if unverified:
        story.append(Paragraph(
            f'<font color="{WARN.hexval()}"><b>Check before use:</b></font> {len(unverified)} citation(s) were '
            f"not found in the evidence the agent fetched and are marked with ? : {escape(', '.join(unverified))}",
            st["quote"]))
    story += blocks

    # numbered references
    by_id = {c["id"].lower(): c for c in citations}
    refs = []
    for n, cid in enumerate(citer.order, 1):
        c = by_id.get(cid.lower())
        kind = TYPE_LABEL.get(cid.split(":")[0].upper() if ":" in cid else "NCT", "")
        if c:
            link = f'<link href="{escape(c.get("url", ""))}" color="#3346C7">{escape(c.get("url", ""))}</link>'
            refs.append(Paragraph(f"<b>[{n}]</b>&nbsp;&nbsp;{escape(_clean(c.get('title') or cid))}. "
                                  f'<font color="{MUTED.hexval()}">{kind}, {escape(cid)}.</font> {link}', st["ref"]))
        else:
            refs.append(Paragraph(f'<b>[{n}]</b>&nbsp;&nbsp;<font color="{WARN.hexval()}">{escape(cid)}: '
                                  f"not found in fetched evidence.</font>", st["ref"]))
    if refs:
        story.append(KeepTogether([Paragraph("References", st["h2"]), refs[0]]))
        story += refs[1:]
    doc.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    return out_path

