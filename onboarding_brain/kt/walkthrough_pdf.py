"""Render a walkthrough document to a real PDF (server-side, via reportlab).

Replaces the old browser print-to-PDF: this produces a proper downloadable file
with consistent layout, no pop-up window, and no client-side dependency.
"""
from __future__ import annotations

import html
import re
from io import BytesIO
from typing import Any

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

_ACCENT = HexColor("#0d6e63")
_INK = HexColor("#14181d")
_BODY = HexColor("#22262b")
_MUTED = HexColor("#6b645c")
_LINE = HexColor("#e3e7ec")


def _inline(text: str) -> str:
    """Escape XML, then map a safe subset of Markdown to reportlab markup."""
    t = html.escape(text or "")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)        # **bold**
    t = re.sub(r"`([^`]+?)`", r'<font face="Courier">\1</font>', t)  # `code`
    return t


def _body_flow(body: str, style: ParagraphStyle, bullet: ParagraphStyle) -> list:
    """Turn a Markdown-ish body into reportlab flowables (paragraphs + bullet lists)."""
    flow: list = []
    for block in re.split(r"\n\s*\n", body or ""):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if lines and all(re.match(r"^\s*[-*]\s+", ln) for ln in lines):
            items = [ListItem(Paragraph(_inline(re.sub(r"^\s*[-*]\s+", "", ln)), bullet))
                     for ln in lines]
            flow.append(ListFlowable(items, bulletType="bullet", leftIndent=14, bulletColor=_ACCENT))
        else:
            block = re.sub(r"^#{1,6}\s*", "", block)      # strip md heading hashes
            flow.append(Paragraph(_inline(block.replace("\n", " ")), style))
        flow.append(Spacer(1, 5))
    return flow


def build_walkthrough_pdf(doc: dict[str, Any]) -> bytes:
    """Build a PDF (bytes) from a WalkthroughResponse-shaped dict."""
    buf = BytesIO()
    pdf = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
        topMargin=20 * mm, bottomMargin=18 * mm,
        title=doc.get("title") or "Project Walkthrough", author="Cortex",
    )
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=21, textColor=_INK, spaceAfter=4, alignment=0)
    h2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=14, textColor=_ACCENT,
                        spaceBefore=16, spaceAfter=5)
    body = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=10.5, leading=15.5, textColor=_BODY)
    bullet = ParagraphStyle("Bul", parent=body)
    label = ParagraphStyle("Label", parent=body, fontSize=8.5, textColor=_ACCENT,
                           spaceBefore=4, spaceAfter=1)
    meta = ParagraphStyle("Meta", parent=body, fontSize=9, textColor=_MUTED)

    flow: list = [Paragraph(_inline(doc.get("title") or "Project Walkthrough"), h1)]
    stack = doc.get("stack") or []
    if stack:
        flow.append(Paragraph("<b>Stack:</b> " + _inline(", ".join(stack)), meta))
    flow += [Spacer(1, 8), HRFlowable(width="100%", thickness=1, color=_LINE), Spacer(1, 4)]

    for s in doc.get("sections") or []:
        flow.append(Paragraph(_inline(s.get("title") or ""), h2))
        takeaways = s.get("takeaways") or []
        if takeaways:
            flow.append(Paragraph("KEY TAKEAWAYS", label))
            flow.append(ListFlowable(
                [ListItem(Paragraph(_inline(t), bullet)) for t in takeaways],
                bulletType="bullet", leftIndent=14, bulletColor=_ACCENT))
            flow.append(Spacer(1, 6))
        flow += _body_flow(s.get("body"), body, bullet)
        files = s.get("files") or []
        if files:
            flow.append(Paragraph("Files: " + _inline(", ".join(files)), meta))
        flow.append(Spacer(1, 10))

    pdf.build(flow)
    return buf.getvalue()
