"""
scripts/build_report.py - Build the project report (.docx) in the university template.

    python scripts/build_report.py
        --template "reports/Summer Internship Report Template 3.docx"
        --source reports/source --out reports/DERMA-GUARD_Project_Report.docx

The template's front matter (title page with logos, declaration, certificate, merit
certificate page, acknowledgement, contents / tables / figures / abbreviations, abstract)
is kept and filled in; everything from "1 INTRODUCTION" onwards is replaced by the
chapters in reports/source/*.md, written in a small markdown dialect:

    # Title              chapter (numbered, new page, 16 pt bold, centred)
    ## Title             section x.y   (14 pt bold)
    ### Title            subsection x.y.z (12 pt bold)
    plain lines          paragraph (Times New Roman 12, justified, 1.5 spacing)
    - item / 1. item     bullet / numbered list
    Table: caption       followed by a | pipe | table |
    ![caption](path)     figure, centred, numbered "Fig x.n"
    ```  ...  ```        code block (Consolas 9)
    $$ text $$           display equation (centred, italic)

Page numbers in the contents / lists are filled in afterwards with Word (pywin32),
if Word is installed; otherwise they are left blank.
"""
from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
FONT = "Times New Roman"


# ------------------------------------------------------------------ helpers
def set_run(run, size=12, bold=False, italic=False, font=FONT, color=None):
    run.font.name = font
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def add_inline(par, text, size=12, base_bold=False, font=FONT):
    """**bold**, *italic* and `code` inside a paragraph."""
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            set_run(par.add_run(tok[2:-2]), size, True, font=font)
        elif tok.startswith("`"):
            set_run(par.add_run(tok[1:-1]), size - 1, base_bold, font="Consolas")
        elif tok.startswith("*") and len(tok) > 2:
            set_run(par.add_run(tok[1:-1]), size, base_bold, True, font=font)
        else:
            set_run(par.add_run(tok), size, base_bold, font=font)


def fmt(par, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.5, before=0, after=3, indent=None):
    pf = par.paragraph_format
    par.alignment = align
    pf.line_spacing = spacing
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    if indent is not None:
        pf.left_indent = indent


def page_break(doc):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def shade(cell, hexcolor):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def set_borders(table, sz=4, color="808080"):
    """Single-line borders on every cell edge (the template has no 'Table Grid' style)."""
    tblPr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), str(sz))
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), color)
        borders.append(e)
    tblPr.append(borders)


def keep_with_next(par):
    par.paragraph_format.keep_with_next = True


# ------------------------------------------------------------------ builder
class Builder:
    def __init__(self, doc):
        self.doc = doc
        self.ch = 0
        self.sec = 0
        self.sub = 0
        self.nfig = 0
        self.ntab = 0
        self.toc = []        # (level, number, title)
        self.figs = []       # (label, caption)
        self.tabs = []
        self.prefix = "0"
        self.n_app = 0

    def chapter(self, title, numbered=True):
        new_page = bool(self.toc)        # chapter 1 gets a section break from Word instead
        if numbered:
            self.ch += 1
            self.sec = self.sub = self.nfig = self.ntab = 0
            label = f"{self.ch} {title.upper()}"
            self.prefix = str(self.ch)
        else:
            label = title.upper()
            self.nfig = self.ntab = 0
            if title.lower().startswith("appendix"):
                self.n_app += 1
                self.prefix = f"A{self.n_app}"
        p = self.doc.add_paragraph()
        set_run(p.add_run(label), 16, True)
        fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.5, 0, 18)
        keep_with_next(p)
        # "page break before" on the heading itself never produces a blank page,
        # unlike a separate page-break paragraph that can land at the top of a new page
        p.paragraph_format.page_break_before = new_page
        self.toc.append((1, str(self.ch) if numbered else "", title, label))

    def section(self, title):
        self.sec += 1
        self.sub = 0
        num = f"{self.ch}.{self.sec}"
        p = self.doc.add_paragraph()
        set_run(p.add_run(f"{num}  {title}"), 14, True)
        fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.5, 12, 6)
        keep_with_next(p)
        self.toc.append((2, num, title, f"{num}  {title}"))

    def subsection(self, title):
        self.sub += 1
        num = f"{self.ch}.{self.sec}.{self.sub}"
        p = self.doc.add_paragraph()
        set_run(p.add_run(f"{num}  {title}"), 12, True)
        fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.5, 8, 4)
        keep_with_next(p)

    def para(self, text):
        p = self.doc.add_paragraph()
        add_inline(p, text)
        fmt(p)

    def bullet(self, text, numbered=None):
        p = self.doc.add_paragraph()
        prefix = f"{numbered}. " if numbered else "• "
        set_run(p.add_run(prefix), 12)
        add_inline(p, text)
        fmt(p, spacing=1.5, after=2, indent=Cm(0.9))
        p.paragraph_format.first_line_indent = Cm(-0.5)

    def equation(self, text):
        p = self.doc.add_paragraph()
        set_run(p.add_run(text), 12, italic=True, font="Cambria Math")
        fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.5, 4, 8)

    def code(self, lines):
        t = self.doc.add_table(rows=1, cols=1)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        set_borders(t, sz=2, color="BFBFBF")
        c = t.rows[0].cells[0]
        shade(c, "F2F2F2")
        c.paragraphs[0].text = ""
        for i, line in enumerate(lines):
            p = c.paragraphs[0] if i == 0 else c.add_paragraph()
            set_run(p.add_run(line if line else " "), 8.5, font="Consolas")
            fmt(p, WD_ALIGN_PARAGRAPH.LEFT, 1.0, 0, 0)
        self.doc.add_paragraph()

    def figure(self, caption, path):
        path = (ROOT / path) if not Path(path).is_absolute() else Path(path)
        if not path.exists():
            self.para(f"[figure missing: {path.name}]")
            return
        self.nfig += 1
        label = f"Fig {self.prefix}.{self.nfig}"
        p = self.doc.add_paragraph()
        from PIL import Image as _I
        with _I.open(path) as im:
            ratio = im.height / im.width
        if 14.5 * ratio > 11.0:          # tall figures/screenshots: cap the height
            p.add_run().add_picture(str(path), height=Cm(11.0))
        else:
            p.add_run().add_picture(str(path), width=Cm(14.5))
        fmt(p, WD_ALIGN_PARAGRAPH.CENTER, 1.0, 6, 2)
        keep_with_next(p)
        c = self.doc.add_paragraph()
        set_run(c.add_run(f"{label}: "), 11, True)
        add_inline(c, caption, size=11)
        fmt(c, WD_ALIGN_PARAGRAPH.CENTER, 1.15, 0, 10)
        self.figs.append((label, caption, f"{label}: "))

    def table(self, caption, rows):
        self.ntab += 1
        label = f"Table {self.prefix}.{self.ntab}"
        c = self.doc.add_paragraph()
        set_run(c.add_run(f"{label}: "), 11, True)
        add_inline(c, caption, size=11)
        fmt(c, WD_ALIGN_PARAGRAPH.CENTER, 1.15, 6, 4)
        keep_with_next(c)
        self.tabs.append((label, caption, f"{label}: "))
        ncol = len(rows[0])
        t = self.doc.add_table(rows=len(rows), cols=ncol)
        set_borders(t)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, r in enumerate(rows):
            for j in range(ncol):
                cell = t.rows[i].cells[j]
                cell.paragraphs[0].text = ""
                txt = r[j] if j < len(r) else ""
                add_inline(cell.paragraphs[0], txt, size=10, base_bold=(i == 0))
                fmt(cell.paragraphs[0], WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER,
                    1.0, 1, 1)
                if i == 0:
                    shade(cell, "D9E2F3")
        sp = self.doc.add_paragraph()
        fmt(sp, spacing=1.0, after=4)

    # -------------------------------------------------------------- markdown
    def render(self, md: str):
        lines = md.splitlines()
        i = 0
        para_buf = []

        def flush():
            if para_buf:
                self.para(" ".join(s.strip() for s in para_buf))
                para_buf.clear()

        while i < len(lines):
            ln = lines[i]
            s = ln.strip()
            if s.startswith("```"):
                flush()
                j = i + 1
                block = []
                while j < len(lines) and not lines[j].strip().startswith("```"):
                    block.append(lines[j].rstrip())
                    j += 1
                self.code(block)
                i = j + 1
                continue
            if s.startswith("# "):
                flush()
                t = s[2:].strip()
                self.chapter(t.lstrip("*"), numbered=not t.startswith("*"))
            elif s.startswith("## "):
                flush()
                self.section(s[3:].strip())
            elif s.startswith("### "):
                flush()
                self.subsection(s[4:].strip())
            elif s.startswith("Table:"):
                flush()
                cap = s[6:].strip()
                rows = []
                j = i + 1
                while j < len(lines) and lines[j].strip().startswith("|"):
                    cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                        rows.append(cells)
                    j += 1
                self.table(cap, rows)
                i = j
                continue
            elif s.startswith("!["):
                flush()
                m = re.match(r"!\[(.*)\]\((.*)\)", s)
                self.figure(m.group(1), m.group(2))
            elif s.startswith("$$"):
                flush()
                self.equation(s.strip("$").strip())
            elif s.startswith("- "):
                flush()
                self.bullet(s[2:])
            elif re.match(r"^\d+\. ", s):
                flush()
                n, rest = s.split(". ", 1)
                self.bullet(rest, numbered=n)
            elif not s:
                flush()
            else:
                para_buf.append(s)
            i += 1
        flush()


# ------------------------------------------------------------------ template
def body_start_index(doc):
    for i, p in enumerate(doc.paragraphs):
        if re.match(r"^\s*1\s+INTRODUCTION", p.text):
            return i
    raise SystemExit("could not find '1 INTRODUCTION' in the template")


def replace_text(doc, old, new):
    for p in doc.paragraphs:
        if old in p.text:
            full = p.text.replace(old, new)
            for r in p.runs[1:]:
                r.text = ""
            if p.runs:
                p.runs[0].text = full


def set_par_text(par, text, size=None, bold=None, italic=None):
    runs = par.runs
    if not runs:
        set_run(par.add_run(text), size or 12, bool(bold))
        return
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""
    if size:
        runs[0].font.size = Pt(size)
    if bold is not None:
        runs[0].bold = bold


def fill_table(t, rows, keep_header=True):
    """Replace the data rows of a template table with `rows` (list of lists)."""
    start = 1 if keep_header else 0
    tmpl = copy.deepcopy(t.rows[start]._tr) if len(t.rows) > start else None
    while len(t.rows) > start:
        t._tbl.remove(t.rows[-1]._tr)
    for r in rows:
        tr = copy.deepcopy(tmpl)
        t._tbl.append(tr)
        row = t.rows[-1]
        for j, cell in enumerate(row.cells):
            cell.paragraphs[0].text = ""
            for extra in cell.paragraphs[1:]:
                extra._element.getparent().remove(extra._element)
            txt = r[j] if j < len(r) else ""
            set_run(cell.paragraphs[0].add_run(txt), 11)


def short_caption(cap: str, limit: int = 95) -> str:
    """First sentence of a caption, for the list of tables / figures."""
    first = re.split(r"(?<=[a-z0-9)])\. ", cap.strip())[0].rstrip(".")
    return first if len(first) <= limit else first[:limit].rsplit(" ", 1)[0] + " ..."


def style_list_table(t, header, widths_cm, bold_rows=(), center_cols=()):
    """Front-matter list as a proper bordered table: shaded header, fixed widths."""
    set_borders(t, sz=6, color="7F7F7F")
    tblPr = t._tbl.tblPr
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(lay)
    t.autofit = False
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(t.rows):
        for j, cell in enumerate(row.cells):
            if j < len(widths_cm):
                cell.width = Cm(widths_cm[j])
            par = cell.paragraphs[0]
            if i == 0:
                for extra in cell.paragraphs[1:]:
                    extra._element.getparent().remove(extra._element)
                for r in par.runs:
                    r.text = ""
                set_run(par.add_run(header[j] if j < len(header) else ""), 12, True)
                shade(cell, "D9E2F3")
            elif i in bold_rows:
                for r in par.runs:
                    r.bold = True
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER if (j in center_cols or i == 0) \
                else WD_ALIGN_PARAGRAPH.LEFT
            par.paragraph_format.space_before = Pt(2)
            par.paragraph_format.space_after = Pt(2)
            par.paragraph_format.line_spacing = 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="reports/Summer Internship Report Template 3.docx")
    ap.add_argument("--source", default="reports/source")
    ap.add_argument("--out", default="reports/DERMA-GUARD_Project_Report.docx")
    ap.add_argument("--meta", default="reports/source/meta.json")
    ap.add_argument("--merit-page", action="store_true",
                    help="keep the template's industry certificate-of-merit page (internships)")
    args = ap.parse_args()
    import json
    meta = json.loads((ROOT / args.meta).read_text(encoding="utf-8"))

    doc = Document(str(ROOT / args.template))
    start = body_start_index(doc)
    body = doc.element.body
    # remove everything from "1 INTRODUCTION" to the end, except the final sectPr
    first = doc.paragraphs[start]._element
    # the page break before chapter 1 lives in the previous paragraph; keep it
    node = first
    while node is not None:
        nxt = node.getnext()
        if node.tag != qn("w:sectPr"):
            body.remove(node)
        node = nxt

    # ---------------------------------------------------------------- front matter
    if not args.merit_page:   # a project report has no industry certificate of merit
        for k, par in enumerate(doc.paragraphs):
            if par.text.startswith("INSERT THE CERTIFICATE OF MERIT"):
                nxt = doc.paragraphs[k + 1]
                par._element.getparent().remove(par._element)
                if not nxt.text.strip() and nxt._element.find(".//" + qn("w:br")) is not None:
                    nxt._element.getparent().remove(nxt._element)
                break
    for t in doc.element.body.iter(qn("w:t")):
        if t.text and "<Title of Industry Internship>" in t.text:
            t.text = t.text.replace("<Title of Industry Internship>", "DERMA-GUARD")
        if t.text and "<Times New Roman>" in t.text:
            t.text = re.sub(r"<Times New Roman>.*", "", t.text)
    for c in doc.element.body.iter(qn("w:color")):
        if c.get(qn("w:val"), "").upper() == "CE181E":
            c.set(qn("w:val"), "000000")
    P = doc.paragraphs
    title = meta["title"]
    for p in P:
        t = p.text
        if "<Name>" in t:
            set_par_text(p, f"{meta['name']} ({meta['reg_no']})")
        elif "Title of the industry internship" in t:
            replace_text(doc, "“Title of the industry internship”", f"“{title}”")
            replace_text(doc, "Title of the industry internship", title)
        elif "Candidate    Name (Register No: 00XYZ000)" in t:
            replace_text(doc, "Candidate    Name (Register No: 00XYZ000)",
                         f"{meta['name']} (Register No: {meta['reg_no']})")
        elif t.strip() == "A REPORT":
            set_par_text(p, title, size=16, bold=True)
            q = p.insert_paragraph_before("")
            set_run(q.add_run("A PROJECT REPORT ON"), 14)
            q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif "October - 2026" in t:
            set_par_text(p, meta["date"])
    # acknowledgement: replace the instruction lines, keep the hierarchy names
    for p in doc.paragraphs:
        t = p.text
        if t.startswith("Give your acknowledgement here"):
            set_par_text(p, meta["ack_intro"])
            p.runs[0].bold = False
        elif t.startswith("Follow the following hierarchy"):
            set_par_text(p, meta["ack_hierarchy_intro"])
        elif t.startswith("Parents, Friends"):
            set_par_text(p, meta["ack_close"])
    # abstract
    abs_i = [i for i, p in enumerate(doc.paragraphs) if p.text.startswith("The abstract of the project work")]
    if abs_i:
        p0 = doc.paragraphs[abs_i[0]]
        paras = meta["abstract"]
        set_par_text(p0, paras[0])
        anchor = p0
        for txt in paras[1:]:
            new = copy.deepcopy(anchor._element)
            anchor._element.addnext(new)
            from docx.text.paragraph import Paragraph
            anchor = Paragraph(new, anchor._parent)
            set_par_text(anchor, txt)
        for p in list(doc.paragraphs):
            if p.text.strip() in ("Abstract should cover the following", "Present scenario of the area",
                                  "Limitations/pitfalls of the present scenario.", "Problem addressed",
                                  "Proposed solution"):
                p._element.getparent().remove(p._element)
        kw = anchor._element.getnext()
        from docx.text.paragraph import Paragraph
        newp = copy.deepcopy(anchor._element)
        anchor._element.addnext(newp)
        kp = Paragraph(newp, anchor._parent)
        set_par_text(kp, "Keywords: " + meta["keywords"])
        kp.runs[0].italic = True

    # the template ends the abstract with a page break; chapter 1 gets a section break
    # from Word, so drop page breaks in the empty paragraphs after the keywords line
    tail_front = list(doc.element.body)
    for el in reversed(tail_front):
        if el.tag != qn("w:p"):
            continue
        if "".join(t.text or "" for t in el.iter(qn("w:t"))).strip():
            break
        for br in list(el.iter(qn("w:br"))):
            if br.get(qn("w:type")) == "page":
                br.getparent().remove(br)

    # ---------------------------------------------------------------- chapters
    # citations: [@key] in the sources -> [n], numbered in alphabetical order of first author
    refs = json.loads((ROOT / args.source / "refs.json").read_text(encoding="utf-8"))["refs"]

    def sort_key(k):
        t = refs[k]
        if t.startswith('"'):
            return re.sub(r"[^a-z]", "", t[1:].split(":")[0].lower())
        head = t.split(",")[0].split(" and ")[0].replace(" et al.", "")
        if "MILK" in head:
            return "milk"
        return head.split()[-1].lower() + t.lower()
    order = sorted(refs, key=sort_key)
    num = {k: i + 1 for i, k in enumerate(order)}
    ref_list = "# *References\n\n" + "\n".join(f"{num[k]}. {refs[k]}" for k in order) + "\n"

    b = Builder(doc)
    for f in sorted((ROOT / args.source).glob("*.md")):
        md = ref_list if f.name.startswith("11_references") else f.read_text(encoding="utf-8")
        missing = set(re.findall(r"\[@([a-z0-9]+)\]", md)) - set(num)
        if missing:
            raise SystemExit(f"{f.name}: unknown references {sorted(missing)}")
        md = re.sub(r"\[@([a-z0-9]+)\]", lambda m: f"[{num[m.group(1)]}]", md)
        b.render(md)

    # front-matter headings carry their own page break, so a list that exactly fills a page
    # cannot push an empty page-break paragraph onto a blank page of its own
    heads = ("ACKNOWLEDGEMENT", "CONTENTS", "LIST OF TABLES", "LIST OF FIGURES",
             "LIST OF ABBREVIATIONS", "ABSTRACT")
    P = doc.paragraphs
    for k in range(1, len(P)):
        if P[k].text.strip() in heads:
            j = k - 1
            empties = []
            while j >= 0 and not P[j].text.strip():
                el = P[j]._element
                if el.find(".//" + qn("w:sectPr")) is not None or el.find(".//" + qn("w:drawing")) is not None:
                    break
                empties.append(el)
                j -= 1
            if any(e.find(".//" + qn("w:br")) is not None for e in empties):
                for e in empties:
                    e.getparent().remove(e)
                P[k].paragraph_format.page_break_before = True
    P = doc.paragraphs
    # no empty paragraph after the last table/text: it would spill onto a blank last page
    body = doc.element.body
    while True:
        last = [c for c in body if c.tag == qn("w:p")]
        if not last:
            break
        el = last[-1]
        if "".join(t.text or "" for t in el.iter(qn("w:t"))).strip() or el.find(".//" + qn("w:sectPr")) is not None                 or el.find(".//" + qn("w:drawing")) is not None:
            break
        body.remove(el)
    # Word requires a paragraph after a final table; make it 1 pt so it never needs a page
    tail = OxmlElement("w:p")
    pPr = OxmlElement("w:pPr")
    sp = OxmlElement("w:spacing")
    sp.set(qn("w:before"), "0"); sp.set(qn("w:after"), "0")
    sp.set(qn("w:line"), "20"); sp.set(qn("w:lineRule"), "exact")
    pPr.append(sp)
    rPr = OxmlElement("w:rPr")
    sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "2")
    rPr.append(sz); pPr.append(rPr)
    tail.append(pPr)
    sect = body.find(qn("w:sectPr"))
    (sect.addprevious(tail) if sect is not None else body.append(tail))

    # ---------------------------------------------------------------- lists
    T = doc.tables
    contents = [["", "Title Page", "i"], ["", "Declaration", "ii"], ["", "Certificate", "iii"],
                *([["", "Industry / Merit Certificate", "iv"]] if args.merit_page else []),
                ["", "Acknowledgement", "v"],
                ["", "Table of Contents", "vi"], ["", "List of Tables", "vii"],
                ["", "List of Figures", "viii"], ["", "List of Abbreviations", "ix"],
                ["", "Abstract", "x"]]
    for lvl, num, ttl, _ in b.toc:
        contents.append([num if lvl == 1 else "", ttl if lvl == 1 else f"      {num}  {ttl}", ""])
    fill_table(T[0], contents)
    fill_table(T[1], [[f"{lab}: {short_caption(cap)}", ""] for lab, cap, _ in b.tabs])
    fill_table(T[2], [[f"{lab}: {short_caption(cap)}", ""] for lab, cap, _ in b.figs])
    fill_table(T[3], [[a, e] for a, e in meta["abbreviations"]])
    chapter_rows = [i + 1 for i, r in enumerate(contents) if r[0]]
    style_list_table(T[0], ["Chapter", "Title", "Page"], [2.2, 12.0, 2.2], chapter_rows, (0, 2))
    style_list_table(T[1], ["Title", "Page"], [14.2, 2.2], (), (1,))
    style_list_table(T[2], ["Title", "Page"], [14.2, 2.2], (), (1,))
    style_list_table(T[3], ["Abbreviation", "Expansion"], [4.2, 12.2], (), ())
    if len(T) > 4:   # the empty sample table at the end of the template
        try:
            T[4]._tbl.getparent().remove(T[4]._tbl)
        except Exception:
            pass

    out = ROOT / args.out
    doc.save(str(out))
    print(f"saved -> {out}  ({len(b.toc)} headings, {len(b.figs)} figures, {len(b.tabs)} tables)")

    # ---------------------------------------------------------------- page numbers via Word
    try:
        import win32com.client as win32
        word = win32.gencache.EnsureDispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        d = word.Documents.Open(str(out), False, False, False)   # ConfirmConversions, ReadOnly, AddToRecent

        # 1. Chapter 1 starts a new section: front matter in roman numerals, body from 1
        rng = d.Content
        f = rng.Find
        if f.Execute(FindText=b.toc[0][3], MatchCase=True):
            start = rng.Start
            before = d.Range(max(start - 1, 0), max(start - 1, 0)).Sections(1).Index
            here = d.Range(start, start).Sections(1).Index
            if before == here:                               # no section break there yet
                d.Range(start, start).InsertBreak(2)        # wdSectionBreakNextPage
                start += 1
            body_sec = d.Range(start, start).Sections(1).Index
            for i in range(1, d.Sections.Count + 1):
                sec = d.Sections(i)
                for k in (1, 2, 3):                          # primary, first page, even
                    ft = sec.Footers(k)
                    if i == body_sec:
                        ft.LinkToPrevious = False
                    pn = ft.PageNumbers
                    if i < body_sec:
                        pn.NumberStyle = 2                    # lowercase roman
                    else:
                        pn.NumberStyle = 0                    # arabic
                        if i == body_sec:
                            pn.RestartNumberingAtSection = True
                            pn.StartingNumber = 1
        d.Repaginate()

        roman = lambda n: ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi",  # noqa: E731
                           "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
                           "xxi", "xxii", "xxiii", "xxiv", "xxv"][n] if 0 < n < 26 else str(n)
        body_from = d.Tables(min(4, d.Tables.Count)).Range.End   # skip the front-matter lists

        def page_of(text, frm=None):
            r_ = d.Range(body_from if frm is None else frm, d.Content.End)
            f_ = r_.Find
            f_.ClearFormatting()
            if f_.Execute(FindText=text[:250], MatchCase=True):
                return r_.Information(1)       # wdActiveEndAdjustedPageNumber
            return ""

        def put(cell, text):
            cell.Range.Text = str(text)
            cell.Range.Font.Size = 11
            cell.Range.Font.Bold = False
            cell.Range.Font.Name = "Times New Roman"

        tbl = d.Tables(1)
        front = ["", "DECLARATION", "CERTIFICATE"] + (["INSERT THE CERTIFICATE"] if args.merit_page else []) +                 ["ACKNOWLEDGEMENT", "CONTENTS", "LIST OF TABLES", "LIST OF FIGURES",
                 "LIST OF ABBREVIATIONS", "ABSTRACT"]
        for k, head in enumerate(front):
            pg = 1 if k == 0 else page_of(head, frm=0)
            if pg != "":
                put(tbl.Cell(2 + k, 3), roman(int(pg)))
        r = 1 + len(front)
        for lvl, num, ttl, label in b.toc:
            r += 1
            if r <= tbl.Rows.Count:
                put(tbl.Cell(r, 3), page_of(label))
        for k, (lab, cap, prefix) in enumerate(b.tabs):
            if k + 2 <= d.Tables(2).Rows.Count:
                put(d.Tables(2).Cell(k + 2, 2), page_of(prefix))
        for k, (lab, cap, prefix) in enumerate(b.figs):
            if k + 2 <= d.Tables(3).Rows.Count:
                put(d.Tables(3).Cell(k + 2, 2), page_of(prefix))

        # front-matter lists (contents, tables, figures, abbreviations): single-spaced
        for i in range(1, min(4, d.Tables.Count) + 1):
            pf = d.Tables(i).Range.ParagraphFormat
            pf.SpaceBefore, pf.SpaceAfter, pf.LineSpacingRule = 0, 0, 0
            d.Tables(i).Rows.HeightRule = 0           # wdRowHeightAuto
        # 2. body tables: fit the columns to their content, then to the page width
        for i in range(5, d.Tables.Count + 1):
            t = d.Tables(i)
            if t.Columns.Count > 1:
                t.AllowAutoFit = True
                t.AutoFitBehavior(1)                     # wdAutoFitContent
                t.AutoFitBehavior(2)                     # wdAutoFitWindow
                if t.Columns.Count >= 4:                 # give the label column room
                    try:
                        t.PreferredWidthType = 2         # wdPreferredWidthPercent
                        t.PreferredWidth = 100
                        t.Columns(1).PreferredWidthType = 2
                        t.Columns(1).PreferredWidth = 30
                    except Exception:
                        pass
        d.Repaginate()
        pages = d.ComputeStatistics(2)          # wdStatisticPages
        import os
        pdf = out.with_suffix(".pdf")
        try:                                    # the PDF may be open in a viewer
            if pdf.exists():
                open(pdf, "r+b").close()
        except PermissionError:
            pdf = out.with_name(out.stem + "_updated.pdf")
            print(f"  {out.with_suffix('.pdf').name} is open elsewhere - writing {pdf.name}")
        tmp = out.with_name(out.stem + "_tmp.docx")
        d.SaveAs2(str(tmp), 16)                 # wdFormatDocumentDefault
        d.ExportAsFixedFormat(str(pdf), 17)     # wdExportFormatPDF
        d.Close()
        os.replace(tmp, out)
        word.Quit()
        print(f"page numbers filled; total pages: {pages}; PDF -> {pdf}")
    except Exception as e:
        print(f"(page numbers not filled - Word automation unavailable: {e})")
        try:
            word.Quit()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
