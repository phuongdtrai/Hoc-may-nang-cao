# -*- coding: utf-8 -*-
"""
pdf_parser.py - Giai doan 3 (Ingestion): phan tich PDF ky thuat so bang PyMuPDF.

Moi trang duoc tach thanh cac phan tu theo thu tu doc:
    heading  tieu de (co cap do, dung de dung duong dan muc "section")
    para     doan van hoac 1 muc liet ke
    caption  chu thich hinh/bang ("Hinh 1.2. ...")
    table    bang, da doi sang markdown
    slide    moc bat dau 1 slide (tai lieu in dang handout nhieu slide/trang)
va danh sach hinh: anh raster, hinh ve vector (so do, bieu do) hoac ca slide, kem toa do.

Cac loi PDF thuong gap da xu ly:
  - PDF LaTeX tieng Viet mat dau cach sau chu co 2 dau ("hệthống") -> them lai dau cach
    dua vao khoang trong giua 2 ky tu.
  - Moi dong 1 khoi (PDF LaTeX, Word) -> noi lai thanh doan van theo le phai va khoang cach dong.
  - Dau trang/chan trang lap lai, so trang, muc luc, manh cong thuc toan -> bo.
  - Chu nam trong hinh ve (nhan cua so do) -> khong tron vao doan van, gan vao hinh.
  - Khung slide, so do co khung bi nhan nham la bang -> kiem tra lai truoc khi nhan la bang.
"""
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

try:
    import pymupdf  # PyMuPDF >= 1.24 (nhom dung 1.28.2: cung phien ban -> cung image_id)
except ImportError:  # ban cu
    import fitz as pymupdf

if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()  # tat dong goi y cai goi pymupdf_layout (khong can)

Rect = pymupdf.Rect
# Khong giu ligature ("ﬁ" -> "fi"), khong lay khoi anh khi doc chu, cat theo trang
TEXT_FLAGS = pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_MEDIABOX_CLIP

STRONG_BULLETS = "•◦▪▫■□●○◆◇►▶▸➢➤✓✔✗❖⦁∙·"
CAPTION_RE = re.compile(
    r"^(Hình|HÌNH|Hinh|Fig\.?|Figure|Bảng|BẢNG|Bang|Biểu đồ|BIỂU ĐỒ|Sơ đồ|SƠ ĐỒ|Ảnh|Table)\s*"
    r"\d+([.\-–]\d+)*\s*[.:\-–]?(\s|$)")
TABLE_CAPTION_RE = re.compile(r"^(Bảng|BẢNG|Bang|Table)\s*\d", re.I)
TOC_RE = re.compile(r"((\.\s?){5,}|…{2,}|_{5,})\s*\d{0,4}\s*$")
PART_RE = re.compile(r"^(PHẦN|Phần)\s+([IVXLC]+|\d+|THỨ\s+\S+|thứ\s+\S+)\b")
CHAPTER_RE = re.compile(r"^(CHƯƠNG|Chương|CHUONG|Chuong)\s*([IVXLC]+|\d+)\b")
ROMAN_RE = re.compile(r"^([IVX]{1,5})\s*[.\-–:)]\s*\S")
ARABIC_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,4})\s*[.):]?\s*(?=\S)")
LETTER_RE = re.compile(r"^([a-hđ])\s*[.)]\s*\S")
PAGE_NO_RE = re.compile(r"[-–—\s]*(\d{1,4}|(Trang|trang|Page)\s*\d+|\d{1,4}\s*/\s*\d{1,4})[-–—\s]*")
SENT_END = (".", "!", "?", ":", ";", "…", "”", '"')


# Font Symbol cua Word luu ky hieu toan o vung rieng U+F020-U+F0FF -> doi ve Unicode that
_SYM_GREEK = dict(zip("ABCDEFGHIKLMNOPQRSTUWXYZabcdefghijklmnopqrstuvwxyz",
                      "ΑΒΧΔΕΦΓΗΙΚΛΜΝΟΠΘΡΣΤΥΩΞΨΖαβχδεφγηιϕκλμνοπθρστυϖωξψζ"))
SYMBOL_MAP = {0x22: "∀", 0x24: "∃", 0x27: "∋", 0x2D: "−", 0x40: "≅", 0x5C: "∴", 0x5E: "⊥", 0x7E: "∼",
              0xA2: "′", 0xA3: "≤", 0xA5: "∞", 0xAB: "↔", 0xAC: "←", 0xAD: "↑", 0xAE: "→", 0xAF: "↓",
              0xB0: "°", 0xB1: "±", 0xB2: "″", 0xB3: "≥", 0xB4: "×", 0xB5: "∝", 0xB6: "∂", 0xB7: "•",
              0xB8: "÷", 0xB9: "≠", 0xBA: "≡", 0xBB: "≈", 0xBC: "…", 0xC4: "⊗", 0xC5: "⊕", 0xC6: "∅",
              0xC7: "∩", 0xC8: "∪", 0xC9: "⊃", 0xCA: "⊇", 0xCB: "⊄", 0xCC: "⊂", 0xCD: "⊆", 0xCE: "∈",
              0xCF: "∉", 0xD0: "∠", 0xD1: "∇", 0xD5: "∏", 0xD6: "√", 0xD7: "⋅", 0xD8: "¬", 0xD9: "∧",
              0xDA: "∨", 0xDB: "⇔", 0xDC: "⇐", 0xDD: "⇑", 0xDE: "⇒", 0xDF: "⇓", 0xE0: "◊", 0xE5: "∑",
              0xF2: "∫"}
SYMBOL_MAP.update({ord(k): v for k, v in _SYM_GREEK.items()})


def map_symbol_char(c, font):
    o = ord(c)
    if 0xF020 <= o <= 0xF0FF and "symbol" in font.lower():
        return SYMBOL_MAP.get(o - 0xF000, chr(o - 0xF000) if 0x20 <= o - 0xF000 < 0x7F else c)
    return c


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def clean_text(s):
    s = nfc(s).replace(" ", " ").replace("­", "")
    s = re.sub(r"[-]", "•", s)  # ky tu vung rieng (Wingdings, Symbol): thuong la dau dau dong
    s = re.sub(r"[ \t\r\n\f\v]+", " ", s)
    return s.strip()


def is_formula_fragment(text):
    """Manh cong thuc toan bi tach roi ("n", "→ − ∑", "y c y b x a") -> bo khoi doan van."""
    letters = sum(ch.isalpha() for ch in text)
    if letters <= 3 and len(text) < 25:
        return True
    toks = text.split()
    if len(toks) >= 3 and len(text) < 80:
        single = sum(1 for t in toks if len(t) == 1)
        if single / len(toks) >= 0.7:
            return True
    return False


def suspicious_ratio(text):
    """Ti le ky tu dang ngo (font loi, ma cu TCVN3/VNI) -> cao thi nen OCR trang do."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    bad = 0
    for c in chars:
        o = ord(c)
        if c == "�" or 0xE000 <= o <= 0xF8FF or o < 32 or 0x80 <= o < 0xA0:
            bad += 1
        elif c in "¨¬−Þ©®¦¤¥§«»µ¶¸¹º¼½¾¿×÷ð":  # hay gap khi PDF dung font TCVN3/VNI cu
            bad += 1
    return bad / len(chars)


# ---------------------------------------------------------------------------
# Dong chu
# ---------------------------------------------------------------------------
@dataclass
class Line:
    text: str
    bbox: Rect
    size: float
    bold: float          # ti le ky tu in dam
    italic: float
    block: int
    page: int
    horizontal: bool = True


def _line_text(chars, size):
    """Ghep ky tu; them dau cach khi khoang trong giua 2 ky tu lon hon ~0.18 em (loi mat dau cach o PDF LaTeX)."""
    out, prev, added = [], None, 0
    gap_min = max(0.5, 0.18 * size)
    for ch in chars:
        c = ch["c"]
        if prev is not None and not c.isspace() and not prev["c"].isspace():
            if ch["bbox"][0] - prev["bbox"][2] > gap_min:
                out.append(" ")
                added += 1
        out.append(c)
        prev = ch
    return "".join(out), added


def page_lines(page, page_no, stats=None):
    raw = page.get_text("rawdict", flags=TEXT_FLAGS, sort=True)
    lines = []
    for bno, b in enumerate(raw.get("blocks", [])):
        if b.get("type", 0) != 0:
            continue
        for ln in b.get("lines", []):
            chars, sizes, nb, ni, n = [], Counter(), 0, 0, 0
            for s in ln.get("spans", []):
                font = s.get("font", "")
                isb = bool(s["flags"] & 16) or bool(re.search(r"bold|black|heavy|semibold|demi", font, re.I))
                isi = bool(s["flags"] & 2) or bool(re.search(r"italic|oblique", font, re.I))
                for ch in s.get("chars", []):
                    if "\uf020" <= ch["c"] <= "\uf0ff":
                        ch = dict(ch, c=map_symbol_char(ch["c"], font))
                    chars.append(ch)
                    if not ch["c"].isspace():
                        sizes[round(s["size"], 1)] += 1
                        n += 1
                        nb += isb
                        ni += isi
            if n == 0:
                continue
            size = sizes.most_common(1)[0][0]
            raw_text, added = _line_text(chars, size)
            text = clean_text(raw_text)
            if stats is not None:
                stats["spaces_added"] += added
            if not text:
                continue
            d = ln.get("dir", (1, 0))
            lines.append(Line(text, Rect(ln["bbox"]), size, nb / n, ni / n, bno, page_no,
                              horizontal=abs(d[0]) > 0.9))
    return lines


def _norm_hf(text):
    return re.sub(r"\d+", "#", text.lower()).strip()


# ---------------------------------------------------------------------------
# Phan tu va hinh
# ---------------------------------------------------------------------------
@dataclass
class Element:
    kind: str            # heading | para | caption | table | slide
    text: str
    page: int
    bbox: Rect = None
    level: int = 0       # cap tieu de (1 = lon nhat)
    bullet: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class Figure:
    page: int
    bbox: Rect
    kind: str                    # raster | vector | mixed | slide
    caption: str = ""
    caption_bbox: Rect = None
    text_in_image: str = ""      # chu co san trong PDF nam trong vung hinh (nhan so do, chu tren slide)
    context_before: str = ""
    context_after: str = ""
    slide_number: int = None
    slide_title: str = ""
    n_raster: int = 0
    n_paths: int = 0


def heading_level(text, size, body):
    if PART_RE.match(text):
        return 1
    if CHAPTER_RE.match(text):
        return 2
    if ROMAN_RE.match(text):
        return 3
    m = ARABIC_RE.match(text)
    if m:
        return min(7, 4 + m.group(1).count("."))
    if LETTER_RE.match(text):
        return 8
    if size >= body + 4:
        return 2
    if size >= body + 1.5:
        return 3
    return 6


def _overlap_x(a, b):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    return max(0.0, w) / max(1e-6, min(a.width, b.width))


def _inside(inner, outer, tol=3):
    return (inner.x0 >= outer.x0 - tol and inner.y0 >= outer.y0 - tol and
            inner.x1 <= outer.x1 + tol and inner.y1 <= outer.y1 + tol)


def _center_in(inner, outer, tol=2):
    cx, cy = (inner.x0 + inner.x1) / 2, (inner.y0 + inner.y1) / 2
    return outer.x0 - tol <= cx <= outer.x1 + tol and outer.y0 - tol <= cy <= outer.y1 + tol


def _area(r):
    return max(0.0, r.width) * max(0.0, r.height)


def _iou(a, b):
    inter = Rect(a) & Rect(b)
    if inter.is_empty:
        return 0.0
    return _area(inter) / max(1e-6, _area(a) + _area(b) - _area(inter))


def _merge_rects(rects, gap=10):
    rects = [Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out = []
        while rects:
            r = rects.pop()
            for i, o in enumerate(out):
                grown = Rect(o.x0 - gap, o.y0 - gap, o.x1 + gap, o.y1 + gap)
                if grown.intersects(r):
                    out[i] = o | r
                    changed = True
                    break
            else:
                out.append(r)
        rects = out
    return rects


def _percentile(vals, q):
    vals = sorted(vals)
    if not vals:
        return 0.0
    k = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return vals[k]


def _map_pua_symbol(text):
    return "".join(SYMBOL_MAP.get(ord(c) - 0xF000, c) if "\uf020" <= c <= "\uf0ff" else c for c in text or "")


def table_to_markdown(rows):
    rows = [[clean_text(_map_pua_symbol(c or "")) for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    esc = lambda c: c.replace("|", "\\|")
    out = ["| " + " | ".join(esc(c) for c in rows[0]) + " |", "|" + "---|" * ncol]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Phan tich ca tai lieu
# ---------------------------------------------------------------------------
class PdfDocument:
    """Doc 1 file PDF, phan tich 2 luot: (1) thong ke ca tai lieu, (2) tach phan tu tung trang."""

    HF_ZONE = 0.10  # 10% tren/duoi trang la vung dau trang/chan trang

    def __init__(self, path, doc_id):
        self.path, self.doc_id = path, doc_id
        self.doc = pymupdf.open(path)
        self.n_pages = len(self.doc)
        st = Counter()
        self.lines = [page_lines(p, i + 1, st) for i, p in enumerate(self.doc)]
        self.spaces_added = st["spaces_added"]
        self.page_chars = [sum(len(l.text) for l in ls) for ls in self.lines]
        sizes = Counter()
        for ls in self.lines:
            for l in ls:
                sizes[l.size] += len(l.text)
        self.body_size = sizes.most_common(1)[0][0] if sizes else 11.0
        self._find_header_footer()
        self._count_xrefs()
        self.slide_frames = {i + 1: self._slide_frames(p) for i, p in enumerate(self.doc)}
        pages_with_slides = sum(1 for v in self.slide_frames.values() if len(v) >= 2)
        self.is_slides = self.n_pages > 0 and pages_with_slides >= max(1, 0.6 * self.n_pages)
        self._slide_counter = 0
        self.toc_pages = set()
        self.footnotes_dropped = 0

    # --- luot 1 ---------------------------------------------------------------
    def _find_header_footer(self):
        cnt = Counter()
        for i, ls in enumerate(self.lines):
            h = self.doc[i].rect.height
            seen = set()
            for l in ls:
                if l.bbox.y1 < self.HF_ZONE * h or l.bbox.y0 > (1 - self.HF_ZONE) * h:
                    seen.add(_norm_hf(l.text))
            cnt.update(seen)
        need = max(3, 0.3 * self.n_pages)
        self.header_footer = {t for t, c in cnt.items() if c >= need and len(t) < 120}

    def _count_xrefs(self):
        pages = defaultdict(set)
        for i, p in enumerate(self.doc):
            for info in p.get_image_info(xrefs=True):
                pages[info["xref"]].add(i + 1)
        self.xref_pages = {x: len(v) for x, v in pages.items()}

    def _slide_frames(self, page):
        """Khung slide: hinh chu nhat vien lon (>=25% rong, >=20% cao), >=2 khung tren 1 trang."""
        pr = page.rect
        frames = []
        try:
            drawings = page.get_drawings()
        except Exception:
            return []
        for d in drawings:
            r = Rect(d["rect"])
            if d.get("type") not in ("s", "fs") or not (r.width > 0.25 * pr.width and r.height > 0.2 * pr.height):
                continue
            if r.width > 0.9 * pr.width and r.height > 0.9 * pr.height:
                continue
            if any(_iou(r, f) > 0.9 for f in frames):
                continue
            frames.append(r)
        if len(frames) < 2:
            return []
        for a in frames:
            for b in frames:
                if a is not b and _iou(a, b) > 0.05:
                    return []
        frames.sort(key=lambda r: (round(r.y0 / 20), r.x0))
        return frames

    def is_header_footer(self, line, page_h):
        t = line.text.strip()
        in_zone = line.bbox.y1 < self.HF_ZONE * page_h or line.bbox.y0 > (1 - self.HF_ZONE) * page_h
        if in_zone and PAGE_NO_RE.fullmatch(t):
            return True
        if re.fullmatch(r"(Trang|trang|Page)\s*\d+(\s*/\s*\d+)?", t):
            return True
        return in_zone and _norm_hf(t) in self.header_footer

    # --- luot 2: tung trang ----------------------------------------------------
    def analyze_page(self, pno):
        page = self.doc[pno - 1]
        pr = page.rect
        lines = [l for l in self.lines[pno - 1] if l.horizontal or len(l.text) > 3]
        frames = self.slide_frames.get(pno, []) if self.is_slides else []
        slide_nums = {}
        if frames:  # so thu tu slide in ngay duoi khung
            for l in lines:
                if re.fullmatch(r"\d{1,3}", l.text.strip()):
                    for k, f in enumerate(frames):
                        if 0 <= l.bbox.y0 - f.y1 <= 30 and _overlap_x(l.bbox, f) > 0:
                            slide_nums[k] = int(l.text.strip())
            lines = [l for l in lines if not (re.fullmatch(r"\d{1,3}", l.text.strip())
                                              and not any(_center_in(l.bbox, f) for f in frames))]
        lines = [l for l in lines if not self.is_header_footer(l, pr.height)]
        if not frames:
            lines = self._drop_footnotes(lines, pr.height)

        n_toc = sum(1 for l in lines if TOC_RE.search(l.text))
        has_title = any(re.fullmatch(r"(MỤC LỤC|Mục lục|MỤC LỤC CHI TIẾT)", l.text.strip()) for l in lines)
        if lines and (n_toc >= 4 or n_toc >= 0.3 * len(lines) or (has_title and n_toc >= 1)
                      or (n_toc >= 2 and (pno - 1) in self.toc_pages)):
            self.toc_pages.add(pno)
            return [], []

        tables = [] if frames else self._tables(page, lines)
        figures = self._figures(page, pno, lines, tables, frames, slide_nums)

        # dong chu thuoc bang / hinh -> khong dua vao doan van
        used = set()
        for t in tables:
            for i, l in enumerate(lines):
                if _center_in(l.bbox, t["bbox"]):
                    used.add(i)
        for f in figures:
            if f.kind == "slide":
                continue
            inside = []
            for i, l in enumerate(lines):
                if i in used or len(l.text) > 80 or l.text == f.caption:
                    continue
                if _center_in(l.bbox, f.bbox):
                    used.add(i)
                    inside.append(l.text)
            f.text_in_image = " | ".join(inside)[:1500]
            g = Rect(f.bbox.x0 - 8, f.bbox.y0 - 8, f.bbox.x1 + 8, f.bbox.y1 + 8)
            f.avoid = [Rect(l.bbox) for l in lines if not _center_in(l.bbox, f.bbox) and l.bbox.intersects(g)]

        elements = []
        if frames:
            for k, fr in enumerate(frames):
                self._slide_counter += 1
                num = slide_nums.get(k, self._slide_counter)
                self._slide_counter = num
                img_boxes = [Rect(i["bbox"]) & fr for i in page.get_image_info(xrefs=True)
                             if _center_in(Rect(i["bbox"]), fr) and _area(Rect(i["bbox"]) & fr) < 0.7 * _area(fr)]
                slide_lines = [l for i, l in enumerate(lines) if i not in used and _center_in(l.bbox, fr)
                               and not any(_center_in(l.bbox, b) for b in img_boxes)]
                els = self._elements_from_lines(slide_lines, pno, frame=fr)
                title = next((e.text for e in els if e.kind == "heading"), "")
                elements.append(Element("slide", title, pno, fr, extra={"slide_number": num}))
                elements += els
                for fig in figures:
                    if fig.kind == "slide" and _iou(fig.bbox, fr) > 0.9:
                        fig.slide_number, fig.slide_title = num, title
        else:
            free = [l for i, l in enumerate(lines) if i not in used]
            elements = self._elements_from_lines(free, pno)
            for t in tables:
                elements.append(Element("table", t["markdown"], pno, t["bbox"],
                                        extra={"caption": t.get("caption", ""), "rows": t["rows"], "cols": t["cols"]}))
            elements.sort(key=lambda e: (e.bbox.y0, e.bbox.x0))
        self._attach_context(figures, elements)
        return elements, figures

    def _drop_footnotes(self, lines, page_h):
        body = self.body_size
        main = [l for l in lines if l.size >= body - 1.0]
        if not main:
            return lines
        last_main = max(l.bbox.y1 for l in main)
        foot = sorted([l for l in lines if l.size <= body - 1.8 and l.bbox.y0 >= last_main - 1 and l.bbox.y0 > 0.6 * page_h],
                      key=lambda l: l.bbox.y0)
        if not foot or not re.match(r"^[\d*¹²³⁴⁵⁶⁷⁸⁹(\[]", foot[0].text):
            return lines
        self.footnotes_dropped += len(foot)
        drop = {id(l) for l in foot}
        return [l for l in lines if id(l) not in drop]

    def _tables(self, page, lines):
        out = []
        try:
            found = page.find_tables()
        except Exception:
            return out
        for t in found.tables:
            bbox = Rect(t.bbox)
            try:
                rows = t.extract()
            except Exception:
                continue
            cells = [c for r in rows for c in r]
            filled = [c for c in cells if c and str(c).strip()]
            if t.row_count < 2 or t.col_count < 2 or len(filled) < 0.5 * len(cells):
                continue
            if sum(len(str(c)) for c in filled) < 20:
                continue
            # so do co khung + chu thich "Hinh" ngay duoi -> la hinh, khong phai bang
            below = sorted([l for l in lines if 0 <= l.bbox.y0 - bbox.y1 <= 45 and _overlap_x(l.bbox, bbox) > 0.3],
                           key=lambda l: l.bbox.y0)
            if below and CAPTION_RE.match(below[0].text) and not TABLE_CAPTION_RE.match(below[0].text):
                continue
            above = [l for l in lines if 0 <= bbox.y0 - l.bbox.y1 <= 40 and TABLE_CAPTION_RE.match(l.text)]
            cap = above[-1].text if above else ""
            md = table_to_markdown(rows)
            if md:
                out.append({"bbox": bbox, "markdown": md, "caption": cap, "rows": t.row_count, "cols": t.col_count})
        return out

    def _figures(self, page, pno, lines, tables, frames, slide_nums):
        pr = page.rect
        parea = _area(pr)
        figs = []
        infos = page.get_image_info(xrefs=True)
        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []

        def decorative(info):
            return self.xref_pages.get(info["xref"], 0) >= 3 and _area(Rect(info["bbox"]) & pr) < 0.15 * parea

        if frames:  # tai lieu slide: slide nao co hinh/anh/so do -> cat ca slide
            for k, f in enumerate(frames):
                rasters = [Rect(i["bbox"]) & f for i in infos
                           if not decorative(i) and _center_in(Rect(i["bbox"]), f)]
                rasters = [r for r in rasters if _area(r) > 0.02 * _area(f)]
                paths = [d for d in drawings if _center_in(Rect(d["rect"]), f) and _area(Rect(d["rect"])) < 0.8 * _area(f)
                         and Rect(d["rect"]).width > 2 and Rect(d["rect"]).height > 2]
                if sum(_area(r) for r in rasters) >= 0.06 * _area(f) or len(paths) >= 8:
                    fig = Figure(pno, Rect(f), "slide", slide_number=slide_nums.get(k),
                                 n_raster=len(rasters), n_paths=len(paths))
                    fig.text_in_image = " | ".join(l.text for l in lines if _center_in(l.bbox, f))[:1500]
                    figs.append(fig)
            return figs

        cands = []
        for info in infos:
            r = Rect(info["bbox"]) & pr
            if r.is_empty or r.width < 40 or r.height < 25 or _area(r) < 0.006 * parea:
                continue
            if decorative(info):
                continue  # logo, hoa van lap lai nhieu trang
            if _area(r) > 0.85 * parea and self.page_chars[pno - 1] > 200:
                continue  # anh nen ca trang
            cands.append(("raster", r))
        try:
            clusters = page.cluster_drawings(drawings=drawings) if drawings else []
        except Exception:
            clusters = []
        for c in clusters:
            r = Rect(c) & pr
            if r.is_empty or r.width < 40 or r.height < 20 or min(r.width, r.height) < 8:
                continue
            if _area(r) > 0.85 * parea:
                continue
            if any(_iou(r, t["bbox"]) > 0.3 or _inside(r, t["bbox"]) for t in tables):
                continue
            inside_d = [d for d in drawings if _inside(Rect(d["rect"]), r, tol=2)]
            n_items = sum(len(d.get("items", [])) for d in inside_d)
            if len(inside_d) < 5 and n_items < 6:
                continue  # 1 khung, 1 duong ke, dau can/gach phan so cua cong thuc
            if not self._looks_like_figure(r, lines):
                continue
            cands.append(("vector", r))
        if not cands:
            return figs
        merged = _merge_rects([r for _, r in cands], gap=12)
        for r in merged:
            kinds = {k for k, rr in cands if _inside(rr, r, tol=13)}
            kind = "raster" if kinds == {"raster"} else ("vector" if kinds == {"vector"} else "mixed")
            if kind != "raster" and not self._looks_like_figure(r, lines):
                continue  # sau khi gop lai bao ca doan van/cong thuc -> khong phai hinh
            # nhan chu nam vat qua mep hinh: chi nhan dong ngan, phan lon nam trong hinh
            blk = Counter(l.block for l in lines)
            near = Rect(r.x0 - 6, r.y0 - 6, r.x1 + 6, r.y1 + 6)
            core = Rect(r)
            for l in lines:
                if len(l.text) <= 60 and not CAPTION_RE.match(l.text):
                    if _area(l.bbox & r) > 0.5 * _area(l.bbox):
                        r = r | l.bbox
                    elif (len(l.text) <= 40 and blk[l.block] == 1 and l.bbox.intersects(near)
                          and not ARABIC_RE.match(l.text) and not l.text.endswith(":")):
                        centered = abs((l.bbox.x0 + l.bbox.x1) / 2 - (core.x0 + core.x1) / 2) < 0.15 * core.width + 15
                        below = l.bbox.y0 >= core.y1 - 2
                        if not (centered and below):  # dong can giua ngay duoi hinh -> de lam chu thich
                            r = r | l.bbox
            fig = Figure(pno, Rect(r), kind,
                         n_raster=sum(1 for k, rr in cands if k == "raster" and _inside(rr, r, tol=13)),
                         n_paths=sum(1 for d in drawings if _inside(Rect(d["rect"]), r, tol=2)))
            figs.append(fig)
        # chu thich: bo qua dong da nam trong hinh khac (vd nhan "Gnome Shell" cua anh ben tren)
        kept = []
        for fig in figs:
            others = [o.bbox for o in figs if o is not fig]
            cand_lines = [l for l in lines if not any(_center_in(l.bbox, b) for b in others)]
            self._caption(fig, cand_lines)
            if fig.kind != "raster" and not CAPTION_RE.match(fig.caption or ""):
                if self._is_math_region(fig.bbox, lines):
                    continue
                if fig.kind == "vector" and _area(fig.bbox) < 0.025 * parea and fig.n_paths < 10:
                    continue  # manh net ve nho (mui ten tren ky hieu, gach chan) khong co chu thich
            kept.append(fig)
        kept.sort(key=lambda f: (f.bbox.y0, f.bbox.x0))
        return kept

    @staticmethod
    def _is_math_region(r, lines):
        """Ma tran, cong thuc dung dau ngoac ve bang net -> phan lon la so, ky hieu, chu cai don."""
        toks = [t for l in lines if _center_in(l.bbox, r) for t in l.text.split()]
        if len(toks) < 4:
            return False
        mathy = sum(1 for t in toks if len(t) == 1 or re.fullmatch(r"[\d.,\-−+*/=()\[\]|{}<>≤≥±×÷∑∏√∂∇∈⊗⊕'’]+", t)
                    or re.search(r"[=+×⊗∑∏√≤≥]", t))
        return mathy >= 0.6 * len(toks)

    @staticmethod
    def _looks_like_figure(r, lines):
        """Vung co qua nhieu chu (doan van, cong thuc dai, chu duoc to nen) thi khong phai hinh."""
        inside = [l for l in lines if _center_in(l.bbox, r)]
        n_chars = sum(len(l.text) for l in inside)
        if n_chars > 700 or sum(len(l.text) > 110 for l in inside) >= 2 or sum(len(l.text) > 60 for l in inside) >= 4:
            return False
        text_area = sum(_area(l.bbox & r) for l in inside)
        if n_chars >= 250 and text_area >= 0.35 * _area(r):
            return False
        if inside and n_chars >= 60 and n_chars / len(inside) >= 45 and text_area >= 0.5 * _area(r):
            return False  # dong chu dai duoc to nen (highlight)
        return True

    def _caption(self, fig, lines):
        r = fig.bbox
        page_w = self.doc[fig.page - 1].rect.width
        block_lines = Counter(l.block for l in lines)
        # 1) chu thich co so ("Hinh 5.10. ..."): trong hinh, ngay duoi (<=70pt) hoac ngay tren (<=40pt)
        best = None
        for l in lines:
            if not CAPTION_RE.match(l.text) or TABLE_CAPTION_RE.match(l.text) or _overlap_x(l.bbox, r) <= 0.2:
                continue
            if _center_in(l.bbox, r):
                d = 0
            elif -2 <= l.bbox.y0 - r.y1 <= 70:
                d = l.bbox.y0 - r.y1
            elif -2 <= r.y0 - l.bbox.y1 <= 40:
                d = 10 + (r.y0 - l.bbox.y1)
            else:
                continue
            if best is None or d < best[0]:
                best = (d, l)
        if best is not None:
            l = best[1]
            fig.caption, fig.caption_bbox = l.text, Rect(l.bbox)
            if _center_in(l.bbox, r) and l.bbox.y0 > r.y0 + 0.6 * r.height:
                fig.bbox = Rect(r.x0, r.y0, r.x1, l.bbox.y0 - 1)  # cat chu thich ra khoi anh
            elif l.bbox.y0 >= r.y1 - 2:
                # nhan chu nam giua hinh va chu thich (vd "Anh thu nhan", "Anh mong muon")
                for m in lines:
                    if (m is not l and len(m.text) <= 60 and m.bbox.y0 >= r.y1 - 2 and m.bbox.y1 <= l.bbox.y0 + 2
                            and _overlap_x(m.bbox, r) > 0.5):
                        fig.bbox = fig.bbox | m.bbox
            return
        # 2) chu thich khong danh so: 1 dong ngan, can giua, dung rieng 1 khoi, ngay duoi (<=60pt) hoac tren (<=25pt)
        cx = (r.x0 + r.x1) / 2
        cands = []
        for l in lines:
            if len(l.text) > 120 or l.text.endswith((",", ";", ":")) or block_lines[l.block] > 1:
                continue
            if re.match(r"^(\d+(\.\d+)*\.?\s|[IVX]{1,5}\.\s|[a-zđ]\)\s|(Chương|Phần|Mục)\s)", l.text):
                continue  # tieu de muc / buoc danh so, khong phai chu thich hinh
            if l.bbox.width > 0.8 * r.width or l.bbox.width > 0.75 * page_w:
                continue
            vis = [ch for ch in l.text if not ch.isspace()]
            if not vis or sum(ch.isalpha() for ch in vis) < 0.75 * len(vis) or re.search(r"[=≤≥<>+±∑]", l.text):
                continue
            if abs((l.bbox.x0 + l.bbox.x1) / 2 - cx) > 0.15 * r.width + 15:
                continue
            if 0 <= l.bbox.y0 - r.y1 <= 60:
                cands.append((l.bbox.y0 - r.y1, l))
            elif 0 <= r.y0 - l.bbox.y1 <= 25:
                cands.append((5 + r.y0 - l.bbox.y1, l))
        if cands:
            l = min(cands, key=lambda x: x[0])[1]
            fig.caption, fig.caption_bbox = l.text, Rect(l.bbox)

    def _attach_context(self, figures, elements):
        paras = [e for e in elements if e.kind in ("para", "caption", "heading") and e.bbox is not None]
        for f in figures:
            if f.kind == "slide":
                continue
            before = [e for e in paras if e.bbox.y1 <= f.bbox.y0 + 2 and e.text != f.caption]
            after = [e for e in paras if e.bbox.y0 >= f.bbox.y1 - 2 and e.text != f.caption]
            if before:
                f.context_before = before[-1].text[-400:]
            if after:
                f.context_after = after[0].text[:400]

    def _elements_from_lines(self, lines, pno, frame=None):
        """Gom dong -> tieu de / doan van / chu thich / muc liet ke."""
        if not lines:
            return []
        body = self.body_size
        slide = frame is not None
        long_lines = [l for l in lines if len(l.text) > 25] or lines
        col_left = _percentile([l.bbox.x0 for l in long_lines], 0.1)
        col_right = _percentile([l.bbox.x1 for l in long_lines], 0.9)
        col_w = max(1.0, col_right - col_left)
        block_len = defaultdict(int)
        block_allbold = defaultdict(lambda: True)
        for l in lines:
            block_len[l.block] += len(l.text)
            block_allbold[l.block] &= l.bold >= 0.6

        title_ids = set()
        if slide:
            # tieu de slide = (cac) dong tren cung cua khung, chu to hon han phan than slide hoac dong dam dung rieng.
            # Slide noi tiep khong co tieu de (dong dau la gach dau dong / chu thuong) -> khong lay tieu de.
            by_y = sorted(lines, key=lambda l: l.bbox.y0)
            sizes = sorted(s for l in by_y[1:] for s in [l.size] * max(1, len(l.text)))  # co chu phan than slide
            med = sizes[len(sizes) // 2] if sizes else 0.0
            first, last_y1, big = None, None, False
            for l in by_y:
                t = l.text
                ok = (l.bbox.y0 <= frame.y0 + 0.3 * frame.height and len(t) <= 90
                      and t[0] not in STRONG_BULLETS and not re.match(r"^[\-–—+*]\s", t))
                if ok and first is None:
                    big = l.size >= med + 1.0
                    ok = big or (l.bold >= 0.6 and block_len[l.block] <= 90)
                elif ok:  # dong thu 2 cua tieu de: cung co chu, sat ngay ben duoi, cung kieu dam
                    ok = (abs(l.size - first.size) <= 0.6 and l.bbox.y0 - last_y1 <= 1.2 * l.size
                          and (big or l.bold >= 0.6))
                if not ok:
                    break
                first = first or l
                last_y1 = l.bbox.y1
                title_ids.add(id(l))

        out, cur, prev = [], None, None
        for l in lines:
            t = l.text
            if TOC_RE.search(t) or is_formula_fragment(t):
                continue
            is_cap = bool(CAPTION_RE.match(t))
            strong_bul = t[0] in STRONG_BULLETS
            dash_bul = bool(re.match(r"^[\-–—+*]\s+\S", t))
            prev_full = prev is not None and prev.bbox.x1 >= col_right - 0.12 * col_w
            if slide:
                is_head = id(l) in title_ids and (t[0].isupper() or t[0].isdigit())
            else:
                is_head = (not is_cap and not strong_bul and not dash_bul and len(t) <= 160
                           and not (block_allbold[l.block] and block_len[l.block] > 220)
                           and ((l.bold >= 0.6 and len(t) <= 130) or l.size >= body + 1.5)
                           and not t.endswith((",", ";")) and len(re.sub(r"\W", "", t)) >= 2
                           and not (prev_full and cur is not None and cur.kind == "para"
                                    and 0 <= l.bbox.y0 - prev.bbox.y1 <= 0.6 * l.size and l.bold < 0.6)
                           and (t[0].isupper() or t[0].isdigit() or bool(LETTER_RE.match(t)) or t[0] in "\"“(")
                           and not (cur is not None and cur.kind == "para" and prev is not None
                                    and not prev.text.endswith(SENT_END)
                                    and 0 <= l.bbox.y0 - prev.bbox.y1 <= 0.6 * l.size))
            # dau gach "–" o dau dong co the chi la dau cau noi tiep dong truoc
            if dash_bul and prev is not None and prev_full and not prev.text.endswith(SENT_END) and cur is not None \
                    and cur.kind == "para":
                dash_bul = False
            is_bul = strong_bul or dash_bul
            if is_head:
                lv = max(heading_level(t, l.size, body), 5) if slide else heading_level(t, l.size, body)
                numbered = bool(PART_RE.match(t) or CHAPTER_RE.match(t) or ROMAN_RE.match(t)
                                or ARABIC_RE.match(t) or LETTER_RE.match(t))
                if (cur is not None and cur.kind == "heading" and prev is not None
                        and abs(l.bbox.y0 - prev.bbox.y1) < 1.6 * max(l.size, prev.size)
                        and (cur.level == lv
                             or (cur.level in (1, 2) and len(cur.text) <= 16 and not numbered)
                             or (not numbered and (cur.text.endswith(("-", "–")) or prev.bbox.width >= 0.6 * col_w)))
                        and not cur.text.endswith((".", "?", "!"))):
                    cur.text = f"{cur.text} {t}".strip()  # tieu de 2 dong / "Chuong 1:" + ten chuong
                    cur.bbox = cur.bbox | l.bbox
                else:
                    cur = Element("heading", t, pno, Rect(l.bbox), level=lv)
                    out.append(cur)
                prev = l
                continue
            if is_cap:
                cur = Element("caption", t, pno, Rect(l.bbox))
                out.append(cur)
                prev = l
                continue
            cont = False
            if cur is not None and cur.kind == "para" and prev is not None and not is_bul:
                gap = l.bbox.y0 - prev.bbox.y1
                if l.block == prev.block:
                    cont = True
                elif -3 <= gap <= 0.9 * max(l.size, prev.size):
                    indented = l.bbox.x0 > col_left + 0.04 * col_w + 6 and not cur.bullet
                    cont = (prev_full or not prev.text.endswith(SENT_END)) and not (indented and not prev_full)
            if cont:
                joiner = "" if cur.text.endswith("-") and not cur.text.endswith(" -") else " "
                cur.text = f"{cur.text}{joiner}{t}"
                cur.bbox = cur.bbox | l.bbox
            else:
                text = re.sub(r"^[" + re.escape(STRONG_BULLETS) + r"\-–—+*]\s*", "", t) if is_bul else t
                if text:
                    cur = Element("para", text, pno, Rect(l.bbox), bullet=is_bul)
                    out.append(cur)
            prev = l
        return out

    # --- anh -------------------------------------------------------------------
    def render_crop(self, fig, out_path, dpi=150, margin=4, max_pixels=1.3e6):
        page = self.doc[fig.page - 1]
        b = fig.bbox
        r = Rect(b.x0 - margin, b.y0 - margin, b.x1 + margin, b.y1 + margin) & page.rect
        for a in getattr(fig, "avoid", []) or []:  # khong cat lem dong chu ngay sat hinh
            cx, cy = (a.x0 + a.x1) / 2, (a.y0 + a.y1) / 2  # dong chu nam phia nao cua hinh (theo tam)
            if cy <= b.y0:
                r.y0 = max(r.y0, min(b.y0, a.y1 + 0.5))
            elif cy >= b.y1:
                r.y1 = min(r.y1, max(b.y1, a.y0 - 0.5))
            elif cx <= b.x0:
                r.x0 = max(r.x0, min(b.x0, a.x1 + 0.5))
            elif cx >= b.x1:
                r.x1 = min(r.x1, max(b.x1, a.x0 - 0.5))
        # hinh nho (vd slide in 4 trang/to) -> phong to toi thieu ~1000 diem anh canh dai (toi da 300 dpi)
        scale = max(dpi / 72, min(300 / 72, 1000 / max(1.0, r.width, r.height)))
        if r.width * r.height * scale * scale > max_pixels:  # Qwen2-VL tu thu nho ve ~0.8 trieu diem anh
            scale = (max_pixels / (r.width * r.height)) ** 0.5
        pix = page.get_pixmap(clip=r, matrix=pymupdf.Matrix(scale, scale), alpha=False)
        pix.save(out_path)
        return pix.width, pix.height

    def render_page(self, pno, dpi=200):
        return self.doc[pno - 1].get_pixmap(dpi=dpi, alpha=False)

    def page_image_cover(self, pno):
        """Ti le dien tich trang bi anh raster che (de nhan biet trang scan)."""
        page = self.doc[pno - 1]
        pr = page.rect
        rects = [Rect(i["bbox"]) & pr for i in page.get_image_info()]
        rects = [r for r in rects if not r.is_empty]
        if not rects:
            return 0.0
        return min(1.0, sum(_area(r) for r in _merge_rects(rects, gap=0)) / _area(pr))
