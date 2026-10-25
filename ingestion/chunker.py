# -*- coding: utf-8 -*-
"""
chunker.py - Giai doan 3: chia van ban thanh chunk giu cau truc tai lieu.

Nguyen tac:
  - Cat theo tieu de (muc, tieu muc): chunk khong vat qua 2 muc khac nhau, tru khi muc qua ngan
    (< min_chars) thi gop voi muc ke tiep cho du y.
  - Muc dai thi cat theo doan van, doan qua dai thi cat theo cau; giua 2 chunk lien tiep cua cung
    muc lap lai ~overlap ky tu (cau cuoi chunk truoc) de khong mat y o ranh gioi.
  - Bang la chunk rieng (markdown), bang dai cat theo hang, lap lai dong tieu de.
  - Tai lieu dang slide: moi slide 1 chunk.
  - max_chars mac dinh 1200 ky tu: bang dung do dai "Ngữ cảnh" luc fine-tune (Buoc 2.4), nen
    chunk tra ve tu Qdrant co cung kich thuoc voi ngu canh mo hinh da hoc.

Moi chunk: {chunk_id, doc_id, type, text, section, page_number, pages, slide_number, source, n_chars}
"""
import re

SENT_SPLIT = re.compile(r"(?<=[.!?…;])\s+(?=\S)")
NUMBERED_HEADING = re.compile(r"^(Phần|PHẦN|Chương|CHƯƠNG|Bài|BÀI|[IVX]{1,5}\.|\d+(\.\d+)*\.?)\s")


def _split_long(text, max_chars):
    """Cat 1 doan qua dai thanh cac phan <= max_chars theo cau (cau qua dai thi theo dau phay)."""
    sents = []
    for s in SENT_SPLIT.split(text):
        while len(s) > max_chars:
            cut = s.rfind(", ", 0, max_chars)
            if cut < max_chars // 3:
                cut = s.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            sents.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            sents.append(s)
    parts, cur = [], ""
    for s in sents:
        if cur and len(cur) + 1 + len(s) > max_chars:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def _tail(text, n):
    """Lay cac cau cuoi cua text, tong <= n ky tu (de lam phan lap giua 2 chunk)."""
    if n <= 0:
        return ""
    sents = SENT_SPLIT.split(text.replace("\n", " "))
    out = ""
    for s in reversed(sents):
        if len(out) + len(s) + 1 > n:
            break
        out = f"{s} {out}".strip()
    return out


def _table_pieces(md, caption, max_chars):
    lines = md.split("\n")
    head, rows = lines[:2], lines[2:]
    prefix = (caption + "\n") if caption else ""
    pieces, cur = [], []
    for r in rows:
        cand = prefix + "\n".join(head + cur + [r])
        if cur and len(cand) > max_chars:
            pieces.append(prefix + "\n".join(head + cur))
            cur = [r]
        else:
            cur.append(r)
    if cur or not pieces:
        pieces.append(prefix + "\n".join(head + cur))
    return pieces


class Chunker:
    def __init__(self, doc_id, max_chars=1200, min_chars=300, overlap=150):
        self.doc_id, self.max_chars, self.min_chars, self.overlap = doc_id, max_chars, min_chars, overlap
        self.chunks = []
        self.stack = []        # [(level, text)]
        self._reset()
        self.slide = None

    def _reset(self):
        self.parts, self.pages, self.sources = [], set(), set()
        self.section = None
        self.has_content = False
        self.overlap_only = False

    def _len(self):
        return sum(len(p) + 1 for p in self.parts)

    def path(self):
        names = [re.sub(r"\s+", " ", t)[:100] for _, t in self.stack]
        p = " > ".join(names)
        return p if len(p) <= 300 else "… > " + " > ".join(names[-2:])

    def _emit(self, text, pages, ctype="text", sources=None, section=None):
        text = text.strip()
        if not text:
            return
        pages = sorted(pages)
        self.chunks.append({
            "chunk_id": f"{self.doc_id}_c{len(self.chunks):04d}",
            "doc_id": self.doc_id,
            "type": ctype,
            "text": text,
            "section": section if section is not None else self.path(),
            "page_number": pages[0] if pages else None,
            "pages": pages,
            "slide_number": self.slide,
            "source": "+".join(sorted(sources or {"pdf"})),
            "n_chars": len(text),
        })

    def flush(self, keep_overlap=False):
        if self.parts and not self.overlap_only:
            text = "\n".join(self.parts)
            self._emit(text, self.pages, sources=self.sources, section=self.section)
            tail = _tail(self.parts[-1], self.overlap) if keep_overlap else ""
            last_page = max(self.pages) if self.pages else None
            self._reset()
            if tail:
                self.parts = [tail]
                self.pages = {last_page} if last_page else set()
                self.section = self.path()
                self.has_content = True
                self.overlap_only = True
        else:
            self._reset()

    def _add(self, text, page, source, is_heading=False):
        if is_heading:
            if not self.has_content:  # chunk bat dau bang (cac) tieu de -> lay muc sau cung
                self.section = self.path()
        else:
            if self.section is None:
                self.section = self.path()
            self.has_content = True
        self.parts.append(text)
        if page:
            self.pages.add(page)
        self.sources.add(source)
        self.overlap_only = False

    def add(self, el):
        kind = el.kind
        page = el.page
        source = el.extra.get("source", "pdf") if el.extra else "pdf"
        if kind == "slide":
            self.flush()
            # giu tieu de danh so ("1.1.1 Lich su ...") cho cac slide noi tiep khong co tieu de;
            # tieu de khong danh so chi thuoc ve slide cua no
            self.stack = [(lv, t) for lv, t in self.stack if NUMBERED_HEADING.match(t)]
            self.slide = el.extra.get("slide_number")
            return
        if kind == "heading":
            if self._len() >= self.min_chars or self.overlap_only:
                self.flush()
            level = el.level
            if self.slide is not None and not NUMBERED_HEADING.match(el.text):
                level = max(level, 9)  # tieu de slide khong danh so = muc con cua tieu de danh so gan nhat
            while self.stack and self.stack[-1][0] >= level:
                self.stack.pop()
            self.stack.append((level, el.text))
            self._add(el.text, page, source, is_heading=True)
            return
        if kind == "table":
            self.flush()
            cap = el.extra.get("caption", "")
            for piece in _table_pieces(el.text, cap, int(self.max_chars * 1.5)):
                self._emit(piece, {page}, ctype="table", sources={source})
            return
        text = ("- " + el.text) if el.bullet else el.text
        if len(text) > self.max_chars:
            for piece in _split_long(text, self.max_chars):
                self._append_text(piece, page, source)
            return
        self._append_text(text, page, source)

    def _append_text(self, text, page, source):
        if self.parts and self._len() + len(text) > self.max_chars and self._len() >= self.min_chars:
            self.flush(keep_overlap=True)
        elif self.parts and self._len() + len(text) > self.max_chars * 1.25:
            self.flush(keep_overlap=True)
        self._add(text, page, source)

    def finish(self):
        self.flush()
        return self._merge_tiny(self.chunks)

    def _merge_tiny(self, chunks, tiny=120):
        """Chunk qua ngan (vd 1 tieu de cuoi trang) -> gop vao chunk ke tiep cung slide."""
        out = []
        i = 0
        while i < len(chunks):
            c = chunks[i]
            if (c["type"] == "text" and c["n_chars"] < tiny and i + 1 < len(chunks)
                    and chunks[i + 1]["type"] == "text" and chunks[i + 1]["slide_number"] == c["slide_number"]
                    and (c["slide_number"] is None)):
                n = chunks[i + 1]
                n["text"] = c["text"] + "\n" + n["text"]
                n["pages"] = sorted(set(c["pages"]) | set(n["pages"]))
                n["page_number"] = n["pages"][0]
                n["n_chars"] = len(n["text"])
                i += 1
                continue
            out.append(c)
            i += 1
        for k, c in enumerate(out):
            c["chunk_id"] = f"{c['doc_id']}_c{k:04d}"
        return out
