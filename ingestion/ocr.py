# -*- coding: utf-8 -*-
"""
ocr.py - Giai doan 3: nhanh OCR tieng Viet, tach biet voi nhanh PyMuPDF.

Dung cho:
  1) Trang scan / anh chup (khong co lop chu) va trang co lop chu bi loi font -> doc lai bang OCR.
  2) Chu nam trong anh raster (anh chup man hinh, bieu do) -> them "ocr_text" cho hinh,
     giup tim kiem anh bang noi dung chu trong anh.

Engine:
  paddle     PaddleOCR 3.x - engine chinh (muc 3.2 / 6.2 bao cao giua ky)
             pip install paddlepaddle paddleocr
             lang="vi" -> PP-OCRv6 (phat hien dong + nhan dang); lan dau tu tai mo hinh (can Internet).
  tesseract  Tesseract 5 + goi "vie" - du phong khi chua cai duoc PaddleOCR
             pip install pytesseract, cai Tesseract (Windows: ban UB Mannheim, tick Vietnamese).

    engine = get_engine("auto")           # paddle neu co, khong thi tesseract, khong co gi -> None
    lines = engine.read(pil_image)       # [OcrLine(text, box=(x0, y0, x1, y1) theo pixel, score)]
    paras = lines_to_paragraphs(lines)   # [OcrLine] da gom thanh doan
"""
import os
import re
import shutil
import statistics
import tempfile
import unicodedata
from dataclasses import dataclass


@dataclass
class OcrLine:
    text: str
    box: tuple          # (x0, y0, x1, y1) pixel
    score: float = 1.0


def _nfc(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "")).strip()


class PaddleEngine:
    name = "paddleocr"

    def __init__(self, lang="vi", ocr_version=None, device=None):
        from paddleocr import PaddleOCR  # import cham: chi khi can

        kw = dict(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False,
                  use_textline_orientation=False)
        if ocr_version:
            kw["ocr_version"] = ocr_version
        if device:
            kw["device"] = device
        self.ocr = PaddleOCR(**kw)
        self.detail = f"PaddleOCR lang={lang}" + (f" {ocr_version}" if ocr_version else "")

    @staticmethod
    def _get(res, key):
        try:
            return res[key]
        except Exception:
            pass
        try:
            return res.json["res"][key]
        except Exception:
            return None

    def read(self, img):
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            img.convert("RGB").save(path)
            results = self.ocr.predict(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        out = []
        for res in results or []:
            texts = self._get(res, "rec_texts") or []
            scores = self._get(res, "rec_scores")
            boxes = self._get(res, "rec_boxes")
            polys = self._get(res, "rec_polys")
            for i, t in enumerate(texts):
                t = _nfc(t)
                if not t:
                    continue
                if boxes is not None and len(boxes) > i:
                    x0, y0, x1, y1 = [float(v) for v in boxes[i]][:4]
                elif polys is not None and len(polys) > i:
                    xs = [float(p[0]) for p in polys[i]]
                    ys = [float(p[1]) for p in polys[i]]
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                else:
                    x0 = y0 = x1 = y1 = 0.0
                sc = float(scores[i]) if scores is not None and len(scores) > i else 1.0
                out.append(OcrLine(t, (x0, y0, x1, y1), sc))
        return out


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang="vie", psm=3, cmd=None):
        import pytesseract

        cmd = cmd or os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
        if not cmd and os.name == "nt":
            for c in (r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
                if os.path.exists(c):
                    cmd = c
                    break
        if not cmd:
            raise RuntimeError("khong tim thay chuong trinh tesseract")
        pytesseract.pytesseract.tesseract_cmd = cmd
        langs = pytesseract.get_languages(config="")
        if lang not in langs:
            raise RuntimeError(f"Tesseract chua co goi ngon ngu '{lang}' (dang co: {', '.join(langs)})")
        self.pt, self.lang, self.psm = pytesseract, lang, psm
        self.detail = f"Tesseract {pytesseract.get_tesseract_version()} lang={lang}"

    def read(self, img):
        d = self.pt.image_to_data(img.convert("RGB"), lang=self.lang, config=f"--oem 1 --psm {self.psm}",
                                  output_type=self.pt.Output.DICT)
        groups = {}
        for i, w in enumerate(d["text"]):
            if not str(w).strip() or float(d["conf"][i]) < 0:
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            g = groups.setdefault(key, {"w": [], "box": [1e9, 1e9, 0, 0], "conf": []})
            g["w"].append(str(w))
            x, y, ww, hh = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            b = g["box"]
            g["box"] = [min(b[0], x), min(b[1], y), max(b[2], x + ww), max(b[3], y + hh)]
            g["conf"].append(float(d["conf"][i]))
        out = []
        for key in sorted(groups):
            g = groups[key]
            out.append(OcrLine(_nfc(" ".join(g["w"])), tuple(g["box"]), sum(g["conf"]) / len(g["conf"]) / 100))
        return out


def get_engine(name="auto", lang_paddle="vi", lang_tess="vie", device=None, verbose=True):
    """Tra ve engine OCR hoac None (kem ly do in ra man hinh)."""
    errors = []
    order = ["paddle", "tesseract"] if name == "auto" else [name]
    for n in order:
        try:
            if n == "paddle":
                return PaddleEngine(lang=lang_paddle, device=device)
            if n == "tesseract":
                return TesseractEngine(lang=lang_tess)
            errors.append(f"{n}: khong ho tro")
        except Exception as e:  # thieu thu vien / mo hinh
            errors.append(f"{n}: {type(e).__name__}: {str(e)[:200]}")
    if verbose:
        print("  !! Khong khoi tao duoc OCR -> " + " | ".join(errors))
    return None


def sort_reading_order(lines):
    """Sap xep tren-xuong, trai-phai; cac dong co tam doc gan nhau coi la cung hang."""
    if not lines:
        return []
    hs = [max(1.0, l.box[3] - l.box[1]) for l in lines]
    tol = 0.5 * statistics.median(hs)
    rows = []
    for l in sorted(lines, key=lambda l: (l.box[1] + l.box[3]) / 2):
        cy = (l.box[1] + l.box[3]) / 2
        if rows and abs(rows[-1][0] - cy) <= tol:
            rows[-1][1].append(l)
        else:
            rows.append([cy, [l]])
    out = []
    for _, row in rows:
        out += sorted(row, key=lambda l: l.box[0])
    return out


def lines_to_paragraphs(lines, min_score=0.5):
    """Gom dong OCR thanh doan van theo khoang cach dong va le trai."""
    lines = [l for l in sort_reading_order(lines) if l.score >= min_score and l.text]
    if not lines:
        return []
    h_med = statistics.median(max(1.0, l.box[3] - l.box[1]) for l in lines)
    x_right = max(l.box[2] for l in lines)
    paras = []
    for l in lines:
        if paras:
            p = paras[-1]
            last = p["last"]
            gap = l.box[1] - last.box[3]
            same_col = abs(l.box[0] - p["box"][0]) < 3 * h_med or last.box[2] > x_right - 2 * h_med
            if -0.3 * h_med <= gap <= 0.8 * h_med and same_col and abs((l.box[3] - l.box[1]) - h_med) < 0.6 * h_med:
                p["texts"].append(l.text)
                b = p["box"]
                p["box"] = (min(b[0], l.box[0]), min(b[1], l.box[1]), max(b[2], l.box[2]), max(b[3], l.box[3]))
                p["last"] = l
                p["scores"].append(l.score)
                continue
        paras.append({"texts": [l.text], "box": tuple(l.box), "last": l, "scores": [l.score],
                      "height": l.box[3] - l.box[1]})
    out = []
    for p in paras:
        o = OcrLine(" ".join(p["texts"]), p["box"], sum(p["scores"]) / len(p["scores"]))
        o.n_lines = len(p["texts"])
        o.line_height = p["height"]
        out.append(o)
    out_h = statistics.median([o.line_height for o in out]) if out else h_med
    for o in out:
        o.is_heading = o.n_lines == 1 and o.line_height >= 1.3 * out_h and len(o.text) <= 100
    return out
