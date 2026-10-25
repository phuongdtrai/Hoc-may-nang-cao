# -*- coding: utf-8 -*-
"""
build_training_data.py - Buoc 2.4 (ban 4): chuan hoa du lieu train cho LLM/VLM

Chay tu thu muc goc cua repo:
    pip install pymupdf pillow
    python ingestion/ingest.py          # Giai doan 3: cat hinh ra data/processed/ingest/images/
    python build_training_data.py       # tu tao anh hoi (ingestion/tao_anh_hoi.py) cho cau image_query neu thieu

Dau vao
  data/raw/domain_docs/qa_thu_cong.csv        CSV xuat tu sheet QA_Data (bat buoc). Cot:
                                              STT, doc_id, page, question, answer, loai, nguoi_soan, ghi_chu,
                                              image_id, tap
      loai = text         dap an dua vao doan van ban -> llm_*.jsonl
      loai = image        cau hoi bang chu, dap an la 1 hinh trong tai lieu:
                            - co image_id: anh = hinh cat images/<image_id>.png, "Ngữ cảnh" = trang/muc/chu thich
                            - khong co image_id (7 cau vong 1): anh = ca trang PDF, luon vao train
      loai = image_query  cau hoi la anh: nguoi dung gui anh chup 1 hinh trong tai lieu roi hoi ve no;
                          anh = anh_hoi/<image_id>.jpg (hinh cat bien doi giong anh chup),
                          "Ngữ cảnh" = trang/muc/chu thich cua hinh + doan van ban cung trang
      tap = train / val   (cau image, image_query) de trong thi script tu chia theo trang
  data/processed/ingest/images.jsonl + images/  hinh do ingestion/ingest.py cat ra (can cho cau co image_id)
  data/processed/synthetic_qa_filtered.json   QA tong hop da loc o Buoc 2.3 (tuy chon)
  data/raw/domain_docs/<doc_id>.pdf           PDF goc, dat ten dung theo doc_id (tuy chon):
                                              dung de gan "Ngữ cảnh" cho cau text / image_query
                                              va render anh trang cho cau image khong co image_id
  data/raw/domain_docs/qa_anh.csv             (cu, tab QA_Anh da bo) chi doc khi QA_Data chua co cot image_id

Dau ra (data/processed/)
  llm_train.jsonl, llm_val.jsonl              {"instruction", "input", "output", "meta"}
  vlm_train.jsonl                             {"image", "instruction", "input", "question", "answer", "meta"}:
                                              anh ca trang (vong 1) + phan train cua cau image va image_query
  vlm_val.jsonl                               phan val cua cau image va image_query - KHONG dung khi train
  qa_anh_truy_hoi.jsonl                       cau image: cau hoi chu -> image_id (do tim anh, Giai doan 5)
  anh_hoi_truy_hoi.jsonl                      cau image_query: anh hoi -> image_id / doc_id / trang (do tim bang anh)
  vlm_pending_images.json                     cau anh chua co PDF / chua co hinh cat
  images/<doc_id>_p<trang>.png                anh trang PDF dung cho VLM
  bao_cao_du_lieu.txt                         thong ke + danh sach cau nen soat lai

Quy uoc "page": so trang cua file PDF, dem tu 1 (trang dau tien cua file = 1).
Trich nhieu trang ghi "71,72". Neu Google Sheets da lo bien "7,8" thanh so thap phan
(7.8, 19.2 ...) thi script tu doc lai thanh cac trang 7-8, 19-20.
"""
import csv
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "finetune"))
sys.path.insert(0, os.path.join(_ROOT, "ingestion"))
import prompt_format as pf  # noqa: E402  (hop dong prompt dung chung: INSTR_IMAGE_CTX, image_context)

try:
    import tao_anh_hoi  # noqa: E402  (tao anh hoi cho cau image_query)
except ImportError:
    tao_anh_hoi = None

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:
    try:
        import fitz  # PyMuPDF ban cu
    except ImportError:
        fitz = None

# ------------------------------------------------------------------ cau hinh
SEED = 42
VAL_RATIO = 0.10            # 90/10 train/validation
MAX_CTX_CHARS = 1200        # do dai toi da cua "Ngữ cảnh" (ky tu), ~500-700 token
LOW_COVERAGE = 0.35         # duoi nguong nay -> ngu canh it khop dap an, nen soat lai so trang
IMG_DPI = 120               # do phan giai anh trang cho VLM
SYNTHETIC_PAGE_0_BASED = True   # generate_synthetic_qa.py (Buoc 2.3) ghi trang dem tu 0

VAL_RATIO_ANH = 0.20        # cau image / image_query chua ghi cot "tap" (va trang khong co cau text) chia 80/20 theo trang
LOAI_HOP_LE = ("text", "image", "image_query")

CSV_PATH = "data/raw/domain_docs/qa_thu_cong.csv"
QA_ANH_PATH = "data/raw/domain_docs/qa_anh.csv"     # cu: tab QA_Anh (da gop vao QA_Data)
INGEST_DIR = "data/processed/ingest"       # dau ra cua ingestion/ingest.py (Giai doan 3)
SYN_PATH = "data/processed/synthetic_qa_filtered.json"
PDF_DIR = "data/raw/domain_docs"
OUT_DIR = "data/processed"

INSTR_CTX = pf.INSTR_CTX            # "Trả lời câu hỏi dựa trên ngữ cảnh sau."
INSTR_NOCTX = pf.INSTR_NOCTX        # "Trả lời câu hỏi sau."
INSTR_IMAGE_CTX = pf.INSTR_IMAGE_CTX  # "Trả lời câu hỏi dựa trên hình ảnh và ngữ cảnh sau."
FIG_NUM = re.compile(r"\b(?:Hình|Fig\.?|Figure)\s+(\d+(?:\.\d+)*)", re.IGNORECASE)

STOPWORDS = set("""
là của và các những được có trong cho với không một này đó thì mà để như theo từ về khi ra
nào gì sao thế hay hoặc cũng đã sẽ đang rất nhiều phải bị do vì nên nếu lại còn chỉ tại trên
dưới ai đâu người cái việc đến đi vào cùng nhau ở hơn sự gồm những
""".split())

IMPERATIVE = re.compile(r"^(nêu|trình bày|phân biệt|liệt kê|kể|giải thích|cho ví dụ|cho xem|so sánh|hãy|mô tả|tóm tắt)\b",
                        re.IGNORECASE)


# ------------------------------------------------------------------ tien ich van ban
def norm_space(s):
    s = unicodedata.normalize("NFC", s or "")
    s = s.replace(" ", " ")
    s = re.sub(r"[ \t\r\n]+", " ", s)
    return s.strip()


def words(s):
    return re.findall(r"\w+", s.lower())


def content_unigrams(s):
    return {w for w in words(s) if w not in STOPWORDS}


def bigrams(s):
    w = words(s)
    return {(a, b) for a, b in zip(w, w[1:]) if not (a in STOPWORDS and b in STOPWORDS)}


def question_key(q):
    return " ".join(words(q))


def parse_pages(raw):
    """'71' -> [71]; '71,72' -> [71, 72]; '12-14' -> [12, 13, 14]; '19,2' (Sheets lam mat so 0) -> [19, 20]."""
    s = norm_space(str(raw))
    if not s:
        return []
    m = re.fullmatch(r"(\d+)\s*[-–]\s*(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a <= b <= a + 10:
            return list(range(a, b + 1))
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if len(nums) == 2 and re.fullmatch(r"\d+[.,]\d+", s) and nums[1] < nums[0]:
        b = nums[1]
        while 0 < b < nums[0]:
            b *= 10
        nums[1] = b
    out = []
    for n in nums:
        if n not in out:
            out.append(n)
    return out


# ------------------------------------------------------------------ doc PDF
def clean_page_text(txt):
    lines = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln or re.fullmatch(r"\d{1,4}", ln):  # bo dong trong va dong chi co so trang
            continue
        lines.append(ln)
    s = " ".join(lines)
    s = re.sub(r"([\"”»])\d{1,2}\b", r"\1", s)  # bo so chu thich dinh sau dau ngoac kep
    return norm_space(s)


class PdfLibrary:
    """Tim PDF theo doc_id trong PDF_DIR (khong phan biet hoa thuong), cache text tung trang."""

    def __init__(self, pdf_dir):
        self.paths = {}
        if fitz is not None and os.path.isdir(pdf_dir):
            for fn in os.listdir(pdf_dir):
                if fn.lower().endswith(".pdf"):
                    self.paths[os.path.splitext(fn)[0].lower()] = os.path.join(pdf_dir, fn)
        self._docs, self._text = {}, {}

    def has(self, doc_id):
        return doc_id.lower() in self.paths

    def _doc(self, doc_id):
        key = doc_id.lower()
        if key not in self._docs:
            self._docs[key] = fitz.open(self.paths[key])
        return self._docs[key]

    def n_pages(self, doc_id):
        return len(self._doc(doc_id))

    def page_text(self, doc_id, page):
        key = (doc_id.lower(), page)
        if key not in self._text:
            d = self._doc(doc_id)
            self._text[key] = clean_page_text(d[page - 1].get_text("text")) if 1 <= page <= len(d) else ""
        return self._text[key]

    def render(self, doc_id, page, out_path):
        if not os.path.exists(out_path):
            self._doc(doc_id)[page - 1].get_pixmap(dpi=IMG_DPI).save(out_path)


# ------------------------------------------------------------------ chon ngu canh
def split_units(text):
    units = []
    for u in re.split(r"(?<=[.!?…;])\s+", text):
        u = u.strip()
        if not u:
            continue
        while len(u) > MAX_CTX_CHARS // 2:  # cau qua dai -> cat theo dau phay
            cut = u.rfind(", ", 0, MAX_CTX_CHARS // 2)
            if cut <= 0:
                cut = MAX_CTX_CHARS // 2
            units.append(u[:cut + 1].strip())
            u = u[cut + 1:].strip()
        if u:
            units.append(u)
    return units


def pick_context(page_text, question, answer):
    """Chon doan lien tuc <= MAX_CTX_CHARS khop dap an nhat (giong mot chunk RAG tra ve)."""
    units = split_units(page_text)
    if not units:
        return "", 0.0
    a_uni, a_bi, q_uni = content_unigrams(answer), bigrams(answer), content_unigrams(question)
    feats = [(len(u) + 1, content_unigrams(u), bigrams(u)) for u in units]
    best = (-1.0, 0, 0)
    for i in range(len(units)):
        length, uni, bi = 0, set(), set()
        for j in range(i, len(units)):
            length += feats[j][0]
            if length > MAX_CTX_CHARS and j > i:
                break
            uni |= feats[j][1]
            bi |= feats[j][2]
            score = 2 * len(a_bi & bi) + len(a_uni & uni) + 0.5 * len(q_uni & uni)
            if score > best[0]:
                best = (score, i, j)
    _, i, j = best
    length = sum(f[0] for f in feats[i:j + 1])
    while True:  # mo rong 2 phia cho du kich thuoc chunk
        grown = False
        if i > 0 and length + feats[i - 1][0] <= MAX_CTX_CHARS:
            i -= 1
            length += feats[i][0]
            grown = True
        if j + 1 < len(units) and length + feats[j + 1][0] <= MAX_CTX_CHARS:
            j += 1
            length += feats[j][0]
            grown = True
        if not grown:
            break
    ctx = " ".join(units[i:j + 1])
    if len(ctx) > MAX_CTX_CHARS:
        ctx = ctx[:MAX_CTX_CHARS].rsplit(" ", 1)[0] + " …"
    coverage = len(a_uni & content_unigrams(ctx)) / max(1, len(a_uni))
    return ctx, coverage


# ------------------------------------------------------------------ doc du lieu vao
def load_manual_csv(path):
    items, skipped = [], []
    if not os.path.exists(path):
        print(f"  (Chua co {path} - hay xuat CSV tu sheet QA_Data truoc)")
        return items, skipped
    with open(path, encoding="utf-8-sig", newline="") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            stt = norm_space(row.get("STT"))
            q, a = norm_space(row.get("question")), norm_space(row.get("answer"))
            if stt.upper() in ("VÍ DỤ", "VI DU") or not q or not a:
                skipped.append((line_no, stt, "dong vi du hoac thieu question/answer"))
                continue
            page_raw = norm_space(row.get("page"))
            loai = (norm_space(row.get("loai")) or "text").lower().replace(" ", "_")
            doc_id = norm_space(row.get("doc_id"))
            if doc_id.lower().endswith(".pdf"):  # ghi nham ten file -> bo duoi .pdf
                doc_id = doc_id[:-4]
            items.append({
                "source": "thu_cong", "line": line_no, "stt": stt,
                "doc_id": doc_id, "page_raw": page_raw,
                "pages": parse_pages(page_raw),
                "question": q, "answer": a,
                "loai": loai if loai in LOAI_HOP_LE else "text", "loai_goc": loai,
                "image_id": norm_space(row.get("image_id")),
                "tap": norm_space(row.get("tap")).lower(),
                "nguoi_soan": norm_space(row.get("nguoi_soan")),
                "ghi_chu": norm_space(row.get("ghi_chu")),
                "context": "",
            })
    return items, skipped


def load_synthetic_json(path):
    items = []
    if not os.path.exists(path):
        print(f"  (Chua co {path} - bo qua QA tong hop)")
        return items
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for k, row in enumerate(data):
        q, a = norm_space(row.get("question")), norm_space(row.get("answer"))
        if not q or not a:
            continue
        doc_id = os.path.splitext(norm_space(row.get("doc_id")))[0]  # "TRIET_001.pdf" -> "TRIET_001"
        pages = parse_pages(row.get("page", ""))
        if SYNTHETIC_PAGE_0_BASED:
            pages = [p + 1 for p in pages]
        items.append({
            "source": "tong_hop", "line": k, "stt": "",
            "doc_id": doc_id, "page_raw": ",".join(map(str, pages)), "pages": pages,
            "question": q, "answer": a, "loai": "text", "loai_goc": "text", "image_id": "", "tap": "",
            "nguoi_soan": "", "ghi_chu": "", "context": norm_space(row.get("context")),
        })
    return items


def load_qa_anh(path):
    """qa_anh.csv (sheet QA_Anh): moi dong la 1 cau hoi co dap an la 1 hinh (image_id) + mo ta bang chu."""
    items, skipped = [], []
    if not os.path.exists(path):
        print(f"  (Chua co {path} - bo qua cau hoi co dap an la hinh)")
        return items, skipped
    with open(path, encoding="utf-8-sig", newline="") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            stt = norm_space(row.get("STT"))
            image_id = norm_space(row.get("image_id"))
            q, a = norm_space(row.get("question")), norm_space(row.get("answer"))
            status = norm_space(row.get("trang_thai")).lower()
            if not image_id or not q or not a:
                skipped.append((line_no, stt, "thieu image_id/question/answer"))
                continue
            if status in ("loại", "loai", "bỏ", "bo"):
                skipped.append((line_no, stt, "trang_thai = loại"))
                continue
            page_raw = norm_space(row.get("page"))
            items.append({
                "source": "qa_anh", "line": line_no, "stt": stt, "image_id": image_id,
                "doc_id": norm_space(row.get("doc_id")), "page_raw": page_raw, "pages": parse_pages(page_raw),
                "question": q, "answer": a, "loai": "image", "loai_goc": "image",
                "tap": norm_space(row.get("tap")).lower(),
                "nguoi_soan": norm_space(row.get("nguoi_soan")), "ghi_chu": norm_space(row.get("ghi_chu")),
                "context": "",
            })
    return items, skipped


def load_ingest_images(ingest_dir):
    """images.jsonl cua ingestion -> {image_id: ban ghi hinh}."""
    path = os.path.join(ingest_dir, "images.jsonl")
    figs = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    figs[r["image_id"]] = r
    return figs


def figure_pages(figs):
    """{(doc_id, "1.2"): {trang}} tu chu thich danh so ("Hình 1.2. ...") cua cac hinh da cat."""
    out = defaultdict(set)
    for r in figs.values():
        m = FIG_NUM.match(r.get("caption_hint") or "")
        if m:
            out[(r["doc_id"].lower(), m.group(1))].add(r["page_number"])
    return out


# ------------------------------------------------------------------ chia train/val
def split_by_page(items, val_ratio, seed):
    """Chia 90/10 trong tung doc_id, theo nhom trang: cac cau cung trang luon o cung 1 tap."""
    rnd = random.Random(seed)
    by_doc = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_doc[it["doc_id"]][it["pages"][0] if it["pages"] else -1].append(it)
    train, val = [], []
    for doc in sorted(by_doc):
        groups = [by_doc[doc][k] for k in sorted(by_doc[doc])]
        rnd.shuffle(groups)
        target = round(sum(len(g) for g in groups) * val_ratio)
        got = 0
        for g in groups:
            if got < target and got + len(g) <= target + max(1, target // 2):
                val.extend(g)
                got += len(g)
            else:
                train.extend(g)
    rnd.shuffle(train)
    rnd.shuffle(val)
    return train, val


def text_page_split(train, val):
    """{(doc_id, trang): "train"/"val"} tu ket qua chia cau text. Trang co cau text val -> "val"."""
    out = {}
    for tap, items in (("train", train), ("val", val)):
        for it in items:
            for p in it["pages"]:
                k = (it["doc_id"].lower(), p)
                if out.get(k) != "val":
                    out[k] = tap
    return out


def split_anh(items, page_tap, forced_train_pages, val_ratio, seed, flags):
    """Chia train/val cho cau image (co image_id) va image_query - theo TRANG, dong bo voi cau text.

    Thu tu uu tien cho moi trang (doc_id, trang cua hinh):
      1. trang da train o vong 1 (anh ca trang cua cau image khong co image_id) -> train
      2. trang co cau text: cung tap voi cau text cua trang do                  -> train/val
      3. cot "tap" tren sheet
      4. con lai: chia 80/20 theo trang (seed co dinh)
    Vi vay mot trang khong bao gio vua o train vua o val, ke ca giua cac loai cau. Neu cot "tap" tren sheet
    khac voi (1)-(2), script dung (1)-(2) va ghi vao bao cao de sua sheet.
    """
    by_page = defaultdict(list)
    for it in items:
        by_page[(it["doc_id"].lower(), it["fig"]["page_number"])].append(it)
    free = []
    for k, group in by_page.items():
        want = sorted({it["tap"] for it in group if it["tap"] in ("train", "val")})
        if k in forced_train_pages:
            tap, why = "train", "trang đã train ở vòng 1 bằng ảnh cả trang"
        elif k in page_tap:
            tap, why = page_tap[k], "theo câu text cùng trang"
        elif len(want) == 1:
            tap, why = want[0], ""
        elif want:
            tap, why = "val", "các câu cùng trang ghi tap khác nhau"
        else:
            free.append(group)
            continue
        for it in group:
            if it["tap"] and it["tap"] != tap:
                flags[f"Cột tap trên sheet khác với cách chia theo trang ({why}) - script đã dùng tap theo trang, "
                      f"nên sửa sheet"].append(dict(it, question=f"[{it['tap']} → {tap}] {it['question']}"))
            it["tap"] = tap
    tr, va = split_by_page([it for g in free for it in g], val_ratio, seed)
    for it in tr:
        it["tap"] = "train"
    for it in va:
        it["tap"] = "val"
    return [it for it in items if it["tap"] == "train"], [it for it in items if it["tap"] == "val"]


# ------------------------------------------------------------------ ghi ket qua
def meta_of(it):
    return {"doc_id": it["doc_id"], "page": it["page_raw"], "source": it["source"], "loai": it["loai"],
            "nguoi_soan": it["nguoi_soan"], "has_context": bool(it["context"])}


def llm_record(it):
    if it["context"]:
        rec = {"instruction": INSTR_CTX,
               "input": f"Ngữ cảnh: {it['context']}\nCâu hỏi: {it['question']}",
               "output": it["answer"]}
    else:
        rec = {"instruction": INSTR_NOCTX, "input": f"Câu hỏi: {it['question']}", "output": it["answer"]}
    rec["meta"] = meta_of(it)
    return rec


def vlm_record_anh(it):
    """Cau hoi co dap an la hinh: anh = hinh cat tu tai lieu, "Ngữ cảnh" = chu thich/muc/trang cua hinh.

    Dinh dang giong het prompt_format.build_rag_messages(question, [image_context(...)], [anh]),
    tuc la giong luc backend Giai doan 6 tra ve 1 hinh kem metadata cua no.
    """
    meta = meta_of(it)
    meta.update(image_id=it["image_id"], tap=it["tap"], stt=it["stt"], kieu="dap_an_la_hinh")
    return {"image": it["image_path"].replace("\\", "/"),
            "instruction": INSTR_IMAGE_CTX,
            "input": f"Ngữ cảnh: {it['context']}\nCâu hỏi: {it['question']}",
            "question": it["question"], "answer": it["answer"], "meta": meta}


def vlm_record_anh_hoi(it):
    """Cau hoi la anh: anh = anh nguoi dung gui (anh_hoi/<image_id>.jpg), "Ngữ cảnh" = hinh tim duoc + doan van.

    Dinh dang giong build_rag_messages(question, pf.image_query_contexts(fig, doan_van), [anh_nguoi_dung]),
    tuc la backend Giai doan 6 tim ra hinh khop voi anh nguoi dung va doan van cung trang roi gui kem.
    """
    meta = meta_of(it)
    meta.update(image_id=it["image_id"], tap=it["tap"], stt=it["stt"], kieu="cau_hoi_la_anh",
                hinh_goc=it["image_path"].replace("\\", "/"))
    return {"image": it["query_image"].replace("\\", "/"),
            "instruction": INSTR_IMAGE_CTX,
            "input": f"Ngữ cảnh: {it['context']}\nCâu hỏi: {it['question']}",
            "question": it["question"], "answer": it["answer"], "meta": meta}


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if fitz is None:
        print("  (Chua cai PyMuPDF - chay: pip install pymupdf. Tam thoi bo qua ngu canh va anh trang.)")
    pdfs = PdfLibrary(PDF_DIR)

    manual, skipped = load_manual_csv(CSV_PATH)
    synthetic = load_synthetic_json(SYN_PATH)
    print(f"Thu cong: {len(manual)} cau (bo qua {len(skipped)} dong) | Tong hop da loc: {len(synthetic)} cau")
    print(f"Tim thay PDF cho: {', '.join(sorted(pdfs.paths)) or '(khong co)'}")

    # 1) bo cau hoi trung lap trong cung doc_id
    seen, items, dups = set(), [], []
    for it in manual + synthetic:
        key = (it["doc_id"].lower(), question_key(it["question"]))
        if key in seen:
            dups.append(it)
            continue
        seen.add(key)
        items.append(it)

    # 2) gan ngu canh tu PDF cho cau text
    flags = defaultdict(list)
    coverage = defaultdict(list)
    for it in items:
        q = it["question"]
        if "?" not in q and not IMPERATIVE.match(q):
            flags["Câu hỏi chưa trọn vẹn (không có dấu ? hoặc bị cắt)"].append(it)
        if it["answer"].endswith((",", ";", ":")):
            flags["Đáp án có vẻ bị cắt dở dang"].append(it)
        if not it["pages"]:
            flags["Không đọc được số trang"].append(it)
            continue
        raw_nums = [int(x) for x in re.findall(r"\d+", it["page_raw"])]
        if len(raw_nums) == 2 and raw_nums[1] < raw_nums[0]:
            flags["Số trang dạng 'a,b' với b < a (đã tự đọc thành trang liền sau) - nên sửa trên sheet"].append(it)
        if it["loai"] != "text" or it["context"] or not pdfs.has(it["doc_id"]):
            continue
        n = pdfs.n_pages(it["doc_id"])
        if any(not 1 <= p <= n for p in it["pages"]):
            flags["Số trang vượt quá số trang của file PDF"].append(it)
            continue
        page_text = " ".join(pdfs.page_text(it["doc_id"], p) for p in it["pages"])
        ctx, cov = pick_context(page_text, q, it["answer"])
        it["context"] = ctx
        coverage[it["doc_id"]].append(cov)
        if cov < LOW_COVERAGE:
            flags[f"Ngữ cảnh ít khớp đáp án (<{LOW_COVERAGE:.0%}) - kiểm tra số trang/nội dung"].append(it)

    for it in items:
        if it["loai_goc"] not in LOAI_HOP_LE:
            flags[f"Cột loai không hợp lệ (chỉ dùng {', '.join(LOAI_HOP_LE)}) - script coi là text"].append(
                dict(it, question=f"[loai = {it['loai_goc']}] {it['question']}"))
        if it["loai"] == "image_query" and not it["image_id"]:
            flags["Câu image_query thiếu image_id - bỏ qua câu này"].append(it)

    # 3) chia train/val cho LLM
    text_items = [it for it in items if it["loai"] == "text"]
    image_items = [it for it in items if it["loai"] == "image" and not it["image_id"]]  # anh ca trang (vong 1)
    anh_rows = [it for it in items if it["loai"] in ("image", "image_query") and it["image_id"]]
    train, val = split_by_page(text_items, VAL_RATIO, SEED)
    write_jsonl([llm_record(it) for it in train], os.path.join(OUT_DIR, "llm_train.jsonl"))
    write_jsonl([llm_record(it) for it in val], os.path.join(OUT_DIR, "llm_val.jsonl"))
    print(f"\nDa ghi {len(train)} cau vao llm_train.jsonl, {len(val)} cau vao llm_val.jsonl")

    # 4) cau loai image khong co image_id (vong 1): render anh ca trang neu co PDF
    figs = load_ingest_images(INGEST_DIR)
    fig_pages = figure_pages(figs)
    forced_train_pages = set()
    for it in image_items:
        # cau hoi nhac "Hình x.y" ma hinh do (theo ingestion) nam o trang khac -> dung trang dung, bao de sua sheet
        for num in dict.fromkeys(FIG_NUM.findall(it["question"] + " " + it["answer"])):
            where = sorted(fig_pages.get((it["doc_id"].lower(), num), ()))
            if len(where) == 1 and where[0] not in it["pages"]:
                flags["Câu loại image nhắc tới Hình x.y nhưng hình nằm ở trang khác (script đã dùng trang đúng) "
                      "- sửa cột page trên sheet"].append(dict(it, question=f"[Hình {num} ở trang {where[0]}] {it['question']}"))
                forced_train_pages |= {(it["doc_id"].lower(), p) for p in it["pages"]}
                it["pages"], it["page_raw"] = where, str(where[0])
        forced_train_pages |= {(it["doc_id"].lower(), p) for p in it["pages"]}
    vlm, pending = [], []
    img_dir = os.path.join(OUT_DIR, "images")
    for it in image_items:
        if pdfs.has(it["doc_id"]) and it["pages"] and 1 <= it["pages"][0] <= pdfs.n_pages(it["doc_id"]):
            os.makedirs(img_dir, exist_ok=True)
            out = os.path.join(img_dir, f"{it['doc_id']}_p{it['pages'][0]}.png")
            pdfs.render(it["doc_id"], it["pages"][0], out)
            vlm.append({"image": out.replace("\\", "/"), "question": it["question"],
                        "answer": it["answer"], "meta": meta_of(it)})
        else:
            pending.append({k: it[k] for k in ("doc_id", "page_raw", "question", "answer", "nguoi_soan", "ghi_chu")})
    n_page_imgs = len(vlm)

    # 5) cau image / image_query co image_id (cot image_id cua QA_Data). Cu: doc qa_anh.csv neu QA_Data chua co.
    if not anh_rows and os.path.exists(QA_ANH_PATH):
        anh_rows, anh_skipped = load_qa_anh(QA_ANH_PATH)
        skipped += [(ln, stt, f"qa_anh.csv: {why}") for ln, stt, why in anh_skipped]
        print(f"  (QA_Data chua co cot image_id - doc {len(anh_rows)} cau tu {QA_ANH_PATH})")
    elif os.path.exists(QA_ANH_PATH):
        print(f"  (Bo qua {QA_ANH_PATH}: cau hoi anh da nam trong QA_Data, cot image_id)")
    need_query = sorted({it["image_id"] for it in anh_rows if it["loai"] == "image_query"})
    query_imgs = {}
    if need_query:
        if tao_anh_hoi is None:
            print("  !! Khong import duoc ingestion/tao_anh_hoi.py - cau image_query se cho anh")
        else:
            query_imgs = tao_anh_hoi.ensure_query_images(need_query, INGEST_DIR)
    anh_ok, seen_anh = [], set()
    for it in anh_rows:
        key = (it["loai"], it["image_id"].lower(), question_key(it["question"]))
        if key in seen_anh:
            dups.append(it)
            continue
        seen_anh.add(key)
        fig = figs.get(it["image_id"])
        path = os.path.join(INGEST_DIR, "images", f"{it['image_id']}.png")
        if fig is None or not os.path.exists(path):
            pending.append({k: it[k] for k in ("stt", "loai", "image_id", "doc_id", "page_raw", "question", "answer")}
                           | {"ly_do": "chưa có hình cắt - chạy python ingestion/ingest.py, hoặc sửa image_id"})
            continue
        if fig["doc_id"].lower() != it["doc_id"].lower() or (it["pages"] and fig["page_number"] not in it["pages"]):
            flags["doc_id/page trên sheet khác với hình theo image_id (script dùng theo image_id)"].append(it)
        it.update(fig=fig, image_path=path, doc_id=fig["doc_id"], pages=[fig["page_number"]],
                  page_raw=str(fig["page_number"]))
        if it["loai"] == "image_query":
            if it["image_id"] not in query_imgs:
                pending.append({k: it[k] for k in ("stt", "loai", "image_id", "doc_id", "page_raw", "question")}
                               | {"ly_do": "chưa tạo được ảnh hỏi - chạy python ingestion/tao_anh_hoi.py"})
                continue
            it["query_image"] = query_imgs[it["image_id"]]
            doan_van = ""
            if pdfs.has(it["doc_id"]) and 1 <= fig["page_number"] <= pdfs.n_pages(it["doc_id"]):
                doan_van, _ = pick_context(pdfs.page_text(it["doc_id"], fig["page_number"]), it["question"],
                                           it["answer"])
            it["context"] = "\n".join(pf.image_query_contexts(fig, doan_van))
        else:
            it["context"] = pf.image_context(fig.get("caption_hint", ""), fig["page_number"], fig.get("section", ""),
                                             fig.get("caption_source", ""), fig.get("slide_number"))
        anh_ok.append(it)
    page_tap = text_page_split(train, val)
    anh_train, anh_val = split_anh(anh_ok, page_tap, forced_train_pages, VAL_RATIO_ANH, SEED, flags)
    rec_of = {"image": vlm_record_anh, "image_query": vlm_record_anh_hoi}
    vlm += [rec_of[it["loai"]](it) for it in anh_train]
    vlm_val = [rec_of[it["loai"]](it) for it in anh_val]
    for name, recs in (("vlm_train.jsonl", vlm), ("vlm_val.jsonl", vlm_val)):
        path = os.path.join(OUT_DIR, name)
        if recs:
            write_jsonl(recs, path)
        elif os.path.exists(path):
            os.remove(path)  # xoa file cu de khong dung nham du lieu lan chay truoc
    truy_hoi = [{"stt": it["stt"], "question": it["question"], "image_id": it["image_id"], "doc_id": it["doc_id"],
                 "page": it["fig"]["page_number"], "tap": it["tap"]} for it in anh_ok if it["loai"] == "image"]
    write_jsonl(truy_hoi, os.path.join(OUT_DIR, "qa_anh_truy_hoi.jsonl"))
    truy_hoi_anh = [{"stt": it["stt"], "query_image": it["query_image"].replace("\\", "/"), "question": it["question"],
                     "image_id": it["image_id"], "doc_id": it["doc_id"], "page": it["fig"]["page_number"],
                     "tap": it["tap"]} for it in anh_ok if it["loai"] == "image_query"]
    write_jsonl(truy_hoi_anh, os.path.join(OUT_DIR, "anh_hoi_truy_hoi.jsonl"))
    with open(os.path.join(OUT_DIR, "vlm_pending_images.json"), "w", encoding="utf-8") as f:
        json.dump(pending, f, ensure_ascii=False, indent=2)
    dem = Counter((it["loai"], it["tap"]) for it in anh_ok)
    print(f"Cau hoi anh -> vlm_train.jsonl ({len(vlm)}): {n_page_imgs} anh ca trang + {dem['image', 'train']} dap an "
          f"la hinh + {dem['image_query', 'train']} cau hoi la anh | vlm_val.jsonl ({len(vlm_val)}): "
          f"{dem['image', 'val']} + {dem['image_query', 'val']} | {len(pending)} cau cho PDF/anh")

    # 6) bao cao
    write_report(manual, synthetic, skipped, dups, items, train, val, vlm, pending, coverage, flags, pdfs,
                 anh=(anh_train, anh_val, n_page_imgs))
    print(f"Bao cao chi tiet: {os.path.join(OUT_DIR, 'bao_cao_du_lieu.txt')}")


def write_report(manual, synthetic, skipped, dups, items, train, val, vlm, pending, coverage, flags, pdfs,
                 anh=((), (), 0)):
    anh_train, anh_val, n_page_imgs = anh
    loai_dem = Counter(it["loai"] for it in items)
    L = []
    L.append("BÁO CÁO DỮ LIỆU TRAIN - Bước 2.4")
    L.append("=" * 60)
    L.append(f"Đầu vào: {len(manual)} câu thủ công ({loai_dem['text']} text, {loai_dem['image']} image, "
             f"{loai_dem['image_query']} image_query), {len(synthetic)} câu tổng hợp; "
             f"bỏ qua {len(skipped)} dòng; bỏ {len(dups)} câu trùng lặp")
    L.append(f"LLM: train {len(train)} | val {len(val)}   ·   VLM: train {len(vlm)} | val {len(anh_val)} | "
             f"chờ ảnh {len(pending)}")
    L.append(f"Cấu hình: chia {1 - VAL_RATIO:.0%}/{VAL_RATIO:.0%} theo nhóm trang trong từng doc_id, seed {SEED}, "
             f"ngữ cảnh tối đa {MAX_CTX_CHARS} ký tự")
    L.append("")
    L.append(f"{'doc_id':<12}{'text':>6}{'image':>7}{'img_q':>7}{'train':>7}{'val':>6}  {'ngữ cảnh':<10}{'độ khớp TB':>11}")
    docs = sorted({it["doc_id"] for it in items})
    tr = Counter(it["doc_id"] for it in train)
    va = Counter(it["doc_id"] for it in val)
    for d in docs:
        nt = sum(1 for it in items if it["doc_id"] == d and it["loai"] == "text")
        ni = sum(1 for it in items if it["doc_id"] == d and it["loai"] == "image")
        nq = sum(1 for it in items if it["doc_id"] == d and it["loai"] == "image_query")
        cov = coverage.get(d)
        L.append(f"{d:<12}{nt:>6}{ni:>7}{nq:>7}{tr[d]:>7}{va[d]:>6}  {'có PDF' if pdfs.has(d) else 'chưa có':<10}"
                 f"{(f'{sum(cov) / len(cov):.0%}' if cov else '-'):>11}")
    L.append("(train/val ở bảng trên là của câu text)")
    L.append("")
    L.append("Ghi chú: doc_id 'chưa có' PDF → câu hỏi vào LLM ở dạng không ngữ cảnh. Chép PDF đặt tên đúng doc_id")
    L.append("vào data/raw/domain_docs/ rồi chạy lại script để có ngữ cảnh và ảnh trang cho câu loại image.")
    if anh_train or anh_val:
        L.append("")
        L.append("CÂU HỎI ẢNH CÓ image_id (image = đáp án là hình, image_query = câu hỏi là ảnh)")
        L.append(f"{'doc_id':<12}{'image tr':>9}{'image val':>10}{'img_q tr':>9}{'img_q val':>10}")
        dem = Counter((it["doc_id"], it["loai"], "train") for it in anh_train)
        dem.update((it["doc_id"], it["loai"], "val") for it in anh_val)
        for d in sorted({k[0] for k in dem}):
            L.append(f"{d:<12}{dem[d, 'image', 'train']:>9}{dem[d, 'image', 'val']:>10}"
                     f"{dem[d, 'image_query', 'train']:>9}{dem[d, 'image_query', 'val']:>10}")
        L.append(f"vlm_train.jsonl = {n_page_imgs} câu image không có image_id (ảnh cả trang, vòng 1) + "
                 f"{len(anh_train)} câu trên; vlm_val.jsonl = {len(anh_val)} câu, không dùng khi train.")
        L.append("Câu image: ảnh = hình cắt (images/), 'Ngữ cảnh' = trang, mục, chú thích của hình (image_context).")
        L.append("Câu image_query: ảnh = ảnh hỏi giống ảnh chụp (anh_hoi/), 'Ngữ cảnh' = hình tìm được + đoạn văn cùng trang.")
        L.append("Chia train/val theo trang và đồng bộ với câu text cùng trang, nên một trang chỉ thuộc một tập.")
        L.append("qa_anh_truy_hoi.jsonl (câu chữ → image_id) và anh_hoi_truy_hoi.jsonl (ảnh → image_id, trang) "
                 "dùng đo Recall@k ở Giai đoạn 5.")
    if skipped:
        L.append("")
        L.append("DÒNG BỊ BỎ QUA")
        for line_no, stt, why in skipped:
            L.append(f"  - dòng CSV {line_no} (STT {stt}): {why}")
    if dups:
        L.append("")
        L.append("CÂU TRÙNG LẶP ĐÃ BỎ (giữ câu xuất hiện trước)")
        for it in dups:
            L.append(f"  - {it['doc_id']} tr.{it['page_raw']} STT {it['stt']}: {it['question']}")
    if flags:
        L.append("")
        L.append("CÂU NÊN SOÁT LẠI TRÊN SHEET")
        for title, lst in flags.items():
            L.append(f"\n[{title}] - {len(lst)} câu")
            for it in lst:
                L.append(f"  - STT {it['stt'] or '-'} | {it['doc_id']} tr.{it['page_raw']} | {it['nguoi_soan'] or '-'} | "
                         f"{it['question'][:110]}")
    with open(os.path.join(OUT_DIR, "bao_cao_du_lieu.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()