# -*- coding: utf-8 -*-
"""
cham_tay.py - Giai doan 4: cham thu cong cau tra loi truoc/sau fine-tune, theo NGU NGHIA va theo NGU CANH.
  ngu nghia: y cua cau tra loi duoc so voi cac y cua dap an chuan theo nghia (dien dat khac van dung,
             dung chu nhung sai nghia la sai) -> R, F1
  ngu canh : moi y con lai doi chieu voi dung ngu canh mo hinh duoc dua (anh, chu thich, doan van tim duoc):
             co trong ngu canh / ngoai ngu canh / mau thuan-bia -> P (do chinh xac theo ngu canh)
Moi chi so tu dong cua eval_truoc_sau.py co ban cham tay tuong ung:
    F1 (dem am tiet trung)      -> F1 theo y = 2PR/(P+R)
    do phu (recall am tiet)     -> R theo y
    thang/hoa/thua (chenh F1)   -> tinh tu F1 theo y + nguoi cham chon thang cau tot hon (cot So sanh A-B)
    ROUGE-L                     -> diem Dien dat 0/1/2 (dem tay chuoi con chung chi lap lai viec cua may)
    thoi gian tra loi           -> van do bang may
  them: nhan dung/sai 0/1/2, ty le y sai/bia, ty le y ngoai ngu canh.

Chay tu thu muc goc repo, can openpyxl (pip install openpyxl==3.1.5, da co trong requirements-ingest.txt):

  1. Tao phieu cham (an ten mo hinh, tron A/B tung cau):
       python finetune/cham_tay.py tao
     Doc  finetune/outputs/danh_gia_v2/so_sanh_truoc_sau.csv  (dau ra eval_truoc_sau.py vong 2)
          finetune/y_dap_an_cham_tay.csv                     (84 cau tap val, dap an da tach san thanh cac y)
          data/processed/vlm_val.jsonl, llm_val.jsonl         (anh va ngu canh mo hinh duoc dua)
     Ghi  finetune/outputs/cham_tay_v2/phieu_cham_1.xlsx ... phieu_cham_4.xlsx   gui moi nguoi 1 file
          finetune/outputs/cham_tay_v2/khoa.json   A/B la mo hinh nao - nguoi dieu phoi giu, KHONG gui, KHONG mo truoc
     Cau hoi chia 4 phan deu nhau (phan nao cung du cac nhom cau); nguoi i cham phan i va i+1 (vong tron) nen moi
     cau duoc 2 nguoi cham, 4 cap nguoi khac nhau. Moi nguoi 42 cau = 84 cau tra loi (~1,5-2 gio).
  2. Moi nguoi doc sheet Huong_dan, cham sheet Cham, gui lai file .xlsx.
  3. Chep cac phieu da cham vao finetune/outputs/cham_tay_v2/da_cham/ (ten file tuy y) roi:
       python finetune/cham_tay.py tong_hop
     -> bao_cao_cham_tay.md   bang chi so tu dong canh ban cham tay, F1 theo nhom cau, nguoi chon cau tot hon,
                              nhan, dien dat, y sai/bia va ngoai ngu canh, do dong thuan, cac cau can xem lai
        ket_qua_cham_tay.csv  tung cau tra loi (da bo an danh), diem tung nguoi
     Chay lai tong_hop bao nhieu lan cung duoc (vd sau khi hai nguoi thong nhat lai cac cau lech nhau).

Tuy chon cua tao: --so_nguoi 2..8 (mac dinh 4), --ten "Phuong,Hai,Tung,Bao Anh" (ghi ten vao phieu va ten file),
                  --seed 42, --ten_ft v2, --so_sanh/--y/--out de dung cho vong sau.
"""
import argparse
import csv
import glob
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SO_SANH = "finetune/outputs/danh_gia_v2/so_sanh_truoc_sau.csv"
DEFAULT_Y = "finetune/y_dap_an_cham_tay.csv"
DEFAULT_OUT = "finetune/outputs/cham_tay_v2"
DEFAULT_RECORDS = ("data/processed/vlm_val.jsonl", "data/processed/llm_val.jsonl")

MO_HINH = ("goc", "finetune")
NGUONG = 0.05  # chenh F1 > 5 diem moi tinh thang/thua, giong eval_truoc_sau.py
TEN_NHOM = {"Ảnh – đáp án là hình, tập val": "Ảnh: đáp án là hình",
            "Ảnh – câu hỏi là ảnh, tập val": "Ảnh: câu hỏi là ảnh",
            "Val – có ngữ cảnh": "Chữ (có ngữ cảnh)"}
TU_DONG = ("f1", "rougeL", "do_phu", "giay")  # cot <ten>_goc, <ten>_finetune cua so_sanh_truoc_sau.csv

# Cot sheet Cham: (khoa, tieu de, do rong). Cac cot Y1..YK chen giua COT_DAU va COT_SAU.
COT_DAU = [("ma", "Mã", 8), ("cau_hoi", "Câu hỏi", 28),
           ("ngu_canh", "Ngữ cảnh mô hình được đưa (ảnh, chú thích, đoạn văn tìm được)", 50),
           ("y", "Các ý của đáp án chuẩn", 40), ("so_y", "Số ý", 5), ("tl", "Câu trả lời", 55)]
COT_SAU = [("them_nc", "Ý thêm trong ngữ cảnh", 8), ("them_ngoai", "Ý thêm ngoài ngữ cảnh", 8),
           ("sai", "Ý sai/bịa", 7), ("nhan", "Nhãn (0/1/2)", 7), ("dien_dat", "Diễn đạt (0/1/2)", 8),
           ("so_sanh", "So sánh A–B", 9), ("ghi_chu", "Ghi chú", 22),
           ("dat", "Ý đạt", 6), ("p", "P", 6), ("r", "R", 6), ("f1", "F1", 6)]
O_NHAP = ("them_nc", "them_ngoai", "sai", "nhan", "dien_dat", "so_sanh", "ghi_chu")
GOP_CAU = ("cau_hoi", "ngu_canh", "y", "so_y", "so_sanh")  # o gop chung 2 dong A/B cua mot cau hoi
ANH_RONG, ANH_CAO = 330, 260  # kich thuoc hien thi anh trong phieu (px); anh luu toi 900 px de phong to van ro
NGU_CANH_TOI_DA = 1600  # ky tu ngu canh hien o sheet Cham; ban day du o sheet Tham_khao
DONG = 15.0  # chieu cao 1 dong chu Calibri 11 (point)


# ---------------------------------------------------------------------------
# Tien ich
# ---------------------------------------------------------------------------
def utf8_stdout():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def rp(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def hien(path):
    """Duong dan de in: tuong doi neu nam trong repo."""
    rel = os.path.relpath(path, ROOT)
    return path if rel.startswith("..") else rel.replace(os.sep, "/")


def norm(s):
    return " ".join(unicodedata.normalize("NFC", "" if s is None else str(s)).split())


def ascii_(s):
    s = s.replace("đ", "d").replace("Đ", "D").replace("–", "-")
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def text_hash(s):
    return hashlib.sha1(norm(s).encode("utf-8")).hexdigest()[:12]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(v):
    try:
        return float(str(v).replace(",", "."))
    except (TypeError, ValueError):
        return None


def need_openpyxl():
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        sys.exit("Thieu thu vien openpyxl: pip install openpyxl==3.1.5")


def ten_nhom(nhom):
    return TEN_NHOM.get(nhom, nhom)


def key_cau(nhom, doc_id, trang, cau_hoi):
    return norm(nhom), norm(doc_id), norm(trang), norm(cau_hoi)


def vn(x, d=2):
    return "–" if x is None else f"{x:.{d}f}".replace(".", ",")


def pt(x):
    """Ti le 0..1 -> diem 0..100, dau phay thap phan."""
    return "–" if x is None else f"{100 * x:.1f}".replace(".", ",")


def pt_dau(x, he_so=100, d=1):
    return "–" if x is None else f"{he_so * x:+.{d}f}".replace(".", ",").replace("-", "−")


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def so_dong(text, rong):
    """Uoc luong so dong khi xuong dong tu dong trong o rong `rong` ky tu."""
    cpl = max(1, int(rong * 1.05))
    return sum(max(1, math.ceil(len(p) / cpl)) for p in str(text or "").split("\n"))


# ---------------------------------------------------------------------------
# Du lieu dau vao
# ---------------------------------------------------------------------------
def load_y(path):
    rows = read_csv(path)
    if not rows:
        sys.exit(f"{path} rong")
    cols = sorted((c for c in rows[0] if re.fullmatch(r"y\d+", c or "")), key=lambda c: int(c[1:]))
    out = []
    for r in rows:
        ys = [norm(r[c]) for c in cols if norm(r.get(c))]
        if not ys:
            sys.exit(f"{path}: cau stt {r.get('stt')} chua tach y nao")
        out.append({"stt": norm(r.get("stt")), "nhom": norm(r["nhom"]), "doc_id": norm(r["doc_id"]),
                    "trang": norm(r["trang"]), "cau_hoi": norm(r["cau_hoi"]), "dap_an": norm(r["dap_an_chuan"]),
                    "y": ys})
    return out


def question_of(rec):
    if rec.get("image"):
        return rec.get("question", "")
    t = rec.get("input", "")
    i = t.rfind("Câu hỏi:")
    return t[i + len("Câu hỏi:"):].strip() if i >= 0 else t


def context_of(rec):
    t = rec.get("input") or ""
    i = t.rfind("Câu hỏi:")
    t = (t[:i] if i >= 0 else t).strip()
    return t[len("Ngữ cảnh:"):].strip() if t.startswith("Ngữ cảnh:") else t


def load_records(paths):
    recs = {}
    for p in paths:
        if not os.path.exists(rp(p)):
            print(f"  (khong thay {p} - phieu se thieu anh/ngu canh cua cac cau trong file nay)")
            continue
        with open(rp(p), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    meta = rec.get("meta") or {}
                    k = ("anh" if rec.get("image") else "chu", norm(question_of(rec)), norm(meta.get("doc_id")),
                         norm(meta.get("page")))
                    recs[k] = rec
    return recs


# ---------------------------------------------------------------------------
# tao: phieu cham an danh
# ---------------------------------------------------------------------------
def chia_phan(cau, n, rnd):
    """Tron trong tung nhom roi chia vong tron -> n phan deu nhau, phan nao cung du cac nhom cau."""
    xep = []
    for g in dict.fromkeys(c["nhom"] for c in cau):
        cs = [c for c in cau if c["nhom"] == g]
        rnd.shuffle(cs)
        xep += cs
    parts = [[] for _ in range(n)]
    for i, c in enumerate(xep):
        parts[i % n].append(c)
    return parts


def doan_ma(ds):
    """['C01','C02','C03','C07'] -> 'C01–C03, C07'"""
    if not ds:
        return ""
    so = sorted(int(m[1:]) for m in ds)
    w = len(ds[0]) - 1
    out, i = [], 0
    while i < len(so):
        j = i
        while j + 1 < len(so) and so[j + 1] == so[j] + 1:
            j += 1
        out.append(f"C{so[i]:0{w}d}" + (f"–C{so[j]:0{w}d}" if j > i else ""))
        i = j + 1
    return ", ".join(out)


def anh_cho_excel(path):
    from PIL import Image as PILImage
    from openpyxl.drawing.image import Image as XLImage

    im = PILImage.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        nen = PILImage.new("RGB", im.size, "white")
        nen.paste(im, mask=im.split()[-1])
        im = nen
    else:
        im = im.convert("RGB")
    s = min(1.0, 900 / max(im.size))
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), PILImage.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    buf.seek(0)
    pil = PILImage.open(buf)  # openpyxl doc lai du lieu tu buf khi luu
    xl = XLImage(pil)
    s = min(ANH_RONG / pil.width, ANH_CAO / pil.height, 2.0)
    xl.width, xl.height = max(1, round(pil.width * s)), max(1, round(pil.height * s))
    return xl


def dat_anh(ws, xl, col0, row0):
    from openpyxl.drawing.spreadsheet_drawing import AnchorMarker, OneCellAnchor
    from openpyxl.drawing.xdr import XDRPositiveSize2D
    from openpyxl.utils.units import pixels_to_EMU

    xl.anchor = OneCellAnchor(_from=AnchorMarker(col=col0, colOff=pixels_to_EMU(4), row=row0,
                                                 rowOff=pixels_to_EMU(4)),
                              ext=XDRPositiveSize2D(pixels_to_EMU(xl.width), pixels_to_EMU(xl.height)))
    ws.add_image(xl)


def huong_dan_rows(so_y_max):
    """Noi dung sheet Huong_dan: (kieu, noi dung). kieu: h2 (muc), p (doan), vd (dong bang vi du)."""
    return [
        ("h2", "Mục đích"),
        ("p", "Các chỉ số tự động (eval_truoc_sau.py) chỉ đếm chữ trùng với đáp án chuẩn: câu đúng ý nhưng diễn đạt "
              "khác bị chấm thấp, câu sai nhưng trùng nhiều chữ lại được điểm. Phiếu này chấm theo ngữ nghĩa và theo "
              "ngữ cảnh:\n"
              "     ngữ nghĩa – ý của câu trả lời được so với các ý của đáp án chuẩn theo nghĩa: diễn đạt khác mà cùng "
              "nghĩa vẫn đúng, dùng đúng chữ mà sai nghĩa vẫn sai;\n"
              "     ngữ cảnh – mỗi ý còn lại được đối chiếu với đúng ngữ cảnh mô hình được đưa (ảnh, chú thích hình, "
              "đoạn văn tìm được): có trong ngữ cảnh, ngoài ngữ cảnh, hay mâu thuẫn/bịa.\n"
              "Từ đó có bản chấm tay của các chỉ số tự động: F1 và độ phủ theo ý (thay F1, độ phủ tự động), điểm "
              "Diễn đạt (thay ROUGE-L), người chấm chọn câu tốt hơn (thay thắng/hòa/thua), thêm nhãn đúng/sai, tỷ lệ ý "
              "bịa và ý ngoài ngữ cảnh. Đáp án chuẩn đã được tách sẵn thành các ý, cả nhóm dùng chung một bản tách. "
              "Mỗi câu hỏi được 2 người chấm độc lập để đo độ đồng thuận."),
        ("h2", "Cách chấm"),
        ("p", "1. Mở sheet Cham. Mỗi câu hỏi chiếm 2 dòng: A và B là câu trả lời của hai mô hình khác nhau (mô hình "
              "gốc và mô hình fine-tune), thứ tự A/B trộn ngẫu nhiên ở từng câu. Đừng đoán mô hình. Chấm riêng từng "
              "câu trả lời theo các bước 3–7; xong cả A và B mới chọn câu tốt hơn ở bước 8."),
        ("p", "2. Đọc câu hỏi, cột Ngữ cảnh và cột Các ý của đáp án chuẩn. Cột Ngữ cảnh là đúng những gì mô hình được "
              "đưa: ảnh (câu hỏi là ảnh thì là ảnh người dùng chụp), chú thích hình, đoạn văn tìm được. Phóng to ảnh "
              "bằng Ctrl + lăn chuột. Đáp án đầy đủ nằm ở sheet Tham_khao."),
        ("p", "3. Đọc câu trả lời và chia thành các ý: mỗi ý là một nhận định có thông tin. Không tính câu chào, câu "
              "nhắc lại câu hỏi, câu đệm; câu gộp hai ý của đáp án tính 2 ý; ý nhắc lại nhiều lần tính 1. Chấm theo "
              "nghĩa, không theo chữ: dùng từ khác, từ đồng nghĩa, đổi thứ tự mà cùng nghĩa vẫn là đúng; dùng đúng chữ "
              "của đáp án nhưng sai nghĩa (đảo quan hệ, nhầm đối tượng, sai số liệu) là sai."),
        ("p", f"4. Đánh dấu từng ý của đáp án ở các cột Ý1…Ý{so_y_max}:\n"
              "     x  – câu trả lời nêu đúng nghĩa ý này;\n"
              "     p  – nêu đúng nhưng thiếu phần quan trọng của ý (tính nửa ý);\n"
              "     để trống – không nêu, hoặc nêu sai (ý nêu sai tính ở bước 5).\n"
              "Ô xám là ý không có (đáp án có ít ý hơn), không điền."),
        ("p", "5. Các ý còn lại của câu trả lời: đối chiếu với cột Ngữ cảnh rồi ghi số lượng vào 3 cột (ghi 0 nếu "
              "không có):\n"
              "     Ý thêm trong ngữ cảnh – đáp án không có nhưng ngữ cảnh hoặc ảnh có nói, hoặc suy ra trực tiếp "
              "được;\n"
              "     Ý thêm ngoài ngữ cảnh – không mâu thuẫn với gì nhưng ngữ cảnh và ảnh không có (kiến thức chung, "
              "hoặc không kiểm tra được);\n"
              "     Ý sai/bịa – mâu thuẫn với ngữ cảnh, ảnh hoặc đáp án, hoặc chi tiết bịa ra: sai số hình, số trang, "
              "đọc sai chữ trong ảnh, thêm bước không có. Biết chắc là sai dù ngữ cảnh không nói cũng tính vào đây. "
              "Ý của đáp án mà nêu sai chi tiết thì không đánh x, tính 1 ý sai."),
        ("p", "6. Nhãn – câu trả lời đúng đến đâu:\n"
              "     2 – đúng trọng tâm câu hỏi, nêu được phần lớn các ý của đáp án (thiếu nhiều nhất 1 ý), không có ý "
              "sai;\n"
              "     1 – trả lời được một phần: có ý đúng nhưng thiếu nhiều ý, hoặc có ý sai/bịa;\n"
              "     0 – không có ý đúng nào (sai hoàn toàn, lạc đề, chỉ bịa, hoặc không trả lời)."),
        ("p", "7. Diễn đạt – chỉ xét cách viết, không xét đúng sai:\n"
              "     2 – mạch lạc, đúng tiếng Việt, dễ hiểu;\n"
              "     1 – hiểu được nhưng lủng củng, lặp ý, lẫn chữ nước ngoài không cần thiết (tên menu, thuật ngữ có "
              "trong tài liệu thì không tính) hoặc bị cắt giữa chừng;\n"
              "     0 – khó hiểu, lặp vô nghĩa, sai ngôn ngữ hoặc không có nội dung."),
        ("p", "8. So sánh A–B (ô gộp chung của cả câu hỏi), điền sau khi chấm xong A và B: chọn A hoặc B nếu câu đó "
              "trả lời người hỏi tốt hơn (đúng hơn, đủ hơn, bám ngữ cảnh hơn); chọn Ngang nếu gần như nhau."),
        ("p", "9. Ghi chú (không bắt buộc): lý do khi phân vân, lỗi lạ (lặp chữ, trả lời bị cắt, sai ngôn ngữ…)."),
        ("p", "Các cột Ý đạt, P, R, F1 tự tính khi đã điền nhãn; ô \"Đã chấm\" ở dòng đầu sheet Cham cho biết tiến "
              "độ."),
        ("h2", "Công thức"),
        ("p", "Ý đạt = số dấu x + 0,5 × số dấu p (ý của đáp án đều lấy từ tài liệu nên luôn có căn cứ)\n"
              "Tổng số ý = Ý đạt + Ý thêm trong ngữ cảnh + Ý thêm ngoài ngữ cảnh + Ý sai/bịa\n"
              "P (độ chính xác theo ngữ cảnh) = (Ý đạt + Ý thêm trong ngữ cảnh) / Tổng số ý\n"
              "R (độ phủ theo nghĩa) = Ý đạt / Số ý của đáp án\n"
              "F1 = 2 × P × R / (P + R); câu trả lời không có ý nào thì F1 = 0."),
        ("h2", "Ví dụ"),
        ("p", "Câu hỏi: \"Nguyên tắc chọn giá trị ít ràng buộc nhất khi gán màu cho Q được minh họa thế nào?\" "
              "Ngữ cảnh: hình hai bản đồ ghi \"Cho phép 1 giá trị cho SA\" và \"Cho phép 0 giá trị cho SA\", chú thích "
              "nói về chọn giá trị ít ràng buộc nhất, không nói gì về quay lui. Đáp án có 4 ý: (1) bối cảnh WA đỏ, NT "
              "xanh lá; (2) gán Q đỏ thì SA còn 1 giá trị; (3) gán Q xanh dương thì SA còn 0 giá trị; (4) kết luận "
              "chọn đỏ cho Q. Năm câu trả lời khác nhau được chấm như sau:"),
        ("vd", ["Câu trả lời", "Ý1", "Ý2", "Ý3", "Ý4", "Thêm trong NC", "Thêm ngoài NC", "Sai/bịa", "Nhãn",
                "Diễn đạt", "P", "R", "F1", "Vì sao"]),
        ("vd", ["Với WA đỏ và NT xanh lá, gán Q đỏ thì SA còn 1 giá trị, gán Q xanh dương thì SA không còn giá trị "
                "nào, nên chọn đỏ cho Q vì nó để lại nhiều lựa chọn nhất cho SA.",
                "x", "x", "x", "x", 1, 0, 0, 2, 2, "1,00", "1,00", "1,00",
                "Đủ 4 ý. \"Để lại nhiều lựa chọn nhất cho SA\" đáp án không có nhưng suy ra trực tiếp từ hình: thêm "
                "trong ngữ cảnh."]),
        ("vd", ["Màu đỏ là lựa chọn tốt hơn cho Q, vì tô xanh dương thì SA hết màu để tô (khi WA đỏ, NT xanh lá).",
                "x", "", "x", "x", 0, 0, 0, 2, 2, "1,00", "0,75", "0,86",
                "Ít chữ trùng đáp án nhưng cùng nghĩa với Ý1, Ý3, Ý4 nên vẫn đánh x (F1 tự động sẽ thấp). Thiếu Ý2."]),
        ("vd", ["Với WA đỏ, NT xanh lá thì nên chọn Q màu đỏ vì khi đó SA vẫn còn giá trị. Cách chọn này giúp giảm "
                "số lần phải quay lui.",
                "x", "p", "", "x", 0, 1, 0, 1, 2, "0,71", "0,63", "0,67",
                "Ý2 chỉ nói \"còn giá trị\", thiếu \"1 giá trị\": p. \"Giảm số lần quay lui\" không sai nhưng ngữ "
                "cảnh không nói: thêm ngoài ngữ cảnh. Không nêu Ý3."]),
        ("vd", ["Khi WA đỏ, NT xanh lá, nếu gán Q đỏ thì SA còn 2 giá trị nên chọn đỏ cho Q.",
                "x", "", "", "x", 0, 0, 1, 1, 2, "0,67", "0,50", "0,57",
                "\"SA còn 2 giá trị\" mâu thuẫn với hình: không đánh Ý2, tính 1 ý sai. Không nêu Ý3."]),
        ("vd", ["Hình 3.2 minh họa thuật toán tìm kiếm theo chiều rộng trên bản đồ Úc.",
                "", "", "", "", 0, 0, 2, 0, 2, "0,00", "0,00", "0,00",
                "Sai số hình và sai nội dung: 2 ý sai. Viết vẫn trôi chảy nên Diễn đạt 2."]),
        ("p", "Nếu câu thứ ba trong bảng là A và câu thứ tư là B thì So sánh A–B chọn A: nhiều ý đúng hơn và không có "
              "ý sai."),
        ("h2", "Lưu ý"),
        ("p", "• Chấm theo ngữ cảnh mô hình được đưa, không theo những gì mình biết thêm: ngữ cảnh không có thì là "
              "\"ngoài ngữ cảnh\", trừ khi chắc chắn sai.\n"
              "• Chấm độc lập: không trao đổi điểm với người khác cho tới khi cả nhóm nộp; không xem điểm tự động "
              "trước khi chấm.\n"
              "• Không sửa cột Mã, không chèn, xóa hay sắp xếp dòng, không đổi tên sheet: máy đọc lại phiếu theo mã.\n"
              "• Khoảng 1 phút một câu trả lời; chấm làm nhiều lần cũng được, nhớ lưu. Dùng Google Sheets thì xong tải "
              "về dạng .xlsx (Tệp > Tải xuống > Microsoft Excel).\n"
              "• Xong gửi lại file cho người điều phối."),
    ]


def build_workbook(so_phieu, ten, cau_cua_nguoi, so_y_max, images, meta):
    from openpyxl import Workbook
    from openpyxl.cell.rich_text import CellRichText, TextBlock
    from openpyxl.cell.text import InlineFont
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Protection, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    def fill(c):
        return PatternFill("solid", start_color=c, end_color=c)

    F_HEAD, F_NHAP_H, F_NHAP, F_XAM, F_TINH_H, F_TINH = (fill("DDEBF7"), fill("FFE699"), fill("FFF8E1"),
                                                         fill("BFBFBF"), fill("E7E6E6"), fill("F2F2F2"))
    mong, dam = Side(style="thin", color="BFBFBF"), Side(style="medium", color="7F7F7F")
    VIEN = Border(left=mong, right=mong, top=mong, bottom=mong)
    VIEN_CUOI = Border(left=mong, right=mong, top=mong, bottom=dam)
    WRAP = Alignment(wrap_text=True, vertical="top")
    WRAP_DAY = Alignment(wrap_text=True, vertical="bottom")
    GIUA = Alignment(horizontal="center", vertical="top", wrap_text=True)
    MO = Protection(locked=False)

    wb = Workbook()
    nq = len(cau_cua_nguoi)
    ten_hien = f" – {ten}" if ten else ""

    # ---------------- Huong_dan ----------------
    hd = wb.active
    hd.title = "Huong_dan"
    widths = [44, 4, 4, 4, 4, 8, 8, 7, 6, 7, 6, 6, 6, 40]
    for i, wd in enumerate(widths, 1):
        hd.column_dimensions[get_column_letter(i)].width = wd
    tong_rong = sum(widths)
    last_col = get_column_letter(len(widths))
    hd["A1"] = f"Phiếu chấm thủ công số {so_phieu}{ten_hien} – câu trả lời trước/sau fine-tune {meta['ten_ft']}"
    hd["A1"].font = Font(bold=True, size=14)
    hd["A2"] = (f"{nq} câu hỏi, {2 * nq} câu trả lời · mã câu {doan_ma([c['ma'] for c in cau_cua_nguoi])} · "
                f"tạo ngày {meta['ngay']}")
    hd["A3"] = "Người chấm (ghi tên):"
    hd["A3"].font = Font(bold=True)
    hd.merge_cells(f"B3:{get_column_letter(len(widths) - 1)}3")
    hd["B3"].fill = F_NHAP
    hd["B3"].protection = MO
    if ten:
        hd["B3"] = ten
    r = 5
    for kieu, nd in huong_dan_rows(so_y_max):
        if kieu == "vd":
            for j, v in enumerate(nd, 1):
                c = hd.cell(row=r, column=j, value=v)
                c.alignment = WRAP if j in (1, len(nd)) else GIUA
                c.border = VIEN
                if nd[0] == "Câu trả lời":
                    c.font = Font(bold=True)
                    c.fill = F_HEAD
            hd.row_dimensions[r].height = max(max(so_dong(v, wd * 0.8) for v, wd in zip(nd, widths)), 2) * DONG + 4
        else:
            hd.merge_cells(f"A{r}:{last_col}{r}")
            c = hd.cell(row=r, column=1, value=nd)
            c.alignment = WRAP
            if kieu == "h2":
                c.font = Font(bold=True, size=12)
            hd.row_dimensions[r].height = so_dong(nd, tong_rong) * DONG + (8 if kieu == "h2" else 4)
        r += 1
    hd.sheet_view.showGridLines = False

    # ---------------- Cham ----------------
    ws = wb.create_sheet("Cham")
    cols = COT_DAU + [(f"y{i}", f"Ý{i}", 4.5) for i in range(1, so_y_max + 1)] + COT_SAU
    ci = {k: i + 1 for i, (k, _, _) in enumerate(cols)}
    L = {k: get_column_letter(i) for k, i in ci.items()}
    rong = {k: wd for k, _, wd in cols}
    for k in ci:
        ws.column_dimensions[L[k]].width = rong[k]
    nhap = {f"y{i}" for i in range(1, so_y_max + 1)} | set(O_NHAP)
    tinh = {"dat", "p", "r", "f1"}
    first, last = 3, 2 + 2 * nq

    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ci["so_y"])
    ws["A1"] = (f"Phiếu chấm số {so_phieu}{ten_hien} · {nq} câu hỏi, {2 * nq} câu trả lời · "
                f"đọc sheet Huong_dan trước khi chấm")
    ws["A1"].font = Font(bold=True, size=12)
    tien_do = ws.cell(row=1, column=ci["tl"],
                      value=f'="Đã chấm: "&COUNTA({L["nhan"]}{first}:{L["nhan"]}{last})&"/{2 * nq} câu trả lời"')
    tien_do.font = Font(bold=True, color="C00000")
    ws.row_dimensions[1].height = 22
    for k, (_, tieu_de, _) in zip(ci, cols):
        c = ws.cell(row=2, column=ci[k], value=tieu_de)
        c.font = Font(bold=True)
        c.alignment = GIUA
        c.border = VIEN
        c.fill = F_NHAP_H if k in nhap else F_TINH_H if k in tinh else F_HEAD
    ws.row_dimensions[2].height = max(so_dong(t, wd * 0.75) for _, t, wd in cols) * DONG + 6

    dv_y = DataValidation(type="list", formula1='"x,p"', allow_blank=True, showErrorMessage=True,
                          errorTitle="Ý của đáp án", error="Điền x (nêu đúng ý), p (đúng một phần) hoặc để trống.")
    dv_so = DataValidation(type="whole", operator="between", formula1="0", formula2="30", allow_blank=True,
                           showErrorMessage=True, errorTitle="Số ý", error="Điền số nguyên ≥ 0 (ghi 0 nếu không có).")
    dv_012 = DataValidation(type="list", formula1='"0,1,2"', allow_blank=True, showErrorMessage=True,
                            errorTitle="Điểm 0/1/2", error="Chỉ điền 0, 1 hoặc 2.")
    dv_ss = DataValidation(type="list", formula1='"A,B,Ngang"', allow_blank=True, showErrorMessage=True,
                           errorTitle="So sánh A–B", error="Chọn A, B hoặc Ngang.")
    for dv in (dv_y, dv_so, dv_012, dv_ss):
        ws.add_data_validation(dv)
    dv_y.add(f"{L['y1']}{first}:{L[f'y{so_y_max}']}{last}")
    dv_so.add(f"{L['them_nc']}{first}:{L['sai']}{last}")
    dv_012.add(f"{L['nhan']}{first}:{L['dien_dat']}{last}")
    dv_ss.add(f"{L['so_sanh']}{first}:{L['so_sanh']}{last}")

    row = first
    for c in cau_cua_nguoi:
        ra, rb = row, row + 1
        so_y = len(c["y"])
        nhan_cau = f"{ten_nhom(c['nhom'])} · {c['doc_id']}, tr. {c['trang']}"
        ws.cell(row=ra, column=ci["cau_hoi"]).value = CellRichText(
            TextBlock(InlineFont(i=True, color="7F7F7F"), nhan_cau + "\n"), TextBlock(InlineFont(b=True), c["cau_hoi"]))
        ngu_canh = c.get("ngu_canh") or "(không có ngữ cảnh chữ)"
        if len(ngu_canh) > NGU_CANH_TOI_DA:
            ngu_canh = ngu_canh[:NGU_CANH_TOI_DA].rstrip() + "… (đầy đủ ở sheet Tham_khao)"
        xl = images.get(c["ma"])
        cell_nc = ws.cell(row=ra, column=ci["ngu_canh"], value=ngu_canh)
        cell_nc.alignment = WRAP_DAY if xl is not None else WRAP
        if c.get("anh_loi"):
            cell_nc.value = f"{c['anh_loi']}\n{ngu_canh}"
        y_text = "\n".join(f"{i}. {y}" for i, y in enumerate(c["y"], 1))
        ws.cell(row=ra, column=ci["y"], value=y_text)
        ws.cell(row=ra, column=ci["so_y"], value=so_y)
        for k in GOP_CAU:
            ws.merge_cells(start_row=ra, start_column=ci[k], end_row=rb, end_column=ci[k])
        h_anh = 0
        if xl is not None:
            dat_anh(ws, xl, ci["ngu_canh"] - 1, ra - 1)
            h_anh = (xl.height + 14) * 0.75
        h_chung = max(h_anh + so_dong(cell_nc.value, rong["ngu_canh"]) * DONG + 6,
                      (so_dong(nhan_cau, rong["cau_hoi"]) + so_dong(c["cau_hoi"], rong["cau_hoi"])) * DONG + 6,
                      so_dong(y_text, rong["y"]) * DONG + 6)
        h_tl = [so_dong(c["tra_loi"][ab], rong["tl"]) * DONG + 6 for ab in "AB"]
        h_a = min(409, max(h_tl[0], h_chung / 2, 2 * DONG))
        h_b = min(409, max(h_tl[1], h_chung - h_a, 2 * DONG))
        ws.row_dimensions[ra].height, ws.row_dimensions[rb].height = h_a, h_b

        for rr, ab in ((ra, "A"), (rb, "B")):
            ws.cell(row=rr, column=ci["ma"], value=f"{c['ma']}-{ab}").font = Font(bold=True)
            ws.cell(row=rr, column=ci["tl"], value=c["tra_loi"][ab])
            for i in range(1, so_y_max + 1):
                cell = ws.cell(row=rr, column=ci[f"y{i}"])
                if i <= so_y:
                    cell.fill, cell.protection = F_NHAP, MO
                else:
                    cell.fill = F_XAM
            for k in O_NHAP:
                if k == "so_sanh" and rr == rb:
                    continue
                cell = ws.cell(row=rr, column=ci[k])
                cell.fill, cell.protection = F_NHAP, MO
            n, y1, yk = L["nhan"], L["y1"], L[f"y{so_y_max}"]
            dat, nc, ngoai, sai = (f"{L[k]}{rr}" for k in ("dat", "them_nc", "them_ngoai", "sai"))
            p, rcol = f"{L['p']}{rr}", f"{L['r']}{rr}"
            tong = f"{dat}+{nc}+{ngoai}+{sai}"
            cong = {
                "dat": f'=IF(${n}{rr}="","",COUNTIF({y1}{rr}:{yk}{rr},"x")+0.5*COUNTIF({y1}{rr}:{yk}{rr},"p"))',
                "p": f'=IF(${n}{rr}="","",IF({tong}=0,0,({dat}+{nc})/({tong})))',
                "r": f'=IF(${n}{rr}="","",{dat}/${L["so_y"]}${ra})',
                "f1": f'=IF(${n}{rr}="","",IF({p}+{rcol}=0,0,2*{p}*{rcol}/({p}+{rcol})))',
            }
            for k, f in cong.items():
                cell = ws.cell(row=rr, column=ci[k], value=f)
                cell.fill = F_TINH
                cell.number_format = "0.0" if k == "dat" else "0.00"
            for k, i in ci.items():
                cell = ws.cell(row=rr, column=i)
                # o gop: LibreOffice lay vien duoi tu o dau, Excel tu o cuoi -> dat ca hai
                cell.border = VIEN_CUOI if rr == rb or k in GOP_CAU else VIEN
                if cell.alignment.wrap_text is not True:
                    giua = (k in nhap and k != "ghi_chu") or k in tinh or k in ("ma", "so_y")
                    cell.alignment = GIUA if giua else WRAP
        row += 2
    ws.freeze_panes = "B3"
    ws.sheet_view.zoomScale = 90
    ws.protection.sheet = True
    ws.protection.formatRows = False
    ws.protection.formatColumns = False
    ws.protection.formatCells = False

    # ---------------- Tham_khao ----------------
    tk = wb.create_sheet("Tham_khao")
    tk_cols = [("Mã câu", 8), ("Câu hỏi", 34), ("Đáp án chuẩn (đầy đủ)", 50),
               ("Ngữ cảnh mô hình được đưa (đầy đủ)", 80), ("Tệp ảnh", 26)]
    for i, (t, wd) in enumerate(tk_cols, 1):
        tk.column_dimensions[get_column_letter(i)].width = wd
        cell = tk.cell(row=1, column=i, value=t)
        cell.font, cell.fill, cell.alignment, cell.border = Font(bold=True), F_HEAD, GIUA, VIEN
    tk.row_dimensions[1].height = 32
    for j, c in enumerate(cau_cua_nguoi, 2):
        vals = [c["ma"], f"{ten_nhom(c['nhom'])} · {c['doc_id']}, tr. {c['trang']}\n{c['cau_hoi']}", c["dap_an"],
                c.get("ngu_canh") or "(không có)", c.get("anh") or ""]
        for i, v in enumerate(vals, 1):
            cell = tk.cell(row=j, column=i, value=v)
            cell.alignment, cell.border = WRAP, VIEN
        tk.row_dimensions[j].height = min(409, max(so_dong(v, wd) for v, (_, wd) in zip(vals, tk_cols)) * DONG + 6)
    tk.freeze_panes = "B2"
    tk.protection.sheet = True
    tk.protection.formatRows = False
    tk.protection.formatColumns = False

    wb.active = 0
    wb.properties.title = f"Phiếu chấm số {so_phieu}"
    wb.calculation.fullCalcOnLoad = True
    return wb


def cmd_tao(args):
    need_openpyxl()
    n = args.so_nguoi
    if not 2 <= n <= 8:
        sys.exit("--so_nguoi phai tu 2 den 8 (moi cau can 2 nguoi cham)")
    ten = [t.strip() for t in args.ten.split(",")] if args.ten else []
    if ten and len(ten) != n:
        sys.exit(f"--ten co {len(ten)} ten nhung --so_nguoi {n}")

    ys = load_y(rp(args.y))
    if not os.path.exists(rp(args.so_sanh)):
        sys.exit(f"Khong thay {args.so_sanh} - chay finetune/eval_truoc_sau.py truoc")
    ss_rows = read_csv(rp(args.so_sanh))
    ss = {key_cau(r["nhom"], r["doc_id"], r["trang"], r["cau_hoi"]): r for r in ss_rows}
    # eval_truoc_sau.py ghi F1, ROUGE-L, do phu theo thang 0-100; phieu cham tinh 0-1
    thang = {k: 100.0 if k != "giay" and any((to_float(r.get(f"{k}_{mh}")) or 0) > 1 for r in ss_rows
                                             for mh in MO_HINH) else 1.0 for k in TU_DONG}
    recs = load_records(args.records)

    cau, thieu, doi, khong_nc = [], [], [], []
    for y in ys:
        r = ss.get(key_cau(y["nhom"], y["doc_id"], y["trang"], y["cau_hoi"]))
        if r is None:
            thieu.append(y["stt"])
            continue
        if norm(r["dap_an_chuan"]) != y["dap_an"]:
            doi.append(y["stt"])
        rec = recs.get(("anh" if y["nhom"].startswith("Ảnh") else "chu", y["cau_hoi"], y["doc_id"], y["trang"]))
        if rec is None:
            khong_nc.append(y["stt"])
        tu_dong = {mh: {k: (None if to_float(r.get(f"{k}_{mh}")) is None
                            else round(to_float(r.get(f"{k}_{mh}")) / thang[k], 6)) for k in TU_DONG}
                   for mh in MO_HINH}
        cau.append(dict(y, tra_loi_mh={"goc": r["tra_loi_goc"], "finetune": r["tra_loi_finetune"]},
                        tu_dong=tu_dong, ngu_canh=context_of(rec) if rec else "",
                        anh=rec.get("image") if rec else None))
    if thieu:
        print(f"  !! {len(thieu)} cau trong {args.y} khong co trong {args.so_sanh} (stt {', '.join(thieu[:10])})"
              " - bo qua")
    if doi:
        print(f"  !! {len(doi)} cau co dap an chuan khac ban da tach y (stt {', '.join(doi[:10])}) - "
              f"nen tach y lai trong {args.y}")
    if khong_nc:
        print(f"  !! {len(khong_nc)} cau khong tim thay trong {', '.join(args.records)} (stt "
              f"{', '.join(khong_nc[:10])}) - phieu se thieu anh/ngu canh cua cac cau nay")
    if len(cau) < n:
        sys.exit("Qua it cau de chia phieu")

    rnd = random.Random(args.seed)
    parts = chia_phan(cau, n, rnd)
    w = max(2, len(str(len(cau))))
    k = 0
    for pi, part in enumerate(parts):
        for c in part:
            k += 1
            c["ma"], c["phan"] = f"C{k:0{w}d}", pi
    ab = [True] * (len(cau) // 2) + [False] * (len(cau) - len(cau) // 2)
    rnd.shuffle(ab)
    tat_ca = [c for part in parts for c in part]
    for c, goc_la_a in zip(tat_ca, ab):
        c["A"], c["B"] = ("goc", "finetune") if goc_la_a else ("finetune", "goc")
        c["tra_loi"] = {"A": c["tra_loi_mh"][c["A"]], "B": c["tra_loi_mh"][c["B"]]}
        c["nguoi"] = sorted({c["phan"] + 1, (c["phan"] + 1) % n + 1})

    out = rp(args.out)
    os.makedirs(os.path.join(out, "da_cham"), exist_ok=True)
    meta = {"so_sanh": args.so_sanh, "y": args.y, "seed": args.seed, "so_nguoi": n, "ten": ten,
            "ten_ft": args.ten_ft, "so_y_toi_da": max(len(c["y"]) for c in cau), "ngay": time.strftime("%d/%m/%Y")}
    khoa = {"meta": meta, "cau": [
        {"ma": c["ma"], "stt": c["stt"], "nhom": c["nhom"], "doc_id": c["doc_id"], "trang": c["trang"],
         "cau_hoi": c["cau_hoi"], "y": c["y"], "A": c["A"], "B": c["B"], "nguoi": c["nguoi"],
         "hash": {"A": text_hash(c["tra_loi"]["A"]), "B": text_hash(c["tra_loi"]["B"])},
         "tu_dong": c["tu_dong"], "tra_loi": c["tra_loi_mh"]} for c in tat_ca]}
    khoa_path = os.path.join(out, "khoa.json")
    if os.path.exists(khoa_path) and not args.ghi_de:
        with open(khoa_path, encoding="utf-8") as f:
            cu = json.load(f)
        if cu.get("cau") != khoa["cau"]:
            sys.exit(f"{khoa_path} da co va khac lan tao nay: cac phieu da gui se khong khop khoa moi.\n"
                     "Neu chac chan muon tao lai tu dau (bo phieu cu), them --ghi_de.")
        meta["ngay"] = cu.get("meta", {}).get("ngay", meta["ngay"])
    with open(khoa_path, "w", encoding="utf-8") as f:
        json.dump(khoa, f, ensure_ascii=False, indent=1)

    print(f"{len(cau)} cau hoi, {sum(len(c['y']) for c in cau)} y; {n} phieu, moi cau 2 nguoi cham")
    for i in range(n):
        so = i + 1
        cua = sorted((c for c in tat_ca if so in c["nguoi"]), key=lambda c: c["ma"])
        images = {}
        for c in cua:
            c.pop("anh_loi", None)
            if not c.get("anh"):
                continue
            p = rp(c["anh"])
            if os.path.exists(p):
                images[c["ma"]] = anh_cho_excel(p)  # moi phieu can doi tuong anh rieng
            else:
                c["anh_loi"] = f"(không thấy ảnh {c['anh']})"
        wb = build_workbook(so, ten[i] if ten else "", cua, meta["so_y_toi_da"], images, meta)
        duoi = "_" + re.sub(r"[^A-Za-z0-9]+", "", ascii_(ten[i])) if ten else ""
        path = os.path.join(out, f"phieu_cham_{so}{duoi}.xlsx")
        wb.save(path)
        print(f"  {hien(path)}: {len(cua)} cau ({2 * len(cua)} cau tra loi), "
              f"ma {ascii_(doan_ma([c['ma'] for c in cua]))}")
    print(f"Khoa A/B: {hien(khoa_path)} - KHONG gui cho nguoi cham")
    print(f"Nhan phieu da cham ve thi chep vao {hien(os.path.join(out, 'da_cham'))}/ "
          f"roi chay: python finetune/cham_tay.py tong_hop" + (f" --out {args.out}" if args.out != DEFAULT_OUT else ""))


# ---------------------------------------------------------------------------
# tong_hop: doc phieu da cham, bo an danh, tinh chi so thu cong
# ---------------------------------------------------------------------------
def doc_dau(v):
    if v is None:
        return 0.0, None
    if isinstance(v, (int, float)):
        return (1.0 if v >= 1 else 0.5 if v > 0 else 0.0), None
    s = norm(v).lower()
    if s in ("", "0", "-", "o"):
        return 0.0, None
    if s in ("x", "1", "v", "✓", "✔"):
        return 1.0, None
    if s in ("p", "/", "½", "0.5", "0,5", "1/2"):
        return 0.5, None
    return 0.0, f"dấu '{v}' không hợp lệ (chỉ x hoặc p)"


def doc_so(v):
    if v is None or norm(v) == "":
        return None, None
    x = float(v) if isinstance(v, (int, float)) else to_float(norm(v))
    if x is None:
        return None, f"'{v}' không phải số"
    if x < 0:
        return None, f"số âm {v}"
    return x, None


def doc_012(v):
    if v is None or norm(v) == "":
        return None
    x = float(v) if isinstance(v, (int, float)) else to_float(norm(v))
    return int(x) if x in (0, 1, 2) else None


def doc_so_sanh(v):
    s = norm(v).lower()
    if not s:
        return None
    if s in ("a", "b"):
        return s.upper()
    if s in ("ngang", "=", "hoa", "hòa", "bang", "bằng", "nhu nhau", "như nhau"):
        return "Ngang"
    return "?"


def doc_bang(path):
    """-> (cac dong cua sheet Cham, ten nguoi cham ghi o Huong_dan)."""
    if path.lower().endswith(".csv"):
        with open(path, encoding="utf-8-sig", newline="") as f:
            return [list(r) for r in csv.reader(f)], ""
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True)
    ws = wb["Cham"] if "Cham" in wb.sheetnames else wb.worksheets[0]
    ten = norm(wb["Huong_dan"]["B3"].value) if "Huong_dan" in wb.sheetnames else ""
    return [list(r) for r in ws.iter_rows(values_only=True)], ten


def tinh_diem(dau, so_y, nc, ngoai, sai):
    dat = sum(dau[:so_y])
    tong = dat + nc + ngoai + sai
    p = (dat + nc) / tong if tong else 0.0
    r = dat / so_y
    return {"dat": dat, "tong": tong, "p": p, "r": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "khong_sai": (dat + nc + ngoai) / tong if tong else 0.0}


def doc_phieu(path, khoa_cau, so_y_max):
    """-> (so phieu, ten nguoi cham, {ma tra loi: diem}, {ma cau: lua chon A/B/Ngang}, canh bao)."""
    ten_file = os.path.basename(path)
    rows, ten = doc_bang(path)
    hi = next((i for i, r in enumerate(rows[:8]) if r and norm(r[0]) == "Mã"), None)
    if hi is None:
        return None, ten, {}, {}, [f"{ten_file}: không thấy dòng tiêu đề 'Mã' – bỏ qua file"]
    head = [norm(h).lower() for h in rows[hi]]

    def col(pred):
        return next((i for i, h in enumerate(head) if pred(h)), None)

    c_tl = col(lambda h: h == "câu trả lời")
    c_y = {int(m.group(1)): i for i, h in enumerate(head) for m in [re.fullmatch(r"ý(\d+)", h)] if m}
    c_nc = col(lambda h: h.startswith("ý thêm") and "trong" in h)
    c_ngoai = col(lambda h: h.startswith("ý thêm") and "ngoài" in h)
    c_sai = col(lambda h: h.startswith("ý sai"))
    c_nhan, c_dd = col(lambda h: h.startswith("nhãn")), col(lambda h: h.startswith("diễn đạt"))
    c_ss, c_gc = col(lambda h: h.startswith("so sánh")), col(lambda h: h.startswith("ghi chú"))
    if None in (c_tl, c_nc, c_ngoai, c_sai, c_nhan) or not c_y:
        return None, ten, {}, {}, [f"{ten_file}: thiếu cột của phiếu chấm – bỏ qua file"]

    m = re.search(r"số\s*(\d+)", norm(rows[0][0] if rows and rows[0] else ""), re.I) or \
        re.search(r"phieu_cham_(\d+)", ten_file)
    so_phieu = int(m.group(1)) if m else None

    def get(r, i):
        return r[i] if i is not None and i < len(r) else None

    out, chon, canh_bao = {}, {}, []
    for r in rows[hi + 1:]:
        ma = norm(get(r, 0)).upper()
        m = re.fullmatch(r"(C\d+)-([AB])", ma)
        if not m:
            continue
        mc, ab = m.groups()
        kc = khoa_cau.get(mc)
        if kc is None:
            canh_bao.append(f"{ten_file}: mã {ma} không có trong khoa.json")
            continue
        ss = doc_so_sanh(get(r, c_ss))
        if ss == "?":
            canh_bao.append(f"{ten_file}: {mc} So sánh '{get(r, c_ss)}' không hợp lệ (A, B hoặc Ngang)")
        elif ss:
            chon[mc] = ss
        nhan, dd = doc_012(get(r, c_nhan)), doc_012(get(r, c_dd))
        dau, loi = [], []
        for i in range(1, so_y_max + 1):
            v, e = doc_dau(get(r, c_y.get(i)))
            if e:
                loi.append(f"Ý{i}: {e}")
            if v and i > len(kc["y"]):
                loi.append(f"Ý{i} được đánh dấu nhưng đáp án chỉ có {len(kc['y'])} ý – bỏ qua")
                v = 0.0
            dau.append(v)
        so = {}
        for k, ci in (("nc", c_nc), ("ngoai", c_ngoai), ("sai", c_sai)):
            so[k], e = doc_so(get(r, ci))
            if e:
                loi.append(e)
        ghi_chu = norm(get(r, c_gc))
        if nhan is None:
            if any(dau) or any(v is not None for v in so.values()) or dd is not None or norm(get(r, c_nhan)):
                canh_bao.append(f"{ten_file}: {ma} chưa có nhãn hợp lệ (0/1/2) – chưa tính")
            continue
        canh_bao += [f"{ten_file}: {ma} {e}" for e in loi]
        if dd is None:
            canh_bao.append(f"{ten_file}: {ma} thiếu Diễn đạt")
        tl = get(r, c_tl)
        if tl is not None and text_hash(tl) != kc["hash"][ab]:
            canh_bao.append(f"{ten_file}: {ma} câu trả lời khác lúc tạo phiếu (phiếu cũ hoặc bị sửa?)")
        nc, ngoai, sai = (so[k] or 0.0 for k in ("nc", "ngoai", "sai"))
        out[ma] = dict(tinh_diem(dau, len(kc["y"]), nc, ngoai, sai), ma_cau=mc, ab=ab, dau=dau[:len(kc["y"])],
                       nc=nc, ngoai=ngoai, sai=sai, nhan=nhan, dien_dat=dd, ghi_chu=ghi_chu)
    for mc in {v["ma_cau"] for v in out.values()}:
        if f"{mc}-A" in out and f"{mc}-B" in out and mc not in chon:
            canh_bao.append(f"{ten_file}: {mc} đã chấm A và B nhưng chưa chọn So sánh A–B")
    return so_phieu, ten, out, chon, canh_bao


def kripp_alpha(units):
    """Krippendorff alpha thang khoang; units: danh sach cac nhom gia tri cua cung 1 doi tuong duoc cham."""
    units = [u for u in units if len(u) >= 2]
    vals = [v for u in units for v in u]
    n = len(vals)
    if n < 4:
        return None
    do = sum(sum((a - b) ** 2 for i, a in enumerate(u) for j, b in enumerate(u) if i != j) / (len(u) - 1)
             for u in units) / n
    s, s2 = sum(vals), sum(v * v for v in vals)
    de = (2 * n * s2 - 2 * s * s) / (n * (n - 1))
    return None if de == 0 else 1 - do / de


def kappa_bac2(pairs, k=3):
    """Cohen kappa trong so bac hai cho diem 0..k-1 cua mot cap nguoi cham."""
    n = len(pairs)
    if n < 2:
        return None
    o = [[0] * k for _ in range(k)]
    for a, b in pairs:
        o[a][b] += 1
    ra = [sum(o[i]) for i in range(k)]
    cb = [sum(o[i][j] for i in range(k)) for j in range(k)]
    wt = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    num = sum(wt[i][j] * o[i][j] for i in range(k) for j in range(k))
    den = sum(wt[i][j] * ra[i] * cb[j] / n for i in range(k) for j in range(k))
    return None if den == 0 else 1 - num / den


def hang(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r, i = [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    n = len(pairs)
    if n < 3:
        return None
    mx, my = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    sxy = sum((a - mx) * (b - my) for a, b in pairs)
    sx, sy = math.sqrt(sum((a - mx) ** 2 for a, _ in pairs)), math.sqrt(sum((b - my) ** 2 for _, b in pairs))
    return None if sx == 0 or sy == 0 else sxy / (sx * sy)


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    return pearson(hang([a for a, _ in pairs]), hang([b for _, b in pairs])) if len(pairs) >= 3 else None


def ktc_chenh(diffs, b=2000, seed=0):
    if len(diffs) < 2:
        return None, None
    rnd = random.Random(seed)
    n = len(diffs)
    ms = sorted(sum(diffs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return ms[int(0.025 * b)], ms[int(0.975 * b) - 1]


def kiem_dinh_dau(thang, thua):
    """p hai phia cua kiem dinh dau (bo cau hoa)."""
    n = thang + thua
    if n == 0:
        return None
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(thang, thua) + 1)) / 2 ** n)


def wtl(diffs, nguong=NGUONG):
    th = sum(1 for d in diffs if d > nguong)
    tu = sum(1 for d in diffs if d < -nguong)
    return th, len(diffs) - th - tu, tu


def rut_gon(s, n=160):
    s = norm(s)
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def bang_md(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in r) + " |" for r in rows]
    return out


def cmd_tong_hop(args):
    need_openpyxl()
    out = rp(args.out)
    khoa_path = os.path.join(out, "khoa.json")
    if not os.path.exists(khoa_path):
        sys.exit(f"Khong thay {khoa_path} - chay 'python finetune/cham_tay.py tao' truoc")
    with open(khoa_path, encoding="utf-8") as f:
        khoa = json.load(f)
    meta = khoa["meta"]
    ft = meta["ten_ft"]
    ten_mh = {"goc": "Gốc", "finetune": f"Fine-tune {ft}"}
    kc = {c["ma"]: c for c in khoa["cau"]}
    thu_muc = rp(args.da_cham) if args.da_cham else os.path.join(out, "da_cham")
    files = sorted(p for p in glob.glob(os.path.join(thu_muc, "*"))
                   if p.lower().endswith((".xlsx", ".csv")) and not os.path.basename(p).startswith("~$"))
    if not files:
        sys.exit(f"Chua co phieu da cham trong {thu_muc} (.xlsx hoac .csv)")

    canh_bao, phieu = [], {}
    for p in files:
        so, ten, rows, chon, cb = doc_phieu(p, kc, meta["so_y_toi_da"])
        canh_bao += cb
        if so is None:
            if rows:
                canh_bao.append(f"{os.path.basename(p)}: không đọc được số phiếu – bỏ qua")
            continue
        if so in phieu:
            cu = phieu[so]
            moi_hon = os.path.getmtime(p) > os.path.getmtime(cu["file"])
            giu, bo = (p, cu["file"]) if moi_hon else (cu["file"], p)
            canh_bao.append(f"Phiếu số {so} có 2 file, dùng {os.path.basename(giu)} (mới hơn), bỏ "
                            f"{os.path.basename(bo)}")
            if not moi_hon:
                continue
        ten_mac_dinh = meta["ten"][so - 1] if meta["ten"] and so <= len(meta["ten"]) else ""
        phieu[so] = {"file": p, "ten": ten or ten_mac_dinh, "rows": rows, "chon": chon}

    # diem[(ma cau, mo hinh)] = {so phieu: diem}; phieu_bau[ma cau] = {so phieu: +1 ft tot hon / 0 / -1 goc tot hon}
    diem, phieu_bau, tien_do = defaultdict(dict), defaultdict(dict), []
    for so, ph in sorted(phieu.items()):
        can = [c["ma"] for c in khoa["cau"] if so in c["nguoi"]]
        la = sorted({v["ma_cau"] for v in ph["rows"].values()} - set(can))
        if la:
            canh_bao.append(f"Phiếu số {so} có mã không được giao cho phiếu này: {', '.join(la[:6])}")
        for v in ph["rows"].values():
            diem[(v["ma_cau"], kc[v["ma_cau"]][v["ab"]])][so] = v
        for mc, ss in ph["chon"].items():
            phieu_bau[mc][so] = 0 if ss == "Ngang" else (1 if kc[mc][ss] == "finetune" else -1)
        tien_do.append((so, ph["ten"], os.path.basename(ph["file"]), len(ph["rows"]), 2 * len(can)))

    gop = {}
    for key, d in diem.items():
        vs = list(d.values())
        g = {k: mean([v[k] for v in vs]) for k in ("f1", "p", "r", "khong_sai", "nhan", "dien_dat", "sai", "ngoai")}
        g["co_sai"] = mean([1.0 if v["sai"] > 0 else 0.0 for v in vs])
        g["co_ngoai"] = mean([1.0 if v["ngoai"] > 0 else 0.0 for v in vs])
        g["n"] = len(vs)
        gop[key] = g
    nhom_order = list(dict.fromkeys(c["nhom"] for c in khoa["cau"]))
    cap_du = [c for c in khoa["cau"] if (c["ma"], "goc") in gop and (c["ma"], "finetune") in gop]
    tong_luot = sum(len(d) for d in diem.values())
    can_luot = 2 * 2 * len(khoa["cau"])

    def theo_nhom(ds=None):
        ds = cap_du if ds is None else ds
        for g in nhom_order + [None]:
            cs = [c for c in ds if g is None or c["nhom"] == g]
            if cs:
                yield ("**Tất cả**" if g is None else ten_nhom(g)), cs

    def g_(c, mh, k):
        return gop[(c["ma"], mh)][k]

    def td(c, mh, k):
        return c["tu_dong"][mh].get(k)

    def bau_tb(c):
        return mean(phieu_bau[c["ma"]].values()) if phieu_bau.get(c["ma"]) else None

    L = [f"# Chấm thủ công – câu trả lời trước/sau fine-tune {ft} (theo ngữ nghĩa và ngữ cảnh)", ""]
    L += [f"{len(khoa['cau'])} câu hỏi tập val × 2 mô hình, {meta['so_nguoi']} người chấm, mỗi câu 2 người; đã chấm "
          f"**{tong_luot}/{can_luot}** lượt; {len(cap_du)} câu có đủ điểm cả hai mô hình được đưa vào so sánh. "
          f"Đáp án chuẩn tách thành {sum(len(c['y']) for c in khoa['cau'])} ý (`{meta['y']}`); thứ tự A/B trộn ngẫu "
          f"nhiên, người chấm không biết câu trả lời của mô hình nào. Câu trả lời và chỉ số tự động lấy từ "
          f"`{meta['so_sanh']}`.", "",
          "Cách chấm: ý của câu trả lời so với các ý của đáp án **theo nghĩa** (diễn đạt khác vẫn đúng, đúng chữ mà "
          "sai nghĩa vẫn sai); các ý còn lại đối chiếu với **ngữ cảnh mô hình được đưa** (ảnh, chú thích, đoạn văn): "
          "có trong ngữ cảnh, ngoài ngữ cảnh, hoặc sai/bịa. Ý đạt = số dấu x + 0,5 × số dấu p; P (theo ngữ cảnh) = "
          "(ý đạt + ý thêm trong ngữ cảnh) / tổng số ý; R (theo nghĩa) = ý đạt / số ý đáp án; F1 = 2PR/(P+R). "
          "Điểm một câu trả lời là trung bình của 2 người chấm. Khoảng tin cậy 95% (KTC) của chênh lệch tính bằng "
          "bootstrap 2000 lần trên các câu hỏi, ghép cặp hai mô hình; thắng/hòa/thua tính theo từng câu, chênh hơn "
          f"{round(NGUONG * 100)} điểm mới tính thắng hoặc thua.", ""]

    # 0. Bang tong hop
    rows = []
    if cap_du:
        def dong(ten_cs, lay, kieu="pt"):
            a, b = mean(lay(c, "goc") for c in cap_du), mean(lay(c, "finetune") for c in cap_du)
            d = None if a is None or b is None else b - a
            if kieu == "pt":
                return [ten_cs, pt(a), pt(b), pt_dau(d)]
            if kieu == "pct":
                return [ten_cs, pct(a), pct(b), "–" if d is None else pt_dau(d, 100, 0) + " điểm %"]
            so_le = 1 if kieu == "giay" else 2
            return [ten_cs, vn(a, so_le), vn(b, so_le), pt_dau(d, 1, so_le)]

        d_td = [td(c, "finetune", "f1") - td(c, "goc", "f1") for c in cap_du]
        d_tc = [g_(c, "finetune", "f1") - g_(c, "goc", "f1") for c in cap_du]
        bau = [x for x in (bau_tb(c) for c in cap_du) if x is not None]
        rows = [
            dong("F1 – tự động (âm tiết trùng)", lambda c, mh: td(c, mh, "f1")),
            dong("F1 – thủ công (ngữ nghĩa + ngữ cảnh)", lambda c, mh: g_(c, mh, "f1")),
            dong("Độ phủ – tự động (recall âm tiết)", lambda c, mh: td(c, mh, "do_phu")),
            dong("Độ phủ – thủ công (R theo nghĩa)", lambda c, mh: g_(c, mh, "r")),
            dong("Độ chính xác – thủ công (P theo ngữ cảnh)", lambda c, mh: g_(c, mh, "p")),
            dong("ROUGE-L – tự động", lambda c, mh: td(c, mh, "rougeL")),
            dong("Diễn đạt – thủ công (0–2)", lambda c, mh: g_(c, mh, "dien_dat"), "so"),
            dong("Nhãn đúng – thủ công (0–2)", lambda c, mh: g_(c, mh, "nhan"), "so"),
            dong("Câu trả lời có ý sai/bịa – thủ công", lambda c, mh: g_(c, mh, "co_sai"), "pct"),
            dong("Câu trả lời có ý ngoài ngữ cảnh – thủ công", lambda c, mh: g_(c, mh, "co_ngoai"), "pct"),
            [f"Thắng/hòa/thua của {ft} – theo F1 tự động", "", "", "/".join(map(str, wtl(d_td)))],
            [f"Thắng/hòa/thua của {ft} – theo F1 thủ công", "", "", "/".join(map(str, wtl(d_tc)))],
            [f"Thắng/hòa/thua của {ft} – người chấm chọn", "", "",
             "/".join(map(str, wtl(bau, 0))) if bau else "–"],
            dong("Thời gian trả lời (giây, đo bằng máy)", lambda c, mh: td(c, mh, "giay"), "giay"),
        ]
    L += ["## 0. Tổng hợp: chỉ số tự động và bản chấm thủ công", "",
          f"Trên {len(cap_du)} câu hỏi có đủ điểm. F1, độ phủ, P, ROUGE-L theo thang 0–100. ROUGE-L chỉ đếm chuỗi "
          "chữ trùng theo thứ tự nên không có bản đếm tay; phần người đánh giá được là Diễn đạt. Thời gian trả lời "
          "đo bằng máy.", ""]
    L += bang_md(["Chỉ số", ten_mh["goc"], ten_mh["finetune"], f"{ft} − gốc"], rows)
    L.append("")

    # 1. F1 thu cong theo nhom
    rows = []
    for ten_g, cs in theo_nhom():
        d = [g_(c, "finetune", "f1") - g_(c, "goc", "f1") for c in cs]
        lo, hi = ktc_chenh(d)
        rows.append([ten_g, len(cs), pt(mean(g_(c, "goc", "f1") for c in cs)),
                     pt(mean(g_(c, "finetune", "f1") for c in cs)),
                     f"{pt_dau(mean(d))} ({pt_dau(lo)} đến {pt_dau(hi)})", "/".join(map(str, wtl(d))),
                     f"{pt(mean(g_(c, 'goc', 'p') for c in cs))} → {pt(mean(g_(c, 'finetune', 'p') for c in cs))}",
                     f"{pt(mean(g_(c, 'goc', 'r') for c in cs))} → {pt(mean(g_(c, 'finetune', 'r') for c in cs))}"])
    L += ["## 1. F1 thủ công theo nhóm câu", ""]
    L += bang_md(["Nhóm câu", "Số câu", f"F1 {ten_mh['goc']}", f"F1 {ten_mh['finetune']}", f"{ft} − gốc (KTC 95%)",
                  "Thắng/Hòa/Thua", "P gốc → ft", "R gốc → ft"], rows)
    L += ["", "P là phần nội dung có căn cứ trong ngữ cảnh trên tổng những gì câu trả lời nêu; R là phần ý của đáp "
              "án được nêu đúng nghĩa.", ""]

    # 2. Nguoi cham chon cau tot hon
    rows = []
    co_bau = [c for c in khoa["cau"] if phieu_bau.get(c["ma"])]
    for ten_g, cs in theo_nhom(co_bau):
        ve = [v for c in cs for v in phieu_bau[c["ma"]].values()]
        th, ho, tu = wtl([bau_tb(c) for c in cs], 0)
        p = kiem_dinh_dau(th, tu)
        rows.append([ten_g, len(cs)] + [pct(sum(1 for v in ve if f(v)) / len(ve)) for f in
                                        (lambda v: v > 0, lambda v: v == 0, lambda v: v < 0)] +
                    [f"{th}/{ho}/{tu}", "–" if p is None else ("< 0,001" if p < 0.001 else vn(p, 3))])
    L += ["## 2. Người chấm chọn câu trả lời tốt hơn", ""]
    if rows:
        L += bang_md(["Nhóm câu", "Số câu", f"Chọn {ft}", "Ngang", "Chọn gốc", f"Thắng/Hòa/Thua của {ft}",
                      "p (kiểm định dấu)"], rows)
        khop = [c for c in co_bau if (c["ma"], "goc") in gop and (c["ma"], "finetune") in gop]

        def cung_chieu(b, d):
            return (b > 0) == (d > NGUONG) and (b < 0) == (d < -NGUONG)

        trung = sum(1 for c in khop if cung_chieu(bau_tb(c), g_(c, "finetune", "f1") - g_(c, "goc", "f1")))
        L += ["", "Ba cột đầu là tỷ lệ trên tất cả lượt chọn. Thắng/hòa/thua theo từng câu lấy trung bình lựa chọn "
                  "của 2 người (một người chọn, một người Ngang vẫn tính nghiêng về bên được chọn). p dưới 0,05 nghĩa "
                  "là chênh lệch thắng/thua khó do ngẫu nhiên.", "",
              f"Lựa chọn của người chấm và kết luận theo F1 thủ công trùng nhau ở {trung}/{len(khop)} câu.", ""]
    else:
        L += ["Chưa có lựa chọn So sánh A–B nào.", ""]

    # 3. Nhan va dien dat
    rows = []
    for ten_g, cs in theo_nhom():
        for mh in MO_HINH:
            nh = [v["nhan"] for c in cs for v in diem[(c["ma"], mh)].values()]
            dd = [v["dien_dat"] for c in cs for v in diem[(c["ma"], mh)].values() if v["dien_dat"] is not None]
            rows.append([ten_g, ten_mh[mh]] + [pct(sum(1 for x in nh if x == k) / len(nh)) for k in (2, 1, 0)] +
                        [vn(mean(nh))] + [pct(sum(1 for x in dd if x == k) / len(dd)) if dd else "–"
                                          for k in (2, 1, 0)] + [vn(mean(dd))])
    L += ["## 3. Nhãn đúng và diễn đạt", "",
          "Tỷ lệ trên tất cả lượt chấm (mỗi câu trả lời 2 lượt). Nhãn: 2 = đúng và đủ ý chính, 1 = đúng một phần, "
          "0 = không có ý đúng. Diễn đạt: 2 = mạch lạc, 1 = lủng củng/lặp/bị cắt, 0 = khó hiểu.", ""]
    L += bang_md(["Nhóm câu", "Mô hình", "Nhãn 2", "Nhãn 1", "Nhãn 0", "Nhãn TB", "Diễn đạt 2", "Diễn đạt 1",
                  "Diễn đạt 0", "Diễn đạt TB"], rows)
    L.append("")

    # 4. Y sai/bia va ngoai ngu canh
    rows = []
    for ten_g, cs in theo_nhom():
        rows.append([ten_g] + [pct(mean(g_(c, mh, "co_sai") for c in cs)) for mh in MO_HINH] +
                    [vn(mean(g_(c, mh, "sai") for c in cs)) for mh in MO_HINH] +
                    [pct(mean(g_(c, mh, "co_ngoai") for c in cs)) for mh in MO_HINH])
    L += ["## 4. Ý sai/bịa và ý ngoài ngữ cảnh", ""]
    L += bang_md(["Nhóm câu", "Có ý sai/bịa: gốc", "Có ý sai/bịa: ft", "Số ý sai TB: gốc", "Số ý sai TB: ft",
                  "Có ý ngoài ngữ cảnh: gốc", "Có ý ngoài ngữ cảnh: ft"], rows)
    L += ["", "\"Có ý …\" là tỷ lệ câu trả lời có ít nhất 1 ý loại đó (trung bình 2 người chấm). Ý ngoài ngữ cảnh "
              "không sai nhưng không có căn cứ trong những gì mô hình được đưa, nên làm giảm P.", ""]

    # 5. So voi chi so tu dong
    rows = []
    for ten_g, cs in theo_nhom():
        rows.append([ten_g, len(cs), pt(mean(td(c, "goc", "f1") for c in cs)),
                     pt(mean(td(c, "finetune", "f1") for c in cs)),
                     pt_dau(mean(td(c, "finetune", "f1") - td(c, "goc", "f1") for c in cs)),
                     pt(mean(g_(c, "goc", "f1") for c in cs)), pt(mean(g_(c, "finetune", "f1") for c in cs)),
                     pt_dau(mean(g_(c, "finetune", "f1") - g_(c, "goc", "f1") for c in cs))])
    L += ["## 5. So với chỉ số tự động", ""]
    L += bang_md(["Nhóm câu", "Số câu", "F1 tự động: gốc", "F1 tự động: ft", "Chênh tự động", "F1 thủ công: gốc",
                  "F1 thủ công: ft", "Chênh thủ công"], rows)
    tl = [(c, mh) for c in cap_du for mh in MO_HINH]
    rows = []
    for ten_cap, a, b in (("F1 tự động – F1 thủ công", "f1", "f1"), ("Độ phủ tự động – R thủ công", "do_phu", "r"),
                          ("ROUGE-L – Diễn đạt", "rougeL", "dien_dat"), ("F1 tự động – Nhãn đúng", "f1", "nhan")):
        x, y = [td(c, mh, a) for c, mh in tl], [g_(c, mh, b) for c, mh in tl]
        rows.append([ten_cap, vn(pearson(x, y)), vn(spearman(x, y))])
    d_td = [td(c, "finetune", "f1") - td(c, "goc", "f1") for c in cap_du]
    d_tc = [g_(c, "finetune", "f1") - g_(c, "goc", "f1") for c in cap_du]
    cung = sum(1 for a, b in zip(d_td, d_tc) if (a > NGUONG) == (b > NGUONG) and (a < -NGUONG) == (b < -NGUONG))
    L += ["", f"Tương quan trên {len(tl)} câu trả lời:", ""]
    L += bang_md(["Cặp chỉ số", "Pearson r", "Spearman ρ"], rows)
    L += ["", f"Kết luận thắng/hòa/thua theo F1 tự động và theo F1 thủ công trùng nhau ở {cung}/{len(cap_du)} câu.",
          ""]
    lech = sorted(((g_(c, mh, "f1") - td(c, mh, "f1"), c, mh) for c, mh in tl), key=lambda t: t[0])
    for tieu_de, ds in (("Người chấm cho cao hơn hẳn F1 tự động (đúng nghĩa nhưng diễn đạt khác đáp án)",
                         [t for t in reversed(lech) if t[0] > 0.2][:5]),
                        ("F1 tự động cao hơn hẳn người chấm (trùng chữ nhưng sai nghĩa, thiếu ý hoặc ngoài ngữ cảnh)",
                         [t for t in lech if t[0] < -0.2][:5])):
        if ds:
            L += [f"**{tieu_de}:**", ""]
            L += bang_md(["Mã", "Mô hình", "Câu hỏi", "Câu trả lời", "F1 tự động", "F1 thủ công"],
                         [[c["ma"], ten_mh[mh], rut_gon(c["cau_hoi"], 90), rut_gon(c["tra_loi"][mh], 160),
                           pt(td(c, mh, "f1")), pt(g_(c, mh, "f1"))] for _, c, mh in ds])
            L.append("")

    # 6. Dong thuan
    hai = [(k, d) for k, d in diem.items() if len(d) >= 2]
    L += ["## 6. Độ đồng thuận giữa người chấm", ""]
    if not hai:
        L += ["Chưa có câu trả lời nào đủ 2 người chấm.", ""]
    else:
        bau2 = [list(d.values()) for d in phieu_bau.values() if len(d) >= 2]
        dd_units = [[v["dien_dat"] for v in d.values() if v["dien_dat"] is not None] for _, d in hai]
        L += [f"{len(hai)} câu trả lời và {len(bau2)} lựa chọn So sánh có đủ 2 người chấm.", ""]
        L += bang_md(["Chỉ số", "Krippendorff α", "Trùng khớp hoàn toàn", "Chênh TB giữa 2 người"], [
            ["F1 thủ công", vn(kripp_alpha([[v["f1"] for v in d.values()] for _, d in hai])), "–",
             pt(mean([max(v["f1"] for v in d.values()) - min(v["f1"] for v in d.values()) for _, d in hai]))
             + " điểm"],
            ["P (theo ngữ cảnh)", vn(kripp_alpha([[v["p"] for v in d.values()] for _, d in hai])), "–",
             pt(mean([max(v["p"] for v in d.values()) - min(v["p"] for v in d.values()) for _, d in hai]))
             + " điểm"],
            ["R (theo nghĩa)", vn(kripp_alpha([[v["r"] for v in d.values()] for _, d in hai])), "–",
             pt(mean([max(v["r"] for v in d.values()) - min(v["r"] for v in d.values()) for _, d in hai]))
             + " điểm"],
            ["Nhãn đúng (0–2)", vn(kripp_alpha([[float(v["nhan"]) for v in d.values()] for _, d in hai])),
             pct(mean([1.0 if len({v["nhan"] for v in d.values()}) == 1 else 0.0 for _, d in hai])), "–"],
            ["Diễn đạt (0–2)", vn(kripp_alpha(dd_units)),
             pct(mean([1.0 if len(set(u)) == 1 else 0.0 for u in dd_units if len(u) >= 2])), "–"],
            ["So sánh A–B", vn(kripp_alpha([[float(x) for x in u] for u in bau2])),
             pct(mean([1.0 if len(set(u)) == 1 else 0.0 for u in bau2])), "–"]])
        cap = defaultdict(list)
        for _, d in hai:
            ks = sorted(d)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    cap[(ks[i], ks[j])].append((d[ks[i]], d[ks[j]]))
        rows = [[f"{a} và {b}", len(ps), pct(mean([1.0 if x["nhan"] == y["nhan"] else 0.0 for x, y in ps])),
                 vn(kappa_bac2([(x["nhan"], y["nhan"]) for x, y in ps])),
                 pt(mean([abs(x["f1"] - y["f1"]) for x, y in ps]))] for (a, b), ps in sorted(cap.items())]
        L += ["", "Theo từng cặp phiếu:", ""]
        L += bang_md(["Cặp phiếu", "Số câu trả lời", "Nhãn trùng", "Kappa có trọng số (nhãn)",
                      "Chênh F1 TB (điểm)"], rows)
        L += ["", "Đọc kết quả: α hoặc kappa từ 0,8 trở lên là đồng thuận tốt, 0,67–0,8 tạm chấp nhận, dưới 0,67 nên "
                  "họp thống nhất lại cách tách ý và cách xếp ý vào trong/ngoài ngữ cảnh rồi chấm lại các câu lệch.",
              ""]

    # 7. Can xem lai
    can = []
    for (mc, mh), d in sorted(diem.items()):
        if len(d) < 2:
            continue
        vs = sorted(d.items())
        f1s, nhs = [v["f1"] for _, v in vs], [v["nhan"] for _, v in vs]
        if max(f1s) - min(f1s) >= 0.3 or max(nhs) - min(nhs) >= 2:
            can.append([f"{mc}-{vs[0][1]['ab']}", "; ".join(f"phiếu {s}: F1 {pt(v['f1'])}, nhãn {v['nhan']}"
                                                            for s, v in vs),
                        "; ".join(f"{s}: {v['ghi_chu']}" for s, v in vs if v["ghi_chu"])])
    nguoc = [mc for mc, d in sorted(phieu_bau.items()) if len(d) >= 2 and max(d.values()) - min(d.values()) == 2]
    lech_nb = []
    for (mc, mh), d in sorted(diem.items()):
        for s, v in sorted(d.items()):
            if v["nhan"] == 2 and (v["r"] < 0.5 or v["sai"] > 0):
                lech_nb.append(f"{mc}-{v['ab']} phiếu {s}: nhãn 2 nhưng R = {vn(v['r'])}, ý sai = {v['sai']:g}")
            elif v["nhan"] == 0 and v["dat"] > 0:
                lech_nb.append(f"{mc}-{v['ab']} phiếu {s}: nhãn 0 nhưng có {v['dat']:g} ý đạt")
    L += ["## 7. Cần xem lại", ""]
    if can:
        L += [f"{len(can)} câu trả lời hai người chấm lệch nhiều (F1 chênh từ 30 điểm, hoặc một người nhãn 0 một người "
              "nhãn 2). Hai người cùng xem lại, mỗi người sửa phiếu của mình rồi chạy lại `tong_hop`:", ""]
        L += bang_md(["Mã", "Điểm", "Ghi chú"], can)
    else:
        L.append("Không có câu trả lời nào hai người chấm lệch nhiều.")
    if nguoc:
        L += ["", f"So sánh A–B ngược nhau (một người chọn A, người kia chọn B): {', '.join(nguoc)}."]
    if lech_nb:
        L += ["", f"Nhãn chưa khớp với các ý đã đánh dấu ({len(lech_nb)} lượt):", ""]
        L += [f"- {x}" for x in lech_nb[:40]] + ([f"- … và {len(lech_nb) - 40} lượt khác"] if len(lech_nb) > 40 else [])
    L.append("")

    # 8. Tien do
    L += ["## 8. Tiến độ và cảnh báo", ""]
    L += bang_md(["Phiếu", "Người chấm", "File", "Câu trả lời đã chấm", "Phải chấm"],
                 [[s, t or "", f, a, b] for s, t, f, a, b in tien_do])
    thieu_phieu = sorted(set(range(1, meta["so_nguoi"] + 1)) - set(phieu))
    if thieu_phieu:
        L += ["", f"Chưa có phiếu số: {', '.join(map(str, thieu_phieu))}."]
    if canh_bao:
        L += ["", f"{len(canh_bao)} cảnh báo khi đọc phiếu:", ""] + [f"- {x}" for x in canh_bao[:60]]
        if len(canh_bao) > 60:
            L.append(f"- … và {len(canh_bao) - 60} cảnh báo khác")
    L.append("")

    bc = os.path.join(out, "bao_cao_cham_tay.md")
    with open(bc, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    kq = os.path.join(out, "ket_qua_cham_tay.csv")
    cot_g = ("f1", "p", "r", "khong_sai", "nhan", "dien_dat", "co_sai", "sai", "co_ngoai", "ngoai")
    with open(kq, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["ma", "mo_hinh", "nhom", "doc_id", "trang", "cau_hoi", "so_y", "so_luot_cham",
                     "f1_thu_cong", "p_ngu_canh", "r_ngu_nghia", "ty_le_y_khong_sai", "nhan_tb", "dien_dat_tb",
                     "co_y_sai", "so_y_sai_tb", "co_y_ngoai_ngu_canh", "so_y_ngoai_tb", "so_sanh_nghieng_ve",
                     "f1_tu_dong", "do_phu_tu_dong", "rougeL_tu_dong", "giay"]
                    + [f"{k}_{i}" for i in (1, 2) for k in ("phieu", "dau", "them_nc", "them_ngoai", "sai", "nhan",
                                                            "dien_dat", "f1", "ghi_chu")] + ["tra_loi"])
        for c in khoa["cau"]:
            b = bau_tb(c)
            for mh in MO_HINH:
                g = gop.get((c["ma"], mh))
                d = sorted(diem.get((c["ma"], mh), {}).items())
                r = [c["ma"] + "-" + ("A" if c["A"] == mh else "B"), mh, c["nhom"], c["doc_id"], c["trang"],
                     c["cau_hoi"], len(c["y"]), len(d)]
                r += [("" if g is None or g[k] is None else round(g[k], 4)) for k in cot_g]
                r.append("" if b is None else round(b * (1 if mh == "finetune" else -1), 4))
                r += [td(c, mh, k) for k in ("f1", "do_phu", "rougeL", "giay")]
                for i in range(2):
                    if i < len(d):
                        s, v = d[i]
                        r += [s, "".join("x" if x == 1 else "p" if x == 0.5 else "." for x in v["dau"]), v["nc"],
                              v["ngoai"], v["sai"], v["nhan"], v["dien_dat"], round(v["f1"], 4), v["ghi_chu"]]
                    else:
                        r += [""] * 9
                r.append(c["tra_loi"][mh])
                wr.writerow(r)

    print(f"Da doc {len(phieu)} phieu, {tong_luot}/{can_luot} luot cham; {len(cap_du)} cau du diem 2 mo hinh")
    for ten_g, cs in theo_nhom():
        a, b = mean(g_(c, "goc", "f1") for c in cs), mean(g_(c, "finetune", "f1") for c in cs)
        print(f"  {ascii_(ten_g.replace('*', '')):<22} n={len(cs):<3} F1 thu cong goc {100 * a:5.1f}  {ft} "
              f"{100 * b:5.1f}  ({100 * (b - a):+.1f})")
    if canh_bao:
        print(f"  {len(canh_bao)} canh bao - xem muc 8 cua bao cao")
    print(f"Bao cao: {hien(bc)}\nTung cau: {hien(kq)}")


def main(argv=None):
    utf8_stdout()
    p = argparse.ArgumentParser(description="Cham thu cong cau tra loi truoc/sau fine-tune (Giai doan 4, Nhom 7)")
    sub = p.add_subparsers(dest="lenh", required=True)
    t = sub.add_parser("tao", help="tao phieu cham an danh tu so_sanh_truoc_sau.csv")
    t.add_argument("--so_sanh", default=DEFAULT_SO_SANH, help="dau ra eval_truoc_sau.py")
    t.add_argument("--y", default=DEFAULT_Y, help="CSV cac cau duoc chon va dap an da tach y (cot y1, y2, ...)")
    t.add_argument("--records", nargs="+", default=list(DEFAULT_RECORDS), help="jsonl co anh va ngu canh")
    t.add_argument("--out", default=DEFAULT_OUT)
    t.add_argument("--so_nguoi", type=int, default=4)
    t.add_argument("--ten", default="", help='ten nguoi cham theo thu tu phieu, vd "Phuong,Hai,Tung,Bao Anh"')
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--ten_ft", default="v2", help="ten mo hinh fine-tune trong bao cao")
    t.add_argument("--ghi_de", action="store_true", help="cho phep tao khoa.json moi khac khoa cu")
    h = sub.add_parser("tong_hop", help="doc phieu da cham, tinh chi so thu cong va do dong thuan")
    h.add_argument("--out", default=DEFAULT_OUT, help="thu muc co khoa.json")
    h.add_argument("--da_cham", default=None, help="thu muc phieu da cham (mac dinh <out>/da_cham)")
    args = p.parse_args(argv)
    (cmd_tao if args.lenh == "tao" else cmd_tong_hop)(args)


if __name__ == "__main__":
    main()
