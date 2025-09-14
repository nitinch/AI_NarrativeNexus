import os
import re
import pandas as pd
from email import policy
from email.parser import BytesParser

# ---------- CONFIG ----------
ROOT = "req_data/20news_18828"
OUT_XLSX = os.path.join("processed", "20news_18828_clean_50.xlsx")
OUT_CSV = os.path.join("processed", "20news_18828_clean_50.csv")
MAX_FILES_PER_CATEGORY = 500
MIN_BODY_LEN = 60  # drop rows shorter than this after cleaning
# ----------------------------

# remove illegal XML characters for Excel / openpyxl
_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def remove_illegal_xml_chars(s):
    if not isinstance(s, str):
        return s
    return _ILLEGAL_XML_RE.sub("", s)

HEADER_PATTERN = re.compile(
    r"^(archive-name|from|subject|path|xref|organization|lines|newsgroups|message-id|keywords|date|sender):",
    re.I,
)

QUOTE_LINE = re.compile(r"^\s*([>|\|])")
REPLY_MARKER = re.compile(r"(writes:|wrote:|In article\s*<)", re.I)
SIG_SEP = re.compile(r"^\s*--\s*$")


def try_extract_from_eml_bytes(content_bytes):
    try:
        msg = BytesParser(policy=policy.default).parsebytes(content_bytes)
        if msg.is_multipart():
            parts = []
            for part in msg.walk():
                ct = part.get_content_type()
                if ct == "text/plain":
                    try:
                        parts.append(part.get_content().strip())
                    except Exception:
                        try:
                            parts.append(part.get_payload(decode=True).decode("latin1", errors="replace").strip())
                        except Exception:
                            continue
            return "\n\n".join([p for p in parts if p])
        else:
            try:
                return msg.get_content().strip()
            except Exception:
                return msg.get_payload(decode=True).decode("latin1", errors="replace").strip()
    except Exception:
        return None


def extract_body_by_heuristic(raw_text):
    txt = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", txt, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]

    lines = []
    for line in body.splitlines():
        if HEADER_PATTERN.match(line):
            continue
        if QUOTE_LINE.match(line):
            continue
        if REPLY_MARKER.search(line):
            continue
        if SIG_SEP.match(line):
            break
        if re.match(r"^\s*[-=]{3,}\s*$", line):
            continue
        lines.append(line)

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    paras = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
    if not paras:
        paras = [" ".join([l.strip() for l in cleaned.splitlines() if l.strip()])]

    for p in paras:
        if len(p) >= MIN_BODY_LEN:
            return p
    return max(paras, key=len) if paras else ""


def extract_main_body_from_file(path):
    try:
        with open(path, "rb") as f:
            raw_bytes = f.read()
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return ""

    text = try_extract_from_eml_bytes(raw_bytes)
    if text:
        text = text.replace("\r\n", "\n")
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        for p in paras:
            if len(p) >= MIN_BODY_LEN and not QUOTE_LINE.match(p):
                p = "\n".join([ln for ln in p.splitlines() if not HEADER_PATTERN.match(ln)])
                return p.strip()
        return extract_body_by_heuristic(text)

    try:
        raw_text = raw_bytes.decode("latin1")
    except Exception:
        raw_text = raw_bytes.decode("utf-8", errors="replace")
    return extract_body_by_heuristic(raw_text)


def collect_eml_files(root_folder, max_files_per_category=MAX_FILES_PER_CATEGORY):
    items = []
    if not os.path.isdir(root_folder):
        return items

    subdirs = [d for d in sorted(os.listdir(root_folder)) if os.path.isdir(os.path.join(root_folder, d))]
    has_valid_subdirs = False
    for d in subdirs:
        full = os.path.join(root_folder, d)
        eml_files = [f for f in sorted(os.listdir(full)) if f.lower().endswith(".eml")]
        if eml_files:
            has_valid_subdirs = True
            for i, fname in enumerate(eml_files):
                if i >= max_files_per_category:
                    break
                items.append({"filename": fname, "category": d, "path": os.path.join(full, fname)})

    if has_valid_subdirs:
        return items

    all_eml = [f for f in sorted(os.listdir(root_folder)) if f.lower().endswith(".eml")]
    for i, fname in enumerate(all_eml):
        if i >= max_files_per_category:
            break
        items.append({"filename": fname, "category": "uncategorized", "path": os.path.join(root_folder, fname)})

    return items


def sanitize_for_excel(value: str) -> str:
    if isinstance(value, str) and value and value[0] in ('=', '+', '-', '@'):
        return "'" + value
    return value


def convert_20ng_to_excel(root_folder=ROOT, out_xlsx=OUT_XLSX, out_csv=OUT_CSV, max_files_per_category=MAX_FILES_PER_CATEGORY):
    os.makedirs("processed", exist_ok=True)
    files = collect_eml_files(root_folder, max_files_per_category)
    if not files:
        print(f"No .eml files found in {root_folder}")
        return

    rows = []
    for info in files:
        path = info["path"]
        filename = info["filename"]
        category = info["category"]
        body = extract_main_body_from_file(path)
        if not body or len(body.strip()) < MIN_BODY_LEN:
            continue
        rows.append({"filename": filename, "category": category, "text": body})

    if not rows:
        print("No usable bodies extracted (all empty or too short).")
        return

    df = pd.DataFrame(rows)

    # Clean string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].apply(lambda v: sanitize_for_excel(v) if isinstance(v, str) else v)
        df[col] = df[col].apply(remove_illegal_xml_chars)

    # Save CSV + Excel
    df.to_csv(out_csv, index=False, encoding="utf-8")
    df.to_excel(out_xlsx, index=False, engine="openpyxl")

    print(f"✅ Saved {len(df)} rows across {df['category'].nunique()} categories → {out_xlsx} and {out_csv}")

if __name__ == "__main__":
    convert_20ng_to_excel()