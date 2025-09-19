import os
import pandas as pd
import re

# ---------------- HELPER: remove illegal XML characters for Excel ----------------
_ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def remove_illegal_xml_chars(s):
    if not isinstance(s, str):
        return s
    return _ILLEGAL_XML_RE.sub("", s)
# -------------------------------------------------------------------------------

# ---------------- MAIN CLEANING FUNCTION ----------------
def clean_body(raw_text):
    if pd.isna(raw_text):
        return ""

    # Split header/body (keep only body if present)
    parts = re.split(r"\n\s*\n", raw_text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else parts[0]

    cleaned_lines = []
    for line in body.splitlines():
        # Skip common metadata fields
        if re.match(
            r"^(archive-name|from|subject|path|xref|organization|lines|newsgroups|message-id|keywords|last-modified|version):",
            line,
            re.I,
        ):
            continue
        # Skip quoted or forwarded text
        if line.strip().startswith((">", "|")):
            continue
        # Stop at signature
        if line.strip().startswith("--"):
            break
        # Skip reply markers
        if re.search(r"In article\s*<.*?>", line, re.I):
            continue
        if re.search(r"writes:|wrote:", line, re.I):
            continue
        cleaned_lines.append(line)

    # Join and clean text
    body = "\n".join(cleaned_lines)
    body = re.sub(r"\S+@\S+", " ", body)                  # remove emails
    body = re.sub(r"http\S+|www\.\S+", " ", body)         # remove URLs
    body = re.sub(r"<[^>]+>", " ", body)                  # remove HTML tags
    body = re.sub(r"[^a-zA-Z0-9\s\.\,\!\?]", " ", body)   # remove special chars
    body = re.sub(r"\n{2,}", "\n", body)                  # normalize newlines
    body = re.sub(r"\s{2,}", " ", body)                   # normalize spaces
    body = body.lower().strip()
    body = _ILLEGAL_XML_RE.sub("", body)                  # remove illegal chars
    return body
# -------------------------------------------------------------------------------

# ---------------- SCRIPT ENTRYPOINT ----------------
if __name__ == "__main__":
    # Input from req_data, output to processed
    input_path = os.path.join("processed", "20news_18828_clean_50.xlsx")
    os.makedirs("processed", exist_ok=True)
    output_path = os.path.join("processed", "20news_18828_final_50.xlsx")

    # Load dataset
    df = pd.read_excel(input_path, engine="openpyxl")

    # Apply cleaning
    df["text"] = df["text"].apply(clean_body)

    # Drop rows where cleaned text is empty or too short
    df["text_len"] = df["text"].apply(lambda t: len(str(t).strip()))
    df = df[df["text_len"] >= 60]
    df = df.drop(columns=["text_len"])

    # Remove illegal characters in all string columns before saving
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].apply(remove_illegal_xml_chars)

    # Save cleaned dataset
    df.to_excel(output_path, index=False, engine="openpyxl")
    df.to_csv("processed/20news_18828_final_50.csv", index=False, encoding="utf-8")

    print(f"✅ Final dataset saved: {len(df)} rows, cleaned text only → {output_path}")
