import re
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

VIDS_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1vUVMffOGz3Eggu4RjZSjSRToQ3Ydc2uHxCjSDdOpvug"
    "/edit?gid=397146063#gid=397146063"
)


def extract_sheet_id_from_url(url):
    for pattern in [r"/spreadsheets/d/([a-zA-Z0-9-_]+)", r"/d/([a-zA-Z0-9-_]+)", r"id=([a-zA-Z0-9-_]+)"]:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract sheet ID from URL: {url}")


def extract_gid_from_url(url):
    for pattern in [r"[?&]gid=(\d+)", r"#gid=(\d+)"]:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _gviz_url(url, out, sheet_name=None, gid=None):
    sheet_id = extract_sheet_id_from_url(url)
    gid = gid or extract_gid_from_url(url)
    base = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:{out}"
    if gid:
        return f"{base}&gid={gid}"
    if sheet_name:
        from urllib.parse import quote
        return f"{base}&sheet={quote(sheet_name)}"
    return base


def read_sheet_as_html(url, sheet_name=None, gid=None):
    response = requests.get(_gviz_url(url, "html", sheet_name, gid), timeout=60)
    response.raise_for_status()
    table = BeautifulSoup(response.text, "html.parser").find("table")
    if not table:
        raise Exception("No table found in HTML response")

    rows = table.find_all("tr")
    headers = [th.get_text(strip=True) for th in rows[0].find_all("th")] if rows else []
    data_rows = rows[1:] if headers else rows
    data = [[td.get_text(strip=True) for td in row.find_all("td")] for row in data_rows]
    data = [row for row in data if row]

    if not headers:
        # gviz emits no <th> for this sheet: the label row is the first populated data row
        for i, row in enumerate(data):
            if sum(bool(c) for c in row) > len(row) / 2:
                headers, data = row, data[i + 1:]
                break

    max_cols = max((len(row) for row in data), default=0)
    for row in data:
        row.extend([""] * (max_cols - len(row)))
    if not headers:
        headers = [f"Column_{i + 1}" for i in range(max_cols)]
    elif len(headers) < max_cols:
        headers += [f"Column_{i + 1}" for i in range(len(headers), max_cols)]
    else:
        headers = headers[:max_cols]

    return pd.DataFrame(data, columns=headers)


def read_sheet_as_csv(url, sheet_name=None, gid=None):
    response = requests.get(_gviz_url(url, "csv", sheet_name, gid), timeout=60)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def read_google_sheet(url, method="csv", sheet_name=None, gid=None):
    if method == "csv":
        return read_sheet_as_csv(url, sheet_name, gid)
    if method == "html":
        return read_sheet_as_html(url, sheet_name, gid)
    raise ValueError("Method must be 'csv' or 'html'")


def extract_file_id_from_url(gdrive_url):
    for pattern in [r"/file/d/([a-zA-Z0-9_-]+)", r"/open\?id=([a-zA-Z0-9_-]+)", r"/uc\?id=([a-zA-Z0-9_-]+)"]:
        match = re.search(pattern, gdrive_url)
        if match:
            return match.group(1)
    return None


def read_gdrive_file_as_text(gdrive_url, encoding="utf-8"):
    file_id = extract_file_id_from_url(gdrive_url)
    if not file_id:
        raise ValueError(f"Could not extract file ID from URL: {gdrive_url}")
    response = requests.get(f"https://drive.google.com/uc?export=download&id={file_id}", timeout=30)
    response.raise_for_status()
    return response.content.decode(encoding)


def parse_srt_content(srt_content):
    subtitles = []
    for block in re.split(r"\n\s*\n", srt_content.strip()):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0])
            time_match = re.match(
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})", lines[1]
            )
            if not time_match:
                continue
            sh, sm, ss, sms = map(int, time_match.groups()[:4])
            eh, em, es, ems = map(int, time_match.groups()[4:])
            subtitles.append({
                "index": index,
                "start_seconds": sh * 3600 + sm * 60 + ss + sms / 1000.0,
                "end_seconds": eh * 3600 + em * 60 + es + ems / 1000.0,
                "text": "\n".join(lines[2:]).strip(),
            })
        except (ValueError, IndexError):
            continue
    return subtitles


def extract_metadata_from_row(row):
    metadata = {}
    if "Name" in row.index:
        metadata["title"] = str(row["Name"])
    if "URL" in row.index:
        metadata["video_url"] = str(row["URL"])
    if "Summary" in row.index:
        metadata["summary"] = str(row["Summary"])
    return metadata


def parse_gdrive_srt_with_metadata(gdrive_url, row, encoding="utf-8"):
    subtitles = parse_srt_content(read_gdrive_file_as_text(gdrive_url, encoding))
    main_metadata = extract_metadata_from_row(row)
    text_list = [s["text"] for s in subtitles]
    metadata_list = [
        {"start_seconds": s["start_seconds"], "end_seconds": s["end_seconds"], "index": s["index"], **main_metadata}
        for s in subtitles
    ]
    return text_list, metadata_list


def proc_BDC_vids_Google_Sheet(url=VIDS_SHEET_URL, method="html",
                               transcript_col="Transcript (with timestamps)"):
    all_text, all_metadata = [], []
    try:
        df = read_google_sheet(url, method=method)
        print(f"Successfully read sheet with {len(df)} rows and {len(df.columns)} columns")

        if transcript_col not in df.columns:
            print(f"Column '{transcript_col}' not found. Available columns: {df.columns.tolist()}")
            return all_text, all_metadata

        df = df[df[transcript_col].str.len() > 0]
        df = df[~df.iloc[:, 0].str.contains("Name|name", na=False)].reset_index(drop=True)

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Reading transcripts"):
            gdrive_url = row[transcript_col]
            if pd.notna(gdrive_url) and "drive.google.com" in str(gdrive_url):
                try:
                    text_list, metadata_list = parse_gdrive_srt_with_metadata(gdrive_url, row)
                    all_text.append(text_list)
                    all_metadata.append(metadata_list)
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
            else:
                print(f"Row {idx}: No valid Google Drive URL found: {gdrive_url}")
    except Exception as e:
        print(f"Error reading sheet: {e}")

    return all_text, all_metadata
