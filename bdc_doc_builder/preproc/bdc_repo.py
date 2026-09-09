import glob
import json
import os
import re
from urllib.parse import quote

import yaml
from tqdm import tqdm


def clean_path(path_str, root_dir):
    cleaned_root = re.sub(r"(?:\.{2}/)*", "", root_dir)
    pattern = f".*?{re.escape(cleaned_root)}"
    cleaned_path = re.sub(pattern, "", path_str.replace("\\", "/"))
    return cleaned_root + cleaned_path.lstrip("/")


def parse_fellow_files(file_path, root_dir):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        yaml_content = content.replace("---\n", "", 1).rsplit("---", 1)[0]
        fellow_data = yaml.safe_load(yaml_content)
        fellow_data["file_path"] = clean_path(file_path, root_dir)
        fellow_data["relative_file_path"] = "/".join(fellow_data["file_path"].split("/")[1:])
        return fellow_data
    except yaml.YAMLError as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def get_fellow_files(fellow_dir, root_dir, base_url=None, remote_file_dir=None):
    fellows = []
    for file_path in tqdm(glob.glob(os.path.join(fellow_dir, "*.md")), desc="Reading fellow files"):
        fellow_data = parse_fellow_files(file_path, root_dir)
        if not fellow_data:
            continue
        if remote_file_dir and "relative_file_path" in fellow_data:
            fellow_data["remote_file_path"] = remote_file_dir + fellow_data["relative_file_path"]
        if base_url and "name" in fellow_data:
            fellow_data["page_url"] = base_url + "about/bdc-fellows/#:~:text=" + quote(fellow_data["name"])
        fellows.append({"metadata": fellow_data, "content": json.dumps(fellow_data, indent=4)})
    return fellows


def parse_simple_mdx_files(file_path, root_dir):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            raise ValueError("File does not contain valid YAML frontmatter")
        metadata = yaml.safe_load(parts[1])
        markdown_content = parts[2].strip()
        metadata["file_path"] = clean_path(file_path, root_dir)
        metadata["relative_file_path"] = "/".join(metadata["file_path"].split("/")[1:])
        return metadata, markdown_content
    except (yaml.YAMLError, ValueError) as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None


def get_data_mdx_files(updates_dir, root_dir, base_url=None, remote_file_dir=None):
    """MDX files directly under the directory plus index.mdx from subdirectories."""
    res = []
    all_mdx_paths = glob.glob(os.path.join(updates_dir, "*.mdx")) + glob.glob(os.path.join(updates_dir, "**/index.mdx"))
    for file_path in tqdm(all_mdx_paths, desc="Parsing MDX files"):
        metadata, content = parse_simple_mdx_files(file_path, root_dir)
        if metadata and content:
            if base_url and "path" in metadata:
                metadata["page_url"] = base_url + str(metadata["path"]).lstrip("/")
            if remote_file_dir and "relative_file_path" in metadata:
                metadata["remote_file_path"] = remote_file_dir + metadata["relative_file_path"]
            res.append({"metadata": metadata, "content": content})
    return res


def clean_mdx(file_path):
    """Strip JSX from a BDC web page MDX, returning (frontmatter, markdown)."""
    tags_to_remove = ["ButtonContainer", "NextStepsCard"]

    with open(file_path, "r", encoding="utf8") as file:
        content = file.read()

    yaml_header = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
    header_dict = yaml.safe_load(yaml_header.group(1)) if yaml_header else {}

    content = re.sub(r"^---\n.*?\n---", "", content, flags=re.DOTALL)

    page_content_match = re.search(r"<PageContent.*?>(.*?)</PageContent>", content, re.DOTALL)
    page_content = page_content_match.group(1) if page_content_match else ""

    floating_content_match = re.search(
        r"<FloatingContentWrapper.*?>(.*?)</FloatingContentWrapper>", page_content, flags=re.DOTALL
    )
    floating_content = floating_content_match.group(1) if floating_content_match else ""
    page_content = re.sub(r"<FloatingContentWrapper.*?</FloatingContentWrapper>", "", page_content, flags=re.DOTALL)

    for tag in tags_to_remove:
        page_content = re.sub(f"<{tag}.*?>.*?</{tag}>", "", page_content, flags=re.DOTALL)

    page_content = re.sub(r'<Link to="([^"]*)"[^>]*>(.*?)</Link>', r"[\2](\1)", page_content)
    cleaned_content = re.sub(r"<.*?>", "", page_content, flags=re.DOTALL)

    # re-insert the floating content after the first h2 section
    sections = re.split(r"\n##\s", cleaned_content)
    if len(sections) > 2:
        sections[2] = sections[2] + "\n\n" + floating_content.strip() + "\n\n"
    cleaned_content = "## ".join(sections)

    return header_dict, cleaned_content.strip()


def get_all_mdx_paths(pages_dir, page_dir_paths, page_file_paths):
    all_paths, relative_paths = [], []

    for dir_path in page_dir_paths:
        full_dir_path = os.path.join(pages_dir, dir_path)
        if os.path.exists(full_dir_path):
            for file in os.listdir(full_dir_path):
                if file.endswith(".mdx"):
                    all_paths.append(os.path.join(full_dir_path, file))
                    relative_paths.append(os.path.join(dir_path, file))

    for file_path in page_file_paths:
        full_file_path = os.path.join(pages_dir, file_path)
        if os.path.exists(full_file_path):
            all_paths.append(full_file_path)
            relative_paths.append(file_path)

    return all_paths, relative_paths
