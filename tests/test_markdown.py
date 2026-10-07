"""Offline self-check for chunk_markdown on fenced code. No network."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # not installed: package sits at the repo root

from r_doc_builder.sources import chunk_markdown


def test_code_fences_are_kept_verbatim():
    """Inside a fence, "# install" is a shell comment, not a header (it used to split the block and
    become the next chunk's hierarchy), and import/export lines and <T> are code, not MDX (they were
    deleted). Outside fences MDX is still stripped."""
    bash = "```bash\n# install the client\nexport TOKEN=x\npip install x\n```"
    java = "~~~java\nimport java.util.List;\nList<String> names;\n~~~"
    text, chunks = chunk_markdown(
        f"import Tabs from '@theme/Tabs';\n# Setup\n<Tabs>\nInstall it:\n{bash}\nIn Java:\n{java}\n</Tabs>\n## Next\nmore")
    assert [c["hierarchy"] for c in chunks] == ["Setup", "Setup, Next"], chunks
    assert bash in chunks[0]["content"] and java in chunks[0]["content"], chunks[0]["content"]
    assert "import Tabs" not in text and "<Tabs>" not in text and "</Tabs>" not in text


if __name__ == "__main__":
    test_code_fences_are_kept_verbatim()
    print("ok")
