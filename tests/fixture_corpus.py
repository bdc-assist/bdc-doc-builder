"""The fake BDC-shaped corpus in examples/fixture, shared by test_pipeline, test_e2e and the fixture
rehearsal (test_fixture_cli). Invented text, with every doc_type and metadata shape real BDC sources produce."""
import pickle
from pathlib import Path

import yaml

FIXTURE = Path(__file__).resolve().parent.parent / "examples" / "fixture"
DOC_TYPES = {"docs", "page", "update", "event", "fellow", "faq", "video"}


def write_extras(dest_dir, source=FIXTURE / "extras.yaml"):
    """examples/fixture/extras.yaml -> <dest_dir>/extras.pkl, in the pipeline's .pkl shape: the metadata
    only networked sources produce (Freshdesk category/folder, YouTube title/date). Committed as YAML, not
    pickle: reviewable, and loading a pickle runs code. Values must be str, int, float or bool: an unquoted
    YAML date loads as a date object, which a push would silently drop."""
    rows = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
    for row in rows:
        bad = {k: v for k, v in row["metadata"].items() if not isinstance(v, (str, int, float, bool))}
        if bad:
            raise ValueError(f"{source}: non-scalar metadata (quote dates): {bad}")
    path = Path(dest_dir) / "extras.pkl"
    with open(path, "wb") as f:
        pickle.dump(rows, f)
    return path
