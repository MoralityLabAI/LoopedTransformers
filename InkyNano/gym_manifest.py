"""Hash and verify the cloud handoff without loading weight tensors."""
import argparse
import hashlib
import json
from pathlib import Path
from inventory import MODEL,REV


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def entries(root):
    return {p.relative_to(root).as_posix():{"bytes":p.stat().st_size,"sha256":sha(p)}
            for p in sorted(root.rglob("*")) if p.is_file() and p!=root/"manifest.json"}


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if (manifest.get("model") != MODEL or manifest.get("revision") != REV
            or not manifest.get("files") or manifest["files"] != entries(root)):
        raise ValueError(f"handoff integrity mismatch: {root}")
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("--exporter-commit")
    p.add_argument("--verify",action="store_true")
    a=p.parse_args()
    manifest=a.root/"manifest.json"
    if a.verify:
        print("verified",len(verify(a.root)["files"]),"files")
    else:
        actual=entries(a.root)
        if not a.exporter_commit or not actual:
            raise ValueError("need exporter commit and files")
        manifest.write_text(json.dumps({"model":MODEL,"revision":REV,"exporter_commit":a.exporter_commit,
                                        "files":actual},indent=2,sort_keys=True)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
