import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import github.Auth
from dotenv import load_dotenv
from github import Github, GithubException

DEFAULT_REPO = "langchain-ai/langgraph"
DEFAULT_BRANCH = "main"
DEFAULT_DOCS_DIR = "docs"
DEFAULT_OUT_DIR = "data/raw"

load_dotenv()


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def write_if_changed(dst: Path, source_url: str, content: bytes) -> bool:
    """
    Write file only if contents changed (based on sha256).
    Returns True if written, False if unchanged.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    hash_new = sha256_bytes(content)
    metainfo_file = dst.with_suffix(dst.suffix + ".meta.json")

    # read existing meta (if any)
    metainfo = {}
    if metainfo_file.exists():
        metainfo = json.loads(metainfo_file.read_text(encoding="utf-8"))

    if dst.exists() and metainfo_file.exists():
        old_hash = metainfo.get("sha256")
        if old_hash == hash_new:
            return False  # unchanged

    dst.write_bytes(content)
    if not metainfo:
        metainfo["source_url"] = source_url
    metainfo["sha256"] = hash_new
    metainfo_file.write_text(
        json.dumps(metainfo, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo in org/name format")
    ap.add_argument("--branch", default=DEFAULT_BRANCH, help="branch to fetch")
    ap.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR, help="docs directory inside repo")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR, help="output directory")
    args = ap.parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("WARNING: GITHUB_TOKEN is not set. You may hit low rate limits for anonymous access.", file=sys.stderr)

    gh = Github(auth=github.Auth.Token(token), per_page=100) if token else Github(per_page=100)

    try:
        repo = gh.get_repo(args.repo)
    except GithubException as e:
        print(f"Failed to access repo {args.repo}: {e}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs_prefix = args.docs_dir.strip("/") + "/"
    md_suffixes = (".md", ".mdx")

    total = 0
    changed = 0
    skipped = 0

    # Efficient way: list the entire tree (recursive) once, then filter paths.
    try:
        tree = repo.get_git_tree(args.branch, recursive=True).tree
    except GithubException as e:
        print(f"Failed to fetch git tree for branch '{args.branch}': {e}", file=sys.stderr)
        sys.exit(2)

    for item in tree:
        # We only care about blobs (files) under docs/ with .md/.mdx
        if item.type != "blob":
            continue
        path = item.path  # e.g. "docs/tutorials/agents.md"
        if not path.startswith(docs_prefix):
            continue
        if not path.lower().endswith(md_suffixes):
            continue

        rel = Path(path[len(docs_prefix):])  # e.g. tutorials/agents.md
        dst = out_dir / rel.with_suffix(".md")  # normalize .mdx -> .md if desired

        gh_url = f"https://github.com/{args.repo}/blob/{args.branch}/{path}"

        try:
            content_file = repo.get_contents(path, ref=args.branch)
            if content_file.encoding == "base64":
                raw = base64.b64decode(content_file.content)
            else:
                raw = content_file.decoded_content or content_file.content.encode("utf-8", errors="ignore")
        except GithubException as e:
            print(f"Skip {path}: {e}", file=sys.stderr)
            continue

        # Prepend source URL if missing
        wrote = write_if_changed(dst, gh_url, raw)
        total += 1
        if wrote:
            changed += 1
            print(f"✓ updated: {rel}")
        else:
            skipped += 1

    # TODO: Support deleting removed files.

    print(f"\nDone. Files scanned: {total}, updated: {changed}, unchanged: {skipped}")
    print(f"Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
