import argparse
import base64
import json
import os
import shutil
import sys
from pathlib import Path

import github.Auth
from dotenv import load_dotenv
from github import Github, GithubException

DEFAULT_REPO = "langchain-ai/langgraph"
DEFAULT_BRANCH = "main"
DEFAULT_OUT_DIR = "data/raw"

load_dotenv()


def write_file_and_metainfo(dst: Path, content: bytes):
    """
    Write file only if contents changed (based on sha256).
    Returns True if written, False if unchanged.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo in org/name format")
    ap.add_argument("--branch", default=DEFAULT_BRANCH, help="branch to fetch")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR, help="output directory")
    args = ap.parse_args()

    # Remove everything before checkout.
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("WARNING: GITHUB_TOKEN is not set. You may hit low rate limits for anonymous access.", file=sys.stderr)

    gh = Github(auth=github.Auth.Token(token), per_page=100) if token else Github(per_page=100)

    try:
        repo = gh.get_repo(args.repo)
    except GithubException as e:
        print(f"Failed to access repo {args.repo}: {e}", file=sys.stderr)
        sys.exit(1)

    docs_prefix = args.docs_dir.strip("/") + "/"
    md_suffixes = (".md", ".mdx")

    total = 0

    # Efficient way: list the entire tree (recursive) once, then filter paths.
    try:
        tree = repo.get_git_tree(args.branch, recursive=True).tree
    except GithubException as e:
        print(f"Failed to fetch git tree for branch '{args.branch}': {e}", file=sys.stderr)
        sys.exit(2)

    manifest = []
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
        write_file_and_metainfo(dst, raw)
        print(f"✓ updated: {rel}")

        total += 1
        manifest.append({"rel": str(rel), "source_url": gh_url})

    print(f"\nDone. Files scanned: {total}")
    print(f"Output dir: {out_dir.resolve()}")

    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
