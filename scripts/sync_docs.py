#!/usr/bin/env python3
"""
Sync documentation from asset/docs to docs folder.

NOTE: This script is for MIGRATION ONLY.
After migration, write all docs directly in docs/ folder.

The docs/ folder is the single source of truth.
"""

import shutil
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSET_DOCS = PROJECT_ROOT / "asset" / "docs"
DOCS_DIR = PROJECT_ROOT / "docs"


def sync_docs():
    """
    Sync documentation files from asset/docs to docs/.

    Priority: docs/ files take precedence over asset/docs/ files.
    Only copies files that don't exist in docs/.
    """

    # Files that already exist in docs/ (these take priority)
    existing_files = set()
    for path in DOCS_DIR.rglob("*"):
        if path.is_file():
            existing_files.add(path.relative_to(DOCS_DIR))

    copied_count = 0
    skipped_count = 0

    # Copy files from asset/docs to docs/ (only if not exists)
    for src_path in ASSET_DOCS.rglob("*"):
        if src_path.is_file():
            rel_path = src_path.relative_to(ASSET_DOCS)
            dst_path = DOCS_DIR / rel_path

            if rel_path in existing_files:
                # Skip - docs/ version takes priority
                print(f"Skipped (exists in docs/): {rel_path}")
                skipped_count += 1
            else:
                # Copy from asset/docs
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {rel_path}")
                copied_count += 1

    # Copy logo to docs/assets
    logo_src = PROJECT_ROOT / "asset" / "logo.png"
    logo_dst = DOCS_DIR / "assets" / "logo.png"
    if logo_src.exists() and not logo_dst.exists():
        logo_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(logo_src, logo_dst)
        print(f"Copied: logo.png -> assets/logo.png")
        copied_count += 1

    print(f"\n✅ Sync complete!")
    print(f"   Copied: {copied_count} files")
    print(f"   Skipped: {skipped_count} files (already exist in docs/)")
    print(f"\n📝 NOTE: docs/ is the source of truth.")
    print(f"   Edit files in docs/, not asset/docs/")


if __name__ == "__main__":
    sync_docs()
