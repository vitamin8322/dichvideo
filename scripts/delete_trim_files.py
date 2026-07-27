from __future__ import annotations

import argparse
import sys
from pathlib import Path


def choose_folder() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"Could not open folder picker: {exc}")
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Choose folder to delete *_trim files")
    root.destroy()
    if not folder:
        return None
    return Path(folder)


def find_trim_files(folder: Path, recursive: bool) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.stem.endswith("_trim")
    )


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def show_matches(files: list[Path], folder: Path, recursive: bool) -> None:
    total_size = sum(path.stat().st_size for path in files)
    scope = "including subfolders" if recursive else "only this folder"
    print()
    print(f"Folder: {folder}")
    print(f"Scope: {scope}")
    print(f"Matched files: {len(files)} ({format_size(total_size)})")
    print()

    preview_count = min(30, len(files))
    for path in files[:preview_count]:
        print(path)
    if len(files) > preview_count:
        print(f"... and {len(files) - preview_count} more")


def confirm_delete(files: list[Path], folder: Path, recursive: bool) -> bool:
    show_matches(files, folder, recursive)
    print()
    answer = input("Delete these files? Type YES to confirm: ").strip()
    return answer == "YES"


def delete_files(files: list[Path]) -> tuple[int, list[tuple[Path, str]]]:
    deleted = 0
    errors: list[tuple[Path, str]] = []
    for path in files:
        try:
            path.unlink()
            deleted += 1
        except Exception as exc:
            errors.append((path, str(exc)))
    return deleted, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete files whose filename stem ends with _trim, such as 0001_trim.wav."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        help="Folder to scan. If omitted, a folder picker will open.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the selected folder, not subfolders.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Delete without asking for confirmation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show matched files without deleting anything.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    folder = args.folder or choose_folder()
    if folder is None:
        print("No folder selected.")
        return 1

    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Folder does not exist: {folder}")
        return 1

    recursive = not args.no_recursive
    files = find_trim_files(folder, recursive)
    if not files:
        print(f"No *_trim files found in: {folder}")
        return 0

    if args.dry_run:
        show_matches(files, folder, recursive)
        print()
        print("Dry run only. No files were deleted.")
        return 0

    if not args.yes and not confirm_delete(files, folder, recursive):
        print("Canceled. No files were deleted.")
        return 0

    deleted, errors = delete_files(files)
    print()
    print(f"Deleted: {deleted}")
    if errors:
        print(f"Failed: {len(errors)}")
        for path, message in errors[:20]:
            print(f"- {path}: {message}")
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
