#!/usr/bin/env python3
"""
Titan Cleanup Script

Removes:
- runs/ directory (test results)
- titan.db and related files (pattern database)
- context.db and related files (memory keeper)
- __pycache__ directories
- .pyc files

Usage:
    python cleanup.py           # Interactive mode (asks what to clean)
    python cleanup.py --force   # Force delete everything without confirmation
    python cleanup.py --dry-run # Show what would be deleted
    python cleanup.py --all     # Clean everything (with confirmation)
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


# Cleanup categories
CLEANUP_CATEGORIES = {
    "runs": {
        "name": "Test runs",
        "description": "runs/ directory with backtest results",
        "dirs": ["runs"],
        "files": [],
    },
    "patterns": {
        "name": "Pattern database",
        "description": "titan.db and related files",
        "dirs": [],
        "files": ["titan.db", "titan.db-shm", "titan.db-wal", "titan.db-journal"],
    },
    "context": {
        "name": "Context database",
        "description": "context.db (memory keeper)",
        "dirs": [],
        "files": ["context.db", "context.db-shm", "context.db-wal", "context.db-journal"],
    },
    "cache": {
        "name": "Python cache",
        "description": "__pycache__ directories and .pyc files",
        "dirs": ["__pycache__"],
        "files": [],
        "patterns": ["*.pyc", "*.pyo"],
    },
}

# Legacy compatibility
DIRS_TO_DELETE = ["runs", "__pycache__"]
FILES_TO_DELETE = [
    "titan.db", "titan.db-shm", "titan.db-wal", "titan.db-journal",
    "context.db", "context.db-shm", "context.db-wal", "context.db-journal",
]
PATTERNS_TO_DELETE = ["*.pyc", "*.pyo", "__pycache__"]


def get_size_str(size_bytes: int) -> str:
    """Convert bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_dir_size(path: Path) -> int:
    """Calculate total size of directory."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except (PermissionError, OSError):
        pass
    return total


def find_items_by_category(base_path: Path, categories: list) -> tuple:
    """Find items to delete based on selected categories."""
    dirs_found = []
    files_found = []
    total_size = 0

    for cat_key in categories:
        cat = CLEANUP_CATEGORIES.get(cat_key, {})

        # Check directories
        for dir_name in cat.get("dirs", []):
            dir_path = base_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                size = get_dir_size(dir_path)
                if dir_path not in [d[0] for d in dirs_found]:
                    dirs_found.append((dir_path, size))
                    total_size += size

        # Check files
        for file_name in cat.get("files", []):
            file_path = base_path / file_name
            if file_path.exists() and file_path.is_file():
                size = file_path.stat().st_size
                if file_path not in [f[0] for f in files_found]:
                    files_found.append((file_path, size))
                    total_size += size

        # Check patterns (recursive)
        for pattern in cat.get("patterns", []):
            for match in base_path.rglob(pattern):
                if match.is_dir() and match not in [d[0] for d in dirs_found]:
                    size = get_dir_size(match)
                    dirs_found.append((match, size))
                    total_size += size
                elif match.is_file() and match not in [f[0] for f in files_found]:
                    size = match.stat().st_size
                    files_found.append((match, size))
                    total_size += size

        # Special case: recursive __pycache__ for cache category
        if cat_key == "cache":
            for pycache in base_path.rglob("__pycache__"):
                if pycache.is_dir() and pycache not in [d[0] for d in dirs_found]:
                    size = get_dir_size(pycache)
                    dirs_found.append((pycache, size))
                    total_size += size

    return dirs_found, files_found, total_size


def find_items_to_delete(base_path: Path) -> tuple:
    """Find all items that would be deleted (all categories)."""
    return find_items_by_category(base_path, list(CLEANUP_CATEGORIES.keys()))


def select_categories_interactive(base_path: Path) -> list:
    """Interactive menu to select what to clean."""
    print("\n" + "=" * 60)
    print("  TITAN CLEANUP - Select what to clean")
    print("=" * 60)

    # Calculate sizes for each category
    category_sizes = {}
    for key in CLEANUP_CATEGORIES:
        _, _, size = find_items_by_category(base_path, [key])
        category_sizes[key] = size

    # Display options
    print("\n  Available categories:\n")
    options = list(CLEANUP_CATEGORIES.keys())
    for i, key in enumerate(options, 1):
        cat = CLEANUP_CATEGORIES[key]
        size = category_sizes[key]
        size_str = get_size_str(size) if size > 0 else "empty"
        status = f"({size_str})" if size > 0 else "(empty)"
        print(f"    {i}. [{key:10}] {cat['name']:20} {status}")
        print(f"                      {cat['description']}")

    print(f"\n    A. All of the above")
    print(f"    Q. Quit (cancel)")

    # Get user choice
    print()
    choice = input("  Enter numbers separated by comma (e.g., 1,2) or A for all: ").strip().upper()

    if choice == 'Q' or choice == '':
        return []

    if choice == 'A':
        return options

    # Parse selected numbers
    selected = []
    try:
        for num in choice.split(','):
            idx = int(num.strip()) - 1
            if 0 <= idx < len(options):
                selected.append(options[idx])
    except ValueError:
        print("  Invalid input. Please enter numbers or 'A'.")
        return []

    return selected


def print_summary(dirs_found: list, files_found: list, total_size: int) -> None:
    """Print summary of items to delete."""
    print("\n" + "=" * 60)
    print("  TITAN CLEANUP SUMMARY")
    print("=" * 60)

    if dirs_found:
        print("\n  Directories to delete:")
        for path, size in dirs_found:
            rel_path = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
            print(f"    {rel_path}/ ({get_size_str(size)})")

    if files_found:
        print("\n  Files to delete:")
        for path, size in files_found:
            rel_path = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
            print(f"    {rel_path} ({get_size_str(size)})")

    print("\n" + "-" * 60)
    print(f"  Total items: {len(dirs_found)} directories, {len(files_found)} files")
    print(f"  Total size:  {get_size_str(total_size)}")
    print("=" * 60)


def delete_items(dirs_found: list, files_found: list, verbose: bool = True) -> tuple:
    """Delete found items. Returns (deleted_count, error_count)."""
    deleted = 0
    errors = 0

    # Delete files first
    for file_path, _ in files_found:
        try:
            file_path.unlink()
            if verbose:
                print(f"  [DELETED] {file_path}")
            deleted += 1
        except Exception as e:
            print(f"  [ERROR] {file_path}: {e}")
            errors += 1

    # Delete directories (sorted by depth, deepest first)
    dirs_sorted = sorted(dirs_found, key=lambda x: len(x[0].parts), reverse=True)
    for dir_path, _ in dirs_sorted:
        try:
            shutil.rmtree(dir_path)
            if verbose:
                print(f"  [DELETED] {dir_path}/")
            deleted += 1
        except Exception as e:
            print(f"  [ERROR] {dir_path}: {e}")
            errors += 1

    return deleted, errors


def main():
    parser = argparse.ArgumentParser(
        description="Clean up Titan test runs and database files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cleanup.py              # Interactive mode (asks what to clean)
    python cleanup.py --all        # Clean all categories (with confirmation)
    python cleanup.py --force      # Delete everything without confirmation
    python cleanup.py --dry-run    # Show what would be deleted
    python cleanup.py --runs       # Clean only test runs
    python cleanup.py --patterns   # Clean only pattern database
    python cleanup.py --context    # Clean only context database
    python cleanup.py --cache      # Clean only Python cache
        """
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Delete everything without confirmation"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Clean all categories (with confirmation)"
    )
    parser.add_argument(
        "--runs",
        action="store_true",
        help="Clean runs/ directory"
    )
    parser.add_argument(
        "--patterns",
        action="store_true",
        help="Clean titan.db pattern database"
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="Clean context.db memory keeper"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Clean __pycache__ and .pyc files"
    )
    parser.add_argument(
        "--path", "-p",
        type=str,
        default=".",
        help="Base path to clean (default: current directory)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()
    base_path = Path(args.path).resolve()

    if not base_path.exists():
        print(f"Error: Path does not exist: {base_path}")
        sys.exit(1)

    # Determine which categories to clean
    selected_categories = []

    # Check for specific category flags
    if args.runs:
        selected_categories.append("runs")
    if args.patterns:
        selected_categories.append("patterns")
    if args.context:
        selected_categories.append("context")
    if args.cache:
        selected_categories.append("cache")

    # If --all or --force, clean everything
    if args.all or args.force:
        selected_categories = list(CLEANUP_CATEGORIES.keys())

    # If no specific categories and not --all/--force, show interactive menu
    if not selected_categories and not args.dry_run:
        selected_categories = select_categories_interactive(base_path)
        if not selected_categories:
            print("\n  Aborted.")
            sys.exit(0)

    # For dry-run without specific categories, show all
    if args.dry_run and not selected_categories:
        selected_categories = list(CLEANUP_CATEGORIES.keys())

    # Find items to delete
    dirs_found, files_found, total_size = find_items_by_category(base_path, selected_categories)

    if not dirs_found and not files_found:
        print("\n  Nothing to clean up!")
        sys.exit(0)

    # Print summary
    if not args.quiet:
        print_summary(dirs_found, files_found, total_size)

    # Dry run mode
    if args.dry_run:
        print("\n  [DRY RUN] No files were deleted.")
        sys.exit(0)

    # Confirmation (unless --force)
    if not args.force:
        print()
        response = input("  Proceed with deletion? [y/N]: ").strip().lower()
        if response not in ('y', 'yes'):
            print("\n  Aborted.")
            sys.exit(0)

    # Delete
    print("\n  Deleting...")
    deleted, errors = delete_items(dirs_found, files_found, verbose=not args.quiet)

    # Summary
    print("\n" + "-" * 60)
    print(f"  Deleted: {deleted} items")
    if errors:
        print(f"  Errors:  {errors} items")
    print(f"  Freed:   {get_size_str(total_size)}")
    print("=" * 60)

    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
