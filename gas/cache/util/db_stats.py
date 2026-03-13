import argparse
from pathlib import Path

from gas.cache.content_db import ContentDB


# Color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def format_size(bytes_size):
    """Format file size in bytes as a human-readable string."""
    if bytes_size == 0:
        return "0 B"

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def get_directory_size(path: Path) -> int:
    """Get the total size of a directory in bytes."""
    total_size = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
    except (PermissionError, OSError):
        pass
    return total_size


def get_disk_usage_breakdown(base_dir: Path) -> dict:
    """Get disk space usage breakdown by subdirectories."""
    breakdown = {}

    try:
        # Check if base directory exists
        if not base_dir.exists():
            return {"error": f"Base directory {base_dir} does not exist"}

        # Get total size of base directory
        total_size = get_directory_size(base_dir)
        breakdown["total_size"] = total_size
        breakdown["total_size_formatted"] = format_size(total_size)

    except Exception as e:
        breakdown["error"] = str(e)

    return breakdown


def get_media_type_breakdown(db: ContentDB) -> dict:
    """Get media breakdown by type (real, synthetic, semisynthetic)."""
    import sqlite3

    breakdown = {
        "image": {"real": 0, "synthetic": 0, "semisynthetic": 0},
        "video": {"real": 0, "synthetic": 0, "semisynthetic": 0},
    }

    try:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT modality, media_type, COUNT(*) as count 
                FROM media 
                GROUP BY modality, media_type
            """
            )

            for modality, media_type, count in cursor.fetchall():
                if modality in breakdown and media_type in breakdown[modality]:
                    breakdown[modality][media_type] = count

    except Exception as e:
        print(f"Warning: Could not get media type breakdown: {e}")

    return breakdown


def get_dataset_media_breakdown(db: ContentDB) -> dict:
    """Get dataset media breakdown by modality and media type."""
    import sqlite3

    breakdown = {
        "image": {"real": 0, "synthetic": 0, "semisynthetic": 0},
        "video": {"real": 0, "synthetic": 0, "semisynthetic": 0},
    }

    try:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT modality, media_type, COUNT(*) as count 
                FROM media 
                WHERE prompt_id IS NULL
                GROUP BY modality, media_type
            """
            )

            for modality, media_type, count in cursor.fetchall():
                if modality in breakdown and media_type in breakdown[modality]:
                    breakdown[modality][media_type] = count

    except Exception as e:
        print(f"Warning: Could not get dataset media breakdown: {e}")

    return breakdown


def get_media_by_source_type(db: ContentDB) -> dict:
    """Get media breakdown by source type: scraper, generated, dataset, or miner."""
    import sqlite3

    breakdown = {
        "scraper": {"image": 0, "video": 0},
        "generated": {"image": 0, "video": 0},
        "dataset": {"image": 0, "video": 0},
        "miner": {"image": 0, "video": 0},
    }

    try:
        with sqlite3.connect(db.db_path) as conn:
            # Get media directly by source_type column
            cursor = conn.execute(
                """
                SELECT source_type, modality, COUNT(*) as count
                FROM media 
                WHERE modality IN ('image', 'video') 
                AND source_type IN ('scraper', 'generated', 'dataset', 'miner')
                GROUP BY source_type, modality
            """
            )

            for source_type, modality, count in cursor.fetchall():
                if modality in ["image", "video"] and source_type in breakdown:
                    breakdown[source_type][modality] = count

    except Exception as e:
        print(f"Warning: Could not get media by source type: {e}")

    return breakdown


def get_model_name_breakdown(db: ContentDB) -> dict:
    """Get detailed breakdown of model names for generated media (source_type='generated')."""
    import sqlite3

    breakdown = {}
    total_generated = 0

    try:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT model_name, COUNT(*) as count
                FROM media 
                WHERE source_type = 'generated' AND model_name IS NOT NULL
                GROUP BY model_name
                ORDER BY count DESC
            """
            )

            for model_name, count in cursor.fetchall():
                breakdown[model_name] = count
                total_generated += count

    except Exception as e:
        print(f"Warning: Could not get model name breakdown: {e}")

    return breakdown, total_generated


def get_dataset_name_breakdown(db: ContentDB) -> dict:
    """Get detailed breakdown of dataset names for dataset media (source_type='dataset')."""
    import sqlite3

    breakdown = {}
    total_dataset = 0

    try:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT dataset_name, COUNT(*) as count
                FROM media 
                WHERE source_type = 'dataset' AND dataset_name IS NOT NULL
                GROUP BY dataset_name
                ORDER BY count DESC
            """
            )

            for dataset_name, count in cursor.fetchall():
                breakdown[dataset_name] = count
                total_dataset += count

    except Exception as e:
        print(f"Warning: Could not get dataset name breakdown: {e}")

    return breakdown, total_dataset


def get_miner_breakdown(db: ContentDB) -> dict:
    """Get detailed breakdown of miner hotkeys for miner media (source_type='miner')."""
    import sqlite3

    breakdown = {}
    total_miner = 0

    try:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT hotkey, uid, verified, COUNT(*) as count
                FROM media 
                WHERE source_type = 'miner' AND hotkey IS NOT NULL
                GROUP BY hotkey, uid, verified
                ORDER BY uid, verified DESC
            """
            )

            for hotkey, uid, verified, count in cursor.fetchall():
                # Truncate hotkey for display (first 8 and last 8 characters)
                if len(hotkey) > 16:
                    display_hotkey = f"{hotkey[:8]}...{hotkey[-8:]}"
                else:
                    display_hotkey = hotkey
                
                verification_status = "✅ verified" if verified else "⏳ unverified"
                key = f"UID {uid} ({display_hotkey}) - {verification_status}"
                breakdown[key] = count
                total_miner += count

    except Exception as e:
        print(f"Warning: Could not get miner breakdown: {e}")

    return breakdown, total_miner


def get_miner_verification_stats(db: ContentDB) -> dict:
    """Get verification statistics for miner media."""
    import sqlite3
    
    stats = {
        "verified": {"image": 0, "video": 0, "total": 0},
        "unverified": {"image": 0, "video": 0, "total": 0},
        "total": {"image": 0, "video": 0, "total": 0}
    }
    
    try:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT modality, verified, COUNT(*) as count
                FROM media 
                WHERE source_type = 'miner'
                GROUP BY modality, verified
            """
            )
            
            for modality, verified, count in cursor.fetchall():
                if modality in ["image", "video"]:
                    status = "verified" if verified else "unverified"
                    stats[status][modality] = count
                    stats[status]["total"] += count
                    stats["total"][modality] += count
                    stats["total"]["total"] += count
                    
    except Exception as e:
        print(f"Warning: Could not get miner verification stats: {e}")
        
    return stats


def print_detailed_breakdowns(db: ContentDB):
    """Print detailed breakdowns of model names and dataset names."""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}🔍 Detailed Breakdowns{Colors.END}")
    print("=" * 60)

    # Get model name breakdown
    model_breakdown, total_generated = get_model_name_breakdown(db)

    if model_breakdown:
        print(f"\n{Colors.CYAN}🤖 Model Names (Source Type: Generated){Colors.END}")
        print("-" * 50)
        for model_name, count in model_breakdown.items():
            percentage = (count / total_generated * 100) if total_generated > 0 else 0
            print(
                f"{Colors.GREEN}• {model_name}:{Colors.END} {count:,} ({percentage:.1f}%)"
            )
        print(f"{Colors.BOLD}Total Generated:{Colors.END} {total_generated:,}")
    else:
        print(f"\n{Colors.CYAN}🤖 Model Names (Source Type: Generated){Colors.END}")
        print(f"{Colors.YELLOW}No generated media found{Colors.END}")

    # Get dataset name breakdown
    dataset_breakdown, total_dataset = get_dataset_name_breakdown(db)

    if dataset_breakdown:
        print(f"\n{Colors.CYAN}📊 Dataset Names (Source Type: Dataset){Colors.END}")
        print("-" * 50)
        for dataset_name, count in dataset_breakdown.items():
            percentage = (count / total_dataset * 100) if total_dataset > 0 else 0
            print(
                f"{Colors.GREEN}• {dataset_name}:{Colors.END} {count:,} ({percentage:.1f}%)"
            )
        print(f"{Colors.BOLD}Total Dataset:{Colors.END} {total_dataset:,}")
    else:
        print(f"\n{Colors.CYAN}📊 Dataset Names (Source Type: Dataset){Colors.END}")
        print(f"{Colors.YELLOW}No dataset media found{Colors.END}")

    # Get miner breakdown
    miner_breakdown, total_miner = get_miner_breakdown(db)

    if miner_breakdown:
        print(f"\n{Colors.CYAN}👨‍💼 Miners (Source Type: Miner){Colors.END}")
        print("-" * 50)
        for miner_key, count in miner_breakdown.items():
            percentage = (count / total_miner * 100) if total_miner > 0 else 0
            print(
                f"{Colors.GREEN}• {miner_key}:{Colors.END} {count:,} ({percentage:.1f}%)"
            )
        print(f"{Colors.BOLD}Total Miner:{Colors.END} {total_miner:,}")
    else:
        print(f"\n{Colors.CYAN}👨‍💼 Miners (Source Type: Miner){Colors.END}")
        print(f"{Colors.YELLOW}No miner media found{Colors.END}")

    # Get miner verification stats
    verification_stats = get_miner_verification_stats(db)
    
    if verification_stats["total"]["total"] > 0:
        print(f"\n{Colors.CYAN}🔍 Miner Media Verification Status{Colors.END}")
        print("-" * 50)
        
        verified_total = verification_stats["verified"]["total"]
        unverified_total = verification_stats["unverified"]["total"]
        total_total = verification_stats["total"]["total"]
        
        verified_pct = (verified_total / total_total * 100) if total_total > 0 else 0
        unverified_pct = (unverified_total / total_total * 100) if total_total > 0 else 0
        
        print(f"{Colors.GREEN}✅ Verified:{Colors.END} {verified_total:,} ({verified_pct:.1f}%)")
        print(f"   • Images: {verification_stats['verified']['image']:,}")
        print(f"   • Videos: {verification_stats['verified']['video']:,}")
        
        print(f"{Colors.YELLOW}⏳ Unverified:{Colors.END} {unverified_total:,} ({unverified_pct:.1f}%)")
        print(f"   • Images: {verification_stats['unverified']['image']:,}")
        print(f"   • Videos: {verification_stats['unverified']['video']:,}")
        
        print(f"{Colors.BOLD}Total Miner Media:{Colors.END} {total_total:,}")


def print_colored_table(db_path: Path, base_dir: Path, detailed: bool = False):
    """Print a clean, colored, tabular output of database statistics."""
    try:
        db = ContentDB(db_path)
        stats = db.get_stats()

        # Get all the breakdown data
        media_type_breakdown = get_media_type_breakdown(db)
        source_type_breakdown = get_media_by_source_type(db)
        dataset_breakdown = get_dataset_media_breakdown(db)

        total_media = stats["total_media"]

        # Get miner verification stats for the table
        verification_stats = get_miner_verification_stats(db)
        
        # Calculate totals for each category
        totals = {
            "image": {
                "real": 0,
                "synthetic": 0,
                "semisynthetic": 0,
                "dataset": 0,
                "scraper": 0,
                "generated": 0,
                "miner": 0,
                "miner_verified": 0,
                "miner_unverified": 0,
                "total": 0,
            },
            "video": {
                "real": 0,
                "synthetic": 0,
                "semisynthetic": 0,
                "dataset": 0,
                "scraper": 0,
                "generated": 0,
                "miner": 0,
                "miner_verified": 0,
                "miner_unverified": 0,
                "total": 0,
            },
        }

        # Fill in the data
        for modality in ["image", "video"]:
            # Media type breakdown (real, synthetic, semisynthetic)
            for media_type in ["real", "synthetic", "semisynthetic"]:
                totals[modality][media_type] = media_type_breakdown[modality].get(
                    media_type, 0
                )

            # Source type breakdown (dataset, scraper, generated, miner)
            totals[modality]["dataset"] = source_type_breakdown["dataset"].get(modality, 0)
            totals[modality]["scraper"] = source_type_breakdown["scraper"].get(modality, 0)
            totals[modality]["generated"] = source_type_breakdown["generated"].get(modality, 0)
            totals[modality]["miner"] = source_type_breakdown["miner"].get(modality, 0)
            
            # Miner verification breakdown
            totals[modality]["miner_verified"] = verification_stats["verified"].get(modality, 0)
            totals[modality]["miner_unverified"] = verification_stats["unverified"].get(modality, 0)

            # Calculate total for this modality
            totals[modality]["total"] = sum(
                totals[modality][col] for col in ["real", "synthetic", "semisynthetic"]
            )

        # Print header
        print(f"{Colors.BOLD}{Colors.HEADER}📊 GAS Database Statistics{Colors.END}")
        print(f"{Colors.CYAN}📍 Database: {db_path}{Colors.END}")
        print("╔" + "═" * 155 + "╗")

        # Print table header
        columns = [
            "real",
            "synthetic",
            "semisynthetic",
            "dataset",
            "scraper",
            "generated",
            "miner_verified",
            "miner_unverified",
            "total",
        ]
        header = f"║ {Colors.BOLD}{'Modality':<10}{Colors.END}"
        for i, col in enumerate(columns):
            if col == "total":
                header += f"{Colors.BOLD}{col.title():<8}"
            elif col == "miner_verified":
                header += f" {Colors.BOLD}{'Miner ✅':<12}"
            elif col == "miner_unverified":
                header += f" {Colors.BOLD}{'Miner ⏳':<12}"
            else:
                header += f" {Colors.BOLD}{col.title():<12}"
        header += f" ║"
        print(header)
        print("╟" + "─" * 155 + "╢")

        # Print table rows
        for modality in ["image", "video"]:
            row = f"║ {Colors.BOLD}{modality.title():<10}{Colors.END}"
            for i, col in enumerate(columns):
                count = totals[modality][col]
                if col == "total":
                    cell = f"{count:<7,}"
                else:
                    percentage = (count / total_media * 100) if total_media > 0 else 0
                    cell = f"{count:>4,} | {percentage:.1f}%"
                
                # Format cell consistently
                if col == "total":
                    row += f"   {cell:>8}"
                else:
                    cell = f"{cell:<11}"
                    row += f" {cell:<11}"

            row += f" ║"
            print(row)

        print("╚" + "═" * 155 + "╝")

        # Print summary statistics
        print(f"\n{Colors.BOLD}{Colors.GREEN}📈 Summary Statistics{Colors.END}")
        print("=" * 40)

        # Get additional stats
        total_prompts = stats["total_prompts"]
        database_size_mb = stats["database_size_mb"]

        # Calculate search queries count
        try:
            import sqlite3

            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM prompts WHERE content_type = 'search_query'"
                )
                total_queries = cursor.fetchone()[0]
        except Exception:
            total_queries = 0

        # Get disk space usage
        disk_usage = get_disk_usage_breakdown(base_dir)
        disk_size = disk_usage.get("total_size_formatted", "Unknown")

        print(f"{Colors.CYAN}1. Total Prompts:{Colors.END} {total_prompts:,}")
        print(f"{Colors.CYAN}2. Total Search Queries:{Colors.END} {total_queries:,}")
        print(
            f"{Colors.CYAN}3. Database File Size:{Colors.END} {database_size_mb:.2f} MB"
        )
        print(f"{Colors.CYAN}4. Disk Space Usage:{Colors.END} {disk_size}")

        # Additional useful stats
        print(f"\n{Colors.YELLOW}📊 Additional Statistics{Colors.END}")
        print("-" * 30)
        print(f"{Colors.CYAN}Total Media:{Colors.END} {total_media:,}")
        print(
            f"{Colors.CYAN}Average Prompt Usage:{Colors.END} {stats['average_prompt_usage']:.2f}"
        )

        # Show breakdown by content type
        print(f"\n{Colors.YELLOW}📋 Content Type Breakdown{Colors.END}")
        print("-" * 30)
        for content_type, count in stats["prompt_counts"].items():
            print(f"{Colors.CYAN}{content_type.title()}:{Colors.END} {count:,}")

        # Show detailed breakdowns if requested
        if detailed:
            print_detailed_breakdowns(db)

    except Exception as e:
        print(f"{Colors.RED}❌ Error reading database: {e}{Colors.END}")
        return False

    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check database statistics")
    parser.add_argument(
        "--db-path",
        type=str,
        default="~/.cache/sn34/prompts.db",
        help="Path to the prompt database",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="~/.cache/sn34",
        help="Base directory for cache system",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed breakdown of model names and dataset names",
    )

    args = parser.parse_args()

    db_path = Path(args.db_path).expanduser()
    base_dir = Path(args.base_dir).expanduser()

    if db_path.exists():
        print_colored_table(db_path, base_dir, detailed=args.detailed)
    else:
        print(f"{Colors.RED}❌ Database not found: {db_path}{Colors.END}")
        print(
            "💡 The database will be created automatically when prompts are first added."
        )

    print(f"\n{Colors.GREEN}✅ Statistics check completed!{Colors.END}")


if __name__ == "__main__":
    main()
