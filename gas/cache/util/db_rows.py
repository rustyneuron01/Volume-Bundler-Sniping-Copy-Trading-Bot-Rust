"""
Script to display the first N rows of either the prompts or media table.
"""

import argparse
import sys
import sqlite3
import json
from pathlib import Path
from datetime import datetime


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


def format_timestamp(timestamp):
    """Format timestamp as human-readable string."""
    if timestamp is None:
        return "N/A"
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError, OSError):
        return str(timestamp)


def truncate_string(s, max_length):
    """Truncate string to max_length, adding '...' if truncated."""
    if s is None:
        return "N/A"
    s = str(s)
    if len(s) <= max_length:
        return s
    return s[: max_length - 3] + "..."


def display_prompts_table(db_path: Path, n_rows: int):
    """Display the first N rows of the prompts table."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """
                SELECT id, content, content_type, created_at, used_count, last_used, source_media_id
                FROM prompts 
                ORDER BY created_at DESC 
                LIMIT ?
            """,
                (n_rows,),
            )
            rows = cursor.fetchall()

            if not rows:
                print(f"{Colors.YELLOW}No prompts found in the database.{Colors.END}")
                return

            print(
                f"{Colors.BOLD}{Colors.HEADER}💬 Prompts Table - First {len(rows)} rows{Colors.END}"
            )
            print("=" * 150)

            # Print header
            print(
                f"{Colors.BOLD}{Colors.CYAN}{'ID':<12} {'Content Type':<12} {'Content':<50} {'Created':<20} {'Used Count':<10} {'Last Used':<20}{Colors.END}"
            )
            print("-" * 150)

            # Print rows
            for row in rows:
                id_short = row["id"][:8] + "..." if row["id"] else "N/A"
                content_preview = truncate_string(row["content"], 47)
                created_str = format_timestamp(row["created_at"])
                last_used_str = format_timestamp(row["last_used"])
                used_count = row["used_count"] or 0

                print(
                    f"{id_short:<12} {row['content_type']:<12} {content_preview:<50} {created_str:<20} {used_count:<10} {last_used_str:<20}"
                )

            print("=" * 150)

    except Exception as e:
        print(f"{Colors.RED}❌ Error reading prompts table: {e}{Colors.END}")


def display_media_table(db_path: Path, n_rows: int, source_type_filter: str = None, miner_uid_filter: int = None, last_24h_filter: bool = False, filepaths_only: bool = False, include_prompts: bool = False):
    """Display the first N rows of the media table."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build query with filters
            conditions = []
            params = []
            
            if source_type_filter:
                conditions.append("m.source_type = ?")
                params.append(source_type_filter)
            
            if miner_uid_filter is not None:
                conditions.append("m.uid = ?")
                params.append(miner_uid_filter)
            
            if last_24h_filter:
                # 24 hours = 24 * 60 * 60 = 86400 seconds
                twenty_four_hours_ago = datetime.now().timestamp() - 86400
                conditions.append("m.created_at >= ?")
                params.append(twenty_four_hours_ago)
            
            where_clause = ""
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
            
            if filepaths_only or include_prompts:
                # Select file_path and potentially prompt content
                if include_prompts:
                    query = f"""
                        SELECT m.file_path, p.content as prompt_content
                        FROM media m
                        LEFT JOIN prompts p ON m.prompt_id = p.id
                        {where_clause}
                        ORDER BY m.created_at DESC 
                        LIMIT ?
                    """
                else:
                    query = f"""
                        SELECT file_path
                        FROM media m
                        {where_clause}
                        ORDER BY m.created_at DESC 
                        LIMIT ?
                    """
            else:
                # Select all fields for normal display
                query = f"""
                    SELECT id, prompt_id, file_path, modality, media_type, source_type,
                           model_name, download_url, scraper_name, dataset_name, 
                           dataset_source_file, dataset_index, created_at, generation_args,
                           mask_path, timestamp, resolution, file_size, format, uid, hotkey,
                           verified, failed_verification, rewarded
                    FROM media m
                    {where_clause}
                    ORDER BY m.created_at DESC 
                    LIMIT ?
                """
            
            params.append(n_rows)
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                filter_parts = []
                if source_type_filter:
                    filter_parts.append(f"source_type='{source_type_filter}'")
                if miner_uid_filter is not None:
                    filter_parts.append(f"uid={miner_uid_filter}")
                if last_24h_filter:
                    filter_parts.append("last 24 hours")
                
                filter_msg = f" with filters: {', '.join(filter_parts)}" if filter_parts else ""
                print(
                    f"{Colors.YELLOW}No media found in the database{filter_msg}.{Colors.END}"
                )
                return

            # Handle filepaths-only and include-prompts modes
            if filepaths_only or include_prompts:
                filter_parts = []
                if source_type_filter:
                    filter_parts.append(f"source_type='{source_type_filter}'")
                if miner_uid_filter is not None:
                    filter_parts.append(f"uid={miner_uid_filter}")
                if last_24h_filter:
                    filter_parts.append("last 24 hours")
                
                filter_info = f" (filtered by: {', '.join(filter_parts)})" if filter_parts else ""
                
                if include_prompts:
                    print(f"{Colors.BOLD}{Colors.HEADER}📁 File Paths with Associated Prompts{filter_info}{Colors.END}")
                    print("=" * 120)
                    for i, row in enumerate(rows, 1):
                        print(f"{Colors.CYAN}{i:3}. File:{Colors.END} {row['file_path']}")
                        # Handle prompt content safely
                        try:
                            prompt_content = row['prompt_content'] if 'prompt_content' in row.keys() else None
                        except (KeyError, TypeError):
                            prompt_content = None
                            
                        if prompt_content and prompt_content.strip():
                            # Truncate long prompts for readability
                            truncated_prompt = truncate_string(prompt_content, 100)
                            print(f"{Colors.YELLOW}     Prompt:{Colors.END} {truncated_prompt}")
                        else:
                            print(f"{Colors.YELLOW}     Prompt:{Colors.END} {Colors.RED}No associated prompt{Colors.END}")
                        print()  # Add spacing between entries
                else:
                    print(f"{Colors.BOLD}{Colors.HEADER}📁 File Paths{filter_info}{Colors.END}")
                    print("=" * 80)
                    for row in rows:
                        print(row["file_path"])
                return

            # Print header for normal display
            filter_parts = []
            if source_type_filter:
                filter_parts.append(f"source_type='{source_type_filter}'")
            if miner_uid_filter is not None:
                filter_parts.append(f"uid={miner_uid_filter}")
            if last_24h_filter:
                filter_parts.append("last 24 hours")
            
            filter_info = f" (filtered by: {', '.join(filter_parts)})" if filter_parts else ""
            print(
                f"{Colors.BOLD}{Colors.HEADER}🎬 Media Table - First {len(rows)} rows{filter_info}{Colors.END}"
            )
            print("=" * 200)

            # Determine which columns to show based on source type
            if source_type_filter == "miner":
                # Show miner-specific columns
                print(
                    f"{Colors.BOLD}{Colors.CYAN}{'ID':<12} {'UID':<6} {'Hotkey':<20} {'File Path':<35} {'Modality':<10} {'Type':<12} {'Verified':<10} {'Created':<20} {'Size':<12}{Colors.END}"
                )
                print("-" * 200)

                # Print miner-specific info for each row
                for row in rows:
                    id_short = row["id"][:8] + "..." if row["id"] else "N/A"
                    uid_str = str(row["uid"]) if row["uid"] is not None else "N/A"
                    hotkey_short = (
                        row["hotkey"][:8] + "..." + row["hotkey"][-8:] 
                        if row["hotkey"] and len(row["hotkey"]) > 16 
                        else row["hotkey"] or "N/A"
                    )
                    file_path_short = truncate_string(
                        Path(row["file_path"]).name if row["file_path"] else "N/A", 30
                    )
                    created_str = format_timestamp(row["created_at"])
                    verified_str = "✅ Yes" if row["verified"] else "⏳ No"

                    # Format file size
                    size_str = "N/A"
                    if row["file_size"]:
                        if row["file_size"] < 1024:
                            size_str = f"{row['file_size']}B"
                        elif row["file_size"] < 1024 * 1024:
                            size_str = f"{row['file_size']/1024:.1f}KB"
                        else:
                            size_str = f"{row['file_size']/(1024*1024):.1f}MB"

                    print(
                        f"{id_short:<12} {uid_str:<6} {hotkey_short:<20} {file_path_short:<35} {row['modality'] or 'N/A':<10} {row['media_type'] or 'N/A':<12} {verified_str:<10} {created_str:<20} {size_str:<12}"
                    )
            else:
                # Show standard columns for non-miner data
                print(
                    f"{Colors.BOLD}{Colors.CYAN}{'ID':<12} {'Prompt ID':<12} {'File Path':<35} {'Modality':<10} {'Type':<12} {'Source':<10} {'Created':<20} {'Size':<12}{Colors.END}"
                )
                print("-" * 200)

                # Print basic info for each row
                for row in rows:
                    id_short = row["id"][:8] + "..." if row["id"] else "N/A"
                    prompt_id_short = (
                        row["prompt_id"][:8] + "..." if row["prompt_id"] else "N/A"
                    )
                    file_path_short = truncate_string(
                        Path(row["file_path"]).name if row["file_path"] else "N/A", 30
                    )
                    created_str = format_timestamp(row["created_at"])

                    # Format file size
                    size_str = "N/A"
                    if row["file_size"]:
                        if row["file_size"] < 1024:
                            size_str = f"{row['file_size']}B"
                        elif row["file_size"] < 1024 * 1024:
                            size_str = f"{row['file_size']/1024:.1f}KB"
                        else:
                            size_str = f"{row['file_size']/(1024*1024):.1f}MB"

                    print(
                        f"{id_short:<12} {prompt_id_short:<12} {file_path_short:<35} {row['modality'] or 'N/A':<10} {row['media_type'] or 'N/A':<12} {row['source_type'] or 'N/A':<10} {created_str:<20} {size_str:<12}"
                    )

            print("=" * 200)

    except Exception as e:
        print(f"{Colors.RED}❌ Error reading media table: {e}{Colors.END}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Display first N rows of prompts or media table"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="~/.cache/sn34/prompts.db",
        help="Path to the prompt database",
    )
    parser.add_argument(
        "--table",
        type=str,
        choices=["prompts", "media"],
        required=True,
        help="Table to display (prompts or media)",
    )
    parser.add_argument(
        "--rows", type=int, default=10, help="Number of rows to display (default: 10)"
    )
    parser.add_argument(
        "--source-type",
        type=str,
        choices=["scraper", "dataset", "generated", "miner"],
        help="Filter media table by source type (only applies to media table)",
    )
    parser.add_argument(
        "--miner-uid",
        type=int,
        help="Filter media table by specific miner UID (only applies to media table with source-type=miner)",
    )
    parser.add_argument(
        "--last-24h",
        action="store_true",
        help="Filter media table to show only entries from the last 24 hours (only applies to media table)",
    )
    parser.add_argument(
        "--filepaths-only",
        action="store_true",
        help="Display only file paths (only applies to media table)",
    )
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help="Include associated prompt content with file paths (only applies to media table)",
    )

    args = parser.parse_args()

    # Validate source-type is only used with media table
    if args.source_type and args.table != "media":
        print(
            f"{Colors.RED}❌ --source-type can only be used with --table media{Colors.END}"
        )
        sys.exit(1)
    
    # Validate miner-uid is only used with media table and source-type=miner
    if args.miner_uid and (args.table != "media" or args.source_type != "miner"):
        print(
            f"{Colors.RED}❌ --miner-uid can only be used with --table media and --source-type miner{Colors.END}"
        )
        sys.exit(1)
    
    # Validate last-24h is only used with media table
    if args.last_24h and args.table != "media":
        print(
            f"{Colors.RED}❌ --last-24h can only be used with --table media{Colors.END}"
        )
        sys.exit(1)
    
    # Validate filepaths-only is only used with media table
    if args.filepaths_only and args.table != "media":
        print(
            f"{Colors.RED}❌ --filepaths-only can only be used with --table media{Colors.END}"
        )
        sys.exit(1)
    
    # Validate include-prompts is only used with media table
    if args.include_prompts and args.table != "media":
        print(
            f"{Colors.RED}❌ --include-prompts can only be used with --table media{Colors.END}"
        )
        sys.exit(1)

    db_path = Path(args.db_path).expanduser()

    if not db_path.exists():
        print(f"{Colors.RED}❌ Database not found: {db_path}{Colors.END}")
        print(
            "💡 The database will be created automatically when prompts are first added."
        )
        sys.exit(1)

    print(f"{Colors.CYAN}📍 Database: {db_path}{Colors.END}")
    filter_parts = []
    if args.source_type:
        filter_parts.append(f"source_type: {args.source_type}")
    if args.miner_uid:
        filter_parts.append(f"uid: {args.miner_uid}")
    if args.last_24h:
        filter_parts.append("last 24 hours")
    if args.filepaths_only:
        filter_parts.append("filepaths only")
    if args.include_prompts:
        filter_parts.append("with prompts")
    
    filter_info = f" ({', '.join(filter_parts)})" if filter_parts else ""

    print(
        f"{Colors.CYAN}📊 Displaying first {args.rows} rows from {args.table} table{filter_info}{Colors.END}"
    )
    print()

    if args.table == "prompts":
        display_prompts_table(db_path, args.rows)
    elif args.table == "media":
        display_media_table(db_path, args.rows, args.source_type, args.miner_uid, args.last_24h, args.filepaths_only, args.include_prompts)

    print(f"\n{Colors.GREEN}✅ Table display completed!{Colors.END}")


if __name__ == "__main__":
    main()
