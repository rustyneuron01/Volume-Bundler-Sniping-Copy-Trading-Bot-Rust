#!/usr/bin/env python3
"""
Standalone C2PA verification helper for testing.

Usage:
    python neurons/generator/helper/verify_c2pa.py <file_path>
    python neurons/generator/helper/verify_c2pa.py image.png
    python neurons/generator/helper/verify_c2pa.py video.mp4 --verbose
    python neurons/generator/helper/verify_c2pa.py *.png  # Multiple files
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from gas.verification.c2pa_verification import verify_c2pa, C2PAVerificationResult


def print_result(filepath: str, result: C2PAVerificationResult, verbose: bool = False):
    """Print verification result in a readable format."""
    status = "✅ PASSED" if (result.verified and result.is_trusted_issuer) else "❌ FAILED"

    print(f"\n{'='*60}")
    print(f"File: {filepath}")
    print(f"Status: {status}")
    print(f"{'='*60}")

    print(f"  verified:          {result.verified}")
    print(f"  signature_valid:   {result.signature_valid}")
    print(f"  is_self_signed:    {result.is_self_signed}")
    print(f"  is_trusted_issuer: {result.is_trusted_issuer}")
    print(f"  issuer:            {result.issuer}")
    print(f"  cert_issuer:       {result.cert_issuer}")
    print(f"  ai_generated:      {result.ai_generated}")

    if result.error:
        print(f"  error:             {result.error}")

    if result.validation_errors:
        print(f"  validation_errors: {result.validation_errors}")

    if verbose and result.manifest_data:
        print(f"\n  Raw manifest data:")
        print(json.dumps(result.manifest_data, indent=4))


def run_verification(files: List[str], verbose: bool = False, as_json: bool = False) -> bool:
    """Run C2PA verification on files.

    Args:
        files: List of file paths to verify
        verbose: Show raw manifest data
        as_json: Output results as JSON

    Returns:
        True if all files passed, False otherwise
    """
    results = []
    any_failed = False

    for filepath in files:
        path = Path(filepath)
        if not path.exists():
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            any_failed = True
            continue

        result = verify_c2pa(path)
        passed = result.verified and result.is_trusted_issuer

        if not passed:
            any_failed = True

        if as_json:
            results.append({
                "file": str(path),
                "result": result.to_dict()
            })
        else:
            print_result(str(path), result, verbose=verbose)

    if as_json:
        print(json.dumps(results, indent=2))

    return not any_failed


def main():
    parser = argparse.ArgumentParser(
        description="Test C2PA verification on local files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s image.png
    %(prog)s video.mp4 --verbose
    %(prog)s /path/to/content/*.png
    %(prog)s --json image.png  # Output as JSON
        """
    )
    parser.add_argument("files", nargs="+", help="File(s) to verify")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show raw manifest data")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    all_passed = run_verification(args.files, verbose=args.verbose, as_json=args.json)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
