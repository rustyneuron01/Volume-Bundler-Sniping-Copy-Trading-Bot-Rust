#!/bin/bash
#
# WARNING: This script DELETES ALL FILES and the bucket "sn34-bucket".
# Do not run unless you intend to permanently remove all data in that bucket.
#
# Usage:
#   export B2_KEY_ID="005e1c121470cf1000000000b"
#   export B2_APP_KEY="K005zGyBdCUlcxo9qzFfnseU6XR4xzk"
#   ./b2_delete_bucket.sh
#
# Or pass inline (less safe: key visible in process list):
#   B2_KEY_ID="..." B2_APP_KEY="..." ./b2_delete_bucket.sh
#
set -euo pipefail

B2_KEY_ID="${B2_KEY_ID:-}"
B2_APP_KEY="${B2_APP_KEY:-}"

if [[ -z "$B2_KEY_ID" || -z "$B2_APP_KEY" ]]; then
    echo "ERROR: Set B2_KEY_ID and B2_APP_KEY (or pass them when running)."
    echo "  export B2_KEY_ID='your_key_id'"
    echo "  export B2_APP_KEY='your_app_key'"
    exit 1
fi

# 1) Authorize (B2 CLI)
if command -v b2 &>/dev/null; then
    echo "Authorizing B2..."
    b2 account authorize "$B2_KEY_ID" "$B2_APP_KEY"
else
    echo "ERROR: 'b2' CLI not found. Install: https://www.backblaze.com/b2/docs/quick_command_line.html"
    exit 1
fi

# 2) Delete all files and bucket
BUCKET_NAME="sn34-bucket"

if command -v b3 &>/dev/null; then
    echo "Using 'b3' to delete bucket $BUCKET_NAME (all files + versions)..."
    while true; do
        b3 rm --recursive --versions --bypass-governance --threads 128 --no-progress "b3://$BUCKET_NAME" 2>/dev/null || true
        if b3 bucket delete "$BUCKET_NAME" 2>/dev/null; then
            echo "DELETE COMPLETE: $BUCKET_NAME removed"
            break
        fi
        echo "Still deleting... retrying in 3s"
        sleep 3
    done
elif command -v b2 &>/dev/null; then
    echo "Using 'b2' to delete all files (--versions) then bucket $BUCKET_NAME..."
    while true; do
        b2 rm --recursive --versions --bypass-governance --threads 128 "b2://$BUCKET_NAME" 2>/dev/null || true
        b2 cancel-all-unfinished-large-files "$BUCKET_NAME" 2>/dev/null || true
        if b2 bucket delete "$BUCKET_NAME" 2>/dev/null; then
            echo "DELETE COMPLETE: $BUCKET_NAME removed"
            break
        fi
        echo "Still deleting... retrying in 3s"
        sleep 3
    done
else
    echo "ERROR: Need 'b3' or 'b2' CLI. Install B2: https://www.backblaze.com/b2/docs/quick_command_line.html"
    exit 1
fi
