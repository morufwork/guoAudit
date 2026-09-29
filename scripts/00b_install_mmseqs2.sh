#!/usr/bin/env bash
# Downloads the official MMseqs2 static binary (no sudo/apt required) into
# tools/mmseqs/. the sequence-cluster split (sequence homology clustering) depends on this.
# Not tracked in git (32MB binary) — re-run this script after a fresh clone.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$REPO_ROOT/tools"
mkdir -p "$TOOLS_DIR"

RELEASE_TAG=$(curl -sSL "https://api.github.com/repos/soedinglab/MMseqs2/releases/latest" | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": "([^"]+)".*/\1/')
echo "Installing MMseqs2 release ${RELEASE_TAG} (AVX2 build) into ${TOOLS_DIR}/mmseqs"

curl -sSL "https://github.com/soedinglab/MMseqs2/releases/download/${RELEASE_TAG}/mmseqs-linux-avx2.tar.gz" \
  -o "$TOOLS_DIR/mmseqs-linux-avx2.tar.gz"
tar xzf "$TOOLS_DIR/mmseqs-linux-avx2.tar.gz" -C "$TOOLS_DIR"
rm "$TOOLS_DIR/mmseqs-linux-avx2.tar.gz"

"$TOOLS_DIR/mmseqs/bin/mmseqs" version
echo "MMseqs2 installed at $TOOLS_DIR/mmseqs/bin/mmseqs"
