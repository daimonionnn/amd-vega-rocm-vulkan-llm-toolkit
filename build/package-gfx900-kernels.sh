#!/bin/bash
#
# Package the gfx900 Tensile kernels out of AMD's ROCm 6.3.4 rocBLAS package.
#
# Why this exists: ROCm 7.x ships no gfx900 device code, and every ROCm path in
# this repo depends on the 6.3.4 kernels. Today each user re-does the same dig --
# fetch a 2024 .deb from repo.radeon.com, unpack it, find the right files. This
# produces exactly the file set build/build-llamacpp-rocm7-baremetal.sh installs,
# with provenance and checksums, so it can be published once and downloaded.
#
# The files are taken from AMD's package, not from this machine's /opt/rocm, so
# what is published is traceable to a named package version rather than to
# whatever happens to be installed here. The .deb is verified against the SHA256
# in the repository's own Packages index before anything is extracted.
#
# Usage:
#   bash build/package-gfx900-kernels.sh [output-dir]     # default: dist/
#
# Produces, in the output directory:
#   rocblas-gfx900-6.3.4.tar.gz   kernels + README recording where they came from
#   SHA256SUMS                    checksum of that tarball
#
# rocBLAS is MIT-licensed (AMD), so these build artefacts are redistributable;
# the tarball carries that notice.

set -euo pipefail

ROCM634_REPO="${ROCM634_REPO:-https://repo.radeon.com/rocm/apt/6.3.4}"
ROCM634_DISTRO="${ROCM634_DISTRO:-jammy}"   # Ubuntu 22.04 — the last with 6.3.4
OUT_DIR="${1:-dist}"
PKG_NAME="rocblas-gfx900-6.3.4"

for tool in wget dpkg-deb sha256sum tar; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "✗  Missing required tool: $tool" >&2
        exit 1
    fi
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO_ROOT/$OUT_DIR"
OUT_DIR="$(cd "$REPO_ROOT/$OUT_DIR" && pwd)"

WORK="$(mktemp -d /tmp/gfx900-pkg.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

echo "─── Step 1/4: Locating rocblas in ${ROCM634_REPO} (${ROCM634_DISTRO}) ───"
PKGLIST_URL="${ROCM634_REPO}/dists/${ROCM634_DISTRO}/main/binary-amd64/Packages"
wget -q "$PKGLIST_URL" -O "$WORK/Packages"

# One stanza, not the whole file: Filename/Version/SHA256 must come from the
# same record or the verification below compares unrelated packages.
awk '/^Package: rocblas$/{f=1} f{print} f && /^$/{exit}' "$WORK/Packages" > "$WORK/stanza"
PKG_VERSION="$(awk '/^Version:/{print $2; exit}' "$WORK/stanza")"
PKG_PATH="$(awk '/^Filename:/{print $2; exit}' "$WORK/stanza")"
PKG_SHA256="$(awk '/^SHA256:/{print $2; exit}' "$WORK/stanza")"

if [ -z "$PKG_PATH" ] || [ -z "$PKG_SHA256" ]; then
    echo "✗  No rocblas record with a checksum in $PKGLIST_URL" >&2
    exit 1
fi
echo "  package: rocblas $PKG_VERSION"
echo "  file:    $PKG_PATH"

echo "─── Step 2/4: Downloading and verifying ───"
DEB_URL="${ROCM634_REPO}/${PKG_PATH}"
DEB_FILE="$WORK/$(basename "$PKG_PATH")"
wget -q --show-progress "$DEB_URL" -O "$DEB_FILE"

# Verify against the index before unpacking anything.
echo "$PKG_SHA256  $DEB_FILE" | sha256sum -c - >/dev/null
echo "  ✓ SHA256 matches the repository index: $PKG_SHA256"

echo "─── Step 3/4: Extracting gfx900 files ───"
mkdir -p "$WORK/extracted"
dpkg-deb -x "$DEB_FILE" "$WORK/extracted"

STAGE="$WORK/$PKG_NAME"
mkdir -p "$STAGE/library"
mapfile -t GFX_FILES < <(find "$WORK/extracted" -name '*gfx900*' -type f | sort)
if [ "${#GFX_FILES[@]}" -eq 0 ]; then
    echo "✗  No gfx900 files inside $(basename "$DEB_FILE")" >&2
    exit 1
fi
cp "${GFX_FILES[@]}" "$STAGE/library/"

# The lazy index is what ROCm 7 reads first; without it the first GEMM dies with
# "Illegal seek for GPU arch: gfx900". A tarball missing it would look complete.
if [ ! -f "$STAGE/library/TensileLibrary_lazy_gfx900.dat" ]; then
    echo "✗  TensileLibrary_lazy_gfx900.dat is missing — the set would be unusable" >&2
    exit 1
fi

FILE_COUNT="${#GFX_FILES[@]}"
RAW_SIZE="$(du -sh "$STAGE/library" | cut -f1)"
echo "  $FILE_COUNT files, $RAW_SIZE"

( cd "$STAGE" && find library -type f | sort | xargs sha256sum > MANIFEST.sha256 )

cat > "$STAGE/README.md" <<EOF
# rocBLAS gfx900 Tensile kernels, from ROCm 6.3.4

Prebuilt GEMM kernels for **gfx900** (Vega 10 — Radeon Vega 56/64, Instinct MI25,
Radeon Pro V340) and, with \`HSA_OVERRIDE_GFX_VERSION=9.0.0\`, **gfx90c** (Vega 8
and other Ryzen APU iGPUs).

ROCm 7.x ships none of these. Dropping them into a ROCm 7 install makes rocBLAS
work on that hardware again: it scans its library directory at run time and picks
up whatever architectures it finds.

## Provenance

| | |
| --- | --- |
| Source package | \`rocblas\` $PKG_VERSION |
| Repository | $ROCM634_REPO ($ROCM634_DISTRO) |
| File | $PKG_PATH |
| Package SHA256 | \`$PKG_SHA256\` (verified against the repository index before extraction) |
| Files extracted | $FILE_COUNT, $RAW_SIZE |
| Packaged | $(date -u +%Y-%m-%d) by \`build/package-gfx900-kernels.sh\` |

Per-file checksums are in \`MANIFEST.sha256\`. Nothing here was rebuilt, renamed
or modified — these are AMD's own files, copied out of that package.

## Install

\`\`\`bash
tar xzf $PKG_NAME.tar.gz
sudo cp $PKG_NAME/library/* /opt/rocm/lib/rocblas/library/
\`\`\`

Then check that rocBLAS sees them:

\`\`\`bash
ls /opt/rocm/lib/rocblas/library/ | grep -c gfx900    # expect $FILE_COUNT
\`\`\`

\`TensileLibrary_lazy_gfx900.dat\` is the one file that must not be left out.
ROCm 7 reads that index first, and without it the first GEMM fails with
\`Illegal seek for GPU arch: gfx900\`.

## What this works with

- **Classic ROCm 7.0–7.2.** Tested on 7.2.0.
- **Not AMD's modular packages** (\`amdrocm-core\` 7.13+). Their ROCr rejects
  \`HSA_OVERRIDE_GFX_VERSION\`, so an APU cannot present itself as gfx900 there. A discrete Vega 10 needs no
  override, and issue #1 reports these files working on one under modular 7.14.1.
- **ROCm 7.14: untested here.** Its rocBLAS lays the library out per
  architecture (\`library/gfx900/\`) where 6.3.4 and 7.2 use one flat directory,
  and the 7.14 work in this repo used AMD's own 182 gfx900 files from their 7.14
  wheel, never these. Issue #1 reports these 6.3.4 files working under AMD's
  modular rocBLAS 7.14.1 on a discrete Vega 10.

On an APU, export \`HSA_OVERRIDE_GFX_VERSION=9.0.0\` so gfx90c loads the gfx900
kernels. A discrete Vega 10 needs no override.

## License

rocBLAS is MIT-licensed, Copyright (C) Advanced Micro Devices, Inc. These are
build artefacts of that source and carry the same license.

## Where this comes from

https://github.com/daimonionnn/amd-vega-rocm-vulkan-llm-toolkit — measurements,
the llama.cpp flash-attention patch for GCN5, and what else on the ROCm stack
does or does not need this treatment.
EOF

echo "─── Step 4/4: Packing ───"
tar czf "$OUT_DIR/$PKG_NAME.tar.gz" -C "$WORK" "$PKG_NAME"
( cd "$OUT_DIR" && sha256sum "$PKG_NAME.tar.gz" > SHA256SUMS )

echo ""
echo "  ✓ $OUT_DIR/$PKG_NAME.tar.gz  ($(du -h "$OUT_DIR/$PKG_NAME.tar.gz" | cut -f1))"
echo "  ✓ $OUT_DIR/SHA256SUMS"
echo ""
cat "$OUT_DIR/SHA256SUMS"
