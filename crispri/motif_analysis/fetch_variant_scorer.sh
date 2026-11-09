#!/bin/bash
# Clone kundajelab/variant-scorer into lib/ next to this script.
# chrombpnet_bed_counts_contribs.py imports utils.shap_utils from its src/ tree,
# and shap_utils.py imports a sibling `generators` package, so the whole src/
# directory is needed, not shap_utils.py alone.
#
# Usage (from any directory, with the ChromBPNet environment active):
#     bash fetch_variant_scorer.sh [commit]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$SCRIPT_DIR/lib"
VS_DIR="$LIB_DIR/variant-scorer"
COMMIT="${1:-}"

mkdir -p "$LIB_DIR"
rm -rf "$VS_DIR"

echo "Cloning kundajelab/variant-scorer into $VS_DIR ..."
git clone https://github.com/kundajelab/variant-scorer.git "$VS_DIR"
if [ -n "$COMMIT" ]; then
    git -C "$VS_DIR" checkout "$COMMIT"
fi
echo "variant-scorer at commit: $(git -C "$VS_DIR" rev-parse HEAD)"

# Check that the import used by chrombpnet_bed_counts_contribs.py resolves.
python - <<PYEOF
import sys
sys.path.insert(0, "$VS_DIR/src")
from utils.shap_utils import shuffle_several_times, combine_mult_and_diffref
print("Imported utils.shap_utils OK")
PYEOF
