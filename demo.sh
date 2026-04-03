#!/usr/bin/env bash
# Demo: cognitive-cache finding the right files for a real GitHub issue
#
# Clones Textualize/rich at the commit BEFORE the fix for issue #4006,
# runs cognitive-cache to predict which files are relevant, then
# reveals which files were actually modified in the fix.
#
# Run: uv run bash demo.sh

set -e

REPO_URL="https://github.com/Textualize/rich.git"
BASE_COMMIT="27701029082052c6541031b6f139097230253900"
ISSUE_TITLE="fix for infinite loop in split_graphemes"
ISSUE_BODY="Fix for issue #3958. The split_graphemes function in cells.py enters an infinite loop on certain Unicode input."

DEMO_DIR=$(mktemp -d)
trap 'rm -rf "$DEMO_DIR"' EXIT

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           cognitive-cache: real-world demo                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Repository:  Textualize/rich"
echo "  Issue:       #4006"
echo "  Task:        $ISSUE_TITLE"
echo ""
echo "  cognitive-cache will analyze the codebase and predict which"
echo "  files need to be modified to fix this issue."
echo ""

echo "→ Cloning rich at the pre-fix commit..."
git clone --quiet "$REPO_URL" "$DEMO_DIR/rich" 2>/dev/null
cd "$DEMO_DIR/rich"
git checkout --quiet "$BASE_COMMIT"
cd - > /dev/null
FILE_COUNT=$(find "$DEMO_DIR/rich/rich" "$DEMO_DIR/rich/tests" -name '*.py' 2>/dev/null | wc -l)
echo "  Done. $FILE_COUNT Python files to analyze."
echo ""

echo "→ Running cognitive-cache..."
echo ""

cognitive-cache select \
    --repo "$DEMO_DIR/rich" \
    --task "$ISSUE_TITLE. $ISSUE_BODY" \
    --budget 12000

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Ground truth (files modified in the actual fix PR):"
echo ""
echo "    ✓ rich/cells.py"
echo "    ✓ tests/test_cells.py"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
