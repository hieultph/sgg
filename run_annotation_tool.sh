#!/bin/bash
# Script to run SGG-Annotate tool with PSG COCO dataset

set -e

echo "======================================================================"
echo "🎨 SGG-Annotate Tool - COCO Relations Annotation"
echo "======================================================================"

# Paths
TOOL_DIR="tools/SGG-Annotate"
COCO_JSON="datasets/psg/coco/coco/annotations/instances_train2017.json"
IMAGES_DIR="datasets/psg/coco/coco/train2017"
PREDICATES_LIST="psg_predicates_list.json"

# Check if files exist
if [ ! -f "$COCO_JSON" ]; then
    echo "❌ Error: COCO JSON not found at $COCO_JSON"
    exit 1
fi

if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ Error: Images directory not found at $IMAGES_DIR"
    exit 1
fi

if [ ! -f "$PREDICATES_LIST" ]; then
    echo "⚠️  Warning: Predicates list not found at $PREDICATES_LIST"
    echo "   You'll need to import it manually in the tool"
fi

echo ""
echo "📋 Configuration:"
echo "  COCO JSON:    $COCO_JSON"
echo "  Images:       $IMAGES_DIR"
echo "  Predicates:   $PREDICATES_LIST"
echo ""
echo "======================================================================"
echo ""

# Check if annotation tool exists
if [ ! -f "$TOOL_DIR/main_coco.py" ]; then
    echo "❌ Error: Annotation tool not found at $TOOL_DIR/main_coco.py"
    exit 1
fi

cd "$TOOL_DIR"

echo "🚀 Starting SGG-Annotate tool..."
echo ""
echo "Instructions:"
echo "  1. Click 'Load COCO JSON' and select:"
echo "     ../../$COCO_JSON"
echo ""
echo "  2. When asked for images folder, select:"
echo "     ../../$IMAGES_DIR"
echo ""
echo "  3. Click 'Import Relationship List' and select:"
echo "     ../../$PREDICATES_LIST"
echo ""
echo "  4. Start annotating relationships!"
echo ""
echo "======================================================================"
echo ""

python main_coco.py

echo ""
echo "✅ Annotation tool closed"
echo "   Output saved to: $TOOL_DIR/output_coco_relations/"
