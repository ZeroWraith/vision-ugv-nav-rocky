#!/usr/bin/env bash
# Download RUGD dataset and extract the rocky (scene_03) sequence.
# Full dataset is ~5.3 GB. Only the rocky folder is kept.
set -euo pipefail

DEST_DIR="$(dirname "$0")/../data/rugd/scene_03"
mkdir -p "$DEST_DIR"

RGB_DIR="$DEST_DIR/rgb"
if [ -d "$RGB_DIR" ] && [ "$(ls -A "$RGB_DIR" 2>/dev/null)" ]; then
    echo "Rocky frames already present at $RGB_DIR — skipping download."
    exit 0
fi

ZIP_URL="http://rugd.vision/data/RUGD_frames-with-annotations.zip"
ZIP_PATH="/tmp/rugd_frames.zip"

echo "Downloading RUGD dataset (~5.3 GB)..."
wget -q --show-progress -O "$ZIP_PATH" "$ZIP_URL"

echo "Extracting rocky sequence..."
# The zip contains RUGD_frames-with-annotations/rocky/*.png
unzip -q -j "$ZIP_PATH" "RUGD_frames-with-annotations/rocky/*" -d "$RGB_DIR"

# Rename frames to sequential format if needed
cd "$RGB_DIR"
i=0
for f in $(ls *.png 2>/dev/null | sort); do
    newname=$(printf "frame_%04d.png" $i)
    if [ "$f" != "$newname" ]; then
        mv "$f" "$newname"
    fi
    i=$((i + 1))
done
cd - > /dev/null

rm -f "$ZIP_PATH"

# Create meta.json with RUGD camera intrinsics
cat > "$DEST_DIR/meta.json" <<'EOF'
{
  "width": 640,
  "height": 480,
  "fps": 10,
  "K": [  381.362, 0.0, 320.5,
          0.0, 381.362, 240.5,
          0.0, 0.0, 1.0 ],
  "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
  "camera_height": 1.2,
  "pitch_deg": 0.0
}
EOF

echo "Done. Rocky frames ready at $RGB_DIR ($(ls "$RGB_DIR"/*.png 2>/dev/null | wc -l) frames)"
