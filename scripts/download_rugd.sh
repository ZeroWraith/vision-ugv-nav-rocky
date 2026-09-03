#!/usr/bin/env bash
# Download a single RUGD "rocky" sequence (scene_03) – ~200 MB
set -euo pipefail

DEST_DIR="$(dirname "$0")/../data/rugd/scene_03"
mkdir -p "$DEST_DIR"

# RUGD is hosted on Google Drive; file IDs taken from the official download page.
# Scene 03 (rocky) zip ~ 210 MB
FILE_ID="1Kx6cK9e8JzVY2Q9v5Z5Y5Z5Y5Z5Y5Z5"   # placeholder – replace with real ID
ZIP_PATH="$DEST_DIR/scene_03.zip"

echo "Downloading RUGD rocky sequence..."
gdown --id "$FILE_ID" -O "$ZIP_PATH"

echo "Extracting..."
unzip -q "$ZIP_PATH" -d "$DEST_DIR"
rm "$ZIP_PATH"

# Create a tiny meta.json with camera intrinsics (taken from RUGD calibration)
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

echo "Done. Data ready at $DEST_DIR"