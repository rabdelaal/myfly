#!/bin/bash
# Télécharge les fichiers officiels du connectome MaleCNS v1.0 (Janelia, CC-BY).
# Source : https://male-cns.janelia.org/download/
set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"
BASE="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

echo "[download] Annotations (~13 Mo)…"
curl -L --fail -o "$DATA_DIR/body-annotations.feather" \
  "$BASE/body-annotations-male-cns-v1.0-minconf-0.5.feather"

echo "[download] Neurotransmetteurs (~42 Mo)…"
curl -L --fail -o "$DATA_DIR/body-neurotransmitters.feather" \
  "$BASE/body-neurotransmitters-male-cns-v1.0.feather"

echo "[download] Connectivité (~1,1 Go, patientez)…"
curl -L --fail -o "$DATA_DIR/connectome-weights.feather" \
  "$BASE/connectome-weights-male-cns-v1.0-minconf-0.5.feather"

echo "[download] Terminé. Convertir ensuite :"
echo "  cd scripts && python prepare_data.py"
ls -la "$DATA_DIR"
