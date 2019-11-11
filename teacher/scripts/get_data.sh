#!/usr/bin/env bash

## 256x256 pixel tiles and adjusted ROI files:
#gsutil -m cp -r gs://planespotting-data-public/tiles_from_USGS_photos ./data/
#gsutil -m cp -r gs://planespotting-data-public/tiles_from_USGS_photos_eval ./data/

# large aerial photographs and ROI files:
DATA_DIR="/home/fchouteau/repositories/tp_isae/data/raw"
rm -r ${DATA_DIR}
mkdir ${DATA_DIR}
mkdir ${DATA_DIR}/trainval/
mkdir ${DATA_DIR}/eval/
gsutil -m cp -r gs://planespotting-data-public/USGS_public_domain_photos/* ${DATA_DIR}/trainval/
gsutil -m cp -r gs://planespotting-data-public/USGS_public_domain_photos_eval/* ${DATA_DIR}/eval/

# Remove outliers and unlabelled
rm ${DATA_DIR}/trainval/USGS_TUC1l.jpg
rm ${DATA_DIR}/trainval/USGS_TUC1l.json
rm ${DATA_DIR}/trainval/USGS_TUC1s.jpg
rm ${DATA_DIR}/trainval/USGS_TUC1s.json
rm ${DATA_DIR}/trainval/USGS_TUC2s.jpg
rm ${DATA_DIR}/trainval/USGS_TUC2s.json
rm ${DATA_DIR}/trainval/USGS_TUC3s.jpg
rm ${DATA_DIR}/trainval/USGS_TUC3s.json
rm ${DATA_DIR}/trainval/USGS_TUC4s.jpg
rm ${DATA_DIR}/trainval/USGS_TUC4s.json
rm ${DATA_DIR}/trainval/USGS_DMA.jpg
rm ${DATA_DIR}/trainval/USGS_DMA2.jpg

python ./scripts/process_raw_to_input_format.py