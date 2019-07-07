#!/usr/bin/env bash
set -eux
set -o pipefail

source activate open-mmlab

cd ../../

# single-gpu testing
#python tools/test.py configs/terror_post/faster_rcnn_r50_fpn_1x.py \
#    checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth \
#    --eval bbox

# multi-gpu testing
./tools/dist_test.sh configs/terror_post/faster_rcnn_r50_fpn_1x.py \
        checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth \
        4 --eval bbox