#!/usr/bin/env bash
set -eux
set -o pipefail

source activate open-mmlab

cd ../../

# single GPU testing
#python tools/test.py configs/terror_post/faster_rcnn_dconv_c3-c5_r50_fpn_1x.py \
#    work_dirs/faster_rcnn_dconv_c3-c5_r50_fpn_1x/2019-07-02-15-50/epoch_1.pth \
#    --eval bbox

# multiple GPU testing
./tools/dist_test.sh configs/terror_post/faster_rcnn_dconv_c3-c5_r50_fpn_1x.py \
    work_dirs/faster_rcnn_dconv_c3-c5_r50_fpn_1x/2019-07-02-15-50/epoch_1.pth \
    4 --eval bbox --out work_dirs/faster_rcnn_dconv_c3-c5_r50_fpn_1x/2019-07-02-15-50/results.pkl