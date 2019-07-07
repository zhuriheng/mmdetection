#!/usr/bin/env bash
set -eux
set -o pipefail

source activate open-mmlab

cd ../../

nohup ./tools/dist_train.sh configs/terror_post/faster_rcnn_dconv_c3-c5_r50_fpn_1x.py 4 --validate \
        > terror_post_faster_rcnn_dconv_c3-c5_r50_fpn_1x_v0.3.log 2>&1 &
