#!/usr/bin/env bash
set -eux
set -o pipefail

source activate open-mmlab

cd ../../

work=terror_post
arch=faster_rcnn_ohem_r101_fpn_1x

nohup ./tools/dist_train.sh configs/${work}/${arch}.py 4 --validate \
        > ${work}_${arch}_v0.13.log 2>&1 &
