
import time
import argparse
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector

parser = argparse.ArgumentParser(description='mmdetection infer')
parser.add_argument('config_file', help='config file path', type=str)
parser.add_argument('checkpoint_file', help='checkpoint file path', type=str)
parser.add_argument('image_lst', help='image list file', type=str)

args = parser.parse_args()

config_file = args.config_file
checkpoint_file = args.checkpoint_file

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

imgs = []
with open(args.image_lst, 'r') as f:
    for line in f:
        imgs.append(line.strip().split("\t")[0] + '.jpg')

# test a list of images and write the results to image files
start = time.time()
for img in tqdm(imgs):
    result = inference_detector(model, img)
end = time.time()

print("Total time: {}".format(end-start))
print("Average time: {}".format((end-start)/len(imgs)))