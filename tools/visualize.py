import cv2
import os
import argparse
from glob import glob
import numpy as np

from toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, \
        VOTDataset, NFSDataset, VOTLTDataset

# argparse check dir function
def dir_path(path):
	"""checks if given path is a path of a directory

    :param path: path of the directory
    """
	if not os.path.isdir(path):
		os.makedirs(path, exist_ok=True)
	return path

parser = argparse.ArgumentParser(description='Visualize tracking results')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument('--tracker_path', '-p', type=dir_path, help='tracker result path')
required.add_argument('--dataset', '-d', type=str, help='dataset name')
required.add_argument('--tracker_name', '-t', default='', type=dir_path, help='tracker name')
optional.add_argument('--video', '-v', default='', type=str, help='Show only that video')

def main(args):
	tracker = os.path.join(args.tracker_path, args.dataset, args.tracker_name)

	root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                            '../testing_dataset'))
	root = os.path.join(root, args.dataset)

	if 'VOT2019-LT' in args.dataset:
		dataset = VOTLTDataset(args.dataset, root)
		for v_idx, video in enumerate(dataset):
			if args.video != '':
            	# visualize one special video
				if video.name != args.video:
					continue
			# read bounding boxes from text-file
			filepath = os.path.join(tracker, 'longterm', video.name, video.name+'_001.txt')
			pred_bbox_list = []
			with open(filepath, 'r') as f:
				lines = f.readlines()
				# pred_bbox_list = [line.rstrip() for line in lines]
				for line in lines:
					line = line.rstrip()
					line = [float(number) for number in line.split(',')]
					pred_bbox_list.append(line)
			for idx, (img, gt_bbox) in enumerate(video):
				if idx == 0:
					cv2.destroyAllWindows()
				elif idx > 0:
					gt_bbox = list(map(int, gt_bbox))
					pred_bbox = list(map(int, pred_bbox_list[idx]))
					if gt_bbox != [0]:
						cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
									(gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
					cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
									(pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
					cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
					cv2.putText(img, 'ground truth', (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
					cv2.putText(img, 'tracking results', (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
					cv2.imshow(video.name, img)
					cv2.waitKey(1)



			


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)