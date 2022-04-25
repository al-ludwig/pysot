import os
import argparse
from cv2 import mean
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time

# argparse check function
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"{path} is not a valid dir")

parser = argparse.ArgumentParser(description='Create graphs')
parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('--type', help='select what graph to export')
required.add_argument('--output', required=True, type=dir_path, help='specify the path where the graph shall be saved')

def get_timestamp():
    return time.strftime("%Y%m%d")

def split_list(a_list):
    i_half = len(a_list) // 2
    return a_list[:i_half], a_list[i_half:]

def fps_eao():
    fps = {
        'pytorch_resnet50_fp32': 11.1,
        'trt_resnet50_fp32': 13.1,
        'trt_resnet50_fp16': 49.9,
        'trt_resnet50_int8': 65.7
    }
    eao = {
        'pytorch_resnet50_fp32': 0.413,
        'trt_resnet50_fp32': 0.392,
        'trt_resnet50_fp16': 0.389,
        'trt_resnet50_int8': 0.347
    }

    markers = split_list(Line2D.filled_markers)
    plt.ylabel('EAO')
    plt.xlabel('FPS')
    plt.grid(linestyle='dashed')
    plt.ylim(bottom=0, top=0.6)
    plt.xlim(left=0, right=70)
    for f in fps:
        plt.plot(fps[f], eao[f], marker=markers[1][4+list(fps.keys()).index(f)], label=f, linestyle='None')
    plt.legend()
    plt.title('Performance on Xavier, VOT2018 dataset')
    plt.hlines(eao['pytorch_resnet50_fp32'], 0, 70, colors='k', linestyles='dashed')
    # plt.show()
    plt.savefig(os.path.join(args.output, 'fps_eao_' + str(get_timestamp()) + '.png'))

def ar_raw(args):
    acc = {
        'pytorch_resnet50_fp32': 0.602,
        'trt_resnet50_fp32': 0.603,
        'trt_resnet50_fp16': 0.603,
        'trt_resnet50_int8': 0.606
    }
    rob = {
        'pytorch_resnet50_fp32': 0.243,
        'trt_resnet50_fp32': 0.239,
        'trt_resnet50_fp16': 0.248,
        'trt_resnet50_int8': 0.281
    }

    markers = split_list(Line2D.filled_markers)
    plt.ylabel('Accuracy')
    plt.xlabel('Robustness')
    plt.grid(linestyle='dashed')
    # plt.ylim(bottom=0, top=1)
    # plt.xlim(left=0, right=1)
    for f in rob:
        plt.plot(rob[f], acc[f], marker=markers[1][4+list(rob.keys()).index(f)], label=f, linestyle='None')
    plt.legend()
    plt.title('Performance on Xavier, VOT2018 dataset')
    # plt.show()
    plt.savefig(os.path.join(args.output, 'ar_raw_' + str(get_timestamp()) + '.png'))

def eao_challenges():
    camera_motion = {
        'pytorch_resnet50_fp32': 0.375,
        'trt_resnet50_fp32': 0.363,
        'trt_resnet50_fp16': 0.362,
        'trt_resnet50_int8': 0.312
    }
    illum_change = {
        'pytorch_resnet50_fp32': 0.445,
        'trt_resnet50_fp32': 0.46,
        'trt_resnet50_fp16': 0.454,
        'trt_resnet50_int8': 0.461
    }
    motion_change = {
        'pytorch_resnet50_fp32': 0.412,
        'trt_resnet50_fp32': 0.4,
        'trt_resnet50_fp16': 0.388,
        'trt_resnet50_int8': 0.394
    }
    size_change = {
        'pytorch_resnet50_fp32': 0.452,
        'trt_resnet50_fp32': 0.456,
        'trt_resnet50_fp16': 0.448,
        'trt_resnet50_int8': 0.368
    }
    occlusion = {
        'pytorch_resnet50_fp32': 0.346,
        'trt_resnet50_fp32': 0.327,
        'trt_resnet50_fp16': 0.302,
        'trt_resnet50_int8': 0.264
    }

    markers = split_list(Line2D.filled_markers)
    challenges = ['camera_motion']
    # challenges = ['camera_motion', 'illum_change', 'motion_change', 'size_change', 'occlusion']
    plt.xlabel('EAO')
    plt.grid(linestyle='dashed')
    colors = ['b', 'g', 'r', 'c']
    for t, c in zip(camera_motion, colors):
        plt.plot(camera_motion[t], ['camera_motion'], color=c, marker=markers[1][4+list(camera_motion.keys()).index(t)], label=t, linestyle='None')
    for t, c in zip(illum_change, colors):
        plt.plot(illum_change[t], ['illum_change'], color=c, marker=markers[1][4+list(illum_change.keys()).index(t)], linestyle='None')
    for t, c in zip(motion_change, colors):
        plt.plot(motion_change[t], ['motion_change'], color=c, marker=markers[1][4+list(motion_change.keys()).index(t)], linestyle='None')
    for t, c in zip(size_change, colors):
        plt.plot(size_change[t], ['size_change'], color=c, marker=markers[1][4+list(size_change.keys()).index(t)], linestyle='None')
    for t, c in zip(occlusion, colors):
        plt.plot(occlusion[t], ['occlusion'], color=c, marker=markers[1][4+list(occlusion.keys()).index(t)], linestyle='None')
    plt.legend()
    plt.title('Performance on Xavier, VOT2018 dataset')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(args.output, 'eao_challenges_' + str(get_timestamp()) + '.png'))

def overlap_video():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    video_name = 'soccer1'
    trt_prefix = 'fp32_01_02_2022_16_04'
    pytorch_prefix = 'fp32_17_02_2022_16_00'
    overlaps_path1 = os.path.realpath(os.path.join(cur_dir, '../experiments/siamrpn_r50_l234_dwxcorr/results/VOT2018/trt_model_' + trt_prefix + '/baseline/' + video_name + '/' + video_name + '_overlaps.txt'))
    overlaps_path2 = os.path.realpath(os.path.join(cur_dir, '../experiments/siamrpn_r50_l234_dwxcorr/results/VOT2018/baseline_xavier_' + pytorch_prefix + '/baseline/' + video_name + '/' + video_name + '_overlaps.txt'))

    overlaps_file1 = open(overlaps_path1, 'r')
    overlaps_file2 = open(overlaps_path2, 'r')
    overlaps_str1 = overlaps_file1.readlines()
    overlaps_str2 = overlaps_file2.readlines()

    overlaps1 = []
    for o in overlaps_str1:
        o = float(o)
        print(o)
        if o == 2:
            o = 0
        overlaps1.append(o)
    
    overlaps2 = []
    for o in overlaps_str2:
        o = float(o)
        if o == 2:
            o = 0
        overlaps2.append(o)

    print(sum(overlaps1) / len(overlaps1))
    print(sum(overlaps2) / len(overlaps2))
    
    lower = 0
    upper = 1
    supper1 = []
    slower1 = []
    supper2 = []
    slower2 = []
    for o in overlaps1:
        supper1.append(np.ma.masked_where(o > upper, o))
        slower1.append(np.ma.masked_where(o <= lower, o))
    for o in overlaps2:
        supper2.append(np.ma.masked_where(o > upper, o))
        slower2.append(np.ma.masked_where(o <= lower, o))

    frames = np.arange(1, len(overlaps1)+1)
    f = plt.figure()
    f.set_figwidth(16)
    f.set_figheight(7)
    plt.plot(frames, supper1, color='r', linestyle='dashed', label='Failure trt model')
    plt.plot(frames, slower1, color='g', label='trt model')
    plt.plot(frames, supper2, color='b', linestyle='dashed', label='Failure pytorch model')
    plt.plot(frames, slower2, color='m', label='pytorch model')
    plt.xlim(1, len(overlaps1))
    plt.legend()
    plt.title('Drift of overlap - video ' + video_name)
    plt.ylabel('Overlap')
    plt.xlabel('Frames')
    plt.savefig(os.path.join(args.output, 'overlap_' + video_name + '_' + trt_prefix + '_' + pytorch_prefix + '.png'))
    plt.show()

def main(args):
    if args.type == 'fps_eao':
        fps_eao()
    elif args.type == 'ar_raw':
        ar_raw(args)
    elif args.type == 'eao_challenges':
        eao_challenges()
    elif args.type == 'overlap_video':
        overlap_video()
    else:
        print('ERROR: Type "' + str(args.type) + '" is not supported!')
    return

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)