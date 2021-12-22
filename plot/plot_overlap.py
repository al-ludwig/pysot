import os
import matplotlib.pyplot as plt
import numpy as np

path1_overlaps = os.path.normpath('..\experiments\siamrpn_r50_l234_dwxcorr\\results\VOT2018\\trt_model_TF32_21_12_2021_23_01\\baseline\\gymnastics2\\gymnastics2_overlaps.txt')
path2_overlaps = os.path.normpath('..\experiments\siamrpn_r50_l234_dwxcorr\\results\VOT2018\\trt_model_fp16_18_12_2021_16_09\\baseline\\gymnastics2\\gymnastics2_overlaps.txt')

overlaps1 = []
overlaps2 = []

with open(path1_overlaps, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        line = float(line)
        if line > 1:
            overlaps1.append(0)
        else:
            overlaps1.append(line)

with open(path2_overlaps, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        line = float(line)
        if line > 1:
            overlaps2.append(0)
        else:
            overlaps2.append(line)

assert len(overlaps1) == len(overlaps2)

x_entries = range(1, len(overlaps1)+1)

plt.figure()
plt.title("Comparing overlaps")
plt.ylim([0, 1])
plt.xlim([0, len(overlaps1)])
plt.ylabel("Overlap in %")
plt.xlabel("Frame")
plt.plot(x_entries, overlaps1, 'r--', label='FP32')
plt.plot(x_entries, overlaps2, 'b--', label='FP16')
plt.legend()
plt.show()