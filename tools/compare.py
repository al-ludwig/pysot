import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)
logging.info('Started')

bb0 = np.load("..\\bb0.npy")
bb1 = np.load("..\\bb1.npy")
bb2 = np.load("..\\bb2.npy")

zf0 = np.load("..\\zf0.npy")
zf1 = np.load("..\\zf1.npy")
zf2 = np.load("..\\zf2.npy")

zf0_onnx = np.load("..\\zf0_onnx.npy")
zf1_onnx = np.load("..\\zf1_onnx.npy")
zf2_onnx = np.load("..\\zf2_onnx.npy")

r0 = zf0 - bb0
r1 = zf1 - bb1
r2 = zf2 - bb2

pt_onnx0 = zf0 - zf0_onnx
pt_onnx1 = zf1 - zf1_onnx
pt_onnx2 = zf2 - zf2_onnx

onnx_trt0 = zf0_onnx - bb0
onnx_trt1 = zf1_onnx - bb1
onnx_trt2 = zf2_onnx - bb2

print("Vergleich pt - onnx:")
print("Mittelwert von zf0 - zf0_onnx = " + str(pt_onnx0.mean()))
print("Maximale Diff zwischen zf0 und zf0_onnx = " + str(np.amax(pt_onnx0)))
print("Mittelwert von zf1 - zf1_onnx = " + str(pt_onnx1.mean()))
print("Maximale Diff zwischen zf1 und zf1_onnx = " + str(np.amax(pt_onnx1)))
print("Mittelwert von zf2 - zf2_onnx = " + str(pt_onnx2.mean()))
print("Maximale Diff zwischen zf2 und zf2_onnx = " + str(np.amax(pt_onnx2)))

print("\nVergleich onnx - trt:")
print("Mittelwert von zf0_onnx - bb0 = " + str(onnx_trt0.mean()))
print("Maximale Diff zwischen zf0_onnx und bb0 = " + str(np.amax(onnx_trt0)))
print("Mittelwert von zf1_onnx - bb1 = " + str(onnx_trt1.mean()))
print("Maximale Diff zwischen zf1_onnx und bb1 = " + str(np.amax(onnx_trt1)))
print("Mittelwert von zf2_onnx - bb2 = " + str(onnx_trt2.mean()))
print("Maximale Diff zwischen zf2_onnx und bb2 = " + str(np.amax(onnx_trt2)))

print("\nVergleich pt - trt:")
print("Mittelwert von zf0 - bb0 = " + str(r0.mean()))
print("Maximale Diff zwischen zf0 und bb0 = " + str(np.amax(r0)))
print("Mittelwert von zf1 - bb1 = " + str(r1.mean()))
print("Maximale Diff zwischen zf1 und bb1 = " + str(np.amax(r1)))
print("Mittelwert von zf2 - bb2 = " + str(r2.mean()))
print("Maximale Diff zwischen zf2 und bb2 = " + str(np.amax(r2)))

# plt.hist(np.ravel(r0), bins=50, range=(r0.min(), r0.max()))
# plt.show()

if not np.any(r0):
    print("bb0 and zf0 are the same!")

if not np.any(r1):
    print("bb1 and zf1 are the same!")

if not np.any(r2):
    print("bb2 and zf2 are the same!")