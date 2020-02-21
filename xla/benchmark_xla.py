#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.contrib.compiler import xla
import time

input_shape = (1, 7, 7, 1)
out_shape_hw = (14, 14)
output_shape = (1, 14, 14, 1)

input_numpy = np.random.normal(loc=0.0, scale=256.0, size=input_shape).astype("float32")

n_runs = 1

USE_XLA = True

def model_resize(x):
    #FIXME: XLA requires size/shape/dimension to be a compile-time constant.
    #Therefore, had to hardcode the output shape here.
    return tf1.image.resize(x, size=(14,14), \
                            method=tf1.image.ResizeMethod.BILINEAR, \
                            align_corners=True)

def construct_bilinear_kernel_matrix(w0, w1):
    # w0 - input image width
    # w1 - output image width
    # assume image is squared.

    # The unit length between pixels in the interpolated image.
    unit1 = (np.float(w0) - 1) / (w1 - 1)

    assert (w1 > w0) and (w0 & 1) and (w1 & 1), "ERROR: only verified when w1 > w0 (i.e., upsampling) and both shapes are odd numbers."

    T = np.zeros(shape=(w0, w1))
    for y in range(0, w1):
        raw_dis = y * unit1
        l = int(np.floor(raw_dis))
        r = int(np.ceil(raw_dis))
        delta = raw_dis -l

        if y == 0 or y == (w1 - 1):
            T[l, y] = 1.0 - delta
        else:
            T[l, y] = 1.0 - delta
            T[r, y] = delta

    T[w0 >> 1, w1 >> 1] = 1.0

    return T

def run_tf():
    input_box = tf.constant(input_numpy, dtype=tf.float32)
    if USE_XLA == True:
      output_box = xla.compile(computation=model_resize, inputs=(input_box,))[0]
    else:
      output_box = tf1.image.resize(input_box, size=out_shape_hw, \
                                    method=tf1.image.ResizeMethod.BILINEAR, \
                                    align_corners=True)
    with tf.Session() as sess:
        # Avoid code start
        for i in range(n_runs):
            sess.run(output_box)

        start = time.time()
        for i in range(n_runs):
            sess.run(output_box)
        stop = time.time()

        # Get the output in numpy array format
        output = output_box.eval()

    return output

output = run_tf()
print(output)

# verify a faster matmul-based resize bilinear approach
A = np.random.normal(loc=100.0, scale=256.0, size=(7,7)).astype("float32")
T = construct_bilinear_kernel_matrix(7, 33)
O = np.matmul(np.transpose(T), np.matmul(A, T))


output_tensor = tf1.image.resize(np.reshape(A, newshape=(7,7,1)), \
                                 size=(33,33), method=tf1.image.ResizeMethod.BILINEAR, \
                                 align_corners=True)
with tf.Session() as sess:
    sess.run(output_tensor)
    tf_output = np.squeeze(output_tensor.eval())
assert np.allclose(tf_output, O, atol=1), "ERROR: numerical mismatch."
print("[SUCCESS] Tensorflow and my matmul approach matched!")
