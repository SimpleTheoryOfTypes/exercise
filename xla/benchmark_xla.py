#!usr/bin/env python3
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
