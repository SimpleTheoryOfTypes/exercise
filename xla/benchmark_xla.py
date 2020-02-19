#!usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import time

input_shape = (1, 7, 7, 1)
out_shape_hw = (14, 14)
output_shape = (1, 14, 14, 1)

input_numpy = np.random.normal(loc=0.0, scale=256.0, size=input_shape).astype("float32")

n_runs = 1

def run_tf():
    input_box = tf.constant(input_numpy, dtype=tf.float32)
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
