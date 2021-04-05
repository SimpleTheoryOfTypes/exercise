#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import time

H = 112
W = 112
strides = 2
pw = 3 # pooling window 3x3

A = np.random.normal(loc=1.0, scale=6.0, size=(H, W)).astype("float32")
output_tensor = tf.nn.max_pool(np.reshape(A, newshape=(1,H,W,1)), \
                               ksize=pw, strides=strides, padding="SAME")
                                
with tf.Session() as sess:
    sess.run(output_tensor)
    tf_output = np.squeeze(output_tensor.eval())
print(tf_output.shape)


def my_max_pool(A, ksize, strides, padding="SAME"):
  H, W = A.shape
  B = np.zeros(shape=(H//2, W//2)).astype("float32")
  for x in range(0,W,strides):
      for y in range(0,H,strides):
          max_pix = float('-inf') # set to -inf initially.
          for i in range(x, x + pw, 1):
              for j in range(y, y + pw, 1):
                  if (i < W and j < H):
                      max_pix = max(max_pix, A[i][j])
  
          B[x//2][y//2] = max_pix

  return B

B = my_max_pool(A, ksize=pw, strides=strides, padding="SAME")
assert np.allclose(tf_output, B, atol=1e-7), "ERROR: numerical mismatch."
print("[SUCCESS] Tensorflow and my max pool approach matched!")

if True:
  # Resize bilinear tryout on a real image.
  from PIL import Image
  im1 = Image.open("lena.png")
  im1.show()
  np_im1 = np.array(im1).astype(np.float32)
  print(np_im1)
  print(np_im1.shape)
  
  A = np_im1
  Result = my_max_pool(A, ksize=3, strides=2, padding="SAME")
  print("Image size after max pool: ", Result.shape)
  new_im = Image.fromarray(Result, mode='L')
  new_im.save("__output_max_pool.png")
  new_im.show()
