import numpy as np
import tensorflow as tf

scalar_x = 4

array_1d = np.array([1,2,3,4,5.99])
array_2d = np.array([(1,2,3,4,5.99),(2,3,4,5,6.99),(3,4,5,6,7.99)])
array_3d = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])

scalar_x_tf = tf.convert_to_tensor(scalar_x, name='scalar_converted')
array_1d_tf = tf.convert_to_tensor(array_1d, dtype=tf.float32)
array_2d_tf = tf.convert_to_tensor(array_2d)
array_3d_tf = tf.convert_to_tensor(array_3d)

print(scalar_x_tf)
print(array_1d_tf)
print(array_2d_tf)
print(array_3d_tf)