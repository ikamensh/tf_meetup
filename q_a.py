import tensorflow as tf
import numpy as np

# Question Task 2:
# 1. Create two 0-d tensors x and y randomly selected from the range [-1, 1).
# 2. Return x + y if x < y, x - y if x > y, 0 otherwise.
# 3. Look up tf.case().

x = tf.random_uniform([], -1, 1)
y = tf.random_uniform([], -1, 1)

sum = lambda: tf.add(x,y)
diff = lambda: tf.subtract(x,y)

f = tf.case([(tf.less(x,y), sum),
             (tf.greater(x,y),diff)],
            default=lambda:tf.constant(0.))

for i in range(10):
    with tf.Session() as sess:
        print(sess.run([x,y,f]))

# QuestionTask 3
# 1.Create the tensor x of the value [[0, -2, -1], [0, 1, 2]] and y as a tensor of zeros with the same shape as x.
# 2.Return a boolean tensor that yields Trues if x equals y element-wise.
# 3.Hint: Look up tf.equal().

x = tf.convert_to_tensor([[0, -2, -1],[0, 1, 2]])
y = tf.zeros_like(x)

eq = tf.equal(x,y)

with tf.Session() as sess:
    print(sess.run(eq))

# Question Task 4
# 1.Create the tensor x of value
# [29.05088806, 27.61298943, 31.19073486, 29.35532951,30.97266006, 26.67541885, 38.08450317, 20.74983215, 34.94445419, 34.45999146, 29.06485367, 36.01657104, 27.88236427, 20.56035233, 30.20379066, 29.51215172,33.71149445, 28.59134293, 36.05556488, 28.66994858].
#
# 2.Get the indices of elements in x whose values are greater than 30.
# 3.Hint: Use tf.where().
# 4.Then extract elements whose values are greater than 30.
# 5.Hint: Use tf.gather().

val = [29.05088806, 27.61298943, 31.19073486, 29.35532951,30.97266006, 26.67541885, 38.08450317, 20.74983215, 34.94445419, 34.45999146, 29.06485367, 36.01657104, 27.88236427, 20.56035233, 30.20379066, 29.51215172,33.71149445, 28.59134293, 36.05556488, 28.66994858]
x = tf.convert_to_tensor(val)
# indexes = tf.where(tf.greater(x, 30), [True]*len(val), [False]*len(val))
indexes = tf.where(tf.greater(x, 30))

with tf.Session() as sess:
    print(sess.run(indexes))

elements = tf.gather(x, indexes)

with tf.Session() as sess:
    print(sess.run(elements))


# Question Task 5
# 1.Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,2, ..., 6
# 2.Use tf.range() and tf.diag().

x = tf.range(1,7)
d = tf.diag(x)

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(d))


# Question Task 7
# 1.Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# 2.Return the unique elements in x
# 3.use tf.unique(). Keep in mind that tf.unique() returns a tuple.

val = [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9]
x = tf.convert_to_tensor(val)
unique, idx = tf.unique(x)
val_back = tf.gather(unique, idx)

with tf.Session() as sess:
    print(sess.run(unique))
    print(sess.run(idx))
    print(sess.run(val_back))






