import tensorflow as tf
import numpy as np

with tf.variable_scope('new_variables', reuse=tf.AUTO_REUSE):
    matrix_new = tf.get_variable('matrix',dtype=tf.float32 ,shape=(2,2), initializer=tf.ones_initializer)

#So we need to create op which will be executed inside of the Session
assign_op = matrix_new.assign_add(tf.fill([2,2], 4.))
with tf.Session() as sess:
    #Init
    sess.run(tf.global_variables_initializer())
    #Here the matrix is printed as we created the variable(no change)
    print(sess.run(matrix_new))
    # Run the op that we created - changing the value of the matrix new
    print('-'*10)
    sess.run(assign_op)
    #Second run of the variable gives us new value
    print(sess.run(matrix_new))

    print('-' * 10)
    sess.run(tf.global_variables_initializer())
    print(sess.run(matrix_new))