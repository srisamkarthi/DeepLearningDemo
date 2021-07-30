from __future__ import print_function

import tensorflow as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
# Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!')

# Start tf session
sess = tf.compat.v1.Session()

# Run the op
print(sess.run(hello))

sess.close
