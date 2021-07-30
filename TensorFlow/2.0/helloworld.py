import tensorflow as tf

# Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.

helloworld = tf.constant("hello, TensorFlow")
print("Tensor:", helloworld)
print("Value :", helloworld.numpy())

