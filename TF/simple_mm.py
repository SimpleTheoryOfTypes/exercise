import tensorflow as tf
from tensorflow.python.tools import freeze_graph

# The following snippet was tested on TF 1.14.0.
print(tf.__version__)

x = tf.placeholder(dtype=tf.float32, shape=[3,2], name='input')
b = tf.Variable(initial_value=[[7,8],[9,10]], shape=[2,2], dtype=tf.float32, name='weight')
c = tf.matmul(x,b, name='mm/output')

# Reference: https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125#.y9zvdd6f9
# We can check easily that we are indeed in the default graph
print(c.graph == tf.get_default_graph())

with tf.compat.v1.Session() as sess:
  # Run initializer first.
  init=tf.global_variables_initializer()
  sess.run(init)

  # Run my graph.
  print(sess.run(c, feed_dict={x: [[1,2], [3,4], [5,6]]}))

  # Reference: https://leimao.github.io/blog/Save-Load-Inference-From-TF-Frozen-Graph/
  tf.io.write_graph(graph_or_graph_def=sess.graph_def, logdir='./', name='gai.pbtxt', as_text=True)

  # Checkpoint my graph before freezing.
  tf.compat.v1.train.Saver().save(sess, './mm-all')

  # Begin freezing my graph.
  graph = tf.get_default_graph()
  input_graph_def = graph.as_graph_def()
  output_node_names = ['mm/output']
  output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names)

  # For some models, we would like to remove training nodes
  output_graph_def = tf.compat.v1.graph_util.remove_training_nodes(output_graph_def, protected_nodes=None)

  pb_filepath = './frozen_model.pb'
  with tf.io.gfile.GFile(name=pb_filepath, mode='wb') as f:
    f.write(output_graph_def.SerializeToString())

# Verify frozen graph by reloading and re-running.
print('Loading model...')
mm_graph = tf.Graph()
isess = tf.InteractiveSession(graph = mm_graph)

with tf.io.gfile.GFile(pb_filepath, 'rb') as f:
  graph_def = tf.compat.v1.GraphDef()
  graph_def.ParseFromString(f.read())

print('Check out the input placeholders:')
nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
for node in nodes:
    print(node)

# Define input tensor
x = tf.placeholder(dtype=tf.float32, shape=[3,2], name='input')

tf.import_graph_def(graph_def, {'input': x})

print('Model loading complete!')

# run the imported model.
print(mm_graph._nodes_by_name)
output_tensor = mm_graph.get_tensor_by_name("import/mm/output:0")
output = isess.run(output_tensor, feed_dict = {x: [[1,2], [3,4], [5,6]]})
print(output)
