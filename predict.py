pb_path = "pb/face.pb"

output_graph_def = tf.GraphDef()
with open(pb_path,"rb") as f:
    output_graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(output_graph_def, name="")
  
node_in = sess.graph.get_tensor_by_name("input_node_name")
model_out = sess.graph.get_tensor_by_name("out_node_name")
  
feed_dict = {node_in:in_data}
pred = sess.run(model_out, feed_dict)
