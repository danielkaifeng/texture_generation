import tensorflow as tf
import os

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def latest_checkpoint(path):
    with open(os.path.join(path, "checkpoint"),'r') as f1:
        txt = f1.readline()
        point = txt.strip().replace('model_checkpoint_path: ','').replace("\"",'')
        print('\n---------->read checkpoint %s\n' % point)
    return point

def restore_checkpoint(sess):
    saver=tf.train.Saver()
    point = latest_checkpoint('log')
    saver.restore(sess,"log/%s"%point)
    return saver
   

def load_assigned_checkpoint(var_name):
    init_checkpoint = latest_checkpoint('log')

    tvars = tf.trainable_variables()
    var_list = [v for v in tvars if var_name not in v.name]
    assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(var_list, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    print("encoder network inited!")

