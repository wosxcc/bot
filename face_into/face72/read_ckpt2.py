# from tensorflow.python import pywrap_tensorflow
# import os
# model_dir = './faceres/log/'
# checkpoint_path = os.path.join(model_dir, "model.ckpt-400.data-00000-of-00001")
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print("tensor_name: ", key)

import tensorflow as tf
import os

logdir='./face72/facell/'

from tensorflow.python import pywrap_tensorflow

ckpt = tf.train.get_checkpoint_state(logdir)

# global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
# print('global_step',global_step)
reader = pywrap_tensorflow.NewCheckpointReader(ckpt.model_checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key).shape)