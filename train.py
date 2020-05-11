from sys import argv
from PIL import Image
from model import *
import random
import numpy as np
#from sklearn.model_selection import train_test_split
from tfrc import get_input_data
import os

from utils import *

from tensorflow.python.framework import graph_util
import json

batch_size = 10
dropout = 0.3
learning_rate = 0.00001
epochs = 900000
size = 256
s = 1024
l2_weight = 100
seq_len = s * s * 3
read_log = argv[1] == "log"

d_iter = 1
g_iter = 1

def read_test_img(path):
    names = os.listdir(path)
    
    imgs = []
    for name in names:
        im = Image.open(os.path.join(path, name)).convert('RGB')
        im = im.resize((128, int(im.size[1]* 128/im.size[0])))

        imgs.append((np.float32(np.array(im)) - 128)/ 128)

    img1 = random_crop(imgs, 64, False)
    img2 = random_crop(imgs, 64, False)
    imgs = np.concatenate((img1, img2), axis=0)

    return imgs


def save_images(arr, name):
    im = Image.fromarray(arr)
    path = os.path.join('images', '%s.png' % name) 
    im.save(path)

def random_crop(x, s, narrow=True):
    lst = []
   
    for xi in x:
        if narrow:
            w = random.randint(100, xi.shape[0] - s - 100)
            h = random.randint(100, xi.shape[1] - s - 100)
        else:
            w = random.randint(10, xi.shape[0] - s)
            h = random.randint(10, xi.shape[1] - s)
        buf = xi[w:w+s, h:h+s]
        lst.append(buf)
    return np.array(lst)


model = build_network(learning_rate, size, l2_weight)
#prob = tf.nn.softmax(logits)

#path = "sample_images"
#test_x = read_test_img(path)

if __name__ == "__main__":
    output_dir = "gen_output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    #tf_filepath = "/data/ai-datasets/201-FFHQ/tfrecords/face256.tfrecords"
    tf_filepath = "/data/ai-datasets/201-FFHQ/tfrecords/face1024.tfrecords"
    img = get_input_data(tf_filepath, seq_len, batch_size, 10)

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    with tf.Session() as sess:
        #tf_writer = tf.summary.FileWriter('graph_log', sess.graph)
        sess.run(init_global)
        sess.run(init_local)
        if read_log:           
            saver = restore_checkpoint(sess)
            #load_assigned_checkpoint('encoder')

        step = 0
        img_src = None
        while step < epochs:
            step += 1

            if img_src is None or step % 10 == 0:
                img_src = sess.run(img)
                img_src = np.reshape(img_src, [-1, s, s, 3])

            img_arr = random_crop(img_src, size)

            label = (img_arr - 128) / 128

            """
            x = []
            for arr in img_arr:
                im = Image.fromarray(np.uint8(arr))
                im = im.resize((64,64))
                #im = im.resize((size,size))
                x.append((np.float32(np.array(im)) - 128 )/ 128)
                #x.append(np.array(im))
            """

            z = np.random.uniform(-1., 1., size=[batch_size, 128])
            #if random.randint(0,1) == 1:
            #    x = np.array(list(map(np.fliplr, x.tolist())))

            #sess.run(model.R_opt, feed_dict={model.x: x,  model.label: label, model.dropout: dropout, model.is_train: True})
            #for n in range(d_iter):
            #if step % 3 == 0:
            sess.run(model.D_opt, feed_dict={model.z: z,  model.label: label, model.dropout: dropout, model.is_train: True})
            #sess.run(model.G_opt, feed_dict={model.z: z,  model.label: label, model.dropout: dropout, model.is_train: True})
            for n in range(g_iter):
                sess.run(model.G_opt, feed_dict={model.z: z,  model.dropout: dropout, model.is_train: True})
        
            if step % 10 == 0: 
                #l2_loss, g_loss, d_loss = sess.run([model.l2_loss, model.G_loss, model.D_loss], feed_dict={model.x: x, model.label: label, model.dropout: dropout, model.is_train: False})
                #print("Epoch %d/%d - R_loss: %.4f\t G/D loss: %.4f\t%.4f" % (step+1, epochs, l2_loss, g_loss, d_loss))
                g_loss, d_loss = sess.run([model.G_loss, model.D_loss], feed_dict={model.z: z, model.label: label, model.dropout: dropout, model.is_train: False})
                print("Epoch %d/%d - G/D loss: %.4f\t%.4f" % (step+1, epochs, g_loss, d_loss))

                if step % 1000 == 0:    
                    checkpoint_filepath='log/step-%d.ckpt' % step
                    saver.save(sess,checkpoint_filepath)
                    print('checkpoint saved!')

                if step % 20 == 0:
                    #x2 = np.concatenate([test_x, x[:2]], axis=0)
                    g_sample = sess.run(model.logit2, feed_dict={model.z: z, model.dropout: 0, model.is_train: False})
                    #g_sample = sess.run(model.logits, feed_dict={model.x: x, model.dropout: 0, model.is_train: False})

                    for ii, val in enumerate(g_sample):
                        img_arr = (val+1) * 128 
                        im = Image.fromarray(img_arr.astype('uint8'))
                        im.save("%s/%d_pred.jpg" % (output_dir, ii))

                    """
                    for ii, val in enumerate(label):
                        img_arr = (val+1) * 128 
                        im = Image.fromarray(img_arr.astype('uint8'))
                        im.save("%s/%d_label.jpg" % (output_dir, ii))

                    for ii, val in enumerate(x2):
                        img_arr = (val+1) * 128 
                        im = Image.fromarray(img_arr.astype('uint8'))
                        im.save("%s/%d_input.jpg" % (output_dir, ii))
                    """
        output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names=["x", "y", 'dropout', 'logit','is_train', 'feature'])
        with tf.gfile.FastGFile('./pb/face.pb', mode='wb') as f:
                f.write(output_graph_def.SerializeToString())
        print('\nmodel protobuf saved!\n')
