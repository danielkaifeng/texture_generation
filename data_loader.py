from PIL import Image
from sys import argv
import numpy as np
from tfrc import *

def get_img_data(image_path):
	im = Image.open(image_path) 
	im = im.convert('RGB')
	data = np.array(im.getdata())
	#data.shape (16384, 3)
	return data.flatten()


if __name__ == "__main__":
	"""
	filelist = argv[1]
	tf_filepath = "data/ocean.tfrecords"
	#writer = tf.io.TFRecordWriter("./data/orange_train.tfrecords")
	writer = tf.io.TFRecordWriter(tf_filepath)

	with open(filelist,'r') as f1:	
		txt = f1.readlines()
	for line in txt:
		image_path = line.strip()
		data = get_img_data(image_path)
		data = data/255.
		write_tfrecords(data, writer)
	#np.savetxt("data/train_data.txt", total_data, fmt='%.4f', delimiter=',')
	writer.close()
	"""

	#tf_filepath = "/data/ai-datasets/201-FFHQ/tfrecords/face1024.tfrecords"
	tf_filepath = "/data/ai-datasets/201-FFHQ/tfrecords/face256.tfrecords"
	data = get_input_data(tf_filepath, 256*256*3, 2)
	sess = tf.Session()
	for i in range(2):
		print(sess.run(data))
	sess.close()














