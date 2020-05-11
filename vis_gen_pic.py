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
	data = get_input_data("data/train.tfrecords", 49152, 2)
	sess = tf.Session()
	val = sess.run(data)
	
	img_arr = np.reshape(val[1], [128, 128, 3]) * 255
	#print(img_arr.shape)
	im = Image.fromarray(img_arr.astype('uint8'))
	im.show()

	im.save("gen_output/1.jpg")














