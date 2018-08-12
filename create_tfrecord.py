import tensorflow as tf
import numpy as np
import random
import os
from datetime import datetime
import sys
import argparse

class ImageDecoder(object):
	"""Helper class for decoding images in Tensorflow.

	Used to check if image data format is valid.
	"""

	def __init__(self):
		self._sess = tf.Session()
		self._encoded_jpeg = tf.placeholder(dtype=tf.string)
		self._decoded_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

	def decode_jpeg(self, encoded_jpeg):
		image = self._sess.run(self._decoded_jpeg, 
				feed_dict={self._encoded_jpeg: encoded_jpeg})

		assert len(image.shape) == 3
		assert image.shape[2] == 3
		return image

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def _int64_feature_list(values):
	return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
	return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _to_sequence_example(image_path, labels, decoder):
	"""Builds a SequenceExample proto for an image-label pair.

	Args:
		image_path: str, image file path.
		image_id: int, image id.
		labels: list of int, image labels.
		decoder: ImageDecoder.

	Returns:
		A SequenceExample proto if image is valid, None if invalid.
	"""

	with tf.gfile.FastGFile(image_path, 'r') as f:
		encoded_image = f.read()

	try:
		decoder.decode_jpeg(encoded_image)
	except (tf.errors.InvalidArgumentError, AssertionError):
		print("Skipping file with invalid JPEG data: {}".format(image_path))
		return None

	context = tf.train.Features(feature={
		"image_path": _bytes_feature(image_path),
		"image_data": _bytes_feature(encoded_image),
	})

	feature_lists = tf.train.FeatureLists(feature_list={
		"image_label": _int64_feature_list(labels),
	})

	sequence_example = tf.train.SequenceExample(
		context=context,
		feature_lists=feature_lists)

	return sequence_example

	
def _generate_tfrecord_file(image_dir, imglist_file, imglabel_file, output_file):
	"""Generate tfrecord file.

	Args:
		image_dir: str, image dir.
		imglist_file: str, the file stores image list. each line format: "xxx.jpg 2". 2 is the count of label 1.
		imglabel_file: str, the file stores image labels. each line format: "1 0 1 0 ... 1 0" 
		output_file: str, target file to save tfrecord.

	"""

	# prepare dir
	output_dir = os.path.dirname(output_file)
	if output_dir and not os.path.exists(output_dir):
		os.mkdir(output_dir)

	# load img names and labels
	imglist = np.loadtxt(imglist_file, dtype=str)
	labels = np.loadtxt(imglabel_file, dtype=str)
	assert len(imglist) == len(labels)

	# shuffle
	tmp = np.hstack((imglist, labels))
	np.random.shuffle(tmp)
	imglist = tmp[:,:imglist.shape[1]]
	labels = tmp[:, imglist.shape[1]:].astype(float).astype(int)
	
	counter = 0
	decoder = ImageDecoder()
	writer = tf.python_io.TFRecordWriter(output_file)
	for i in range(len(imglist)):
		# check labels count
		assert int(imglist[i][1]) == np.sum(labels[i]), "label count of {} is error".format(imglist[i][0])

		# generate sequence example
		image_path = os.path.join(image_dir, imglist[i][0])
		sequence_example = _to_sequence_example(image_path, labels[i], decoder)

		# write into file
		if sequence_example is not None:
			writer.write(sequence_example.SerializeToString())
			counter += 1
		
		# log
		if not counter % 50:
			print("{} : Processed {} items".format(datetime.now(), counter))
			sys.stdout.flush()

	writer.close()

	print("Finished. Save to: {}".format(output_file))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_dir', required=True, help='in which place raw image files saved')
	parser.add_argument('--imglist_file', required=True, help='file containing image names and label count')
	parser.add_argument('--imglabel_file', required=True, help='file containing image labels')
	parser.add_argument('--output_file', required=True, help='in which file tfrecord will be stored')
	parser.add_argument('--gpu', default="", help='in which file tfrecord will be stored')
	
	args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
	_generate_tfrecord_file(args.image_dir, args.imglist_file, args.imglabel_file, args.output_file)
