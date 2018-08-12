import tensorflow as tf
from multi_label_classification_model import MultiLabelClassificationModel, ModelConfig
import os
import numpy as np

class InferenceConfig(object):
	def __init__(self):
		# checkpoint path.
		self.checkpoint_path = 'path to multi-label-classification model checkpoint dir or file'

		# in which dir inference result file will be saved in.
		self.inference_dir = 'some dir'

		# in which file inference result will be saved in 
		self.inference_file_name = 'some file name'

		# gpu id
		self.gpu_id = "some GPU ID"

def get_test_image_list():
	return ['path to image.jpg']


def main():
	# -- start : modify the model_config and train_config --
	# model config
	model_config = ModelConfig(
		mode="inference",
		label_count=35)
	model_config.resize_height = 290
	model_config.resize_width = 290

	# inference config
	inference_config = InferenceConfig()
	# -- end: modify the model_config and train_config --

	# set log level
	tf.logging.set_verbosity(tf.logging.INFO)

	# inference images list
	images = get_test_image_list()

	# no need to use GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = inference_config.gpu_id

	g = tf.Graph()
	with g.as_default():
		# build model
		model = MultiLabelClassificationModel(model_config)
		model.build()
			
		# find checkpoint file
		if tf.gfile.IsDirectory(inference_config.checkpoint_path):
			checkpoint_file = tf.train.latest_checkpoint(inference_config.checkpoint_path)
			if not checkpoint_file:
				raise ValueError("No checkpoint file found in {}".format(inference_config.checkpoint_path))

		else:
			checkpoint_file = inference_config.checkpoint_path 

		# make sure inference_dir is ready
		if not tf.gfile.IsDirectory(inference_config.inference_dir):
			tf.logging.info("Creating result directory: %s", inference_config.inference_dir)
			tf.gfile.MakeDirs(inference_config.inference_dir)

		with tf.Session(graph=g) as sess:

			# Set up the saver for restoring model checkpoint.
			saver = tf.train.Saver()
			tf.logging.info("Loading model from checkpoint: %s", checkpoint_file)
			saver.restore(sess, checkpoint_file)
			tf.logging.info("Successfully loaded checkpoint: %s", checkpoint_file)
			g.finalize()

			result = []
			for image_path in images:
				tf.logging.info("inference file: {}".format(image_path))

				# read image
				with tf.gfile.GFile(image_path, 'r') as f:
					encoded_image = f.read()
	
				# run model
				sigmoid_result = sess.run(model.sigmoid_result, feed_dict={model.image_feed: encoded_image})

				after_mean = np.mean(sigmoid_result, axis=0)
				result.append(after_mean)

	# save the result
	file_path = os.path.join(inference_config.inference_dir, inference_config.inference_file_name)

	tf.logging.info("save result to: %s", file_path)

	#print(result)
	np.savetxt(file_path, result, fmt='%.6f')
	
if __name__ == '__main__':
	main()
