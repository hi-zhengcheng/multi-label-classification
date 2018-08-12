import tensorflow as tf
from multi_label_classification_model import MultiLabelClassificationModel, ModelConfig
import os

class TrainConfig(object):
	def __init__(self):
		# dir to save checkpoint
		self.train_dir = 'some dir'

		# GPU ID
		self.gpu_id = "some GPU id or empty to use no GPU"
		
		# learning rate
		self.learning_rate = 0.001

		# params for learning rate exponential_decay
		self.decay_steps = 1000
		self.decay_rate = 0.9
		self.staircase = True

		# optimizer
		self.optimizer = "Adam"	
		self.clip_gradients = 5.0

		# saver
		self.max_to_keep = 20
		self.keep_checkpoint_every_n_hours = 0.2

		# train steps
		self.number_of_steps = 100000
		

def main():
	# -- start : modify the model_config and train_config --
	# model config
	model_config = ModelConfig(
		mode="train", 
		label_count=35,
		base_network_checkpoint="path to resnet_v2_50.ckpt",
		input_tfrecord_file="path to train.tfrecords")

	# train config
	train_config = TrainConfig()
	# -- end: modify the model_config and train_config --

	
	# set GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = train_config.gpu_id 

	# set log level
	tf.logging.set_verbosity(tf.logging.INFO)

	# prepare train dir
	if not tf.gfile.IsDirectory(train_config.train_dir):
		tf.logging.info("Creating training directory: %s", train_config.train_dir)
		tf.gfile.MakeDirs(train_config.train_dir)

	g = tf.Graph()
	with g.as_default():

		# build model
		model = MultiLabelClassificationModel(model_config)
		model.build()

		def _learning_rate_decay_fn(learning_rate, global_step):
			return tf.train.exponential_decay(
				learning_rate,
				global_step,
				decay_steps=train_config.decay_steps,
				decay_rate=train_config.decay_rate,
				staircase=train_config.staircase)
		
		learning_rate_decay_fn = _learning_rate_decay_fn

		# Set up the training ops
		train_op = tf.contrib.layers.optimize_loss(
			loss=model.total_loss,
			global_step=model.global_step,
			learning_rate=train_config.learning_rate,
			optimizer=train_config.optimizer,
			clip_gradients=train_config.clip_gradients,
			learning_rate_decay_fn=learning_rate_decay_fn)

		# Set up the saver for saving and restoring model checkpoints.
		saver = tf.train.Saver(
			max_to_keep=train_config.max_to_keep,
			keep_checkpoint_every_n_hours=train_config.keep_checkpoint_every_n_hours)

		# Run training.
		tf.contrib.slim.learning.train(
			train_op,
			train_config.train_dir,
			log_every_n_steps=1,
			graph=g,
			global_step=model.global_step,
			number_of_steps=train_config.number_of_steps,
			init_fn=model.base_network_init_fn,
			saver=saver)


if __name__ == "__main__":
	main()
