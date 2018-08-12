from tensorflow.python.framework import ops
import tensorflow as tf
slim = tf.contrib.slim
from resnet_v2 import resnet_v2_50
import numpy as np


class ModelConfig(object):
	def __init__(self, mode, label_count, base_network_checkpoint=None, input_tfrecord_file=None):
		"""Initialize model config.

		Args:
			mode: str, train, eval or inference.
			label_count: int, total count of different labels.
			base_network_checkpoint: make sure it's not None when train on first time.
			input_tfrecord_file: make sure it's not None when train or eval.
		"""
		# mode
		self.mode = mode

		# label count
		self.label_count = label_count

		# Must be provided when starting training for the first time
		self.base_network_checkpoint = base_network_checkpoint

		# inputs tf recourd file. If mode is train or eval, make sure it's not None
		self.input_tfrecord_file = input_tfrecord_file

		# target image size
		self.target_height = 224
		self.target_width = 224

		# resize image size. make sure larger than target image size
		self.resize_height = 255
		self.resize_width = 255

		# l2 weight decay
		self.use_regularizer= True
		self.weight_decay = 0.0001
		
		# batch size
		self.batch_size = 32

		# base network
		self.train_base_network = False
		
		# keys in sequence example 
		self.image_path_key = "image_path"
		self.image_data_key = "image_data"
		self.image_label_key = "image_label"

		# number of threads do enqueue operations
		self.num_threads = 1


class MultiLabelClassificationModel(object):
	"""One simple multi label image classification implementation"""


	def __init__(self, model_config):
		"""multi lable classification model initializer.

		Args:
			model_config: instance if ModelConfig class.
		"""
		# model config
		self.model_config = model_config

		# A float32 Tensor with shape [batch_size, height, width, channels]
		self.images = None

		# Instance of tf.placeholder, str of image content
		self.image_feed = None

		# An int32 Tensor with shape [batch_size, label_count]
		self.target_labels = None

		# Function to restore the base network from checkpoint
		self.base_network_init_fn = None

		# Global step Tensor
		self.global_step = None

		# sigmoid result, used in inference
		self.sigmoid_result = None

		# loss function
		self.sigmoid_cross_entropy_loss = None 
		self.total_loss = None


	def process_image(self, encoded_image, thread_id=0):
		"""Decodes and processes an image string.

		Args:
			encoded_image: A scalar string Tensor, the encoded image.
			thread_id: thread id used to do image augmentation.

		Returns:
			A float32 Tensor of shape [height, width, 3]; the processed image.
		"""
		with tf.name_scope("decode_image", values=[encoded_image]):
			# Decode image into float32 Tensor of shape [?, ?, 3] with values in [0, 1)
			image = tf.image.decode_jpeg(encoded_image, channels=3)
			image = tf.image.convert_image_dtype(image, dtype=tf.float32)

			if self.model_config.mode == "train":
				# resize image
				image = tf.image.resize_images(image, 
					size=[self.model_config.resize_height, self.model_config.resize_width], 
					method=tf.image.ResizeMethod.BILINEAR)
			
				# crop to target size
				image = tf.random_crop(image, 
					[self.model_config.target_height, self.model_config.target_width, 3])
				
				# do some modifications
				image = tf.image.random_flip_left_right(image)
				image = tf.image.random_brightness(image, max_delta=32. / 255.)
				image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
				image = tf.image.random_hue(image, max_delta=0.032)
				image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

				# The random_* ops do not necessarily clamp.
				image = tf.clip_by_value(image, 0.0, 1.0)

			else:
				image = tf.image.resize_images(image, 
					size=[self.model_config.target_height, self.model_config.target_width], 
					method=tf.image.ResizeMethod.BILINEAR)

			# Rescale to [-1, 1] instead of [0, 1]
			image = tf.subtract(image, 0.5)
			image = tf.multiply(image, 2.0)

			return image

	def over_sample_image(self, encoded_image):
		image = tf.image.decode_jpeg(encoded_image, channels=3)
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		image = tf.image.resize_images(image, 
			size=[self.model_config.resize_height, self.model_config.resize_width], 
			method=tf.image.ResizeMethod.BILINEAR)
	
		im_shape = tf.shape(image)
		crop1 = tf.slice(image, 
			[0, 0, 0], 
			[self.model_config.target_height, self.model_config.target_width, 3], 
			name="crop_1")

		crop2 = tf.slice(image, 
			[0, self.model_config.resize_width - self.model_config.target_width, 0], 
			[self.model_config.target_height, self.model_config.target_width, 3], 
			name="crop_2")

		crop3 = tf.slice(image, 
			[self.model_config.resize_height - self.model_config.target_height, 0, 0], 
			[self.model_config.target_height, self.model_config.target_width, 3], 
			name="crop_3")

		crop4 = tf.slice(image, 
			[self.model_config.resize_height - self.model_config.target_height, self.model_config.resize_width - self.model_config.target_width, 0], 
			[self.model_config.target_height, self.model_config.target_width, 3], 
			name="crop_4")

		crop5 = tf.slice(image, 
			[(self.model_config.resize_height - self.model_config.target_height)/2, (self.model_config.resize_width - self.model_config.target_width)/2, 0], 
			[self.model_config.target_height, self.model_config.target_width, 3], 
			name="crop_5")

		reverse1 = tf.reverse(crop1, [1], name="reverse_1")
		reverse2 = tf.reverse(crop2, [1], name="reverse_2")
		reverse3 = tf.reverse(crop3, [1], name="reverse_3")
		reverse4 = tf.reverse(crop4, [1], name="reverse_4")
		reverse5 = tf.reverse(crop5, [1], name="reverse_5")

		images = tf.stack([
			crop1,
			crop2,
			crop3,
			crop4,
			crop5,
			reverse1,
			reverse2,
			reverse3,
			reverse4,
			reverse5])

		return images
		

	def build_inputs(self):
		"""Read Input, preprocessing and batching.

		Outputs:
			self.image_feed
			self.images
			self.target_labels
		"""
		if self.model_config.mode == "inference":
			# input image as string
			self.image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
			#image = self.process_image(self.image_feed)
			#images = tf.expand_dims(image, 0)
			images = self.over_sample_image(self.image_feed)
			#tf.logging.info("images type: {}, shape: {}".format(type(images), images.shape))
			target_labels = None
		else:
			filename_queue = tf.train.string_input_producer([self.model_config.input_tfrecord_file])
			reader = tf.TFRecordReader()
			_, serialized_example = reader.read(filename_queue)
			
			# parse sequence example
			context, sequence = tf.parse_single_sequence_example(
				serialized_example,
				context_features={
					self.model_config.image_path_key: tf.FixedLenFeature([], dtype=tf.string),
					self.model_config.image_data_key: tf.FixedLenFeature([], dtype=tf.string),
				},
				sequence_features={
					self.model_config.image_label_key: tf.FixedLenSequenceFeature([], dtype=tf.int64),
				})		

			image_path = context[self.model_config.image_path_key]
			encoded_image = context[self.model_config.image_data_key]
			image_label = sequence[self.model_config.image_label_key]
			image_label = tf.cast(image_label, tf.float32)
			
			image = self.process_image(encoded_image)
			
			if self.model_config.mode == "train":
				# use shuffle
				min_after_dequeue = self.model_config.batch_size * 3
				capacity = min_after_dequeue + (1 + self.model_config.num_threads) * self.model_config.batch_size
				images, target_labels = tf.train.shuffle_batch(
					[image, image_label],
					self.model_config.batch_size,
					min_after_dequeue=min_after_dequeue,
					num_threads=self.model_config.num_threads,
					capacity=capacity,
					shapes=[[self.model_config.target_height, self.model_config.target_width, 3], [self.model_config.label_count]])
					
			else:
				# eval, not use shuffle
				capacity = (1 + self.model_config.num_threads) * self.model_config.batch_size
				images, target_labels = tf.train.batch(
					[image, image_label],
					self.model_config.batch_size,
					num_threads=self.model_config.num_threads,
					capacity=capacity,
					dynamic_pad=True)

		self.images = images
		self.target_labels = target_labels


	def resnet_arg_scope(self):
		""""Defines the default ResNet arg scope
		
		Args:
			self.model_config.train_base_network
			self.model_config.mode

		Returns:
			resnet arg scope
		"""

		is_base_network_training = self.model_config.train_base_network and (self.model_config.mode == "train")
		batch_norm_decay=0.997
		batch_norm_epsilon=1e-5
		batch_norm_scale=True
		batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS
		weight_decay=0.0001 
		activation_fn=tf.nn.relu
		use_batch_norm=True

		if use_batch_norm:
			batch_norm_params = {
				"is_training": is_base_network_training,
				"trainable": self.model_config.train_base_network,
				'decay': batch_norm_decay,
				'epsilon': batch_norm_epsilon,
				'scale': batch_norm_scale,
				'updates_collections': batch_norm_updates_collections,
				'fused': None,  # Use fused batch norm if possible.
			}
		else:
			batch_norm_params = {}

		if self.model_config.train_base_network:
			weights_regularizer = slim.l2_regularizer(weight_decay)
		else:
			weights_regularizer = None

		with slim.arg_scope([slim.conv2d],
			trainable=self.model_config.train_base_network,
			weights_regularizer=weights_regularizer,
			weights_initializer=slim.variance_scaling_initializer(),
			activation_fn=activation_fn,
			normalizer_fn=slim.batch_norm if use_batch_norm else None,
			normalizer_params=batch_norm_params):

			with slim.arg_scope([slim.batch_norm], **batch_norm_params):
				with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
					return arg_sc

					
	def build_model(self):
		"""Builds the model.

		Inputs:
			self.images
			self.target_labels
			self.model_config.train_base_network
			self.model_config.mode

		Outputs:
			self.sigmoid_result (inference)
			self.cross_entropy (train, eval)
			self.total_loss
		"""

		# build resnet_50
		arg_scope = self.resnet_arg_scope()
		is_base_network_training = self.model_config.train_base_network and (self.model_config.mode == "train")
		with slim.arg_scope(arg_scope):
			net, end_points = resnet_v2_50(self.images, is_training=is_base_network_training, global_pool=False)

		with tf.variable_scope('my_sub_network'):
			if self.model_config.use_regularizer:
				weights_regularizer = slim.l2_regularizer(self.model_config.weight_decay)
			else:
				weights_regularizer = None

			net = slim.conv2d(net, 
				self.model_config.label_count, 
				[1, 1], 
				weights_regularizer=weights_regularizer, 
				scope="my_conv_1")

			net = slim.conv2d(net, 
				self.model_config.label_count, 
				[1, 1], 
				weights_regularizer=weights_regularizer, 
				scope="my_conv_2")

			net = slim.conv2d(net, 
				self.model_config.label_count, [7, 7], 
				weights_regularizer=weights_regularizer, 
				padding="VALID", 
				activation_fn=None, 
				scope="my_conv_3")

			logits = tf.squeeze(net, [1, 2], name="logits")
		
		self.sigmoid_result = tf.sigmoid(logits, name="sigmoid_result")

		if self.model_config.mode == "inference":
			return
		
		tf.logging.info("Before:--- GraphKeys.LOSSES len: {}; GraphKeys.REGULARIZATION_LOSSES len: {}".format(len(tf.losses.get_losses()), len(tf.losses.get_regularization_losses())))

		# By default, all the losses in 'tf.losses' are collected into the GraphKeys.LOSSES collection.
		sigmoid_cross_entropy_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.target_labels, logits=logits, scope="sigmoid_cross_entropy")

		# get_total_loss get both GraphKeys.LOSSES and GraphKeys.REGULARIZATION_LOSSES.
		total_loss = tf.losses.get_total_loss()

		tf.logging.info("After:--- GraphKeys.LOSSES len: {}; GraphKeys.REGULARIZATION_LOSSES len: {}".format(len(tf.losses.get_losses()), len(tf.losses.get_regularization_losses())))

		# Add summaries.
		tf.summary.scalar("losses/sigmoid_cross_entropy_loss", sigmoid_cross_entropy_loss)
		tf.summary.scalar("losses/total_loss", total_loss)

		self.sigmoid_cross_entropy_loss = sigmoid_cross_entropy_loss 
		self.total_loss = total_loss

		
	def setup_base_network_initializer(self):
		"""Sets up the function to restore base network variables from checkpoint."""
		if self.model_config.mode != "inference" and self.model_config.base_network_checkpoint != None:
			# restore base network model
			tf.logging.info("Restoring resnet_v2_50 from checkpoint file: {}".format(self.model_config.base_network_checkpoint))
			base_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet_v2_50")
			saver = tf.train.Saver(base_network_variables)
			def restore_fn(sess):
				saver.restore(sess, self.model_config.base_network_checkpoint)

			self.base_network_init_fn = restore_fn


	def setup_global_step(self):
		"""Sets up the global step Tensor."""
		self.global_step = tf.Variable(
			initial_value=0,
			name="global_step",
			trainable=False,
			collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])


	def build(self):
		"""Creates all ops for train, eval and inference."""
		self.build_inputs()
		self.build_model()
		
		self.setup_base_network_initializer()
		self.setup_global_step()

