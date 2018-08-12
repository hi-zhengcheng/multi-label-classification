import tensorflow as tf
import os
import argparse


def build_batch_readout(data_path, batch_size, num_threads, image_size=224, is_training=False):
	# Put tfrecord file name (or names) into a list, pass it to a queue
	# Here, i don't set num_epochs to get cycled unlimited data.
	filename_queue = tf.train.string_input_producer([data_path])
	
	# Define the reader, and read the next record
	# Actually, it doesn't really read the next record, just adds one read operation in the graph
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)

	# parse sequence example
	image_path_key = "image_path"
	image_data_key = "image_data"
	image_label_key = "image_label"
	context, sequence = tf.parse_single_sequence_example(
		serialized_example, 
		context_features={
			image_path_key: tf.FixedLenFeature([], dtype=tf.string),
			image_data_key: tf.FixedLenFeature([], dtype=tf.string),
		},
		sequence_features={
			image_label_key: tf.FixedLenSequenceFeature([], dtype=tf.int64),
		})

	image_path = context[image_path_key]
	encoded_image = context[image_data_key]
	image_label = sequence[image_label_key]

	# do some log
	encoded_image = tf.Print(encoded_image, data=[image_path], message='imgpath: ')

	# decode image. must add 'channels=3' to let img tensor has channel dimen info
	img = tf.image.decode_jpeg(encoded_image, channels=3)

	# can also to image augmentation here
	img = tf.image.convert_image_dtype(img, dtype=tf.float32)
	resize_height = image_size
	resize_width = image_size
	img = tf.image.resize_images(img, size=[resize_height, resize_width], method=tf.image.ResizeMethod.BILINEAR)
		
	if is_training:
		image_batch, label_batch = tf.train.shuffle_batch(
			[img, image_label],
			batch_size,
			min_after_dequeue=10,
			capacity=10 + (num_threads + 1) * batch_size,
			num_threads=num_threads,
			shapes=[(image_size, image_size, 3), (52,)])
	else:
		# If dynamic_pad is False, you must ensure that either:
		# (i) the shapes argument is passed
		# (ii) all of the tensors in tensors must have fully-defined shapes
		# Since we have resize image to same size, and label info also have the save size, 
		# it's OK to set true. Otherwize, we have specify the label count info
		image_batch, label_batch = tf.train.batch(
			[img, image_label],
			batch_size,
			num_threads=num_threads,
			capacity=batch_size * (num_threads + 1),
			dynamic_pad=True)

	return [image_batch, label_batch]
	
	
def main(data_path, batch_size, num_threads):

	images = []
	labels = []

	with tf.Session() as sess:
		# save the graph to visualize purpose
		writer = tf.summary.FileWriter('./tmp', sess.graph)

		batch_data = build_batch_readout(data_path, batch_size, num_threads, is_training=True)

		# Initialize all global and local variables
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		sess.run(init_op)

		# create coordinator and all QueueRunner objects
		# A QueueRunner will control the asynchronous execution of enqueue operations to ensure that our queues never run dry
		# tf.train.batch and tf.train.string_input_producer automatically add their QueueRunners into current Graph's QUEUE_RUNNER collectio
		# A coordinator object helps to make sure that all the threads we created by QueueRunner stop together
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		for batch_idx in range(5):
			image, label = sess.run(batch_data)
			images.extend(image)
			labels.extend(label)

		# stop the threads
		coord.request_stop()

		# wait for threads to stop
		coord.join(threads)

	# verify if data is OK
	print(labels[0])

	with tf.Session() as sess:
		img = tf.image.convert_image_dtype(images[0], dtype=tf.uint8)
		img = tf.image.encode_jpeg(img)
		net = tf.write_file('./tmp/check_img.jpg', img)
		sess.run(net)
	
	


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default="", help='in which file tfrecord will be stored')
	parser.add_argument('--data_path', required=True, help='TFRecord file path')
	parser.add_argument('--batch_size', type=int, default=2, help='batch size')
	parser.add_argument('--num_threads', type=int, default=2, help='number of threads enqueuing tensors')
	args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
	main(args.data_path, args.batch_size, args.num_threads)
