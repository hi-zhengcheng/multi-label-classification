import tensorflow as tf
import os
import time
from multi_label_classification_model import MultiLabelClassificationModel, ModelConfig

class EvalConfig(object):
	def __init__(self):
		# total count of examples in eval dataset
		self.num_eval_examples = 291

		# checkpoint dir
		self.checkpoint_dir = "some path"

		# check min global step
		self.min_global_step = 200

		self.batch_size = 32

		# eval 
		self.eval_dir = os.path.join(self.checkpoint_dir, 'eval')
		self.eval_interval_secs = 30
	
	
last_eval_checkpoint_step = None

def evaluate_model(sess, model, global_step, summary_writer, summary_op, eval_config):
	"""computes loss

	Args:
		sess: Session object
		model: Instance of MultiLabelClassificationModel
		global_step: Integer; global step of the model checkpoint
		summary_writer: Instance of FileWriter.
		summary_op: Op for generating model summaries.
		eval_config: Instance of EvalConfig
	"""
	summary_str = sess.run(summary_op)
	summary_writer.add_summary(summary_str, global_step)

	num_eval_batches = eval_config.num_eval_examples / eval_config.batch_size
	
	sum_losses = 0.0
	start_time = time.time()
	for i in range(num_eval_batches):
		loss = sess.run(model.total_loss)
		tf.logging.info("Computed losses for {} of {} batches: {}".format(i + 1, num_eval_batches, loss))
		sum_losses += loss

	ave_loss = sum_losses / num_eval_batches
	tf.logging.info("Ave loss: {}".format(ave_loss))

	summary = tf.Summary()	
	value = summary.value.add()
	value.simple_value = ave_loss
	value.tag = "AveLoss"
	summary_writer.add_summary(summary, global_step)


def run_once(model, saver, summary_writer, summary_op, eval_config):
	"""Evaluates the latest model checkpoint.

	Args:
		model: Instance of MultiLabelClassificationModel to evaluate
		saver: Instance of tf.train.Saver for restoring the model Variables
		summary_writer: Instance of FileWriter.
		summary_op: Op for generating model summaries.
		eval_config: Instance of EvalConfig.
	"""
	global last_eval_checkpoint_step
	model_path = tf.train.latest_checkpoint(eval_config.checkpoint_dir)
	if not model_path:
		tf.logging.info("Skipping evaluation. No checkpoint found in: %s", eval_config.checkpoint_dir)
		return

	with tf.Session() as sess:
		# Load model from checkpoint
		tf.logging.info("Loading model from checkpoint: %s", model_path)
		saver.restore(sess, model_path)
		global_step = tf.train.global_step(sess, model.global_step.name)
		tf.logging.info("Successfully loaded %s at global step = %d.", os.path.basename(model_path), global_step)

		if last_eval_checkpoint_step == global_step:
			tf.logging.info("No new checkpoint created")
			return
		
		last_eval_checkpoint_step = global_step

		# check min global step
		if global_step < eval_config.min_global_step:
			tf.logging.info("Skipping evaluation. Global step = %d < %d", global_step, eval_config.min_global_step)
			return

		# Start the queue runners
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		# Run evaluation on the latest checkpoint
		try:
			evaluate_model(
				sess=sess,
				model=model,
				global_step=global_step,
				summary_writer=summary_writer,
				summary_op=summary_op,
				eval_config=eval_config)
		except Exception as e:
			tf.logging.error("Evaluation failed.")
			coord.request_stop(e)

		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=10)
		


def main():
	# -- start : modify the model_config and train_config --
	# model config
	model_config = ModelConfig(
		mode="eval",
		label_count=35,
		input_tfrecord_file="path to val.tfrecords")

	# eval config
	eval_config = EvalConfig()
	eval_config.batch_size = model_config.batch_size
	# -- end: modify the model_config and train_config --


	# no need to use GPU
	os.environ['CUDA_VISIBLE_DEVICES'] = ""

	# set log level
	tf.logging.set_verbosity(tf.logging.INFO)

	# prepare eval dir
	if not tf.gfile.IsDirectory(eval_config.eval_dir):
		tf.logging.info("Creating eval directory: %s", eval_config.eval_dir)
		tf.gfile.MakeDirs(eval_config.eval_dir)


	g = tf.Graph()
	with g.as_default():
		# build model
		model = MultiLabelClassificationModel(model_config)
		model.build()

		# Create saver to restore model variables
		saver = tf.train.Saver()

		# Create the summary operation and the summary writer.
		summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(eval_config.eval_dir, graph=g)

		g.finalize()

		while True:
			start = time.time()
			tf.logging.info("Starting evaluation at " + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
			run_once(model, saver, summary_writer, summary_op, eval_config)
			time_to_next_eval = start + eval_config.eval_interval_secs - time.time()
			if time_to_next_eval > 0:
				time.sleep(time_to_next_eval)


if __name__ == '__main__':
	main()
