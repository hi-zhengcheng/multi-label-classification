import numpy as np
from sklearn import metrics
import argparse


def compute_optimal_threshold(y, score):
	"""Compyte one label's optimal threshold score.

	Args:
		y: array, ground truth array, 1 for positive, 0 for negative.
		score: array, model computed scores, in range [0, 1]

	Returns:
		optimal threshold score value
	"""
	fpr, tpr, thresholds = metrics.roc_curve(y, score, pos_label=1)

	# just set a large distance than sqrt(2)
	min_distance = 2
	result = thresholds[0]
	perfect_point = np.array([0, 1])
	for i in range(len(thresholds)):
		roc_curve_point = np.array([fpr[i], tpr[i]])
		distance = np.linalg.norm(roc_curve_point - perfect_point)
		if min_distance > distance:
			min_distance = distance
			result = thresholds[i]

	return result


def threshold_calibration(label_file, score_file):
	"""Compute eath label's optimal threshold score by ROC curve.

	Args:
		label_file: contains ground truth lable info. Each line is like this format:
					"1 0 1 0"
		score_file: contains scores computed by the trained model. Each line is like this format:
					"0.12 0.45 0.98 0.45"

	Returns:
		One list containing each label's optimal threshold scores.
	"""
	ys = np.loadtxt(label_file, dtype=int)
	scores = np.loadtxt(score_file)

	assert ys.shape == scores.shape, "label and score have different shape"

	optimal_thresholds = []
	for i in range(ys.shape[1]):
		print("compute labe {}".format(i + 1))
		threshold = compute_optimal_threshold(ys[:,i], scores[:,i])
		optimal_thresholds.append(threshold)

	print("Result is:")
	for item in optimal_thresholds:
		print(item)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='classification threshold calibration')
	parser.add_argument('--label_file', required=True, help='ground truth label file')
	parser.add_argument('--score_file', required=True, help='score file computed by model')
	args = parser.parse_args()
	
	threshold_calibration(args.label_file, args.score_file)
