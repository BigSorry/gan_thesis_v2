import numpy as np
import sklearn.cluster

# Code reference
# https://github.com/msmsajjadi/precision-recall-distributions/blob/master/prd_score.py
def compute_prd(eval_dist, ref_dist, num_angles=1001, epsilon=1e-10):
  if not (epsilon > 0 and epsilon < 0.1):
    raise ValueError('epsilon must be in (0, 0.1] but is %s.' % str(epsilon))
  if not (num_angles >= 3 and num_angles <= 1e6):
    raise ValueError('num_angles must be in [3, 1e6] but is %d.' % num_angles)

  # Compute slopes for linearly spaced angles between [0, pi/2]
  angles = np.linspace(epsilon, np.pi/2 - epsilon, num=num_angles)
  slopes = np.tan(angles)

  # Broadcast slopes so that second dimension will be states of the distribution
  slopes_2d = np.expand_dims(slopes, 1)

  # Broadcast distributions so that first dimension represents the angles
  ref_dist_2d = np.expand_dims(ref_dist, 0)
  eval_dist_2d = np.expand_dims(eval_dist, 0)

  # Compute precision and recall for all angles in one step via broadcasting
  precision = np.minimum(ref_dist_2d*slopes_2d, eval_dist_2d).sum(axis=1)
  recall = precision / slopes

  # handle numerical instabilities leaing to precision/recall just above 1
  max_val = max(np.max(precision), np.max(recall))
  if max_val > 1.001:
    raise ValueError('Detected value > 1.001, this should not happen.')
  precision = np.clip(precision, 0, 1)
  recall = np.clip(recall, 0, 1)

  return precision, recall

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors, datasets
def _cluster_into_bins(eval_data, ref_data, num_clusters):
  cluster_data = np.vstack([eval_data, ref_data])
  kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=num_clusters, n_init=10)
  labels = kmeans.fit(cluster_data).labels_

  eval_labels = labels[:len(eval_data)]
  ref_labels = labels[len(eval_data):]

  eval_bins = np.histogram(eval_labels, bins=num_clusters,
                           range=[0, num_clusters], density=True)[0]
  ref_bins = np.histogram(ref_labels, bins=num_clusters,
                          range=[0, num_clusters], density=True)[0]
  return eval_bins, ref_bins, labels

def compute_prd_from_embedding(eval_data, ref_data, num_clusters=20,
                               num_angles=1001, num_runs=10,
                               enforce_balance=True):

  # if enforce_balance and len(eval_data) != len(ref_data):
  #   raise ValueError(
  #       'The number of points in eval_data %d is not equal to the number of '
  #       'points in ref_data %d. To disable this exception, set enforce_balance '
  #       'to False (not recommended).' % (len(eval_data), len(ref_data)))

  eval_data = np.array(eval_data, dtype=np.float64)
  ref_data = np.array(ref_data, dtype=np.float64)
  precisions = []
  recalls = []
  for _ in range(num_runs):
    eval_dist, ref_dist, labels = _cluster_into_bins(eval_data, ref_data, num_clusters)
    precision, recall = compute_prd(eval_dist, ref_dist, num_angles)
    precisions.append(precision)
    recalls.append(recall)

  precision = np.mean(precisions, axis=0)
  recall = np.mean(recalls, axis=0)

  return precision, recall, labels


def _prd_to_f_beta(precision, recall, beta=1, epsilon=1e-10):
  """Computes F_beta scores for the given precision/recall values.
  The F_beta scores for all precision/recall pairs will be computed and
  returned.
  For precision p and recall r, the F_beta score is defined as:
  F_beta = (1 + beta^2) * (p * r) / ((beta^2 * p) + r)
  Args:
    precision: 1D NumPy array of precision values in [0, 1].
    recall: 1D NumPy array of precision values in [0, 1].
    beta: Beta parameter. Must be positive. The default value is 1.
    epsilon: Small constant to avoid numerical instability caused by division
             by 0 when precision and recall are close to zero.
  Returns:
    NumPy array of same shape as precision and recall with the F_beta scores for
    each pair of precision/recall.
  Raises:
    ValueError: If any value in precision or recall is outside of [0, 1].
    ValueError: If beta is not positive.
  """

  if not ((precision >= 0).all() and (precision <= 1).all()):
    raise ValueError('All values in precision must be in [0, 1].')
  if not ((recall >= 0).all() and (recall <= 1).all()):
    raise ValueError('All values in recall must be in [0, 1].')
  if beta <= 0:
    raise ValueError('Given parameter beta %s must be positive.' % str(beta))

  f_scores = (1 + beta**2) * (precision * recall) / (
      (beta**2 * precision) + recall + epsilon)

  return f_scores


def prd_to_max_f_beta_pair(precision, recall, beta=8):
  """Computes max. F_beta and max. F_{1/beta} for precision/recall pairs.
  Computes the maximum F_beta and maximum F_{1/beta} score over all pairs of
  precision/recall values. This is useful to compress a PRD plot into a single
  pair of values which correlate with precision and recall.
  For precision p and recall r, the F_beta score is defined as:
  F_beta = (1 + beta^2) * (p * r) / ((beta^2 * p) + r)
  Args:
    precision: 1D NumPy array or list of precision values in [0, 1].
    recall: 1D NumPy array or list of precision values in [0, 1].
    beta: Beta parameter. Must be positive. The default value is 8.
  Returns:
    f_beta: Maximum F_beta score.
    f_beta_inv: Maximum F_{1/beta} score.
  Raises:
    ValueError: If beta is not positive.
  """

  if not ((precision >= 0).all() and (precision <= 1).all()):
    raise ValueError('All values in precision must be in [0, 1].')
  if not ((recall >= 0).all() and (recall <= 1).all()):
    raise ValueError('All values in recall must be in [0, 1].')
  if beta <= 0:
    raise ValueError('Given parameter beta %s must be positive.' % str(beta))

  f_beta = np.max(_prd_to_f_beta(precision, recall, beta))
  f_beta_inv = np.max(_prd_to_f_beta(precision, recall, 1/beta))

  return f_beta, f_beta_inv