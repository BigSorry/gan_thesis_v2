import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
import visualize as plotting
from sklearn.metrics import pairwise_distances

def savePickle(path, python_object):
    with open(path, 'wb') as fp:
        pickle.dump(python_object, fp)

def readPickle(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
        return data

def _circleCheck(circles, new_point, radius):
    check = True
    for circle in circles:
        point_inside_circle = (circle[0] - new_point[0])**2 + (circle[1] - new_point[1])**2 < (radius)**2
        if point_inside_circle:
            return False

    return check

def generateCircles(samples, radius=1, dimension=2):
    start_point = np.zeros(dimension)
    dataset = [start_point]
    for sample_index in range(1, samples):
        legit_sample = False
        count = 0
        while legit_sample is False:
            angle = np.pi * np.random.uniform(0, 2)
            old_sample = dataset[sample_index - 1]
            x = old_sample[0] + (radius * np.cos(angle))
            y = old_sample[1] + (radius * np.sin(angle))
            new_sample = np.array([x, y])
            legit_sample = _circleCheck(dataset, new_sample, radius)
            count += 1
            if count == 1000:
                print("Error count reached 1000")
                print(len(dataset))
                return np.array(dataset)

        dataset.append(new_sample)

    return np.array(dataset)

def getDistanceMatrices(real_features, fake_features):
    distance_matrix_real = pairwise_distances(real_features, real_features, metric='euclidean')
    distance_matrix_fake = pairwise_distances(fake_features, fake_features, metric='euclidean')
    distance_matrix_pairs = pairwise_distances(real_features, fake_features, metric='euclidean')
    distance_matrix_real = np.sort(distance_matrix_real, axis=1)
    distance_matrix_fake = np.sort(distance_matrix_fake, axis=1)

    return distance_matrix_real, distance_matrix_fake, distance_matrix_pairs

def getBoundaries(real_features, fake_features, k):
    distance_matrix_real = pairwise_distances(real_features, real_features, metric='euclidean')
    distance_matrix_fake = pairwise_distances(fake_features, fake_features, metric='euclidean')
    distance_matrix_pairs = pairwise_distances(real_features, fake_features, metric='euclidean')
    distance_matrix_real = np.sort(distance_matrix_real, axis=1)
    distance_matrix_fake = np.sort(distance_matrix_fake, axis=1)
    boundaries_real = distance_matrix_real[:, k]
    # Temp adjust for experiment case |F|=2
    if k >= distance_matrix_fake.shape[1]:
        k = distance_matrix_fake.shape[1]-1
    boundaries_fake = distance_matrix_fake[:, k]

    return boundaries_real, boundaries_fake, distance_matrix_pairs

def getScoreMask(boundaries_real, boundaries_fake, distance_matrix_pairs):
    recall_mask = (distance_matrix_pairs < np.expand_dims(boundaries_fake, axis=0)).any(axis=1)
    coverage_mask = (distance_matrix_pairs.min(axis=1) < boundaries_real)

    return recall_mask, coverage_mask

def getScores(distance_matrix_pairs, boundaries_fake, boundaries_real, k_val):
    # Get scores and boundaries
    # TODO Refator
    precision = (distance_matrix_pairs < np.expand_dims(boundaries_real, axis=1)).any(axis=0).mean()
    recall = (distance_matrix_pairs < np.expand_dims(boundaries_fake, axis=0)).any(axis=1).mean()
    # density coverage
    density = (1. / float(k_val)) * (distance_matrix_pairs < np.expand_dims(boundaries_real, axis=1)).sum(
        axis=0).mean()
    coverage = (distance_matrix_pairs.min(axis=1) < boundaries_real).mean()

    return precision, recall, density, coverage

def getLimits(saved_real_sets, saved_fake_sets, radius=10):
    real_np = np.concatenate(np.array(saved_real_sets), axis=0)
    fake_np = np.concatenate(np.array(saved_fake_sets), axis=0)
    all_data = np.concatenate([real_np, fake_np])
    min_x = np.min(all_data[:, 0]) - radius
    max_x = np.max(all_data[:, 0]) + radius
    min_y = np.min(all_data[:, 1]) - radius
    max_y = np.max(all_data[:, 1]) + radius

    return min_x, max_x, min_y, max_y

def checkMinMaxDistance(real_features, fake_features, k=1):
    distance_matrix_real = pairwise_distances(real_features, real_features, metric='euclidean')
    distance_matrix_fake = pairwise_distances(fake_features, fake_features, metric='euclidean')
    distance_matrix_real = np.sort(distance_matrix_real, axis=1)
    distance_matrix_fake = np.sort(distance_matrix_fake, axis=1)
    min_distance_real = distance_matrix_real[:, k]
    max_distances_real = np.max(distance_matrix_real, axis=1)
    min_distance_fake = distance_matrix_fake[:, k]
    max_distances_fake = np.max(distance_matrix_fake, axis=1)

    return min_distance_real, max_distances_real, min_distance_fake, max_distances_fake

def getArea(boundaries):
    return math.pi*(boundaries**2)

def getVolume(boundaries, dimension):
    nominator = (np.pi**(dimension/2))
    input_gamma = int((dimension / 2) + 1)
    denominator = np.math.factorial(input_gamma - 1)
    volume = (nominator/denominator)*(boundaries**dimension)
    return volume

# From https://stackoverflow.com/questions/4247889/area-of-intersection-between-two-circles
def getIntersection(x0, y0, r0, x1, y1, r1):
    rr0 = r0 * r0
    rr1 = r1 * r1
    d = math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))

    # Circles do not overlap
    if d > r1 + r0:
        return 0
    # Circle1 is completely inside circle0
    elif d <= np.abs(r0 - r1) and r0 >= r1:
        return np.pi * rr1
    # Circle0 is completely inside circle1
    elif d <= np.abs(r0 - r1) and r0 < r1:
        return np.pi * rr0
    # Circles partially overlap
    else:
        phi = (np.arccos((rr0 + (d * d) - rr1) / (2 * r0 * d))) * 2
        theta = (np.arccos((rr1 + (d * d) - rr0) / (2 * r1 * d))) * 2
        area1 = 0.5 * theta * rr1 - 0.5 * rr1 * np.sin(theta)
        area2 = 0.5 * phi * rr0 - 0.5 * rr0 * np.sin(phi)
        return area1 + area2

def testVolumes():
    samples = 100
    dimension = 2
    k_val = 2
    mean = np.zeros(dimension)
    cov = np.eye(dimension)
    real_features = np.random.multivariate_normal(mean, cov, size=samples)
    cov *= 0.1
    fake_features = np.random.multivariate_normal(mean, cov, size=samples)
    boundaries_real, boundaries_fake, distance_matrix_pairs = getBoundaries(real_features, fake_features, k_val)
    for i in range(samples):
        real_sample = real_features[i, :]
        x0 = real_sample[0]
        y0 = real_sample[1]
        r0 = boundaries_real[i]
        for j in range(samples):
            if i != j:
                other_sample = real_features[j, :]
                x1 = other_sample[0]
                y1 = other_sample[1]
                r1 = boundaries_real[j]
                intersection = getIntersection(x0, y0, r0, x1, y1, r1)
                k=0

testVolumes()