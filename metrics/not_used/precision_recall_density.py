import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from sklearn.metrics import pairwise_distances

class NNProjector(nn.Module):
    def __init__(self, radius_param):
        super(NNProjector, self).__init__()
        self.radius = radius_param
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x

def getLoss(targets, radius_param, center_param, weight_param):
    distance_function = torch.cdist(targets, center_param, p=2) - radius_param**2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_val = (torch.tensor(0).float()).to(device)
    batch_losses = radius_param**2 + (1/weight_param) * torch.max(distance_function, min_val)
    total_loss = torch.sum(batch_losses)

    return total_loss


def testMethod():
    # Prep data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_vec_real = np.array([0, 0])
    mean_vec_fake = np.array([2, 0])
    real_covariance = np.eye(2)
    samples = 1000
    real_features = np.random.multivariate_normal(mean_vec_real, real_covariance, size=samples)
    fake_features = np.random.multivariate_normal(mean_vec_fake, real_covariance, size=samples)
    real_features_torch = (torch.from_numpy(real_features).float()).to(device)
    fake_features_torch = (torch.from_numpy(fake_features).float()).to(device)
    radius_param = torch.rand(1).to(device)
    center_param = (torch.tensor(np.ones((1, 10))).float()).to(device)
    epochs = 100
    weight_param = 10
    cnn = NNProjector(radius_param).to(device)
    optimizer = optim.Adam(cnn.parameters(),
                            lr=0.001, betas=(0.5, 0.999))
    optimizer.param_groups.append({'params': radius_param})
    for epoch in range(epochs):  # loop over the dataset multiple times
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = cnn(real_features_torch)
        loss = getLoss(output, radius_param, center_param, weight_param)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(loss.item(), radius_param.item())

    real_output = cnn(real_features_torch).cpu().detach().numpy()
    fake_output = cnn(fake_features_torch).cpu().detach().numpy()
    radius = radius_param.item()
    center_param = center_param.cpu().detach().numpy()
    precision = getPrecision(fake_output, center_param, radius)
    recall = getRecall(real_output, fake_output)

    precision_end = precision.mean()
    recall_end = recall.mean()

    print(precision_end)
    print(recall_end)

def getPrecision(fake_features, center_param, trained_radius):
    distances = pairwise_distances(fake_features, center_param, metric="l2")
    classified = np.int32(distances < trained_radius)

    return classified


def getRecall(real_features, fake_features, k=7):
    distance_matrix_real = pairwise_distances(real_features, real_features, metric='euclidean')
    distance_matrix_real = np.sort(distance_matrix_real, axis=1)
    boundaries_real = distance_matrix_real[:, k]
    distance_matrix_pairs = pairwise_distances(real_features, fake_features, metric='euclidean')

    fake_real_min_distance = distance_matrix_pairs.min(axis=1)
    recall = np.int32(fake_real_min_distance < boundaries_real)

    return recall

testMethod()