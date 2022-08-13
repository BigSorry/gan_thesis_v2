import pickle
import numpy as np
import pathlib
import datetime
import torch
import matplotlib.pyplot as plt
import scipy.spatial.distance
import torchvision.transforms as transforms
from PIL import Image
from models.inceptionv3 import *
from torchvision import datasets
import torchvision.models as models


def checkDistances():
    num_real_samples = num_fake_samples = [1000]
    feature_dim = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    mean = 0
    for sample_count in num_real_samples:
        for dim in feature_dim:
            mean_vec = np.array([mean for i in range(dim)])
            cov_matrix = np.eye(dim)
            real_features = np.random.multivariate_normal(mean_vec, cov_matrix, size=sample_count)
            stop = int(np.ceil(dim / 2))
            stop = 3
            cov_matrix2 = np.eye(dim) * np.array([0.1 if i >= stop else 1 for i in range(dim)])
            print(cov_matrix2)
            fake_features = np.random.multivariate_normal(mean_vec, cov_matrix2, size=sample_count)

            distances = scipy.spatial.distance.cdist(real_features, fake_features, 'euclidean')
            distances = distances.flatten()
            plt.figure()
            plt.title(dim)
            plt.hist(distances, bins=50, density=True)
            plt.xlabel("Distance")
            plt.ylabel("Density")

def getLambdas(angle_count, epsilon= 1e-10):
    epsilon = 1e-10
    angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=angle_count)
    lambdas = np.tan(angles)

    return lambdas
# Rows are real samples and columns are generated samples
def createTrainTest(real_features, fake_features):
    all_combined = np.concatenate((real_features, fake_features))
    all_labels = np.concatenate((np.ones(real_features.shape[0]),
                                 np.zeros(fake_features.shape[0])))
    label_vec = np.random.randint(2, size=all_combined.shape[0])
    mixture = all_combined[label_vec == 1, :]
    mixture_labels = all_labels[label_vec == 1]
    test_data = all_combined[label_vec == 0, :]
    test_labels = all_labels[label_vec == 0]


    return mixture, mixture_labels, test_data, test_labels

def savePickle(path, python_object):
    with open(path, 'wb') as fp:
        pickle.dump(python_object, fp)

def readPickle(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
        return data

def createMap(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def plotResults(path_name="./subspaceExperimentOne_1000.pkl"):
    read_dict = readPickle(path_name)
    for method, result_dict in read_dict.items():
        plotting.plotPRCurve(result_dict, f"Method name {method}")

def prepData():
    curve_method_names = ["kmeans_on_mixture", "likelihood_classifier", "classifier_on_mixture"]
    curve_method_names = ["kmeans_on_mixture", "likelihood_classifier"]
    # Lambdas for pr-curves
    point_count = 1001
    epsilon = 1e-10
    angles = np.linspace(epsilon, (np.pi / 2) - epsilon, point_count)
    lambdas = np.tan(angles)

    return curve_method_names, lambdas

def getTransform(name, imsize=299):
    transform = None
    mean = None
    std = None
    if name == "mnist":
        mean = (0.1307,)
        std = (0.3081,)
    elif name == "cifar":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    if mean and std is not None:
        transform = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(imsize, interpolation=Image.BICUBIC),
            transforms.CenterCrop(imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    return transform

def getDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getActivations(dataloader, feature_extractor, feature_output=2048):
    samples = dataloader.dataset.data.shape[0]
    batch_size = dataloader.batch_size
    activations = np.zeros((samples, feature_output))
    device = getDevice()
    feature_extractor = feature_extractor.to(device)
    activation_dict = {i:[] for i in range(10)}
    for i, data in enumerate(dataloader, 0):
        with torch.no_grad():
            input_data = data[0].to(device)
            batch_output = feature_extractor(input_data, conv_space=True).cpu().detach().numpy()
            for index, sample in enumerate(batch_output):
                label = data[1][index].item()
                activation_dict[label].append(sample)

    return activation_dict

def saveActivations(path="./dataset"):
    transform = getTransform(name="mnist")
    batch_size = 32
    pretrained = models.inception_v3(pretrained=True)
    inception = InceptionV3(init_weights=True)
    inception.load_state_dict(pretrained.state_dict())

    trainset = datasets.MNIST(root=path, train=True,
                              download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    testset = datasets.MNIST(root=path, train=False,
                             download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    train_activations = getActivations(trainloader, inception)
    train_path = "./activations/activation_train_mnist.pkl"
    savePickle(train_path, train_activations)

def checkDistanceActivation():
    train_path = "./activations/activation_train_mnist.pkl"
    data = readPickle(train_path)
    x = []
    for key, value in data.items():
        x.extend(value[:1000])

    distances = scipy.spatial.distance.cdist(x, x, 'euclidean')
    distances = distances.flatten()

    plt.figure()
    plt.hist(distances, bins=50, density=True)
    plt.xlabel("Distance")
    plt.ylabel("Density")

