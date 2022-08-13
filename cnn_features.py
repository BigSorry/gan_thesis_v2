import torch
import torchvision
import numpy as np
import models.cnn as cnn
import torch.optim as optim
import training as trn
import experiment_utils as util
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import cluster, datasets, mixture
from sklearn.decomposition import PCA

def testCNN(testloader, cnn, device):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = cnn(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

def getFeatures(dataloader, cnn, device):
    all_features = np.zeros((dataloader.dataset.data.shape[0], 320))
    batch_size = dataloader.batch_size
    with torch.no_grad():
        for start_index, data in enumerate(dataloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            samples = images.shape[0]
            # calculate outputs by running images through the network
            features = cnn.getFeature(images).cpu().detach().numpy()
            begin = start_index * batch_size
            end = begin + samples
            all_features[begin: end, :] = features

    return all_features

def getFeatures2(dataloader, cnn, device):
    all_features = []
    with torch.no_grad():
        for start_index, data in enumerate(dataloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            samples = images.shape[0]
            # calculate outputs by running images through the network
            features = cnn.getFeature(images).cpu().detach().numpy()
            all_features.extend(features)

    return np.array(all_features)

batch_size_train = 128
batch_size_test = 1024
# define how image transformed
image_transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
#image datasets
train_dataset = torchvision.datasets.MNIST('./dataset',
                                           train=True,
                                           download=True,
                                           transform=image_transform)
test_dataset = torchvision.datasets.MNIST('./dataset',
                                          train=False,
                                          download=True,
                                          transform=image_transform)
#data loaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size_train,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size_test,
                                          shuffle=True)

## create model and optimizer
learning_rate = 0.01
momentum = 0.9
save = False
save_path = "./saved_pkl/cnn/cnn_state_dict.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = cnn.CNN().to(device)
optimizer = optim.SGD(feature_extractor.parameters(), lr=learning_rate,
                      momentum=momentum)
if save:
    cnn_states = trn.trainCNN(train_loader, feature_extractor, optimizer,
                              device, epochs=10)
    util.savePickle(save_path, cnn_states)
else:
    cnn_states = util.readPickle(save_path)

test_data = test_loader.dataset.data.numpy()
test_labels = test_loader.dataset.targets.numpy()
train_labels = train_loader.dataset.targets.numpy()

clf = AdaBoostClassifier(n_estimators=256)
for epoch, state_dict in cnn_states.items():
    print(f"CNN trained with epochs {epoch+1}")
    model = cnn.CNN().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    train_features = getFeatures2(train_loader, model, device)
    test_features = getFeatures2(test_loader, model, device)

    clf.fit(train_features, train_labels)
    predicted_labels = clf.predict(test_features)
    errors = np.mean(predicted_labels != test_labels)
    print(errors)
    break

