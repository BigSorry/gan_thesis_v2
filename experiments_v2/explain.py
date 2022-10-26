import pandas as pd
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import experiments_v2.helper_functions as util
import seaborn as sns

def doEval(real_features, fake_features, k_params, circle_iters, percentage_off):
    columns = ["iter", "k_val", "recall", "precision"]
    row_data = []
    samples = real_features.shape[0]
    distance_matrix_real, distance_matrix_fake, distance_matrix_pairs = util.getDistanceMatrices(real_features, fake_features)
    for k in k_params:
        boundaries_real = distance_matrix_real[:, k]
        boundaries_fake = distance_matrix_fake[:, k]
        off_samples = int(samples * percentage_off)
        for i in range(circle_iters):
            off_indices = np.random.choice(samples, off_samples, replace=False)
            boundaries_real_used = boundaries_real.copy()
            boundaries_real_used[off_indices] = 0
            boundaries_fake_used = boundaries_fake.copy()
            boundaries_fake_used[off_indices] = 0
            # Turn off fake samples for Coverage

            special_coverage = util.getCoverageSpecial(distance_matrix_pairs, boundaries_real, off_indices)
            # Turn off circles works only for Precision/Recall
            precision, recall, density, coverage = util.getScores(distance_matrix_pairs, boundaries_fake_used,
                                                                   boundaries_real_used, k)
            row = [i, k, recall, precision]
            row_data.append(row)

    dataframe = pd.DataFrame(columns=columns, data=row_data)
    return dataframe

def plotFetures(data_dict):
    plt.figure()
    plt.title("MNIST PCA 2D")
    for class_nr, feature_data in data_dict.items():
        plt.scatter(feature_data[:, 0], feature_data[:, 1], label=class_nr)
    plt.legend()

# define how image transformed
image_transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
#image datasets
train_dataset = torchvision.datasets.MNIST('./dataset',
                                           train=True,
                                           download=False,
                                           transform=image_transform)
train_set = train_dataset.data.numpy()
train_labels = train_dataset.targets.numpy()

iters = 10
class_samples=100
classes = [0, 1,2,3, 4,5,6,7,8,9]
classes = [2,3]
for i in range(iters):
    data_dict = {}
    sel_samples = np.zeros((class_samples*len(classes), 28*28))
    sel_class = []
    for index, class_nr in enumerate(classes):
        selection = train_labels == class_nr
        class_data = train_set[selection]
        # 1X28x28
        sel_data = class_data[:class_samples].reshape(-1, 28*28)
        sel_samples[index*class_samples:index*class_samples+class_samples, :] = sel_data
        data_dict[class_nr] = sel_data

    pca = PCA(n_components=2)
    features = pca.fit_transform(sel_samples)
    for class_nr in classes:
        data_dict[class_nr] = pca.transform(data_dict[class_nr])

    plotFetures(data_dict)

class_two = data_dict[2]
class_three = data_dict[3]
k_vals = util.getParams(class_samples)
print(k_vals)
score_dataframe = doEval(class_three, class_two, k_vals, 1, 0)
plt.figure()
sns.boxplot(x="k_val", y="recall", data=score_dataframe).set(
    xlabel='K value',
    ylabel='Recall'
)
plt.figure()
sns.boxplot(x="k_val", y="precision", data=score_dataframe).set(
    xlabel='K value',
    ylabel='Precision'
)

plt.show()