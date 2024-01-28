import numpy as np
import torch


# Rows are real samples and columns are generated samples
def createTrainTest(real_features, fake_features):
    mixture_data = np.concatenate([real_features, fake_features])
    mixture_labels = np.concatenate([np.ones(real_features.shape[0]), np.zeros(fake_features.shape[0])])
    real_indices = np.random.choice(real_features.shape[0], real_features.shape[0] // 2)
    fake_indices = np.random.choice(fake_features.shape[0], fake_features.shape[0] // 2) + real_features.shape[0]
    all_indices = np.concatenate([real_indices, fake_indices])
    train_data = mixture_data[all_indices, :]
    test_data = mixture_data[~all_indices, :]
    train_labels = mixture_labels[all_indices]
    test_labels = mixture_labels[~all_indices]

    return train_data, test_data, train_labels, test_labels

def getScores(truth, predictions):
    # A positive result corresponds to rejecting the null hypothesis (sample is fake)
    # Null -> sample is from real data
    correct_mask = truth == predictions
    tp = truth[(correct_mask == True) & (truth == 0)].shape[0]
    fp = truth[(correct_mask == False) & (predictions == 0)].shape[0]

    tn = truth[(correct_mask == True) & (truth == 1)].shape[0]
    fn = truth[(correct_mask == False) & (predictions == 1)].shape[0]

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    return fpr, fnr

def getPrecisionRecallCurve(train, train_labels, test, test_labels,
                            lambdas, threshold_count, classifier):
    classifier.fit(train, train_labels)
    # Estimates for the second class (class 1)
    probability_predictions = classifier.predict_proba(test)[:, 1]
    #predicted = clf.predict(test)
    fVals = probability_predictions
    min_t = np.min(fVals)
    max_t = np.max(fVals)
    thresholds = np.linspace(min_t, max_t, num=threshold_count)
    fValsAndUs = np.array([fVals, test_labels]).T
    errorRates = []
    for t in thresholds:
        fpr = np.sum([(fOfZ >= t) and U == 0 for fOfZ, U in fValsAndUs]) / float(np.sum([U == 0 for fOfZ, U in fValsAndUs]))
        fnr = np.sum([(fOfZ < t) and U == 1 for fOfZ, U in fValsAndUs]) / float(np.sum([U == 1 for fOfZ, U in fValsAndUs]))
        errorRates.append((float(fpr), float(fnr)))

    precision = []
    recall = []
    for slope in lambdas:
        prec = np.min([(slope * fnr) + fpr for fpr, fnr in errorRates])
        precision.append(prec)
        rec = np.min(prec / slope)
        recall.append(rec)

    curve = np.zeros((lambdas.shape[0], 2))
    curve[:, 0] = precision
    curve[:, 1] = recall

    return np.clip(curve, 0, 1), thresholds, probability_predictions


def getPRCurveDiscriminator(train, train_labels, test, test_labels,
                            lambdas, discriminator, device):
    # Estimates for the second class (class 1)
    torch__predictions = discriminator(torch.from_numpy(test).to(device).float())
    numpy_predictions = torch__predictions.cpu().detach().numpy().flatten()
    fVals = numpy_predictions
    min_t = np.min(fVals)
    max_t = np.max(fVals)
    thresholds = np.linspace(min_t, max_t, num=1000)
    fValsAndUs = np.array([fVals, test_labels]).T
    errorRates = []
    for t in thresholds:
        fpr = np.sum([(fOfZ >= t) and U == 0 for fOfZ, U in fValsAndUs]) / float(np.sum([U == 0 for fOfZ, U in fValsAndUs]))
        fnr = np.sum([(fOfZ < t) and U == 1 for fOfZ, U in fValsAndUs]) / float(np.sum([U == 1 for fOfZ, U in fValsAndUs]))
        errorRates.append((float(fpr), float(fnr)))

    precision = []
    recall = []
    for slope in lambdas:
        prec = np.min([(slope * fnr) + fpr for fpr, fnr in errorRates])
        precision.append(prec)
        rec = np.min(prec / slope)
        recall.append(rec)

    curve = np.zeros((lambdas.shape[0], 2))
    curve[:, 0] = precision
    curve[:, 1] = recall



    return np.clip(curve, 0, 1)
