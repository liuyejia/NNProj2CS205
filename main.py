import argparse
import numpy as np
from sklearn import preprocessing
from collections import Counter
import timeit



# def euclidean_distance(sample1, sample2, features):
#     return ((sample1[features] - sample2[features]) ** 2).sum() ** 0.5


# def KNN(train_data, v_data, features, k):
#     distance_list = []
#     for i, data in enumerate(train_data):
#         distance = euclidean_distance(data[1:], v_data[1:], features)
#         distance_list.append((i, distance))
#     distance_list.sort(key=lambda item: item[1])
#     neighbors = [train_data[i][0] for i, _ in distance_list[:k]]
#     return neighbors

def KNN(train_data, valid_data, features, k):
    '''
    Matrix-based KNN for acceleration: reshape train data into (1, train_size, all_feature_size)
    and valid data into (valid_size, 1, all_feature_size), compute distance (valid_size, train_size)

    :param train_data: shape is (train_size, all_feature_size)
    :param valid_data: shape is (valid_size, all_feature_size)
    :param features: List of features
    :param k: Select k neighbors
    :return: Selected neighbors' indices for each valid data, shape is (valid_size, all_feature_size)
    '''
    train_data = train_data[:, features].reshape(1, train_data.shape[0], len(features))
    valid_data = valid_data[:, features].reshape(valid_data.shape[0], 1, len(features))
    distance = ((train_data - valid_data) ** 2).sum(axis=-1) ** 0.5
    neighbors = distance.argsort()[:, :k]
    return neighbors


def cross_one_out_validation(all_data, features, k, folds):
    assert len(features) >= 1
    acc_list = []
    for fold in range(folds):
        valid_start_idx = int(fold * len(all_data) / folds)
        valid_end_idx = int((fold + 1) * len(all_data) / folds)
        train_data = np.concatenate((all_data[:valid_start_idx], all_data[valid_end_idx:]), axis=0)
        valid_data = all_data[valid_start_idx:valid_end_idx]
        # accs = 0
        neighbors = KNN(train_data[:, 1:], valid_data[:, 1:], features, k)
        preds = [Counter([train_data[i][0] for i in n]).most_common(1)[0][0] for n in neighbors]
        labels = valid_data[:, 0]
        accs = (preds == labels).sum()
        # for v_data in valid_data:
        #     neighbors = KNN(train_data, v_data, features, k)
        #     pred = Counter(neighbors).most_common(1)[0][0]
        #     label = v_data[0]
        #     accs += (pred == label)
        acc_list.append(accs / len(valid_data))
    return np.array(acc_list).mean()


def forward_search(all_data, k, folds):
    assert len(all_data.shape) == 2
    assert len(all_data[0]) > 1

    feature_counts = len(all_data[0]) - 1
    features_list = [[i] for i in range(feature_counts)]
    all_features_set = set(range(feature_counts))

    global_best_acc, global_best_features = -np.inf, []
    while True:
        best_acc, best_index = -np.inf, 0

        # Cross valid of features
        for i, features in enumerate(features_list):
            acc = cross_one_out_validation(all_data, features, k, folds)
            if acc > best_acc:
                best_acc = acc
                best_index = i
            # print("Features:{}, Acc:{}".format(features, acc))

        # Find best feature
        best_features = features_list[best_index]
        print("Best features:{}, Acc:{}".format(best_features, best_acc))

        # Update global best
        if best_acc > global_best_acc:
            global_best_acc = best_acc
            global_best_features = best_features

        # Generate new feature list
        if len(features_list[0]) >= feature_counts:
            break
        else:
            features_list = [best_features + [i] for i in all_features_set - set(best_features)]

    print("Global best features:{}, Acc:{}".format(global_best_features, global_best_acc))



def backward_elimination(all_data, k, folds):
    assert len(all_data.shape) == 2
    assert len(all_data[0]) > 1

    feature_counts = len(all_data[0]) - 1
    features_list = [list(range(feature_counts))]

    global_best_acc, global_best_features = -np.inf, []
    while True:
        best_acc, best_index = -np.inf, 0

        # Cross valid of features
        for i, features in enumerate(features_list):
            acc = cross_one_out_validation(all_data, features, k, folds)
            if acc > best_acc:
                best_acc = acc
                best_index = i
            # print("Features:{}, Acc:{}".format(features, acc))

        # Find best feature
        best_features = features_list[best_index]
        print("Best features:{}, Acc: {}".format(best_features, best_acc))

        # Update global best
        if best_acc > global_best_acc:
            global_best_acc = best_acc
            global_best_features = best_features

        # Generate new feature list
        if len(features_list[0]) <= 1:
            break
        else:
            features_list = [list(set(best_features) - set([i])) for i in set(best_features)]


    print("Global best features:{}, Acc:{}".format(global_best_features, global_best_acc))



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='forward')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--folds', type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()

    # load data
    data = np.genfromtxt('datasets/CS205_small_testdata__9.txt')

    # preprocess data
    label = data[:, 0:1]
    data = preprocessing.normalize(data[:, 1:], norm='l2')  # check
    all_data = np.concatenate((label, data), axis=1)

    start = timeit.default_timer()
    # print(all_data[0])
    if args.method == 'forward':
        str = 'Forward Selection'
        print("Your selected search algorithm is: ", str)
        forward_search(all_data, args.k, args.folds)
    elif args.method == 'backward':
        str = 'Backward Elimination'
        print("Your selected search algorithm is: ", str)
        backward_elimination(all_data, args.k, args.folds)
    else:
        raise ValueError("Not implemented method!")
    stop = timeit.default_timer()
    print("total running time: ", stop - start)
