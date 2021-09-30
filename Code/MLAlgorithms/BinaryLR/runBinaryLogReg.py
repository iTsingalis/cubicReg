import os
import dsdl
import argparse
import numpy as np
from mnist import MNIST
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from Code.MLAlgorithms.BinaryLR.BinaryLogReg import BinaryLogReg
from sklearn.datasets import load_svmlight_file

markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D',
           'd', '|', '_']


def relabel(y):
    for idx, item in enumerate(y):
        if item % 2 == 0:
            y[idx] = 0  # even number
        else:
            y[idx] = 1  # odd number
    return y


def mean_list(input_lists):
    max_len = np.max([len(input_list) for input_list in input_lists])
    resized_lists = [input_list + [np.nan] * (max_len - len(input_list)) for input_list in input_lists]
    _mean_list = list(map(np.nanmean, zip(*resized_lists)))

    return _mean_list


def main():
    _OPT_METHODS_ = ['L-BFGS-B', 'Nesterov']

    method_idx = 1

    nesterov_kwargs = {'L0': None, 'L': 200., 'M': None, 'kappa_easy': 0.001,
                       'max_iter': 1000, 'max_sub_iter': 100,
                       'tol': 1e-3, 'stop_criteria': 'nesterov'}

    params = {'alpha': 1e-1, 'atol': 1e-6, 'reltol': 1e-6,
              'maxit': 2000, 'opt_method': _OPT_METHODS_[method_idx], 'verbose': True,
              'kwargs': nesterov_kwargs}

    n_runs = 2  # For mean accuracy
    _method_precision = []
    _method_timings = []
    _acc = []
    for _ in range(n_runs):

        estimator = BinaryLogReg(**params).fit(X_tr, labels_tr)

        pred_labels_tst = estimator.predict(X_tst)
        acc = np.mean(pred_labels_tst == labels_tst)
        _acc.append(acc)

        method_precision = estimator.method_precision
        method_timings = estimator.timings

        _method_timings.append(method_timings)
        _method_precision.append(method_precision)

    print('Method {} Mean Acc. {}'.format(params['opt_method'], np.mean(_acc)))

    mean_method_timings = mean_list(_method_timings)
    mean_method_precision = mean_list(_method_precision)

    plt.plot(mean_method_timings, mean_method_precision, label='L-BFGS', linewidth=1, color='black',
             marker=markers[method_idx], markersize=10)

    plt.yscale('log')
    plt.ylabel(r'$\|\|\nabla f\|\|_2$', fontsize=10)
    plt.xlabel('Time (in seconds)', fontsize=10)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the target language.")

    FLAGS = parser.parse_args()
    X_tst = None
    if 'a9a' in FLAGS.task_name:
        tr_path = os.path.join("../../../Datasets/libsvm/{}".format(FLAGS.task_name), 'a9a.txt')
        X_tr_raw = load_svmlight_file(tr_path)

        X_tr = np.asarray(X_tr_raw[0].todense())
        labels_tr = np.array(X_tr_raw[1], dtype=int)

        labels_tr[labels_tr == -1] = 0

        tst_path = os.path.join("../../../Datasets/libsvm/{}".format(FLAGS.task_name), 'a9a.t.txt')
        X_tst_raw = load_svmlight_file(tst_path)

        X_tst = np.asarray(X_tst_raw[0].todense())
        labels_tst = np.array(X_tst_raw[1], dtype=int)

        labels_tst[labels_tst == -1] = 0
    if 'mnist' in FLAGS.task_name:
        DATASET = 'mnist'
        image_shape = (28, 28)
        mndata = MNIST('../../../Datasets/mnist')

        images_tr, labels_tr = mndata.load_training()
        images_tst, labels_tst = mndata.load_testing()

        X_tr = np.asarray(images_tr).astype(np.float32) / 255.

        labels_tr = np.asarray(labels_tr).astype(np.int)
        X_tr, labels_tr = shuffle(X_tr, labels_tr, random_state=0)

        labels_tr = relabel(labels_tr)

        X_tst = np.asarray(images_tst).astype(np.float32) / 255.
        labels_tst = np.asarray(labels_tst).astype(np.int)
        X_tst, labels_tst = shuffle(X_tst, labels_tst, random_state=0)

        labels_tst = relabel(labels_tst)

        tr_n_samples, _ = X_tr.shape
        tst_n_samples, _ = X_tst.shape
        h, w = image_shape
        n_classes = len(np.unique(labels_tr))
        n_features = h * w

        print("Total dataset: n_tr_samples {}, n_tst_samples {}, n_features (h {} x w {}) {}, n_classes {}"
              .format(tr_n_samples, tst_n_samples, h, w, h * w, n_classes))

    main()
