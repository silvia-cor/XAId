import sys
import numpy as np
import pickle
import tqdm
from multiprocessing import Pool
from functools import partial
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# sometimes the learning method does not converge; this is to suppress a lot of warnings
if not sys.warnoptions:
    import os, warnings

    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')


def ml_experiment(tr_data, val_data, te_data, learner_name, task, model_path, AV_label, unique_labels):
    """
    Manage the Machine Learning (classic) experiment.
    :param tr_data: training data
    :param val_data: validation data
    :param te_data: test data
    :param learner_name: name of the learning method
    :param task: task to perform
    :param model_path: file path where to save the SAV model
    :param AV_label: the author if interest for AV task, None for SAV or AA task
    :param unique_labels: list of unique labels
    :return: predictions and targets
    """
    if os.path.exists(model_path):
        print('Pair classifier found!')
        with open(model_path, 'rb') as pickle_file:
            cls_params = pickle.load(pickle_file)
    else:
        cls_params = _optimization(learner_name, tr_data, val_data)  # optimized hyperparameters
        with open(model_path, 'wb') as pickle_file:
            pickle.dump(cls_params, pickle_file)
    print(cls_params)
    if task == 'SAV':
        y_pred = _classification_class(learner_name, tr_data, val_data, te_data, cls_params)
    else:
        all_probs = _classification_prob(learner_name, tr_data, val_data, te_data, cls_params)
        y_pred = []
        for i, single_test_probs in enumerate(all_probs):
            label_probs = []  # mean probability that the test sample belongs to each label
            for label in unique_labels:
                label_probs.append(np.mean(np.array([pair_probs[1] for j, pair_probs in enumerate(single_test_probs)
                                                     if te_data['pairs_labels'][i][j] == label])))
            y_pred.append(unique_labels[np.argmax(np.array(label_probs))])
        # for AV, simply check if the predicted author is the one of interest
        if AV_label:
            y_pred = [1 if single_y_pred == AV_label else 0 for single_y_pred in y_pred]
    return y_pred, te_data['task_labels']


# optimization of hyper-parameters on the validation set
def _optimization(learner_name, tr_data, val_data):
    if learner_name == 'lr':
        params = {'class_weight': ['balanced', None], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [1000],
                  'random_state': [42]}
    else:
        params = {'class_weight': ['balanced', None], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'random_state': [42]}
    params_grid = ParameterGrid(params)
    print('Creating train / val feature matrix...')
    X_tr, X_val = __feature_extractor(tr_data, val_data)
    print("Training shape: ", X_tr.shape)
    print("Validation shape: ", X_val.shape)
    print('OPTIMIZATION')
    with Pool(processes=12) as p:
        optimization_step = partial(__single_experiment, X_tr, X_val, tr_data['task_labels'], val_data['task_labels'],
                                    learner_name)
        opt_results = list(tqdm.tqdm(p.imap(optimization_step, params_grid), total=len(params_grid)))
    best_result_idx = opt_results.index(max(opt_results, key=lambda result: result))
    print('Best model:', params_grid[best_result_idx])
    return params_grid[best_result_idx]


# classification outputting list of predicted classes (for SAV task)
# training on the train+val set (with optimized hyperparameters)
def _classification_class(learner_name, tr_data, val_data, te_data, cls_params):
    print('CLASSIFICATION')
    if learner_name == 'lr':
        cls = LogisticRegression(**cls_params)
    else:
        cls = SVC(**cls_params)
    print('Creating train+val / test feature matrix...')
    trval_data = {key: tr_data[key] + val_data[key] for key in tr_data}
    X_trval, X_te = __feature_extractor(trval_data, te_data)  # X_te is a single test matrix
    print("Training shape: ", X_trval.shape)
    print("Test shape: ", X_te.shape)
    cls.fit(X_trval, trval_data['task_labels'])
    y_pred = cls.predict(X_te)
    return y_pred


# classification outputting list of lists of probabilities (one list per test sample, two probs per pair) (for AA and AV task)
# training on the train+val set (with optimized hyperparameters)
def _classification_prob(learner_name, tr_data, val_data, te_data, cls_params):
    print('CLASSIFICATION')
    if learner_name == 'lr':
        cls = LogisticRegression(**cls_params)
    else:
        cls = SVC(probability=True, **cls_params)
    print('Creating train+val feature matrix...')
    trval_data = {key: tr_data[key] + val_data[key] for key in tr_data}
    X_trval, X_te = __feature_extractor(trval_data, te_data)  # X_te is a list of test matrixes (one per test sample)
    print("Training shape: ", X_trval.shape)
    print("Test shape (first test matrix): ", X_te[0].shape)
    cls.fit(X_trval, trval_data['task_labels'])
    probs = []
    for single_test_matrix in X_te:
        probs.append(cls.predict_proba(single_test_matrix))
    return probs


# perform single experiment for optimization (for multi-threading)
def __single_experiment(X_tr, X_val, y_tr, y_val, learner_name, params_combination):
    if learner_name == 'lr':
        learner = LogisticRegression(**params_combination)
    else:
        learner = SVC(**params_combination)
    cls = learner.fit(X_tr, y_tr)
    preds = cls.predict(X_val)
    result = f1_score(y_val, preds, average='binary')
    return result


# extract char 3-grams for the classic ML algorithm
def __feature_extractor(tr_data, te_data):
    X_tr, X_te = [], []
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
    vectorizer.fit(tr_data['texts'])
    for pair in tr_data['pairs_texts']:
        X_tr.append(np.absolute(vectorizer.transform([pair[0]]).toarray() -
                                vectorizer.transform([pair[1]]).toarray()))
    X_tr = np.squeeze(np.array(X_tr))
    if isinstance(te_data['pairs_texts'][0], list):  # for AA and AV: list of matrixes (one per test sample)
        for single_test in te_data['pairs_texts']:
            single_test_matrix = []
            for pair in single_test:
                single_test_matrix.append(np.absolute(vectorizer.transform([pair[0]]).toarray() -
                                                      vectorizer.transform([pair[1]]).toarray()))
            single_test_matrix = np.squeeze(np.array(single_test_matrix))
            X_te.append(single_test_matrix)
    else:  # for SAV : single matrix
        for pair in te_data['pairs_texts']:
            X_te.append(np.absolute(vectorizer.transform([pair[0]]).toarray() -
                                    vectorizer.transform([pair[1]]).toarray()))
        X_te = np.squeeze(np.array(X_te))
    return X_tr, X_te
