from ML_classification import ML_experiment
from NN_classification import NN_experiment
import os
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from general.process_data import process_victoria, make_task_pairs
from general.utils import pickled_resource, check_create_output_files, update_output_files
import random

random.seed(42)

dataset_name = 'victoria'
learner_name = 'bert'
task = 'SAV'

assert dataset_name == 'victoria', 'Victoria is the only dataset atm'
assert learner_name in ['svm', 'lr', 'bert'], 'Available methods: svm, lr, bert.'
assert task in ['SAV', 'AV', 'AA'], 'Available tasks: SAV, AV, AA.'

dataset_path = f'../pickles/dataset_{dataset_name}.pickle'
if learner_name in ['svm', 'lr']:
    model_path = f'../models/ML/{dataset_name}_{learner_name}.pickle'
else:
    model_path = f'../models/NN/{dataset_name}_{learner_name}.pickle'
pickle_path = f'../pickles/preds_{dataset_name}_{task}_prova.pickle'
result_path = f'../results/res_{dataset_name}_{task}_prova.csv'

os.makedirs(str(Path(dataset_path).parent), exist_ok=True)
os.makedirs(str(Path(result_path).parent), exist_ok=True)
os.makedirs(str(Path(model_path).parent), exist_ok=True)


# make the samples
texts, labels = pickled_resource(dataset_path, process_victoria)
# make the output files
# df_preds, df_csv = check_create_output_files(pickle_path, result_path, np.unique(labels), task)
# if learner_name in df_csv['Method'].values:
#     print(f'{task} with {learner_name} experiment already done!')
# else:
print(f'----- {task} with {learner_name} experiment -----')
# select random author for AV
unique_labels = list(np.unique(labels))
AV_label = random.sample(unique_labels, 1)[0] if task == 'AV' else None
tr_data, val_data, te_data = make_task_pairs(texts, labels, task, AV_label, unique_labels)
if learner_name in ['lr', 'svm']:
    y_pred, y_te = ML_experiment(tr_data, val_data, te_data, learner_name, task, model_path, AV_label, unique_labels)
else:
    y_pred, y_te = NN_experiment(tr_data, val_data, te_data, task, model_path, AV_label, unique_labels)
if task == 'AA':
    macro_f1 = np.around(f1_score(y_te, y_pred, average='macro'), decimals=3)
    micro_f1 = np.around(f1_score(y_te, y_pred, average='micro'), decimals=3)
    row = {'Method': learner_name, 'Macro-F1': macro_f1, 'micro-F1': micro_f1}
else:
    f1 = np.around(f1_score(y_te, y_pred, average='binary'), decimals=3)
    row = {'Method': learner_name, 'F1': f1}
print(row)
    # df_csv = update_output_files(pickle_path, result_path, df_preds, y_te, y_pred, df_csv, row, learner_name)
# print(df_csv[df_csv["Method"] == learner_name])
