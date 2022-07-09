import os
from pathlib import Path
import pandas as pd
import pickle


# pickle the output of a function
def pickled_resource(pickle_path: str, generation_func: callable, *args, **kwargs):
    if pickle_path is None:
        return generation_func(*args, **kwargs)
    else:
        if os.path.exists(pickle_path):
            return pickle.load(open(pickle_path, 'rb'))
        else:
            instance = generation_func(*args, **kwargs)
            os.makedirs(str(Path(pickle_path).parent), exist_ok=True)
            pickle.dump(instance, open(pickle_path, 'wb'), pickle.HIGHEST_PROTOCOL)
            return instance


# check if the output files exist, and either open them or create the corresponding data structure
# output: pickle for predictions (dictionary), csv for results (dataframe with metrics)
def check_create_output_files(pickle_path, result_path, task):
    if os.path.exists(result_path):
        df_csv = pd.read_csv(result_path, sep=';')
        with open(pickle_path, 'rb') as pickle_file:
            df_preds = pickle.load(pickle_file)
    else:
        if task == 'AA':
            df_csv = pd.DataFrame(columns=['Method', 'Macro-F1', 'micro-F1'])
        else:
            df_csv = pd.DataFrame(columns=['Method', 'F1'])
        df_preds = {}
    return df_preds, df_csv


# update the output files with the data from the last experiment
def update_output_files(pickle_path, result_path, df_preds, y_te, y_pred, df_csv, row, learner_name):
    if 'True' not in df_preds:
        df_preds['True'] = y_te
    df_preds[learner_name] = y_pred
    df_csv = df_csv.append(row, ignore_index=True)
    df_csv.to_csv(path_or_buf=result_path, sep=';', index=False, header=True)
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(df_preds, pickle_file)
    return df_csv