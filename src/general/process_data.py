import csv
import numpy as np
from sklearn.model_selection import train_test_split
import itertools
import random
from torch.utils.data import DataLoader, Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

random.seed(42)


# --------
# general methods for data processing
# --------
def divide_dataset(texts, labels):
    x_trval, x_te, y_trval, y_te = train_test_split(texts, labels, test_size=0.1, random_state=42, stratify=labels)
    x_tr, x_val, y_tr, y_val = train_test_split(x_trval, y_trval, test_size=0.1, random_state=42, stratify=y_trval)
    print('# train samples:', len(y_tr))
    print('# validation samples:', len(y_val))
    print('# test samples:', len(y_te))
    return x_tr, x_val, x_te, y_tr, y_val, y_te


def make_task_pairs(texts, labels, task, AV_label, unique_labels):
    tr_texts, val_texts, te_texts, tr_labels, val_labels, te_labels = divide_dataset(texts, labels)
    print('Creating pairs...')
    tr_data = _SAV_make_pairs(tr_texts, tr_labels, unique_labels, limit_pos_perauthor=1000, limit_neg_tot=5000)
    print('# train pairs:', len(tr_data['pairs_texts']))
    val_data = _SAV_make_pairs(val_texts, val_labels, unique_labels, limit_pos_perauthor=50, limit_neg_tot=250)
    print('# val pairs:', len(val_data['pairs_texts']))
    if task == 'SAV':
        te_data = _SAV_make_pairs(te_texts, te_labels, unique_labels, limit_pos_perauthor=100, limit_neg_tot=500)
        print('# test pairs:', len(te_data['pairs_texts']))
    else:
        te_data = _AA_AV_test_make_pairs(tr_texts, tr_labels, te_texts, te_labels, AV_label, unique_labels,
                                         limit_pos_perauthor=10)
        print('# test pairs:', sum([len(l) for l in te_data['pairs_labels']]))
    return tr_data, val_data, te_data


# make positive and negative pairs for SAV task
# number of pairs is controlled by limit_pos and limit_neg: 0 means making as many as possible
def _SAV_make_pairs(texts, labels, unique_labels, limit_pos_perauthor, limit_neg_tot):
    pairs = list(itertools.combinations([i for i in range(0, len(labels))], 2))
    if limit_pos_perauthor + limit_neg_tot != 0:
        all_possible_neg_pairs = pairs.copy()
        pairs = []
        for label in unique_labels:
            idx_pos_labels = np.where(np.array(labels) == label)[0]  # texts by author
            author_pairs = list(itertools.combinations(idx_pos_labels, 2))  # all possible pos pairs by author
            all_possible_neg_pairs = list(set(all_possible_neg_pairs) - set(author_pairs))
            if limit_pos_perauthor == 0 or len(author_pairs) < limit_pos_perauthor:
                pairs.extend(author_pairs)  # add all positive pairs
            else:
                pairs.extend(random.sample(author_pairs, limit_pos_perauthor))
        if limit_neg_tot == 0 or len(all_possible_neg_pairs) < limit_neg_tot:
            pairs.extend(all_possible_neg_pairs)
        else:
            pairs.extend(random.sample(all_possible_neg_pairs, limit_neg_tot))
    shuffled_pairs = random.sample(pairs, len(pairs))
    # labels for shuffled pairs (same author == 1, different author == 0)
    task_labels = [1 if labels[pair[0]] == labels[pair[1]] else 0 for pair in shuffled_pairs]
    pairs_texts = [(texts[pair[0]], texts[pair[1]]) for pair in shuffled_pairs]
    return {'texts': texts, 'task_labels': task_labels, 'pairs_texts': pairs_texts}


def _AA_AV_test_make_pairs(tr_texts, tr_labels, te_texts, te_labels, AV_label, unique_labels, limit_pos_perauthor):
    pairs_texts = []
    pairs_labels = []
    for i in range(len(te_labels)):
        pairs = []
        for label in unique_labels:
            idx_pos_labels = np.where(np.array(tr_labels) == label)[0]
            all_possible_pos_pairs = [(i, idx_pos_label) for idx_pos_label in
                                      idx_pos_labels]  # pairs with author's texts
            if limit_pos_perauthor == 0 or len(all_possible_pos_pairs) < limit_pos_perauthor:
                pairs.extend(all_possible_pos_pairs)
            else:
                pairs.extend(random.sample(all_possible_pos_pairs, limit_pos_perauthor))
        pairs_texts.append([(te_texts[pair[0]], tr_texts[pair[1]]) for pair in pairs])
        pairs_labels.append([tr_labels[pair[1]] for pair in pairs])
    if AV_label:
        te_labels = [1 if te_label == AV_label else 0 for te_label in te_labels]
    return {'texts': te_texts, 'task_labels': te_labels,
            'pairs_texts': pairs_texts, 'pairs_labels': pairs_labels}


# --------
# methods for NN data processing
# --------

class _SAV_Dataset(Dataset):
    def __init__(self, token_ids, mask_ids, seg_ids, task_labels):
        self.input_ids = token_ids
        self.mask_ids = mask_ids
        self.seg_ids = seg_ids
        self.task_labels = task_labels

    def __len__(self):
        return len(self.task_labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.mask_ids[index], self.seg_ids[index], self.task_labels[index]


def SAV_DataLoader(df_data, tokenizer, batch_size):
    token_ids = []
    seg_ids = []
    mask_ids = []
    for pair in df_data['pairs_texts']:
        pair_token_ids, pair_seg_ids, pair_attention_mask_ids = _bert_combine_pairs(pair, tokenizer)
        token_ids.append(torch.tensor(pair_token_ids))
        seg_ids.append(pair_seg_ids)
        mask_ids.append(pair_attention_mask_ids)
    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    dataset = _SAV_Dataset(token_ids, mask_ids, seg_ids, df_data['task_labels'])
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=5, worker_init_fn=_seed_worker)
    return dataloader


class _AA_AV_test_Dataset(Dataset):
    def __init__(self, token_ids, mask_ids, seg_ids, pairs_labels):
        self.input_ids = token_ids
        self.mask_ids = mask_ids
        self.seg_ids = seg_ids
        self.pairs_labels = pairs_labels

    def __len__(self):
        return len(self.pairs_labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.mask_ids[index], self.seg_ids[index], self.pairs_labels[index]


def AA_AV_test_DataLoader(df_data, tokenizer):
    test_dataloaders = []
    for test_pairs in df_data['pairs_texts']:
        token_ids = []
        seg_ids = []
        mask_ids = []
        for pair in test_pairs:
            pair_token_ids, pair_seg_ids, pair_attention_mask_ids = _bert_combine_pairs(pair, tokenizer)
            token_ids.append(torch.tensor(pair_token_ids))
            seg_ids.append(pair_seg_ids)
            mask_ids.append(pair_attention_mask_ids)
        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        dataset = _AA_AV_test_Dataset(token_ids, mask_ids, seg_ids, df_data['pairs_labels'])
        test_dataloaders.append(DataLoader(dataset, len(test_pairs), num_workers=5, worker_init_fn=_seed_worker))
    return test_dataloaders


# set the seed for the DataLoader worker
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _bert_combine_pairs(pair, tokenizer):
    # truncation required since Bert supports up to 512 tokens max
    a_id = tokenizer.encode(pair[0], max_length=254, truncation=True, add_special_tokens=False)
    b_id = tokenizer.encode(pair[1], max_length=254, truncation=True, add_special_tokens=False)
    pair_token_ids = [tokenizer.cls_token_id] + a_id + [tokenizer.sep_token_id] + b_id + [tokenizer.sep_token_id]
    a_len = len(a_id)
    b_len = len(b_id)
    pair_seg_ids = torch.tensor([0] * (a_len + 2) + [1] * (b_len + 1))  # sentence a and sentence b
    pair_attention_mask_ids = torch.tensor([1] * (a_len + b_len + 3))  # mask padded values
    return pair_token_ids, pair_seg_ids, pair_attention_mask_ids


# --------
# processing methods for each specific dataset
# --------
# victoria dataset
def process_victoria(data_path='../dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv'):
    print('Creating dataset Victoria...')
    texts = []
    labels = []
    with open(data_path, 'r', encoding="latin-1") as data_file:
        csv_reader = csv.reader(data_file, delimiter=',')
        next(csv_reader)  # skip first line
        for row in csv_reader:
            texts.append(row[0])
            labels.append(int(row[1]))
    selected_authors = random.sample(np.unique(labels).tolist(), 5)
    selected_texts = [text for text, label in zip(texts, labels) if label in selected_authors]
    selected_labels = [label for label in labels if label in selected_authors]
    return selected_texts, selected_labels
