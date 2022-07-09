import itertools
import random
import csv
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

random.seed(42)


# --------
# general methods for data processing
# --------

# divide dataset in train, validation and test sets (texts and labels)
# dataset -> trainval (90%) and test (10%)
# trainval -> train (90%) and validation (10%)
def divide_dataset(texts, labels):
    x_trval, x_te, y_trval, y_te = train_test_split(texts, labels, test_size=0.1, random_state=42, stratify=labels)
    x_tr, x_val, y_tr, y_val = train_test_split(x_trval, y_trval, test_size=0.1, random_state=42, stratify=y_trval)
    print('# train samples:', len(y_tr))
    print('# validation samples:', len(y_val))
    print('# test samples:', len(y_te))
    return x_tr, x_val, x_te, y_tr, y_val, y_te


# control the creation of the experiment pairs, given the task
def make_task_pairs(texts, labels, task, AV_label, unique_labels):
    tr_texts, val_texts, te_texts, tr_labels, val_labels, te_labels = divide_dataset(texts, labels)
    print('Creating pairs...')
    tr_data = _sav_make_pairs(tr_texts, tr_labels, unique_labels, limit_pos_perauthor=1000, limit_neg_tot=5000)
    print('# train pairs:', len(tr_data['pairs_texts']))
    val_data = _sav_make_pairs(val_texts, val_labels, unique_labels, limit_pos_perauthor=100, limit_neg_tot=500)
    print('# val pairs:', len(val_data['pairs_texts']))
    if task == 'SAV':
        te_data = _sav_make_pairs(te_texts, te_labels, unique_labels, limit_pos_perauthor=100, limit_neg_tot=500)
        print('# test pairs:', len(te_data['pairs_texts']))
    else:
        te_data = _aa_av_test_make_pairs(tr_texts, tr_labels, te_texts, te_labels, AV_label, unique_labels,
                                         limit_pairs_perauthor=10)
        print('# test pairs:', sum([len(exp) for exp in te_data['pairs_labels']]))
    return tr_data, val_data, te_data


def _sav_make_pairs(texts, labels, unique_labels, limit_pos_perauthor, limit_neg_tot):
    """
    Make positive and negative pairs for SAV task.
    We want limit_pos_perauthor positive (author, author) pairs for each author
    and limit_neg_tot negative (author_x, author_y) pairs in total (in this case, we don't care who are the two authors).
    We use it for train, val e test in SAV experiments, and for train and val in AA and AV experiments
    (the learner has to learn only to distinguish among pos and neg pairs, the real author in test is obtained via probabilities).
    :param texts: list of original texts
    :param labels: list of original labels (the author of each text)
    :param unique_labels: list of unique labels
    :param limit_pos_perauthor: controls the number of positive pairs for each author (0 -> as many as possible)
    :param limit_neg_tot: controls the number of total negative pairs (0 -> as many as possible)
    :return: dictionary with: 'texts' (list of original texts),
    'task_labels' (list of SAV labels: 1 positive pair | 0 negative pair), 'pairs_texts' (list of tuple (text1, text2))
    """
    pairs = list(itertools.combinations([i for i in range(0, len(labels))], 2))  # all possible pairs
    if limit_pos_perauthor + limit_neg_tot != 0:
        all_possible_neg_pairs = pairs.copy()
        pairs = []
        # make limit_pos_perauthor positive pairs for each author
        for label in unique_labels:
            idx_pos_labels = np.where(np.array(labels) == label)[0]  # texts by author
            author_pairs = list(itertools.combinations(idx_pos_labels, 2))  # all possible pos pairs by author
            all_possible_neg_pairs = list(set(all_possible_neg_pairs) - set(author_pairs))
            if limit_pos_perauthor == 0 or len(author_pairs) < limit_pos_perauthor:
                pairs.extend(author_pairs)  # add all pos pairs
            else:
                pairs.extend(
                    random.sample(author_pairs, limit_pos_perauthor))  # add random limit_pos_perauthor pos pairs
        # make limit_neg_tot negative pairs
        # all_possible_neg_pairs is obtained by excluding the author_pairs for each author from the total pairs
        if limit_neg_tot == 0 or len(all_possible_neg_pairs) < limit_neg_tot:
            pairs.extend(all_possible_neg_pairs)  # add all neg pairs
        else:
            pairs.extend(random.sample(all_possible_neg_pairs, limit_neg_tot))  # add random limit_neg_tot neg pairs
    shuffled_pairs = random.sample(pairs, len(pairs))
    # labels for shuffled pairs (same author == 1, different author == 0)
    task_labels = [1 if labels[pair[0]] == labels[pair[1]] else 0 for pair in shuffled_pairs]
    pairs_texts = [(texts[pair[0]], texts[pair[1]]) for pair in shuffled_pairs]
    return {'texts': texts, 'task_labels': task_labels, 'pairs_texts': pairs_texts}


def _aa_av_test_make_pairs(tr_texts, tr_labels, te_texts, te_labels, AV_label, unique_labels, limit_pairs_perauthor):
    """
    Make positive and negative pairs for the test set of AA and AV tasks.
    Given one text sample, AA and AV is achieved by making limit_pos_perauthor pairs for each author, where a pair is
    (test_sample, author_x). In AA, we then select the real author for the test sample as the author that obtains
    the highest mean probabilities in his/her pairs with the test sample; in AV, we check whether the selected author
    is the author of interest.
    Does not require shuffle, since it's the test set.
    :param tr_texts: list of texts from training set
    :param tr_labels: list of labels from training set (the author of each training text)
    :param te_texts: list of texts from test set
    :param te_labels: list of labels from test set (the author of each training text)
    :param AV_label: the author if interest for AV task, None for AA task
    :param unique_labels: list of unique labels
    :param limit_pairs_perauthor: controls the number of pairs with test sample for each author (0 -> as many as possible)
    :return: dictionary with: 'texts' (list of original test texts),
    'task_labels' (list of AA/AV labels: multiclass or binary),
    'pairs_texts' (list of lists of tuple (text1, text2), one list for each test sample),
    'pars_labels' (list of lists of tr_labels, one list for each test sample, to track the author in each pair)
    """
    pairs_texts = []
    pairs_labels = []
    for i in range(len(te_labels)):
        pairs = []
        for label in unique_labels:
            idx_pos_labels = np.where(np.array(tr_labels) == label)[0]  # index of texts by author
            all_possible_pos_pairs = [(i, idx_pos_label) for idx_pos_label in
                                      idx_pos_labels]  # all possible pairs with author's texts
            if limit_pairs_perauthor == 0 or len(all_possible_pos_pairs) < limit_pairs_perauthor:
                pairs.extend(all_possible_pos_pairs)
            else:
                pairs.extend(random.sample(all_possible_pos_pairs, limit_pairs_perauthor))
        pairs_texts.append([(te_texts[pair[0]], tr_texts[pair[1]]) for pair in pairs])
        pairs_labels.append([tr_labels[pair[1]] for pair in pairs])
    if AV_label:
        te_labels = [1 if te_label == AV_label else 0 for te_label in te_labels]
    return {'texts': te_texts, 'task_labels': te_labels,
            'pairs_texts': pairs_texts, 'pairs_labels': pairs_labels}


# --------
# methods for NN data processing
# --------

# class Dataset for SAV data
class _SavDataset(Dataset):
    def __init__(self, input_ids, seg_ids, mask_ids, task_labels):
        self.input_ids = input_ids
        self.seg_ids = seg_ids
        self.mask_ids = mask_ids
        self.task_labels = task_labels

    def __len__(self):
        return len(self.task_labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.seg_ids[index], self.mask_ids[index], self.task_labels[index]


# class Dataset for AA and AV data
class _AaAvTestDataset(Dataset):
    def __init__(self, input_ids, seg_ids, mask_ids, pairs_labels):
        self.input_ids = input_ids
        self.seg_ids = seg_ids
        self.mask_ids = mask_ids
        self.pairs_labels = pairs_labels

    def __len__(self):
        return len(self.pairs_labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.seg_ids[index], self.mask_ids[index], self.pairs_labels[index]


# create Dataloader for SAV data
def SavDataLoader(df_data, tokenizer, batch_size):
    input_ids = []
    seg_ids = []
    mask_ids = []
    for pair in df_data['pairs_texts']:
        pair_input_ids, pair_seg_ids, pair_mask_ids = _bert_combine_pairs(pair, tokenizer)
        input_ids.append(torch.tensor(pair_input_ids))
        seg_ids.append(pair_seg_ids)
        mask_ids.append(pair_mask_ids)
    input_ids = pad_sequence(input_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    dataset = _SavDataset(input_ids, seg_ids, mask_ids, df_data['task_labels'])
    dataloader = DataLoader(dataset, batch_size, num_workers=5, worker_init_fn=_seed_worker)
    return dataloader


# create n Dataloaders for AA nd AV data, one for each test sample
def AaAvTestDataLoader(df_data, tokenizer):
    test_dataloaders = []
    for i, test_pairs in enumerate(df_data['pairs_texts']):
        input_ids = []
        seg_ids = []
        mask_ids = []
        for pair in test_pairs:
            pair_input_ids, pair_seg_ids, pair_mask_ids = _bert_combine_pairs(pair, tokenizer)
            input_ids.append(torch.tensor(pair_input_ids))
            seg_ids.append(pair_seg_ids)
            mask_ids.append(pair_mask_ids)
        input_ids = pad_sequence(input_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        dataset = _AaAvTestDataset(input_ids, seg_ids, mask_ids, df_data['pairs_labels'][i])
        test_dataloaders.append(DataLoader(dataset, len(test_pairs), num_workers=5, worker_init_fn=_seed_worker))
    return test_dataloaders


# set the seed for the DataLoader worker
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# combine the pair of sentences in a form that is BERT-compliant
# BERT can work with pairs of sentences in the form: [CLS] sentence_a [SEP] sentence_b [SEP]
def _bert_combine_pairs(pair, tokenizer):
    # truncation required since Bert supports up to 512 tokens max
    a_id = tokenizer.encode(pair[0], max_length=254, truncation=True, add_special_tokens=False)
    b_id = tokenizer.encode(pair[1], max_length=254, truncation=True, add_special_tokens=False)
    pair_input_ids = [tokenizer.cls_token_id] + a_id + [tokenizer.sep_token_id] + b_id + [tokenizer.sep_token_id]
    a_len = len(a_id)
    b_len = len(b_id)
    pair_seg_ids = torch.tensor([0] * (a_len + 2) + [1] * (b_len + 1))  # seg_ids are used to identify the two sentences
    pair_mask_ids = torch.tensor([1] * (a_len + b_len + 3))  # mask to recognize non-padded values
    return pair_input_ids, pair_seg_ids, pair_mask_ids


# --------
# processing methods for each specific dataset
# --------

# victoria dataset
def process_victoria(data_path):
    print('Creating dataset Victoria...')
    texts = []
    labels = []
    with open(data_path, 'r', encoding="latin-1") as data_file:
        csv_reader = csv.reader(data_file, delimiter=',')
        next(csv_reader)  # skip first line (header)
        for row in csv_reader:
            texts.append(row[0])
            labels.append(int(row[1]))
    selected_authors = random.sample(np.unique(labels).tolist(), 5) # select only n authors (all texts)
    selected_texts = [text for text, label in zip(texts, labels) if label in selected_authors]
    selected_labels = [label for label in labels if label in selected_authors]
    return selected_texts, selected_labels
