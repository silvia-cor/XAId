import copy
import logging
from typing import Tuple, Union

import numpy
import pandas
import spacy
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from alive_progress import alive_bar
from collections import Counter
import random


def dataset_as_sav(data: pandas.DataFrame, sampling_size: Union[int, float] = -1,
                   negative_sampling_size: Union[int, float] = -1,
                   seed: int = 42) -> pandas.DataFrame:
    """
    Preprocess the given `data` for a SAV task.
    Args:
        data: The dataset to preprocess.
        sampling_size: Number (if integer) or percentage (if float) of positive samples for adaptation to AV tasks.
                        Defaults to 1.0 (use all samples).
        negative_sampling_size: Number (if integer) or percentage (if float) of negative samples for adaptation to
                                AV tasks. If percentage, the percentage is computed according to `sampling_size`.
                                To use all samples, set to -1. Defaults to 1.0 (use as many negative samples as
                                positive ones).
        seed: Sampling seed. Defaults to 42.

    Returns:
    """
    dataset = copy.deepcopy(data)
    authors = numpy.unique(dataset.author)
    nr_authors = authors.size
    df = list()
    with alive_bar(nr_authors) as bar:
        for author in authors:
            positive_texts = dataset[dataset.author == author].text.values.tolist()
            negative_texts = dataset[dataset.author != author].text.values.tolist()
            # positive samples
            positive_pairs = list(itertools.permutations(positive_texts, 2))
            positive_pairs = positive_pairs[:len(positive_pairs) // 2]
            positive_df = pandas.DataFrame(positive_pairs, columns=["text_A", "text_B"])
            positive_df = positive_df[positive_df.text_A != positive_df.text_B]
            positive_df["author_A"] = author
            positive_df["author_B"] = author

            # negative samples
            negative_texts_authors = dataset[dataset.author != author].author.values.tolist()
            negative_pairs = list(itertools.product(positive_texts, negative_texts))
            negative_df = pandas.DataFrame(negative_pairs, columns=["text_A", "text_B"])
            negative_df["author_A"] = author
            negative_df["author_B"] = negative_texts_authors * len(positive_texts)

            nr_positive_samples = sampling_size if isinstance(sampling_size, int) \
                else int(positive_df.shape[0] * sampling_size)
            nr_positive_samples = positive_df.shape[0] if sampling_size == -1 else nr_positive_samples
            nr_positive_samples = min(positive_df.shape[0], nr_positive_samples)
            nr_negative_samples = negative_sampling_size if isinstance(negative_sampling_size, int) \
                else int(nr_positive_samples * negative_sampling_size)
            nr_negative_samples = negative_df.shape[0] if negative_sampling_size == -1 else nr_negative_samples
            nr_negative_samples = min(negative_df.shape[0], nr_negative_samples)
            nr_negative_samples_per_author = nr_negative_samples // (nr_authors - 1)

            positive_author_dataframe = positive_df.sample(nr_positive_samples, random_state=seed)
            negative_author_dataframe = negative_df.groupby(["author_A", "author_B"], group_keys=False).apply(
                lambda x: x.sample(min(nr_negative_samples_per_author, x.shape[0]),
                                   random_state=seed))
            positive_author_dataframe["label"] = 1
            negative_author_dataframe["label"] = 0
            author_df = pandas.concat((positive_author_dataframe, negative_author_dataframe))
            df.append(author_df)

            bar()

    df = pandas.concat(df, axis="rows")
    df = df[["text_A", "text_B", "author_A", "author_B", "label"]].reset_index()

    return df


def dataset_as_av(data: pandas.DataFrame, author) -> pandas.DataFrame:
    """
    Preprocess the given `data` for a SAV task.
    Args:
        data: The dataset to preprocess.
        sampling_size: Number (if integer) or percentage (if float) of positive samples for adaptation to AV tasks.
                        Defaults to 1.0 (use all samples).
        negative_sampling_size: Number (if integer) or percentage (if float) of negative samples for adaptation to
                                AV tasks. If percentage, the percentage is computed according to `sampling_size`.
                                To use all samples, set to -1. Defaults to 1.0 (use as many negative samples as
                                positive ones).
        seed: Sampling seed. Defaults to 42.

    Returns:
    """
    data.columns = ["text", "label"]
    labels = data.label.values
    data.label = [1 if label == author else 0 for label in labels]
    return data


def dataset_as_aa(data: pandas.DataFrame, label_encoder=None):
    """
    Preprocess the given `data` for a SAV task.
    Args:
        data: The dataset to preprocess.
        scale_labels: Some models only produce outputs in [0, 1], if `labels_scale` is True, scale the labels to [0, 1].
                        Defaults to False.

    """
    data.columns = ["text", "label"]
    # scale labels
    if label_encoder is None:
        labels = data.label.values
        # scaled_labels = (labels - labels.min()) / (labels.max() - labels.min())
        label_encoder = LabelEncoder()
        data.label = label_encoder.fit_transform(labels)
        return data, label_encoder
    else:
        data.label = label_encoder.transform(data.label.values)
        return data


def preprocess_for_task(data: pandas.DataFrame, task: str, sampling_size: Union[int, float] = 1.,
                        negative_sampling_size: Union[int, float] = 1.,
                        scale_labels=None, seed: int = 42):
    """
    Preprocess the given `data` for the given `task`.
    Args:
        data: The dataset to preprocess.
        task: The task to solve.
        sampling_size: Number (if integer) or percentage (if float) of positive samples for adaptation to AV tasks.
                        Defaults to 1.0 (use all samples).
        negative_sampling_size: Number (if integer) or percentage (if float) of negative samples for adaptation to
                                AV tasks. If percentage, the percentage is computed according to `sampling_size`.
                                To use all samples, set to -1. Defaults to 1.0 (use as many negative samples as
                                positive ones).
        scale_labels: Some models only produce outputs in [0, 1], if `labels_scale` is True, scale the labels to [0, 1].
                        Defaults to False.
        seed: Sampling seed. Defaults to 42.

    Returns:
        A copy of `data` preprocessed according to `task`.
    """

    train_df, test_df = train_test_split(data, test_size=0.1, random_state=seed, stratify=data[['author']])

    if task == "sav":
        train_df = dataset_as_sav(train_df, sampling_size, negative_sampling_size, seed)
        test_df = dataset_as_sav(test_df, int(sampling_size/10), negative_sampling_size, seed)
        return train_df.reset_index(), test_df.reset_index()
    elif task == "av":
        random.seed(seed)
        author = random.choice(numpy.unique(data.author))
        logging.debug(f'\tSelecting AV author: {author}')
        train_df = dataset_as_av(train_df, author)
        test_df = dataset_as_av(test_df, author)
        return train_df.reset_index(), test_df.reset_index()
    else:
        # no preprocessing necessary for Authorship Attribution
        train_df, label_encoder = dataset_as_aa(train_df)
        test_df = dataset_as_aa(test_df, label_encoder=label_encoder)
        return train_df.reset_index(), test_df.reset_index(), label_encoder


def n_grams(dataframe: pandas.DataFrame, ngrams: int = 3, task: str = "sav", analyzer: str = "char",
            vectorizer = None, max_len = None):
    """
    Extract n-gram features from the given `dataframe`.
    Args:
        dataframe: The dataframe. Texts are assumed to be in a "text_A" column or in a "text_A, text_B" column pair.
        ngrams: n-grams to consider. Defaults to 3.
        task: Task: one of "sav", "av", or "aa".
        analyzer: The features to use to train the linear model (algorithm in ["lr", "svm"]). One of "char", "pos",
    Returns:
        A triple: a numpy.ndarray encoding n-gram statistics and the appropriate additional info for the task,
                the labels, and the n_gram vectorizer.
    """
    assert not (analyzer != 'char' and task != 'sav'), "For AV and AA tasks, only char analyzer is available."
    spacy_analyzer = spacy.load("en_core_web_sm")

    if task == "sav":
        # array in the form
        # absolute difference in n-gram stats
        texts_A, texts_B = dataframe.text_A.values.tolist(), dataframe.text_B.values.tolist()
        if analyzer == "pos":
            pos_A = [[token.pos_ for token in spacy_analyzer(t)] for t in texts_A]
            pos_B = [[token.pos_ for token in spacy_analyzer(t)] for t in texts_B]
            joined_pos_A = [" ".join(p) for p in pos_A]
            joined_pos_B = [" ".join(p) for p in pos_B]
            if vectorizer is None:
                logging.debug(f"\tFitting vectorizer on {len(texts_A + texts_B)} texts...")
                vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), sublinear_tf=True)
                vectorizer.fit(joined_pos_A + joined_pos_B)
            vectors_A = vectorizer.transform(joined_pos_A).toarray()
            vectors_B = vectorizer.transform(joined_pos_B).toarray()

        elif analyzer == "word_lengths":
            len_A = [[len(token.text_) for token in spacy_analyzer(t)] for t in texts_A]
            len_B = [[len(token.text_) for token in spacy_analyzer(t)] for t in texts_B]
            flat_len_AB = [item for sublist in (len_A + len_B) for item in sublist]
            if max_len is None:
                max_len = max([k for k, v in Counter(flat_len_AB).items() if v >= 5])
            vectors_A = [[(sum(j >= i for j in l)) / len(l) for i in range(1, max_len)] for l in len_A]
            vectors_B = [[(sum(j >= i for j in l)) / len(l) for i in range(1, max_len)] for l in len_B]

        elif analyzer == "word unigram":
            words_A = [[token.text_ for token in spacy_analyzer(t)] for t in texts_A]
            words_B = [[token.text_ for token in spacy_analyzer(t)] for t in texts_B]
            joined_words_A = [" ".join(p) for p in words_A]
            joined_words_B = [" ".join(p) for p in words_B]
            if vectorizer is None:
                vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), sublinear_tf=True,
                                            token_pattern='(?u)\\b\\w+\\b')  # for 1-letter words
                vectorizer.fit(joined_words_A + joined_words_B)
            vectors_A = vectorizer.transform(words_A).toarray()
            vectors_B = vectorizer.transform(words_B).toarray()

        elif analyzer == "char":
            if vectorizer is None:
                vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=(2, ngrams), sublinear_tf=True)
                vectorizer.fit(texts_A + texts_B)
            vectors_A, vectors_B = vectorizer.transform(texts_A).toarray(), vectorizer.transform(texts_B).toarray()
        data = abs(vectors_A - vectors_B)

    # elif task == "av":
    #     logging.debug("\tFitting vectorizer...")
    #     # array in the form
    #     # n-gram stats, guessed author
    #     texts = dataframe.text.values.tolist()
    #     vectorizer.fit(texts)
    #     logging.debug("\tApplying vectorizer...")
    #     # data = abs(vectorizer.transform(texts).toarray())
    #     data = vectorizer.transform(texts).toarray()  # why abs?
    #     # data = numpy.hstack((data, dataframe.author.values.reshape(-1, 1)))
    else:
        # array in the form
        # n-gram stats
        texts = dataframe.text.values.tolist()
        if vectorizer is None:
            logging.debug("\tFitting vectorizer...")
            vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=(1, ngrams), sublinear_tf=True)
            vectorizer.fit(texts)
        logging.debug("\tApplying vectorizer...")
        # data = abs(vectorizer.transform(texts).toarray())
        data = vectorizer.transform(texts).toarray()

    return data, vectorizer, max_len
