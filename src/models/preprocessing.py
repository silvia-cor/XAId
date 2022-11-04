import copy
import itertools
from typing import Tuple, Union

import numpy
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


def dataset_as_sav(data: pandas.DataFrame, sampling_size: int = -1, negative_sampling_size: int = -1,
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
    for author in authors:
        positive_texts = dataset[dataset.author == author].encoded_texts.values.tolist()
        negative_texts = dataset[dataset.author != author].encoded_texts.values.tolist()
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

        nr_positive_samples = sampling_size if isinstance(sampling_size, int)\
                                            else int(positive_df.shape[0] * sampling_size)
        nr_positive_samples = positive_df.shape[0] if sampling_size == -1 else nr_positive_samples
        nr_positive_samples = min(positive_df.shape[0], nr_positive_samples)
        nr_negative_samples = negative_sampling_size if isinstance(negative_sampling_size, int)\
                                                        else int(positive_df.shape[0] * negative_sampling_size)
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

    df = pandas.concat(df, axis="rows")
    df = df[["text_A", "text_B", "author_A", "author_B", "label"]].reset_index().drop_duplicates()

    return df


def dataset_as_av(data: pandas.DataFrame, sampling_size: int = -1, negative_sampling_size: int = -1,
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
    for author in authors:
        positive_df = dataset[dataset.author == author].copy(deep=True)
        negative_df = dataset[dataset.author != author].copy(deep=True)

        nr_positive_samples = sampling_size if isinstance(sampling_size, int)\
                                            else int(positive_df.shape[0] * sampling_size)
        nr_positive_samples = positive_df.shape[0] if sampling_size == -1 else nr_positive_samples
        nr_positive_samples = min(positive_df.shape[0], nr_positive_samples)
        nr_negative_samples = negative_sampling_size if isinstance(negative_sampling_size, int)\
                                                        else int(positive_df.shape[0] * negative_sampling_size)
        nr_negative_samples = negative_df.shape[0] if negative_sampling_size == -1 else nr_negative_samples
        nr_negative_samples = min(negative_df.shape[0], nr_negative_samples)
        nr_negative_samples_per_author = nr_negative_samples // nr_authors

        # for each author, add the positive documents...
        positive_author_dataframe = positive_df.sample(nr_positive_samples, random_state=seed)
        # ... and sample documents from other authors for negative sampling
        negative_author_dataframe = negative_df.groupby("author", group_keys=False).apply(
                                    lambda x: x.sample(min(nr_negative_samples_per_author, x.shape[0]),
                                                       random_state=seed))
        positive_author_dataframe["label"] = 1
        negative_author_dataframe["label"] = 0
        author_df = pandas.concat((positive_author_dataframe, negative_author_dataframe))
        author_df["author"] = author
        df.append(author_df)

    df = pandas.concat(df, axis="rows")
    df = df[["text", "author", "label"]].reset_index().drop_duplicates()

    return df


def dataset_as_aa(data: pandas.DataFrame, scale_labels: bool = False) -> pandas.DataFrame:
    """
    Preprocess the given `data` for a SAV task.
    Args:
        data: The dataset to preprocess.
        scale_labels: Some models only produce outputs in [0, 1], if `labels_scale` is True, scale the labels to [0, 1].
                        Defaults to False.

    """
    data.columns = ["text", "label"]
    # scale labels
    if scale_labels:
        labels = data.label.values
        # scaled_labels = (labels - labels.min()) / (labels.max() - labels.min())
        encoder = LabelEncoder()
        data.label = encoder.fit_transform(labels)

    return data


def preprocess_for_task(data: pandas.DataFrame, task: str, sampling_size: Union[int, float] = 1., negative_sampling_size: Union[int, float] = 1.,
                        scale_labels: bool = False, seed: int = 42) -> pandas.DataFrame:
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
    if task == "sav":
        return dataset_as_sav(data, sampling_size, negative_sampling_size, seed)
    elif task == "av":
        return dataset_as_av(data, sampling_size, negative_sampling_size, seed)
    else:
        # no preprocessing necessary for Authorship Attribution
        return dataset_as_aa(data, scale_labels=scale_labels)


def n_grams(dataframe: pandas.DataFrame, ngrams: int = 3, task: str = "sav") -> Tuple[numpy.ndarray, numpy.ndarray, TfidfVectorizer]:
    """
    Extract n-gram features from the given `dataframe`.
    Args:
        dataframe: The dataframe. Texts are assumed to be in a "text_A" column or in a "text_A, text_B" column pair.
        ngrams: n-grams to consider. Defaults to 3.
        task: Task: one of "sav", "av", or "aa".
    Returns:
        A triple: a numpy.ndarray encoding n-gram statistics and the appropriate additional info for the task,
                the labels, and the n_gram vectorizer.
    """
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(ngrams, ngrams))

    if task == "sav":
        # array in the form
        # difference in n-gram stats
        texts_A, texts_B = dataframe.text_A.values.tolist(), dataframe.text_B.values.tolist()
        vectorizer.fit(texts_A + texts_B)
        vectors_A, vectors_B = vectorizer.transform(texts_A).toarray(), vectorizer.transform(texts_B).toarray()
        data = vectors_A - vectors_B
    elif task == "av":
        # array in the form
        # n-gram stats, guessed author
        texts = dataframe.text.values.tolist()
        vectorizer.fit(texts)
        data = abs(vectorizer.transform(texts).toarray())
        data = numpy.hstack((data, dataframe.author.values.reshape(-1, 1)))
    elif task == "aa":
        # array in the form
        # n-gram stats
        texts = dataframe.text.values.tolist()
        vectorizer.fit(texts)
        data = abs(vectorizer.transform(texts).toarray())

    return data, dataframe.label.values, vectorizer
