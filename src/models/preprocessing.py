import copy
import logging
from typing import Union
import os

import numpy
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from alive_progress import alive_bar
import re
import itertools
from helpers import splitter


def create_MedLatin(data_path, n_sent=10, cleaning=True):
    """
    :param data_path: path to the dataset directory, default: "../dataset/MedLatin"
    :param n_sent: number of sentences forming a fragment, default: 10
    :param cleaning: trigger the custom cleaning of the texts, default: True
    """
    if not os.path.exists(data_path):
        print('Dataset not found!')
        exit(0)
    data = []
    authors_labels = []
    titles_labels = []
    subs = []
    for d in os.listdir(data_path):
        dir_path = data_path + '/' + d
        for file in os.listdir(dir_path):
            file_path = dir_path + '/' + file
            text = open(file_path, "r", errors='ignore').read()
            author, title = file.split('_', 1)  # get author index by splitting the file name
            if author in ['Dante', 'GiovanniBoccaccio', 'PierDellaVigna', 'BenvenutoDaImola', 'PietroAlighieri']:
                if cleaning:  # if cleaning is requested, the files are cleaned
                    text = _clean_texts(text)
                fragments = splitter(text, n_sent)
                data.extend(fragments)
                # add corresponding title label, one for each fragment
                titles_labels.extend([title] * len(fragments))
                # add corresponding author labels, one for each fragment
                authors_labels.extend([author] * len(fragments))
                subs.extend(['epi'] * len(fragments) if d == 'MedLatinEpi' else ['lit'] * len(fragments))
    data = pandas.DataFrame({'text': data, 'author': authors_labels, 'title': titles_labels,
                             'sub_corpus': subs})
    logging.info(f"\tauthors: {numpy.unique(numpy.array(authors_labels))}")
    logging.info(f"\t#texts: {len(data)}")
    return data


def _clean_texts(text):
    # text = text.lower()
    text = text.replace('v', 'u')
    text = text.replace('j', 'i')
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    text = re.sub('\*.*?\*', "", text)
    text = re.sub('\{.*?\}', "", text)
    text = re.sub('[0-9]', "", text)
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    text = re.sub('\.\s+(?=\.)|\.\.+', "", text)
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    text = re.sub("\(|\)|\[|\]", "", text)
    text = re.sub("\—|\–|\-|\_", "", text)
    text = re.sub("\‹|\›|\»|\«|\=|\/|\\|\~|\§|\*|\#|\@|\^|\“|\”|\‘|\’|\°", "", text)
    text = re.sub("\&dagger;|\&amacr;|\&emacr;|\&imacr;|\&omacr;|\&umacr;|\&lsquo;|\&rsquo;|\&rang;|\&lang;|\&lsqb;",
                  "", text)
    text = re.sub("\?|\!|\:|\;", ".", text)
    text = text.replace("'", "")
    text = text.replace('"', '')
    text = text.replace(".,", ".")
    text = text.replace(",.", ".")
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = re.sub('(\.)+', ".", text)
    text = re.sub('(\,)+', "", text)
    text = text.replace("á", "a")
    text = text.replace("é", "e")
    text = text.replace("í", "i")
    text = text.replace("ó", "o")
    text = text.replace("ç", "")
    text = re.sub("\n+", " ", text)
    text = re.sub("\s+", " ", text)
    return text


def dataset_as_sav(data: pandas.DataFrame, sampling_size: Union[int, float] = 1.,
                   negative_sampling_size: Union[int, float] = 1.,
                   seed: int = 42) -> pandas.DataFrame:
    """
    Preprocess the given `data` for SAV task.
    Args:
        :param data: The dataset to preprocess.
        :param sampling_size: Number (if integer) or percentage (if float) of positive samples for adaptation to AV tasks.
                        Defaults to 1.0 (use all samples).
        :param negative_sampling_size: Number (if integer) or percentage (if float) of negative samples for adaptation to
                                AV tasks. If percentage, the percentage is computed according to `sampling_size`.
                                To use all samples, set to -1. Defaults to 1.0 (use as many negative samples as
                                positive ones).
        :param seed: Sampling seed. Defaults to 42.
    Returns:
        dataframe for the SAV task.
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


def dataset_as_av(data: pandas.DataFrame, author: str) -> pandas.DataFrame:
    """
    Preprocess the given `data` for AV task.
    Args:
        :param data: The dataset to preprocess.
        :param author: The author of interest.
    Returns:
        dataframe for the AV task.
    """
    labels = data.author.values
    data['label'] = [1 if label == author else 0 for label in labels]
    return data


def dataset_as_aa(data: pandas.DataFrame, label_encoder: LabelEncoder = None):
    """
    Preprocess the given `data` for AA task.
    Args:
        :param data: The dataset to preprocess.
        :param label_encoder: The label encoder to standardize the author labels.
    Returns:
        dataframe for the AA task.
    """
    # scale labels
    if label_encoder is None:
        labels = data.author.values
        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(labels)
        return data, label_encoder
    else:
        data['label'] = label_encoder.transform(data.author.values)
        return data


def preprocess_for_task(data: pandas.DataFrame, task: str, sampling_size: Union[int, float] = 1.,
                        negative_sampling_size: Union[int, float] = 1., seed: int = 42):
    """
    Preprocess the given `data` for the given `task`.
    Args:
        :param data: The dataset to preprocess.
        :param task: The task to solve.
        :param sampling_size: Number (if integer) or percentage (if float) of positive samples for adaptation to AV tasks.
                        Defaults to 1.0 (use all samples).
        :param negative_sampling_size: Number (if integer) or percentage (if float) of negative samples for adaptation to
                                AV tasks. If percentage, the percentage is computed according to `sampling_size`.
                                To use all samples, set to -1. Defaults to 1.0 (use as many negative samples as
                                positive ones).
        :param seed: Sampling seed. Defaults to 42.

    Returns:
        A copy of `data` preprocessed according to `task`.
    """

    train_df, test_df = train_test_split(data, test_size=0.1, random_state=seed, stratify=data[['author']])

    if task == "sav":
        train_df = dataset_as_sav(train_df, sampling_size, negative_sampling_size, seed)
        test_df = dataset_as_sav(test_df, int(sampling_size / 10), negative_sampling_size, seed)
        return train_df.reset_index(), test_df.reset_index()
    elif task == "av":
        author = 'Dante'  # author for AV is Dante
        logging.debug(f'\tSelecting AV author: {author}')
        train_df = dataset_as_av(train_df, author)
        test_df = dataset_as_av(test_df, author)
        return train_df.reset_index(), test_df.reset_index()
    else:
        # no preprocessing necessary for Authorship Attribution
        train_df, label_encoder = dataset_as_aa(train_df)
        test_df = dataset_as_aa(test_df, label_encoder=label_encoder)
        return train_df.reset_index(), test_df.reset_index(), label_encoder


def n_grams(dataframe: pandas.DataFrame, ngrams: int = 3, task: str = "sav", vectorizer=None):
    """
    Extract n-gram features from the given `dataframe`.
    Args:
        :param dataframe: The dataframe. Texts are assumed to be in a "text_A" column or in a "text_A, text_B" column pair.
        :param ngrams: n-grams to consider. Defaults to 3.
        :param task: Task: one of "sav", "av", or "aa".
        :param vectorizer: the vectorizer to employ, if available.
    Returns:
        A triple: a numpy.ndarray encoding n-gram statistics and the appropriate additional info for the task,
                the labels, and the n_gram vectorizer.
    """
    if task == "sav":
        # array in the form
        # absolute difference in n-gram stats
        texts_A, texts_B = dataframe.text_A.values.tolist(), dataframe.text_B.values.tolist()
        if vectorizer is None:
            logging.debug("\t\tFitting vectorizer...")
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, ngrams), sublinear_tf=True)
            vectorizer.fit(texts_A + texts_B)
        logging.debug("\t\tApplying vectorizer...")
        vectors_A, vectors_B = vectorizer.transform(texts_A).toarray(), vectorizer.transform(texts_B).toarray()
        data = abs(vectors_A - vectors_B)
    else:
        # array in the form
        # n-gram stats
        texts = dataframe.text.values.tolist()
        if vectorizer is None:
            logging.debug("\t\tFitting vectorizer...")
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, ngrams), sublinear_tf=True)
            vectorizer.fit(texts)
        logging.debug("\t\tApplying vectorizer...")
        data = vectorizer.transform(texts).toarray()
    return data, vectorizer
