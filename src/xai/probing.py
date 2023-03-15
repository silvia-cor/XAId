from __future__ import annotations

import itertools
import logging
import json
from abc import abstractmethod
from collections import defaultdict
from typing import List, Iterable, Dict, Tuple, Sequence

import sys
import numpy
import pandas
import ray
import spacy as spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tune_sklearn import TuneGridSearchCV as SearchAlgorithm
from tqdm import tqdm
from .pla import pla
import pickle
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from kneed import KneeLocator

sys.path.append("../")


class Prober:
    @abstractmethod
    def probe(self, text: Iterable, probes: List):
        pass


class Tagger:
    @abstractmethod
    def tag(self, text: Iterable) -> List:
        pass


class TransformerProber:
    def __init__(self, model, n_authors, seed):
        self.model = model
        self.seed = seed
        self.n_authors = n_authors
        # self.pos_tokens = {"ADJ", "PROPN", "PUNCT", "INTJ", "PART", "NOUN", "ADV", "VERB", "X", "SYM", "NUM", "ADP",
        #                    "SCONJ", "DET", "AUX", "PRON", "CCONJ"}

    def probe(self, data_df, task, label_encoder, output, probe_type: str = "pos", k: int = 5, min_len: int = 5,
              max_len: int = 10, n_jobs: int = 1):
        """Probe the model on the given `text` by using a `model` model.

        Args:
            dataloader: The probing text.
            probe_type: The probe type to check. One of "pos", "ner", "lemma", "stem", "word_length_frequency",
                        "richness".
            k: How many probes to create, defaults to 5.
            min_len: Minimum chain length for "pos" probes.
            max_len: Maximum chain length for "pos" probes.
            n_jobs: Parallelism degree. Defaults to 1.

        Returns:
            The probe, its parametrization, and a validation report.
        """
        assert task == 'aa', "Only AA task supported for probing."
        documents = data_df.text.values
        labels = data_df.label.values
        authors = numpy.unique(labels)

        if probe_type == "pos":
            tagger = POSTagger()
        elif probe_type == "ner":
            tagger = NERTagger()
        elif probe_type == "stem":
            tagger = StemTagger()
        elif probe_type == "lemma":
            tagger = LemmaTagger()
        elif probe_type == "word_length_frequency":
            tagger = WordLengthFrequencyTagger(self.seed, self.n_authors)
        elif probe_type == "richness":
            logging.debug("\t\tBuilding frequency classes...")
            tagger = RichnessFrequencyTagger(documents, self.seed, self.n_authors)
        else:
            raise ValueError("Unknown probe_type")

        ray.init()

        logging.debug("\tBuilding ground truth for probing...")
        logging.debug("\tComputing tags...")
        tags = tagger.tag(documents)
        tags_enriched_dataloader = list(zip(documents, labels, tags))
        texts_and_tags_per_author = {author: [(document, label, document_tags)
                                              for (document, label, document_tags) in
                                              tags_enriched_dataloader
                                              if label == author] for author in authors}
        probes_results = dict()
        probes = dict()

        if probe_type in ["pos", "lemma", "stem", "ner"]:

            logging.debug("\tIterating over authors...")
            for i, author in enumerate(authors):
                true_label = label_encoder.inverse_transform([author])[0]
                logging.debug(f"\t\tAuthor {true_label}")
                author_data = texts_and_tags_per_author[author]
                other_authors_data = list(itertools.chain.from_iterable([texts_and_tags_per_author[other_author]
                                                                         for other_author in authors if
                                                                         other_author != author]))
                probes_results[str(true_label)] = list()
                probes[str(true_label)] = list()
                author_documents = [document for document, _, _ in author_data]
                other_authors_documents = [document for document, _, _ in other_authors_data]
                author_tags = [tags for _, _, tags in author_data]
                other_author_tags = [tags for _, _, tags in other_authors_data]

                if probe_type == "pos":
                    logging.debug("\t\t\tAuthor POS chains....")
                    top_author_pos_chains = self._top_k_chains(author_tags, k=k * k, min_len=min_len, max_len=max_len)
                    logging.debug("\t\t\tOther authors POS chains....")
                    top_other_authors_pos_chains = self._top_k_chains(other_author_tags, k=k * k, min_len=min_len,
                                                                      max_len=max_len)
                    logging.debug("\t\t\tFiltering POS chains....")
                    top_author_pos_chains = [chain for chain in top_author_pos_chains if
                                             chain not in top_other_authors_pos_chains][:k]
                    # construct probe labels
                    logging.debug("\t\t\tProbing POS chains....")
                    for target_pos_chain in tqdm(top_author_pos_chains):
                        author_documents_labels = [int(target_pos_chain in " ".join(document_pos_chain).lower())
                                                   for document_pos_chain in author_tags]
                        other_documents_labels = [int(target_pos_chain in " ".join(document_pos_chain).lower())
                                                  for document_pos_chain in
                                                  other_author_tags[:len(author_documents_labels)]]
                        probing_labels = numpy.array(author_documents_labels + other_documents_labels)

                        # balance classes
                        positive_indexes = numpy.argwhere(probing_labels == 1).squeeze()
                        nr_positive_samples = positive_indexes.size
                        negative_indexes = numpy.argwhere(probing_labels == 0).squeeze()[:nr_positive_samples]
                        indexes = numpy.hstack((positive_indexes, negative_indexes))
                        documents = author_documents + other_authors_documents
                        probing_documents = [documents[i] for i in indexes]
                        probing_labels = probing_labels[indexes]

                        logging.debug("\t\t\tLaunching probe...")
                        classifier, configuration, validation = self._launch_probe(probing_documents, probing_labels,
                                                                                   task,
                                                                                   n_jobs=n_jobs)

                        probes_results[str(true_label)].append((target_pos_chain, configuration, validation))
                        probes[str(true_label)].append((target_pos_chain, classifier))

                else:
                    logging.debug("\t\t\tFiltering top items....")
                    top_author_items = self._top_k_items(author_tags, k=k * k)
                    top_other_authors_roots = self._top_k_items(other_author_tags, k=k * k)
                    top_author_items = [root for root in top_author_items if root not in top_other_authors_roots][:k]

                    # construct probe labels
                    logging.debug("\t\t\tProbing items....")
                    for target_root in tqdm(top_author_items):
                        author_documents_labels = [int(target_root in document_roots)
                                                   for document_roots, _, _ in author_data]
                        other_documents_labels = [int(target_root in document_roots)
                                                  for document_roots, _, _ in
                                                  other_authors_data[:len(author_documents_labels)]]
                        probing_labels = numpy.array(author_documents_labels + other_documents_labels)
                        if numpy.unique(probing_labels).size == 1:
                            continue
                        try:
                            classifier, configuration, validation = self._launch_probe(
                                author_documents + other_authors_documents[:len(author_documents_labels)],
                                probing_labels,
                                task, n_jobs=n_jobs)

                            probes_results[str(true_label)].append((target_root, validation))
                            probes[str(true_label)].append((target_root, classifier))

                            with open(f"{output}.probe.{probe_type}.author{true_label}.{target_root}.validation.json",
                                      "w") as log:
                                json.dump(validation, log)
                        except ValueError:
                            continue

        else:
            if probe_type == "richness":
                probes_results["token_classes"] = tagger.token_classes
                probes_results["token_frequencies"] = {k: int(v) for k, v in tagger.token_frequencies.items()}
            # multi-class
            probing_labels = numpy.array([frequency_cluster for _, _, frequency_cluster in tags_enriched_dataloader])
            try:
                classifier, configuration, validation = self._launch_probe(documents, probing_labels, task,
                                                                           n_jobs=n_jobs)
                probes_results['multiclass'] = (configuration, validation)
                probes['multiclass'] = classifier

                with open(f"{output}.probe.{probe_type}.multiclass.validation.json", "w") as log:
                    json.dump(validation, log)
            except ValueError as e:
                logging.debug(f"Could not get probe to work: {e}")

            # # binary?
            # for i, author in enumerate(authors):
            #     true_label = label_encoder.inverse_transform([author])[0]
            #     logging.debug(f"\t\tAuthor {true_label}")
            #     author_data = texts_and_tags_per_author[author]
            #     other_authors_data = list(itertools.chain.from_iterable([texts_and_tags_per_author[other_author]
            #                                                              for other_author in authors if
            #                                                              other_author != author]))
            #     probes_results[str(true_label)] = list()
            #     probes[str(true_label)] = list()
            #     author_documents = [document for document, _, _ in author_data]
            #     other_authors_documents = [document for document, _, _ in other_authors_data]
            #     author_tags = [tags for _, _, tags in author_data]
            #     other_author_tags = [tags for _, _, tags in other_authors_data]
            #     probing_labels = numpy.array(([1] * len(author_tags)) + ([0] * len(other_author_tags)))
            #     try:
            #         classifier, configuration, validation = self._launch_probe(
            #             author_documents + other_authors_documents, probing_labels,
            #             task, n_jobs=n_jobs)
            #         probes_results[str(true_label)].append((configuration, validation))
            #         probes[str(true_label)].append(classifier)
            #         with open(f"{output}.probe.{probe_type}.author{true_label}.validation.json", "w") as log:
            #             json.dump(validation, log)
            #     except ValueError as e:
            #         logging.debug(f"Could not get probe to work: {e}")

        ray.shutdown()

        return probes_results, probes

    def _launch_probe(self, data, probing_labels: numpy.ndarray, task, n_jobs: int = 1) -> Tuple[
        LogisticRegression, Dict, Dict]:
        """Probe the model on the given `text` by using a `model` model.

        Args:
            text: The probing text.
            probing_labels: The probes to check.
            n_jobs: Parallelism degree. Defaults to 1.

        Returns:
            The probe, its parametrization, and a validation report.
        """
        # Build probing dataset
        logging.debug("\t\t\tConstructing probing dataset...")
        data_df = pandas.DataFrame({'text': data, 'label': probing_labels})
        embedding_data = self.model.encode(data_df, task)
        embedding_data = numpy.array(embedding_data)
        embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(embedding_data, probing_labels,
                                                                                        stratify=probing_labels,
                                                                                        test_size=0.1)
        # Build prober
        hyperparameters = {
            "C": [0.001, 0.01, 0.1, 1, 10],
            "penalty": ["l2"]
        }
        logging.debug("\t\t\tFitting prober...")
        search = SearchAlgorithm(LogisticRegression(max_iter=10000,
                                                    warm_start=True),
                                 param_grid=hyperparameters,
                                 cv=3, early_stopping=False,
                                 max_iters=1,
                                 refit=True,
                                 n_jobs=n_jobs)
        search.fit(embeddings_train, labels_train)
        classifier = search.best_estimator
        configuration = {
            "C": classifier.C,
            "coefficients": classifier.coef_.tolist(),
            "intercept": classifier.intercept_.tolist()
        }

        # Validate prober
        validation = classification_report(labels_test, classifier.predict(embeddings_test), output_dict=True)

        return classifier, configuration, validation

    def _top_k_chains(self, chains: List, k: int = 5, min_len: int = 5, max_len: int = 10) -> List[List]:
        """
        Find the most characteristic features
        Args:
            chains: The probe
            k: How many sequences to select, defaults to 5.
            min_len: Minimum length of chain.
            max_len: Maximum length of chain.

        Returns:
            The top probing sequences per author.
        """
        chain_counter = CountVectorizer(analyzer="word", ngram_range=(min_len, max_len))
        chains_as_str = [" ".join(chain) for chain in chains]
        chain_counter = chain_counter.fit(chains_as_str)
        counted_chains_names = chain_counter.get_feature_names_out()
        counted_chains = chain_counter.transform(chains_as_str)

        mean_counts_per_chain = numpy.ravel(counted_chains.mean(axis=0))
        top_k_mean_chains = counted_chains_names[numpy.argsort(mean_counts_per_chain)[-k:]]

        return top_k_mean_chains

    def _is_subchain_in_chain(self, subchain: List, chain: List) -> bool:
        """
        True if `subchain` is found in `chain`, False otherwise.
        Args:
            subchain: The subchain to find
            chain: The chain where to find the subchain

        Returns:
            True if `subchain` is found in `chain`, False otherwise.
        """
        subchain_length = len(subchain)
        chain_length = len(chain)

        return any([chain[i:i + subchain_length] == subchain
                    for i in range(chain_length - subchain_length + 1)])

    def _top_k_items(self, probes: List[List], k: int = 5) -> List[str]:
        """
        Compute the most characteristic roots (lemmas/stems) among `probes`
        Args:
            probes: The probes to analyze
            k: The number of top roots to return. Defaults to 5

        Returns:
            The top-k roots
        """
        flat_probes = numpy.array([item for document_items in probes for item in document_items])
        roots, frequencies = numpy.unique(flat_probes, return_counts=True)

        return roots[frequencies.argsort()[-k:]]


class POSTagger(Tagger):
    def __init__(self):
        self.tagger = spacy.load("en_core_web_sm")

    def tag(self, text: Iterable) -> List[str] | List[List[str]]:
        """
        Probe the model on the given `text`.
        Args:
            text: The text to analyze. Either a text or an Iterable.

        Returns:
            A list of POS, if a single piece of text is provided, or a list of lists of POS,
            if an Iterable is provided.
        """
        if isinstance(text, str):
            analysis = self.tagger(text)
            pos = [token.pos_ for token in analysis]
        else:
            pos = [self.tag(t) for t in text]

        return pos


class LemmaTagger(Tagger):
    def __init__(self):
        self.prober = spacy.load("en_core_web_sm")

    def tag(self, text: Iterable) -> List[str] | List[List[str]]:
        """
        Probe the model on the given `text`.
        Args:
            text: The text to analyze. Either a text or an Iterable.

        Returns:
            A list of lemmas, if a single piece of text is provided, or a list of lists of lemmas,
            if an Iterable is provided.
        """
        if isinstance(text, str):
            analysis = self.prober(text)
            lemmas = [token.lemma_ for token in analysis]
        else:
            lemmas = [self.tag(t) for t in text]

        return lemmas


class StemTagger(Tagger):
    def __init__(self):
        # self.prober = spacy.load("en_core_web_sm")
        self.stemmer = SnowballStemmer(language='english')

    def tag(self, text: Iterable) -> List[str] | List[List[str]]:
        """
        Probe the model on the given `text`.
        Args:
            text: The text to analyze. Either a text or an Iterable.

        Returns:
            A list of stemmed lemmas, if a single piece of text is provided, or a list of lists of stemmed lemmas,
            if an Iterable is provided.
        """
        if isinstance(text, str):
            stems = [self.stemmer.stem(w) for w in word_tokenize(text)]
        else:
            stems = [self.tag(t) for t in text]
            #     analysis = self.prober(text)
        #     lemmas = [token.root.text for token in analysis.noun_chunks]
        # else:
        #     lemmas = [self.tag(t) for t in text]
        return stems


class NERTagger(Tagger):
    def __init__(self):
        self.prober = spacy.load("en_core_web_sm")

    def tag(self, text: Iterable) -> List[str] | List[List[str]]:
        """
        Probe the model on the given `text`.
        Args:
            text: The text to analyze. Either a text or an Iterable.

        Returns:
            A list of named entities, if a single piece of text is provided, or a list of lists of named entities,
            if an Iterable is provided.
        """
        if isinstance(text, str):
            analysis = self.prober(text)
            lemmas = [token.text for token in analysis.ents]
        else:
            lemmas = [self.tag(t) for t in text]

        return lemmas


class WordLengthFrequencyTagger(Tagger):
    def __init__(self, seed, n_authors):
        self.prober = spacy.load("en_core_web_sm")
        self.max_length = 10000
        self.seed = seed
        self.n_authors = n_authors

    def tag(self, text: Iterable) -> numpy.ndarray:
        """
        Probe the model on the given `text`.
        Args:
            text: The text to analyze. Either a text or an Iterable.

        Returns:
            Top-10 frequencies as computed by K-Means
        """
        if isinstance(text, str):
            analysis = self.prober(text)
            lengths = [len(token.text) for i, token in enumerate(analysis)
                       if not token.text.isdigit()]

            return lengths
        else:
            lengths = [self.tag(t) for t in text]
            counts = [numpy.unique(length, return_counts=True) for length in lengths]
            # may have holes, i.e. frequencies that do not appear, so need to replace them w/ 0
            frequencies = [numpy.array([count_frequencies[numpy.where(count_index == length_value)].item()
                                        if length_value in count_index else 0
                                        for length_value in range(self.max_length)])
                           for count_index, count_frequencies in counts]
            frequencies = numpy.array(frequencies)
            frequencies = frequencies / frequencies.sum()
            # map frequencies to class through clustering
            inertias, clusterings = [], []
            for k in range(2, self.n_authors*2):
                clustering = KMeans(n_clusters=k, max_iter=1000, random_state=self.seed)
                clusterings.append(clustering.fit(frequencies))
                inertias.append(clustering.inertia_)
            kn = KneeLocator(range(2, self.n_authors*2), inertias, curve='convex', direction='decreasing').knee
            self.clustering = clusterings[kn]
            centroids = self.clustering.cluster_centers_
            class_labels = self.clustering.predict(frequencies)
            self.frequency_classes = centroids
            self.frequency_classes = centroids
        return class_labels


class RichnessFrequencyTagger(Tagger):
    """
    Args:
        prober: spacy.lang.en.English: The analyzer.
        token_frequencies: Dict[str,int]: A dictionary token => frequency
        richness_classes: List[int]: Classes of token richness
        token_classes: Dict[str, int]: A dictionary token => richness class
        centroids: numpy.ndarray: After tagging some text, these are the centroids of each frequency distribution
    """

    def __init__(self, corpus: List[str], seed, n_authors):
        self.prober = spacy.load("en_core_web_sm")
        self.token_frequencies = defaultdict(int)
        self.clustering = None
        self.prober.max_length = 1000000
        self.seed = seed
        self.n_authors = n_authors

        # build corpus frequencies
        frequencies_dicts = list()
        lemmas = set()
        nr_splits = (len(corpus) // 10)

        # split into chunks to avoid Outofmemory
        nr_analyzed_docs = 0
        for i in range(nr_splits):
            analyzed_corpuses = [self.prober(c) for c in corpus[i * 10:(i + 1) * 10]]
            lemmatized_corpuses = [numpy.array([token.lemma_ for token in c]) for c in analyzed_corpuses]
            lemmatized_corpus = numpy.hstack(lemmatized_corpuses)
            nr_analyzed_docs += len(analyzed_corpuses)

            ls, fs = numpy.unique(lemmatized_corpus, return_counts=True)
            frequencies_dic = {l: f for l, f in zip(ls, fs)}
            frequencies_dicts.append(frequencies_dic)
            lemmas |= set(frequencies_dic.keys())

        if len(corpus) - (nr_splits - 1) * 10 > 0:
            # self.prober.max_length = len(c_corpus) + 1  # adjust size
            analyzed_corpuses = [self.prober(c) for c in corpus[nr_splits * 10:]]
            lemmatized_corpuses = [numpy.array([token.lemma_ for token in c]) for c in analyzed_corpuses]
            lemmatized_corpus = numpy.hstack(lemmatized_corpuses)
            ls, fs = numpy.unique(lemmatized_corpus, return_counts=True)
            frequencies_dic = {l: f for l, f in zip(ls, fs)}
            frequencies_dicts.append(frequencies_dic)
            lemmas |= set(frequencies_dic.keys())
            nr_analyzed_docs += len(analyzed_corpuses)
        lemmas = numpy.array(list(lemmas))

        for lemma in lemmas:
            self.token_frequencies[lemma] = 0
            for d in frequencies_dicts:
                self.token_frequencies[lemma] += d.get(lemma, 0)
        frequencies = numpy.array([self.token_frequencies[l] for l in lemmas])

        # build frequency curve...
        frequency_curve_arg = numpy.argsort(frequencies)
        frequency_curve = frequencies[frequency_curve_arg]
        frequency_curve_tokens = lemmas[frequency_curve_arg]

        # ...and split it into classes with a PLA
        pla_splits = pla(frequency_curve, error=32.)

        # assign tokens to a richness class
        self.token_classes = dict()
        self.richness_classes = list(range(len(pla_splits)))

        for i, (segment_start, segment_end, _, _) in enumerate(pla_splits):
            for token in frequency_curve_tokens[segment_start:segment_end]:
                if i > 0:
                    assert pla_splits[i][0] == pla_splits[i - 1][1]
                self.token_classes[token] = i
        self.token_classes[frequency_curve_tokens[-1]] = self.token_classes[frequency_curve_tokens[-2]]

    def tag(self, text: Sequence[str]) -> numpy.ndarray:
        """Probe the model on the given `text`.

        Args:
            text: The text to analyze. Either a text or an Iterable.

        Returns:
            Top-10 frequencies as computed by K-Means
        """
        if isinstance(text, str):
            analysis = self.prober(text)
            tokens = [token.lemma_ for i, token in enumerate(analysis) if not token.lemma_.isdigit()]
            tokens_classes = [self.token_classes[token] for token in tokens]
            frequencies = numpy.zeros(len(self.richness_classes), )
            classes, class_frequencies = numpy.unique(tokens_classes, return_counts=True)
            class_frequencies = class_frequencies / len(tokens)
            frequencies[classes] = class_frequencies
            return frequencies
        else:
            self.prober.max_length = max([len(t) for t in text])
            frequencies = numpy.array([self.tag(t) for t in tqdm(text)])
            frequencies = frequencies.reshape((len(text), len(self.richness_classes)))
            inertias, clusterings = [], []
            for k in range(2, self.n_authors*2):
                clustering = KMeans(n_clusters=k, max_iter=1000, random_state=self.seed)
                clusterings.append(clustering.fit(frequencies))
                inertias.append(clustering.inertia_)
            kn = KneeLocator(range(2, self.n_authors*2), inertias, curve='convex', direction='decreasing').knee
            self.clustering = clusterings[kn]
            centroids = self.clustering.cluster_centers_
            classes = self.clustering.predict(frequencies)
            self.frequency_classes = centroids

        return classes


def probe(model, data_df, task, label_encoder, output, probe_type: str = "pos", k: int = 5, seed=42, n_authors=5):
    prober = TransformerProber(model, n_authors, seed)
    logging.debug("Probing...")
    probing_results, probes = prober.probe(data_df, task, label_encoder, output=output, probe_type=probe_type, k=k)

    logging.debug("Dumping info...")
    with open(f"{output}.probe.{probe_type}.json", "w") as log:
        json.dump(probing_results, log)
    for author in probes:
        with open(f"{output}.author{author}_probe_{probe_type}.pickle", "wb") as log:
            pickle.dump(probes[author], log)
    return probing_results
