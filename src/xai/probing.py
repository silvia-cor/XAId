from __future__ import annotations

import logging
import json
from abc import abstractmethod
from typing import List, Iterable, Dict, Tuple, Sequence

import sys
import numpy
import pandas
import ray
import spacy as spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tune_sklearn import TuneGridSearchCV as SearchAlgorithm
from kneed import KneeLocator
from cltk.prosody.lat.macronizer import Macronizer
from cltk.prosody.lat.scanner import Scansion
from tqdm import tqdm

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
    def __init__(self, model, task, seed, n_jobs):
        self.model = model
        self.task = task
        self.seed = seed
        self.n_jobs = n_jobs

    def probe(self, data_df, probe_type: str = "pos", k: int = 5, min_len: int = 3, max_len: int = 5):
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

        probe_types = ['pos', 'sq', 'word_lengths', 'function_words', 'genre']
        assert probe_type in probe_types, f'Unknown probe_type, supported probe_type: {probe_types}'

        documents = data_df.text.values
        authors = data_df.author.values

        ray.init()

        # assign tags to documents
        logging.debug("\tBuilding ground truth for probing...")
        logging.debug("\t\tComputing tags...")
        if probe_type == 'pos':
            tags = POSTagger().tag(documents)
        elif probe_type == 'sq':
            tags = SQTagger().tag(documents)
        elif probe_type == "word_lengths":
            tags = WordLengthFrequencyTagger(self.seed, len(numpy.unique(numpy.array(authors)))).tag(documents)
        elif probe_type == 'function_words':
            tags = FunctionWordsFrequencyTagger(self.seed, len(numpy.unique(numpy.array(authors)))).tag(documents)
        else:
            pass

        probes_results = dict()

        if probe_type in ["pos", 'sq']:
            # computing POS or SQ tags, and select k chains
            logging.debug(f"\t\t\tComputing top {k} chains....")
            analyzer = 'word' if probe_type == 'pos' else 'char'
            top_chains = self._top_k_chains(tags, authors, k, min_len=min_len, max_len=max_len, analyzer=analyzer)
            for target_chain in top_chains:
                probing_labels = [self._is_subchain_in_chain(target_chain, t.lower()) for t in tags]
                _, configuration, validation = self._launch_probe(documents, probing_labels)
                probes_results[target_chain] = (configuration, validation)
        elif probe_type in ['word_lengths', 'function_words']:
            probing_labels = numpy.array(tags)
            _, configuration, validation = self._launch_probe(documents, probing_labels)
            probes_results['multiclass'] = (configuration, validation)
        else:
            probing_labels = [0 if label == 'lit' else 1 for label in numpy.array(data_df.sub_corpus.values)]
            _, configuration, validation = self._launch_probe(documents, probing_labels)
            probes_results['sub_corpus'] = (configuration, validation)
        ray.shutdown()
        return probes_results

    def _launch_probe(self, data, probing_labels) -> Tuple[
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
        embedding_data = self.model.encode(data_df, self.task)
        embedding_data = numpy.array(embedding_data)
        embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(embedding_data, probing_labels,
                                                                                        stratify=probing_labels,
                                                                                        test_size=0.1,
                                                                                        random_state=self.seed)
        # Build prober
        hyperparameters = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "penalty": ["l2"],
            "random_state": [self.seed],
            "class_weight": ['balanced']
        }
        logging.debug("\t\t\tFitting prober...")
        search = SearchAlgorithm(LogisticRegression(max_iter=100000,
                                                    warm_start=True),
                                 param_grid=hyperparameters,
                                 cv=3, early_stopping=False,
                                 refit=True, n_jobs=self.n_jobs)
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

    def _top_k_chains(self, tagged_docs: List, labels: List, k: int = 5, min_len: int = 3, max_len: int = 10,
                      analyzer='word') -> List[
        List]:
        """
        Find the most characteristic chains
        Args:
            chains: The probe
            k: How many sequences to select, defaults to 5.
            min_len: Minimum length of chain.
            max_len: Maximum length of chain.

        Returns:
            The top probing sequences.
        """
        chain_counter = CountVectorizer(analyzer=analyzer, ngram_range=(min_len, max_len),
                                        token_pattern=r"(?u)\b\w+\b").fit(tagged_docs)
        counted_chains = chain_counter.transform(tagged_docs)
        selector = SelectKBest(chi2, k=k).fit(counted_chains, labels)
        top_chains_idx = selector.get_support(indices=True)
        top_chains = [chain_counter.get_feature_names_out()[i] for i in top_chains_idx]
        return top_chains

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


class POSTagger(Tagger):
    def __init__(self):
        self.tagger = spacy.load("la_core_web_lg")

    def tag(self, text):
        """
        Probe the model on the given `text`.
        Args:
            text: The text to analyze. Either a text or an Iterable.

        Returns:
            A list of POS, if a single piece of text is provided, or a list of lists of POS,
            if an Iterable is provided.
        """
        if isinstance(text, str):
            nlp = self.tagger(text)
            pos_tags = [token.pos_ for token in nlp]
            return ' '.join(pos_tags)
        else:
            pos = []
            for i, t in enumerate(tqdm(text, 'pos-tagging', total=len(text))):
                pos.append(self.tag(t))
        return pos


class SQTagger(Tagger):
    def __init__(self):
        self.macronizer = Macronizer('tag_ngram_123_backoff')
        self.tagger = Scansion(clausula_length=100000)

    def tag(self, text):
        if isinstance(text, str):
            sq = self.tagger.scan_text(self.macronizer.macronize_text(text))
            return ' '.join(sq)
        else:
            sq = []
            for i, t in enumerate(tqdm(text, 'metric-scansion', total=len(text))):
                sq.append(self.tag(t))
        return sq


class WordLengthFrequencyTagger(Tagger):
    def __init__(self, seed, n_authors):
        self.tagger = spacy.load("la_core_web_lg")
        self.seed = seed
        self.n_authors = n_authors

    def tag(self, text):
        """
        Probe the model on the given `text`.
        Args:
            text: The text to analyze. Either a text or an Iterable.

        Returns:
            Top-10 frequencies as computed by K-Means
        """
        if isinstance(text, str):
            nlp = self.tagger(text)
            lengths = [len(token.text) for token in nlp if not token.text.isdigit()]
            return lengths
        else:
            lengths_docs = [self.tag(t) for t in text]
            len_max = max(map(max, lengths_docs))  # find longest word in dataset
            histograms = [numpy.histogram(lengths_doc, range=(1., float(len_max)), density=True) for lengths_doc in
                          lengths_docs]  # compute histogram, bin_edges for each doc
            cdf = [numpy.cumsum(histo * numpy.diff(bin_edges)) for histo, bin_edges in
                   histograms]  # compute the Cumulative Distribution Function
            # map frequencies to class through clustering
            inertias, clusterings = [], []
            for k in range(2, self.n_authors * 2):
                clustering = KMeans(n_clusters=k, max_iter=10000, random_state=self.seed)
                clusterings.append(clustering.fit(cdf))
                inertias.append(clustering.inertia_)
            kn = KneeLocator(range(2, self.n_authors * 2), inertias, curve='convex', direction='decreasing').knee
            clustering = clusterings[kn]
            class_labels = clustering.predict(cdf)
        return class_labels


class FunctionWordsFrequencyTagger(Tagger):
    def __init__(self, seed, n_authors):
        self.tagger = spacy.load("la_core_web_lg")
        self.seed = seed
        self.n_authors = n_authors
        self.latin_function_words = ['et', 'in', 'de', 'ad', 'non', 'ut', 'cum', 'per', 'a', 'sed', 'que', 'quia', 'ex',
                                     'sic',
                                     'si', 'etiam', 'idest', 'nam', 'unde', 'ab', 'uel', 'sicut', 'ita', 'enim',
                                     'scilicet',
                                     'nec',
                                     'pro', 'autem', 'ibi', 'dum', 'uero', 'tamen', 'inter', 'ideo', 'propter',
                                     'contra',
                                     'sub',
                                     'quomodo', 'ubi', 'super', 'iam', 'tam', 'hec', 'post', 'quasi', 'ergo', 'inde',
                                     'e',
                                     'tunc',
                                     'atque', 'ac', 'sine', 'nisi', 'nunc', 'quando', 'ne', 'usque', 'siue', 'aut',
                                     'igitur',
                                     'circa', 'quidem', 'supra', 'ante', 'adhuc', 'seu', 'apud', 'olim', 'statim',
                                     'satis',
                                     'ob',
                                     'quoniam', 'postea', 'nunquam', 'semper', 'licet', 'uidelicet', 'quoque', 'uelut',
                                     'quot']

    def tag(self, text):
        """
        Probe the model on the given `text`.
        Args:
            text: The text to analyze. Either a text or an Iterable.

        Returns:
            Top-10 frequencies as computed by K-Means
        """
        if isinstance(text, str):
            nlp = self.tagger(text.lower())
            tokens = [token.text for token in nlp if not token.text.isdigit()]
            fw_freqs = numpy.array([tokens.count(fw) for fw in self.latin_function_words])
            fw_freqs = fw_freqs / len(tokens)  # normalize
            return fw_freqs
        else:
            fw_freqs_docs = [self.tag(t) for t in text]
            # map frequencies to class through clustering
            inertias, clusterings = [], []
            for k in range(2, self.n_authors * 2):
                clustering = KMeans(n_clusters=k, max_iter=10000, random_state=self.seed)
                clusterings.append(clustering.fit(fw_freqs_docs))
                inertias.append(clustering.inertia_)
            kn = KneeLocator(range(2, self.n_authors * 2), inertias, curve='convex', direction='decreasing').knee
            clustering = clusterings[kn]
            class_labels = clustering.predict(fw_freqs_docs)
        return class_labels


def probe(model, data_df, task, output, probe_type, k, min_len, max_len, seed, n_jobs):
    prober = TransformerProber(model, task, seed, n_jobs)
    probing_results = prober.probe(data_df, probe_type=probe_type, k=k, min_len=min_len, max_len=max_len)
    logging.debug("Dumping info...")
    with open(f"{output}.probing.{probe_type}.json", "w") as log:
        json.dump(probing_results, log)
    return probing_results

# class RichnessFrequencyTagger(Tagger):
#     """
#     Args:
#         prober: spacy.lang.en.English: The analyzer.
#         token_frequencies: Dict[str,int]: A dictionary token => frequency
#         richness_classes: List[int]: Classes of token richness
#         token_classes: Dict[str, int]: A dictionary token => richness class
#         centroids: numpy.ndarray: After tagging some text, these are the centroids of each frequency distribution
#     """
#
#     def __init__(self, corpus: List[str], seed, n_authors):
#         self.prober = spacy.load("en_core_web_sm")
#         self.token_frequencies = defaultdict(int)
#         self.clustering = None
#         self.prober.max_length = 1000000
#         self.seed = seed
#         self.n_authors = n_authors
#
#         # build corpus frequencies
#         frequencies_dicts = list()
#         lemmas = set()
#         nr_splits = (len(corpus) // 10)
#
#         # split into chunks to avoid Outofmemory
#         nr_analyzed_docs = 0
#         for i in range(nr_splits):
#             analyzed_corpuses = [self.prober(c) for c in corpus[i * 10:(i + 1) * 10]]
#             lemmatized_corpuses = [numpy.array([token.lemma_ for token in c]) for c in analyzed_corpuses]
#             lemmatized_corpus = numpy.hstack(lemmatized_corpuses)
#             nr_analyzed_docs += len(analyzed_corpuses)
#
#             ls, fs = numpy.unique(lemmatized_corpus, return_counts=True)
#             frequencies_dic = {l: f for l, f in zip(ls, fs)}
#             frequencies_dicts.append(frequencies_dic)
#             lemmas |= set(frequencies_dic.keys())
#
#         if len(corpus) - (nr_splits - 1) * 10 > 0:
#             # self.prober.max_length = len(c_corpus) + 1  # adjust size
#             analyzed_corpuses = [self.prober(c) for c in corpus[nr_splits * 10:]]
#             lemmatized_corpuses = [numpy.array([token.lemma_ for token in c]) for c in analyzed_corpuses]
#             lemmatized_corpus = numpy.hstack(lemmatized_corpuses)
#             ls, fs = numpy.unique(lemmatized_corpus, return_counts=True)
#             frequencies_dic = {l: f for l, f in zip(ls, fs)}
#             frequencies_dicts.append(frequencies_dic)
#             lemmas |= set(frequencies_dic.keys())
#             nr_analyzed_docs += len(analyzed_corpuses)
#         lemmas = numpy.array(list(lemmas))
#
#         for lemma in lemmas:
#             self.token_frequencies[lemma] = 0
#             for d in frequencies_dicts:
#                 self.token_frequencies[lemma] += d.get(lemma, 0)
#         frequencies = numpy.array([self.token_frequencies[l] for l in lemmas])
#
#         # build frequency curve...
#         frequency_curve_arg = numpy.argsort(frequencies)
#         frequency_curve = frequencies[frequency_curve_arg]
#         frequency_curve_tokens = lemmas[frequency_curve_arg]
#
#         # ...and split it into classes with a PLA
#         pla_splits = pla(frequency_curve, error=32.)
#
#         # assign tokens to a richness class
#         self.token_classes = dict()
#         self.richness_classes = list(range(len(pla_splits)))
#
#         for i, (segment_start, segment_end, _, _) in enumerate(pla_splits):
#             for token in frequency_curve_tokens[segment_start:segment_end]:
#                 if i > 0:
#                     assert pla_splits[i][0] == pla_splits[i - 1][1]
#                 self.token_classes[token] = i
#         self.token_classes[frequency_curve_tokens[-1]] = self.token_classes[frequency_curve_tokens[-2]]
#
#     def tag(self, text: Sequence[str]) -> numpy.ndarray:
#         """Probe the model on the given `text`.
#
#         Args:
#             text: The text to analyze. Either a text or an Iterable.
#
#         Returns:
#             Top-10 frequencies as computed by K-Means
#         """
#         if isinstance(text, str):
#             analysis = self.prober(text)
#             tokens = [token.lemma_ for i, token in enumerate(analysis) if not token.lemma_.isdigit()]
#             tokens_classes = [self.token_classes[token] for token in tokens]
#             frequencies = numpy.zeros(len(self.richness_classes), )
#             classes, class_frequencies = numpy.unique(tokens_classes, return_counts=True)
#             class_frequencies = class_frequencies / len(tokens)
#             frequencies[classes] = class_frequencies
#             return frequencies
#         else:
#             self.prober.max_length = max([len(t) for t in text])
#             frequencies = numpy.array([self.tag(t) for t in tqdm(text)])
#             frequencies = frequencies.reshape((len(text), len(self.richness_classes)))
#             inertias, clusterings = [], []
#             for k in range(2, self.n_authors * 2):
#                 clustering = KMeans(n_clusters=k, max_iter=1000, random_state=self.seed)
#                 clusterings.append(clustering.fit(frequencies))
#                 inertias.append(clustering.inertia_)
#             kn = KneeLocator(range(2, self.n_authors * 2), inertias, curve='convex', direction='decreasing').knee
#             self.clustering = clusterings[kn]
#             centroids = self.clustering.cluster_centers_
#             classes = self.clustering.predict(frequencies)
#             self.frequency_classes = centroids
#
#         return classes

# def _top_k_items(self, probes: List[List], k: int = 5) -> List[str]:
#     """
#     Compute the most characteristic roots (lemmas/stems) among `probes`
#     Args:
#         probes: The probes to analyze
#         k: The number of top roots to return. Defaults to 5
#
#     Returns:
#         The top-k roots
#     """
#     flat_probes = numpy.array([item for document_items in probes for item in document_items])
#     roots, frequencies = numpy.unique(flat_probes, return_counts=True)
#
#     return roots[frequencies.argsort()[-k:]]
