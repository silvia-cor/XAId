import nltk
import pickle
import os
from pathlib import Path


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


# tokenize text without punctuation
def tokenize_nopunct(text):
    unmod_tokens = nltk.word_tokenize(text)
    return [token.lower() for token in unmod_tokens if
            any(char.isalpha() for char in token)]  # checks whether all the chars are alphabetic


# ------------------------------------------------------------------------
# functions to split the text into fragments of (n_sentences) sentences
# ------------------------------------------------------------------------

# split text in fragments made of (n_sentences) sentences
def splitter(text, n_sentences):
    sentences = _split_sentences(text)
    return _group_sentences(sentences, n_sentences)


# split text into single sentences
def _split_sentences(text):
    # strip() removes blank spaces before and after string
    sentences = [t.strip() for t in nltk.tokenize.sent_tokenize(text) if t.strip()]
    for i, sentence in enumerate(sentences):
        mod_tokens = tokenize_nopunct(sentence)
        if len(mod_tokens) < 5:  # if the sentence is less than 5 words long, it is...
            if i < len(sentences) - 1:
                # combined with the next sentence
                sentences[i + 1] = sentences[i] + ' ' + sentences[i + 1]
            else:
                # or the previous one if it was the last sentence
                sentences[i - 1] = sentences[i - 1] + ' ' + sentences[i]
            sentences.pop(i)  # and deleted as a standalone sentence
    return sentences


# group sentences into fragments of window_size sentences
# not overlapping
def _group_sentences(sentences, window_size):
    new_fragments = []
    nbatches = len(sentences) // window_size
    if len(sentences) % window_size > 0:
        nbatches += 1
    for i in range(nbatches):
        offset = i * window_size
        new_fragments.append(' '.join(sentences[offset:offset + window_size]))
    return new_fragments
