import random
import math
import numpy as np
from multiprocessing import Pool, cpu_count

from neural_lark.dataset import Example
from neural_lark.train_utils import logger

# adapted from rank_bm25

class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()


class BM25Retriever:
    """
    Wrapper to handle tokenizer/de-tokenizer for BM25
    """
    def __init__(self, examples, tokenizer=None, detokenizer=None, ex2doc=lambda x: x.source, **kwargs):
        self.examples = examples
        if tokenizer is None:
            self.tokenizer = lambda x: x.split(" ")
        else:
            self.tokenizer = tokenizer
        
        if detokenizer is None:
            self.detokenizer = lambda x: " ".join(x)
        else:
            self.detokenizer = detokenizer

        self.ex2doc = ex2doc
        self.corpus2id = {ex2doc(ex):idx for idx, ex in enumerate(examples)}
        self.corpus = [self.tokenizer(ex2doc(ex)) for ex in examples]
        self.bm25 = BM25Okapi(self.corpus, **kwargs)

    def retrieve(self, example, n=5, ret_id=True):
        toks = self.tokenizer(self.ex2doc(example))
        bm_exemplars_l = self.bm25.get_top_n(toks, self.corpus, n=n)
        bm_exemplar_ids = [self.corpus2id[self.detokenizer(toks)] for toks in bm_exemplars_l]
        retrieved_examples = [self.examples[idx] for idx in bm_exemplar_ids]
        if ret_id:
            return retrieved_examples, bm_exemplar_ids
        else:
            return retrieved_examples
    
    def retrieve_by_src(self, src_str, n=5):
        example = Example(idx=None, source=src_str, target="")
        return self.retrieve(example, n=n)

def setup_bm25(train_examples):
    bm25 = BM25Retriever(train_examples, k1=1.5, b=0.75)
    return bm25

def remove_example_from_retrieved_exemplars(input_example, exemplars):
    # this is useful to make sure that the input example is not in the exemplar during training
    new_exemplars = []
    for exemplar in exemplars:
        if exemplar.source != input_example.source:
            new_exemplars.append(exemplar)
    logger.debug(f"retriever: removed {len(exemplars) - len(new_exemplars)} exemplars, {len(new_exemplars)} left")
    return new_exemplars

def retrieve_fn_all(input_example, train_examples, batch_size):
    assert len(train_examples) <= batch_size
    exemplars = train_examples
    exemplars = remove_example_from_retrieved_exemplars(input_example, exemplars)
    return exemplars


def retrieve_fn_rand(input_example, train_examples, batch_size):
    exemplars = random.sample(train_examples, min(len(train_examples), batch_size * 2))
    exemplars = remove_example_from_retrieved_exemplars(input_example, exemplars)
    exemplars = exemplars[:batch_size]
    return exemplars


def retrieve_fn_bm25(input_example, train_examples, batch_size, bm25):
    exemplars = bm25.retrieve(input_example, batch_size * 2, False)
    exemplars = remove_example_from_retrieved_exemplars(input_example, exemplars)
    exemplars = list(reversed(exemplars))
    exemplars = exemplars[-batch_size:]
    return exemplars


retrieve_fn_dict = {
    "rand": retrieve_fn_rand,
    "bm25": retrieve_fn_bm25,
    "all": retrieve_fn_all,
}