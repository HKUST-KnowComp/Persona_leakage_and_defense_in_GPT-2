"""
Copied from nltk.ngrams().
data_dir = sys.argv[1]
Use bertscore+dist+bleu score to evaluate generation quality
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from itertools import chain
import nltk
import bert_score
from bert_score import score
import numpy as np
import pprint
import json
import sys
import logging
import logging.handlers


logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
f_handler = logging.FileHandler(sys.argv[2]+'.log')
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s -  %(message)s"))
logger.addHandler(f_handler)





__all__ = ["ngrams"]

def load_data(data_dir):
    with open(data_dir, 'r') as f:
        #list of dicts
        data = json.load(f)
    cands_list = []
    refs_list = []
    for i,d in enumerate(data):
        cands_list += d['output']
        refs_list += d['ref']
    return data,cands_list,refs_list




def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    """
    Returns a padded sequence of items before ngram extraction.
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']
    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:
        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        #print('his: ',history)
        yield tuple(history)
        del history[0]

def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    #print(distinct_ngrams)
    return len(distinct_ngrams) / len(sentence)



#   hypothesis = [sentence.split() for sentence in f.readlines()]
if __name__ == '__main__':
    data_dir = sys.argv[1]
    data,cands_list,refs_list = load_data(data_dir)
    dist1_score = []
    dist2_score = []
    for sentence in cands_list:
        split_sent = sentence.split()
        dist1_score.append(distinct_n_sentence_level(split_sent, 1))
        dist2_score.append(distinct_n_sentence_level(split_sent, 2))

    
    # bleu
    cands_list_bleu = [sentence.split() for sentence in cands_list] 
    refs_list_bleu = [[sentence.split()] for sentence in refs_list]
    bleu_score = nltk.translate.bleu_score.corpus_bleu(refs_list_bleu, cands_list_bleu,weights=(0.25, 0.25, 0.25, 0.25)) 
    bleu_score_1 = nltk.translate.bleu_score.corpus_bleu(refs_list_bleu, cands_list_bleu,weights=(1, 0, 0, 0)) 
    bleu_score_2 = nltk.translate.bleu_score.corpus_bleu(refs_list_bleu, cands_list_bleu,weights=(0.5, 0.5, 0, 0)) 

    # bert score
    P, R, F1 = score(cands_list, refs_list, lang='en', verbose=False)

    logger.info(f'dist 1: {np.mean(dist1_score)} ')
    logger.info(f'dist 2: {np.mean(dist2_score)} ')
    logger.info(f'bleu1 : {bleu_score_1}')
    logger.info(f'bleu2 : {bleu_score_2}')
    logger.info(f'bleu : {bleu_score}')
    logger.info(f'bert score P : {P.mean()}')
    logger.info(f'bert score R : {R.mean()}')
    logger.info(f'bert score F1 : {F1.mean()}')



    # sentence = 'the cat sat on the mat'
    # sentence = sentence.split()
    # logger.info(sentence)
    # logger.info(distinct_n_sentence_level(sentence, 2))



    # reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
    # candidate = ['the', 'fast', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
    # bleu_score = nltk.translate.bleu_score.sentence_bleu(reference, candidate,weights=(0.25, 0.25, 0.25, 0.25))
    # logger.info(bleu_score)
    



    # #### bert score

    # cands = ['28-year-old chef found dead in San Francisco mall']
    # refs = ['28-Year-Old Chef Found Dead at San Francisco Mall']
    # P, R, F1 = score(cands, refs, lang='en', verbose=True)
    # logger.info(f'bert socre has P: {P} recall: {R} and F1 score {F1}')