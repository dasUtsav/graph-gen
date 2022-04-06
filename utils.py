from torchtext.data.metrics import bleu_score
import sys
from config import config
import time
import math
from nltk.tokenize.treebank import TreebankWordDetokenizer
from paddlenlp.metrics import Distinct

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(config['logs_path_config_2_78k'], "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def calc_time(start):
    now = time.time()
    end = now - start
    mins = math.floor(end / 60)
    end -= mins*60
    return end, mins

def get_pred_words(total_output, idx2word):

    # TODO: test get_pred_words

    decoded_words = []
    
    for sentence in total_output:
        preds = []
        for word in sentence:
            # if idx2word[word.item()] != '<pad>':
                # print("word: {}\n" .format(word.item()))
            preds.append(idx2word[word.item()])
            if idx2word[word.item()] == 'EOS':
                break
        # print("preds shape: {}\n" .format(len(preds)))
        decoded_words.append(preds)

    return decoded_words

def format_candidate(candidate):

    new = []
    for x in candidate:
        temp = []
        for word in x:
            if word == 'EOS':
                break
            temp.append(word)
        new.append(temp)

    return new


def calc_bleu(candidate, reference, flag = 'regular'):

    new = format_candidate(candidate)

    # print("candidate: ", new)

    bleu_1 = bleu_score(new, reference, weights=[1, 0, 0, 0])

    if flag == 'split':
        return bleu_1
    else:
        bleu_2 = bleu_score(new, reference, weights=[0.5, 0.5, 0, 0])
        bleu_3 = bleu_score(new, reference, weights=[0.34, 0.33, 0.33, 0])
        bleu_4 = bleu_score(new, reference, weights=[0.25, 0.25, 0.25, 0.25])

    return bleu_1, bleu_2, bleu_3, bleu_4

def calc_ppl(candidate, ken_model):

    ppl = 0

    new = format_candidate(candidate)

    avg_len = len(new)

    for x in new:
        x = TreebankWordDetokenizer().detokenize(x)
        ppl += ken_model.perplexity(x)

    ppl = ppl / avg_len

    return ppl

def calc_distinct(candidate):


    distinct_1 = Distinct(n_size = 1)
    distinct_2 = Distinct(n_size = 2)

    distinct_1_total = 0
    distinct_2_total = 0

    new = format_candidate(candidate)

    for x in new:
        distinct_1.add_inst(x)
        distinct_1_total += distinct_1.score()
        distinct_2.add_inst(x)
        distinct_2_total += distinct_2.score()

    return distinct_1_total, distinct_2_total
