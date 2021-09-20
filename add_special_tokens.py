from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader
import re

# dataset = 'snli'
# fname = './snli_sentences_all.txt'
# fin = open(fname, 'r')
# snli_data = fin.readlines()
# fin.close()

# text_total = []

# for line in snli_data:
    
#     line = line.lower().strip()
#     line = re.sub(r"([.!?])", r" \1", line)
#     line = re.sub(r"[^a-zA-Z.!?]+", r" ", line)
#     line = line.split(' ')
#     # line = nlp.tokenizer(line)
#     text_total.append(line)


path = 'snli_w2v'

# snli_w2v = Word2Vec(sentences=text_total, vector_size=300, window=5, min_count=1)
# snli_w2v.save(path)


# pad_token = '<pad>'
# EOS_token = 'EOS'
# SOS_token = 'SOS'
# unk_token = '<unk>'

snli_w2v = Word2Vec.load(path)
print(snli_w2v.wv.get_vector("building", norm=True))

# special_tokens = [pad_token, EOS_token, SOS_token, unk_token]

# snli_w2v.build_vocab([special_tokens], update=True)
# snli_w2v.train([special_tokens], total_examples=1, epochs=1)
# snli_w2v.save(path)


