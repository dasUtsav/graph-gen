import pickle
from gensim.models import Word2Vec

fname = 'snli_w2v'

fin = open('snli_gcn_word2idx.pkl', 'rb')
word2idx = pickle.load(fin)
fin.close()

snli_w2v = Word2Vec.load(fname)

word_vecs = snli_w2v.wv

print(word_vecs.get_vector("a"))