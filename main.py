from morphadorner import MorphAdornerToolkit
from generators import BiranGenerator

m = MorphAdornerToolkit('./morph/')
bg = BiranGenerator('./corpora/wiki.vocab.txt', './corpora/wikisimple.vocab.txt', './corpora/wiki.5.bin.txt', './corpora/simplewiki.5.bin.txt', m)
subs = bg.getSubstitutions('./corpora/lexmturk_test.txt')
print(str(subs))
