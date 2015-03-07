from morphadorner import MorphAdornerToolkit
from generators import *

m = MorphAdornerToolkit('./morph/')

yg = YamamotoGenerator(m, '65f439df-0149-4294-bd7f-2d317b3bd00e')
subs = yg.getSubstitutions('./corpora/lexmturk_test.txt')
print(str(subs))
print(str(len(subs.keys())))

mg = MerriamGenerator(m, 'c21550b0-418e-4a52-b85c-76587b8fdc2f')
subs = mg.getSubstitutions('./corpora/lexmturk_test.txt')
print(str(subs))
print(str(len(subs.keys())))

wg = WordnetGenerator(m)
subs = wg.getSubstitutions('./corpora/lexmturk_test.txt')
print(str(subs))
print(str(len(subs.keys())))

bg = BiranGenerator('./corpora/wiki.vocab.txt', './corpora/wikisimple.vocab.txt', './corpora/wiki.5.bin.txt', './corpora/simplewiki.5.bin.txt', m)
subs = bg.getSubstitutions('./corpora/lexmturk_test.txt')
print(str(subs))
print(str(len(subs.keys())))
