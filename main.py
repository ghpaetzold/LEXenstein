from morphadorner import MorphAdornerToolkit
from generators import *
from selectors import *

m = MorphAdornerToolkit('./morph/')

kg = KauchakGenerator(m, './corpora/all.fastalign.pos.txt', './corpora/all.fastalign.forward.txt', './corpora/stop_words.txt')
subs = kg.getSubstitutions('./corpora/lexmturk_test.txt')
print('Kauchak:')
for k in subs.keys():
	print('\tTarget: ' + k)
	for sub in subs[k]:
		print('\t\t' + sub)

#yg = YamamotoGenerator(m, '65f439df-0149-4294-bd7f-2d317b3bd00e')
#subs = yg.getSubstitutions('./corpora/lexmturk_test.txt')
#print('Yamamoto:')
#for k in subs.keys():
#        print('\tTarget: ' + k)
#        for sub in subs[k]:
#                print('\t\t' + sub)

wsdselector = WSDSelector('lesk')
selected = wsdselector.selectCandidates(subs, './corpora/lexmturk_test.txt')
print(str(selected))

wsdselector = WSDSelector('random')
selected = wsdselector.selectCandidates(subs, './corpora/lexmturk_test.txt')
print(str(selected))

wsdselector = WSDSelector('first')
selected = wsdselector.selectCandidates(subs, './corpora/lexmturk_test.txt')
print(str(selected))

wsdselector = WSDSelector('leacho')
selected = wsdselector.selectCandidates(subs, './corpora/lexmturk_test.txt')
print(str(selected))

wsdselector = WSDSelector('wupalmer')
selected = wsdselector.selectCandidates(subs, './corpora/lexmturk_test.txt')
print(str(selected))

wsdselector = WSDSelector('path')
selected = wsdselector.selectCandidates(subs, './corpora/lexmturk_test.txt')
print(str(selected))

#mg = MerriamGenerator(m, 'c21550b0-418e-4a52-b85c-76587b8fdc2f')
#subs = mg.getSubstitutions('./corpora/lexmturk_test.txt')
#print('Merriam:')
#for k in subs.keys():
#        print('\tTarget: ' + k)
#        for sub in subs[k]:
#                print('\t\t' + sub)
#wg = WordnetGenerator(m)
#subs = wg.getSubstitutions('./corpora/lexmturk_test.txt')
#print('Wordnet:')
#for k in subs.keys():
#        print('\tTarget: ' + k)
#        for sub in subs[k]:
#                print('\t\t' + sub)

#bg = BiranGenerator(m, './corpora/wiki.vocab.txt', './corpora/wikisimple.vocab.txt', './corpora/wiki.5.bin.txt', './corpora/simplewiki.5.bin.txt')
#subs = bg.getSubstitutions('./corpora/lexmturk_test.txt')
#print('Biran:')
#for k in subs.keys():
#        print('\tTarget: ' + k)
#        for sub in subs[k]:
#                print('\t\t' + sub)
