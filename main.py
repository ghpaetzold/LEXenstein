from lexenstein.morphadorner import MorphAdornerToolkit
from lexenstein.generators import *
from lexenstein.selectors import *
from lexenstein.features import *

m = MorphAdornerToolkit('./morph/')

fe = FeatureEstimator()
fe.addLexiconFeature('./corpora/basic_words.txt')
fe.addLengthFeature()
fe.addSyllableFeature(m)
fe.addCollocationalFeature('./corpora/wiki.5.bin.txt', 2, 2)
fe.addSentenceProbabilityFeature('./corpora/wiki.5.bin.txt')
fe.addSenseCountFeature()
fe.addSynonymCountFeature()
fe.addHypernymCountFeature()
fe.addHyponymCountFeature()
fe.addMinDepthFeature()
fe.addMaxDepthFeature()
feats = fe.calculateFeatures('./corpora/lexmturk_test.txt')
print(str(feats))

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

voidselector = VoidSelector()
selected = voidselector.selectCandidates(subs, './corpora/lexmturk_test.txt')
print(str(selected))

biranselector = BiranSelector('./corpora/vectors.clean.txt')
selected = biranselector.selectCandidates(subs, './corpora/lexmturk_test.txt', 0.01, 0.75)
print(str(selected))

wordvecselector = WordVectorSelector('./corpora/word_vectors_all.bin')
selected = wordvecselector.selectCandidates(subs, './corpora/lexmturk_test.txt', proportion=0.75, stop_words_file='./corpora/stop_words.txt', onlyInformative=True, keepTarget=True, onePerWord=True)
print(str(selected))

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
