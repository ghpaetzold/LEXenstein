from lexenstein.morphadorner import MorphAdornerToolkit
from lexenstein.generators import *
from lexenstein.evaluators import *
from lexenstein.selectors import *
from lexenstein.features import *
from lexenstein.rankers import *

m = MorphAdornerToolkit('./morph/')


fe = FeatureEstimator()
fe.addLexiconFeature('./corpora/basic_words.txt', 'Simplicity')
fe.addLengthFeature('Complexity')
fe.addSyllableFeature(m, 'Complexity')
fe.addCollocationalFeature('./corpora/wiki.5.bin.txt', 2, 2, 'Simplicity')
fe.addSentenceProbabilityFeature('./corpora/wiki.5.bin.txt', 'Simplicity')
fe.addSenseCountFeature('Simplicity')
fe.addSynonymCountFeature('Simplicity')
fe.addHypernymCountFeature('Simplicity')
fe.addHyponymCountFeature('Simplicity')
fe.addMinDepthFeature('Complexity')
fe.addMaxDepthFeature('Complexity')
#feats = fe.calculateFeatures('./corpora/lexmturk_test.txt')

fe = FeatureEstimator()
fe.addCollocationalFeature('./corpora/simplewiki.5.bin.txt', 0, 0, 'Complexity')
fe.addLengthFeature('Complexity')
fe.addSenseCountFeature('Simplicity')
mr = MetricRanker(fe)
freqs = mr.getRankings('./corpora/lexmturk_all.txt', 0)
lengs = mr.getRankings('./corpora/lexmturk_all.txt', 1)
senss = mr.getRankings('./corpora/lexmturk_all.txt', 2)

test = RankerEvaluator()
t1, t2, t3, r1, r2, r3 = test.rankerIntrinsicEvaluation('./corpora/lexmturk_all.txt', senss)
print('t1: ' + str(t1))
print('r1: ' + str(r1))
print('r2: ' + str(r2))
print('r3: ' + str(r3))


print('###################################################################################################################')
svmr = SVMRanker(fe, '/export/tools/svm-rank/')
svmr.getFeaturesFile('./corpora/lexmturk_test.txt', './corpora/lexmturk_test_svmfeatures.txt')
svmr.getTrainingModel('./corpora/lexmturk_test_svmfeatures.txt', 0.0001, 0.01, 0, './corpora/lexmturk_test_svmmodel.txt')
svmr.getScoresFile('./corpora/lexmturk_test_svmfeatures.txt', './corpora/lexmturk_test_svmmodel.txt', './corpora/lexmturk_test_svmscores.txt')
rankings = svmr.getRankings('./corpora/lexmturk_test_svmfeatures.txt', './corpora/lexmturk_test_svmscores.txt')
print(str(rankings))

br = BoundaryRanker(fe)
br.trainRanker('./corpora/lexmturk_test.txt', 1, 'modified_huber', 'l1', 0.1, 0.1, 0.001)
rankings = br.getRankings('./corpora/lexmturk_test.txt')
print('Boundary')
print(str(rankings))

re = RankerEvaluator()
t1, t2, t3, r1, r2, r3 = re.rankerIntrinsicEvaluation('./corpora/lexmturk_test.txt', rankings)
print(str(t1) + ' ' + str(t2) + ' ' + str(t3) + ' ' + str(r1) + ' ' + str(r2) + ' ' + str(r3))

kg = KauchakGenerator(m, './corpora/all.fastalign.pos.txt', './corpora/all.fastalign.forward.txt', './corpora/stop_words.txt')
subs = kg.getSubstitutions('./corpora/lexmturk_test.txt')
print('Kauchak:')
for k in subs.keys():
	print('\tTarget: ' + k)
	for sub in subs[k]:
		print('\t\t' + sub)

ge = GeneratorEvaluator()
precision, recall = ge.evaluateGenerator('./corpora/lexmturk_test.txt', subs)
print('Precision: ' + str(precision) + ', Recall: ' + str(recall))

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

se = SelectorEvaluator()
precision, recall = se.evaluateSelector('./corpora/lexmturk_test.txt', selected)
print('Precision: ' + str(precision) + ', Recall: ' + str(recall))

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
