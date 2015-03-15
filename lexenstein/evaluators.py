

class GeneratorEvaluator:

	def evaluateGenerator(self, victor_corpus, substitutions):
		#Initialize variables:
		precisionc = 0
		precisiont = 0
		recallc = 0
		recallt = 0
		
		#Calculate measures:
		f = open(victor_corpus)
		for line in f:
			data = line.strip().split('\t')
			target = data[1].strip()
			items = data[3:len(data)]
			candidates = [item.strip().split(':')[1].strip() for item in items]
			if target in substitutions.keys():
				overlap = set(candidates).intersection(set(substitutions[target]))
				recallc += len(overlap)
				if len(overlap)>0:
					precisionc += 1
			precisiont += 1
			recallt += len(candidates)
		f.close()
		
		#Return measures:
		return float(precisionc)/float(precisiont), float(recallc)/float(recallt)

class SelectorEvaluator:

	def evaluateSelector(self, victor_corpus, substitutions):
		#Initialize variables:
		precisionc = 0
		precisiont = 0
		recallc = 0
		recallt = 0
		
		#Calculate measures:
		f = open(victor_corpus)
		index = -1
		for line in f:
			index += 1
		
			data = line.strip().split('\t')
			target = data[1].strip()
			items = data[3:len(data)]
			candidates = [item.strip().split(':')[1].strip() for item in items]
			
			selected = substitutions[index]
			if len(selected)>0:
				overlap = set(candidates).intersection(set(selected))
				recallc += len(overlap)
				if len(overlap)>0:
					precisionc += 1
			precisiont += 1
			recallt += len(candidates)
		f.close()

		#Return measures:
		return float(precisionc)/float(precisiont), float(recallc)/float(recallt)

class RankerEvaluator:

	def rankerIntrinsicEvaluation(self, victor_corpus, rankings):
		#Initialize variables:
		total1 = 0
		total2 = 0
		total3 = 0
		corrects1 = 0
		corrects2 = 0
		corrects3 = 0
		recall1 = 0
		recall2 = 0
		recall3 = 0
		trecall1 = 0
		trecall2 = 0
		trecall3 = 0
	
		#Read data:
		index = -1
		f = open(victor_corpus)
		for data in f:
			index += 1
			line = data.strip().split('\t')
			gold_rankings = {}
			for subst in line[3:len(line)]:
				subst_data = subst.strip().split(':')
				word = subst_data[1].strip()
				ranking = int(subst_data[0].strip())
				gold_rankings[word] = ranking
			ranked_candidates = rankings[index]

			first = gold_rankings[ranked_candidates[0]]
			
			print('')
			print('Gold:')
			for key in gold_rankings.keys():
				print(key + ': ' + str(gold_rankings[key]))
			print('Cands: ' + str(ranked_candidates))
	
			#Get recall sets:
			set1, set2, set3 = self.getRecallSets(line[3:len(line)])
			rankedset1 = set([])
			rankedset2 = set([])
			rankedset3 = set([])
	
			#Calculate TRank 1:
			if first==1:
				rankedset1 = set([ranked_candidates[0]])
				corrects1 += 1
			recall1 += len(rankedset1.intersection(set1))
			trecall1 += len(set1)
			total1 += 1
	
			#Calculate TRank 2:
			if len(gold_rankings.keys())>2:
				rankedset2 = rankedset1.union(set([ranked_candidates[1]]))
				recall2 += len(rankedset2.intersection(set2))
				trecall2 += len(set2)
				if first<=2:
					corrects2 += 1
				total2 += 1
	
			#Calculate TRank 3:
			if len(gold_rankings.keys())>3:
				rankedset3 = rankedset2.union(set([ranked_candidates[2]]))
				recall3 += len(rankedset3.intersection(set3))
				trecall3 += len(set3)
				if first<=3:
					corrects3 += 1
				total3 += 1
	
		#Return measures:
		return float(corrects1)/float(total1), float(corrects2)/float(total2), float(corrects3)/float(total3), float(recall1)/float(trecall1), float(recall2)/float(trecall2), float(recall3)/float(trecall3)
		
	def getRecallSets(self, substs):
		result1 = set([])
		result2 = set([])
		result3 = set([])
		for subst in substs:
			datasubst = subst.strip().split(':')
			word = datasubst[1].strip()
			index = datasubst[0].strip()
			if index=="1":
				result1.add(word)
				result2.add(word)
				result3.add(word)
			elif index=="2":
				result2.add(word)
				result3.add(word)
			elif index=="3":
				result3.add(word)
		return result1, result2, result3

class PipelineEvaluator:

	def evaluatePipeline(self, victor_corpus, rankings):
		#Initialize counting variables:
		total = 0
		totalc = 0
		precise = 0
		
		#Read victor corpus:
		f = open(victor_corpus)
		for i in range(rankings):
			#Get gold candidates:
			data = f.readline().strip().split('\t')
			target = data[1].strip()
			data = data[3:len(data)]
			gold_subs = set([item.strip().split(':')[1].strip() for item in data])
			
			#Get highest ranked candidate:
			first = rankings[i][0]
			
			#Check if it is in gold candidates:
			total += 1
			if first!=target:
				totalc += 1
				if first in gold_subs:
					precise += 1
		
		#Return metrics:
		return float(precise)/float(totalc), float(precise)/float(total), float(totalc)/float(total)