from scipy.stats import *

class IdentifierEvaluator:

	def evaluateIdentifier(self, cwictor_corpus, predicted_labels):
		"""
		Performs an intrinsic evaluation of a Complex Word Identification approach.
	
		@param cwictor_corpus: Path to a training corpus in CWICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param predicted_labels: A vector containing the predicted binary labels of each instance in the CWICTOR corpus.
		@return: Accuracy, Precision, Recall and the F-score between Accuracy and Recall for the substitutions provided as input with respect to the gold-standard in the VICTOR corpus.
		For more information on how the metrics are calculated, please refer to the LEXenstein Manual.
		"""

		gold = [int(line.strip().split('\t')[3]) for line in open(cwictor_corpus)]
		
		#Initialize variables:
		accuracyc = 0.0
		accuracyt = 0.0
		precisionc = 0.0
		precisiont = 0.0
		recallc = 0.0
		recallt = 0.0
		
		#Calculate measures:
		for i in range(0, len(gold)):
			gold_label = gold[i]
			predicted_label = predicted_labels[i]
			if gold_label==predicted_label:
				accuracyc += 1
				if gold_label==1:
					recallc += 1
					precisionc += 1
			if gold_label==1:
				recallt += 1
			if predicted_label==1:
				precisiont += 1
			accuracyt += 1

		try:
			accuracy = accuracyc / accuracyt
		except ZeroDivisionError:
			accuracy = 0
		try:
			precision = precisionc / precisiont
		except ZeroDivisionError:
			precision = 0
		try:
			recall = recallc / recallt
		except ZeroDivisionError:
			recall = 0
		fmean = 0
		gmean = 0
		
		try:
			fmean = 2 * (precision * recall) / (precision + recall)
			gmean = 2 * (accuracy * recall) / (accuracy + recall)
		except ZeroDivisionError:
			fmean = 0
			gmean = 0
		
		#Return measures:
		return accuracy, precision, recall, fmean, gmean
		
class GeneratorEvaluator:

	def evaluateGenerator(self, victor_corpus, substitutions):
		"""
		Performs an intrinsic evaluation of a Substitution Generation approach.
	
		@param victor_corpus: Path to a training corpus in VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param substitutions: A dictionary that assigns target complex words to sets of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		@return: Values for Potential, Precision, Recall and F-measure for the substitutions provided as input with respect to the gold-standard in the VICTOR corpus.
		For more information on how the metrics are calculated, please refer to the LEXenstein Manual.
		"""
		
		#Initialize variables:
		potentialc = 0
		potentialt = 0
		precisionc = 0
		precisiont = 0
		recallt = 0
		
		#Calculate measures:
		f = open(victor_corpus)
		for line in f:
			data = line.strip().split('\t')
			target = data[1].strip()
			items = data[3:len(data)]
			candidates = set([item.strip().split(':')[1].strip() for item in items])
			if target in substitutions:
				overlap = candidates.intersection(set(substitutions[target]))
				precisionc += len(overlap)
				if len(overlap)>0:
					potentialc += 1
				precisiont += len(substitutions[target])
			potentialt += 1
			recallt += len(candidates)
		f.close()
		
		potential = float(potentialc)/float(potentialt)
		precision = float(precisionc)/float(precisiont)
		recall = float(precisionc)/float(recallt)
		fmean = 0.0
		if precision==0.0 and recall==0.0:
			fmean = 0.0
		else:
			fmean = 2*(precision*recall)/(precision+recall)
			
		#Return measures:
		return potential, precision, recall, fmean

class SelectorEvaluator:

	def evaluateSelector(self, victor_corpus, substitutions):
		"""
		Performs an intrinsic evaluation of a Substitution Selection approach.
	
		@param victor_corpus: Path to a training corpus in VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param substitutions: A vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		@return: Values for Potential, Recall, Precision and F-measure for the substitutions provided as input with respect to the gold-standard in the VICTOR corpus.
		For more information on how the metrics are calculated, please refer to the LEXenstein Manual.
		"""
	
		#Initialize variables:
		potentialc = 0
		potentialt = 0
		precisionc = 0
		precisiont = 0
		recallt = 0
		
		#Calculate measures:
		f = open(victor_corpus)
		index = -1
		for line in f:
			index += 1
		
			data = line.strip().split('\t')
			target = data[1].strip()
			items = data[3:len(data)]
			candidates = set([item.strip().split(':')[1].strip() for item in items])
			
			selected = substitutions[index]
			if len(selected)>0:
				overlap = candidates.intersection(set(selected))
				precisionc += len(overlap)
				if len(overlap)>0:
					potentialc += 1
			potentialt += 1
			precisiont += len(selected)
			recallt += len(candidates)
		f.close()

		potential = float(potentialc)/float(potentialt)
		precision = float(precisionc)/float(precisiont)
		recall = float(precisionc)/float(recallt)
		fmean = 0.0
		if precision==0.0 and recall==0.0:
			fmean = 0.0
		else:
			fmean = 2*(precision*recall)/(precision+recall)
			
		#Return measures:
		return potential, precision, recall, fmean

class RankerEvaluator:

	def evaluateRanker(self, victor_corpus, rankings):
		"""
		Performs an intrinsic evaluation of a Substitution Ranking approach.
	
		@param victor_corpus: Path to a training corpus in VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param rankings: A vector of size N, containing a set of ranked substitutions for each instance in the VICTOR corpus.
		@return: Values for TRank-at-1/2/3, Recall-at-1/2/3, Spearman and Pearson correlation for the substitutions provided as input with respect to the gold-standard in the VICTOR corpus.
		For more information on how the metrics are calculated, please refer to the LEXenstein Manual.
		"""
		
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
		all_gold = []
		all_ranks = []
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

			for i in range(0, len(ranked_candidates)):
				word = ranked_candidates[i]
				all_gold.append(gold_rankings[word])
				all_ranks.append(i)

			first = gold_rankings[ranked_candidates[0]]

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

		S, p = spearmanr(all_ranks, all_gold)
		P = pearsonr(all_ranks, all_gold)

		return float(corrects1)/float(total1), float(corrects2)/float(total2), float(corrects3)/float(total3), float(recall1)/float(trecall1), float(recall2)/float(trecall2), float(recall3)/float(trecall3), S, P[0]
		
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
		"""
		Performs a round-trip evaluation of a Substitution Generation, Selection and Ranking approach combined.
	
		@param victor_corpus: Path to a training corpus in VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param rankings: A list of ranked candidates for each instance in the VICTOR corpus, from simplest to most complex.
		One should produce candidates with a Substitution Generation approach, select them for a given VICTOR corpus with a Substitution Selection approach, then rank them with a Substitution Ranking approach.
		@return: Values for Precision, Accuracy and Changed Proportion for the substitutions provided as input with respect to the gold-standard in the VICTOR corpus.
		For more information on how the metrics are calculated, please refer to the LEXenstein Manual.
		"""
	
		#Initialize counting variables:
		total = 0
		totalc = 0
		accurate = 0
		precise = 0
		
		#Read victor corpus:
		f = open(victor_corpus)
		for i in range(0, len(rankings)):
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
					accurate += 1
					precise += 1
			else:
				precise += 1
		
		#Return metrics:
		return float(precise)/float(total), float(accurate)/float(total), float(totalc)/float(total)
		
class PLUMBErr:

	def __init__(self, dataset, complex):
		"""
		Creates a PLUMBErr error categorizer.
		This class implements the strategy introduced in:
		Paetzold, G. H.; Specia, L. PLUMBErr: An Automatic Error Identification Framework for Lexical Simplification. Proceedings of the 1st QATS. 2016.
		One can download BenchLS (dataset) and NNSVocab (complex) from http://ghpaetzold.github.io/data/PLUMBErr.zip
	
		@param dataset: Path to a data in VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param complex: Path to a file containing complex words.
		Each line of the file must contain a single word.
		"""
		self.data = [line.strip().split('\t') for line in open(dataset)]
		self.complex = set([line.strip() for line in open(complex)])
		
	def cumulativeAnalysis(self, identified, selected, ranked):
		"""
		Performs the cumulative error identification analysis of a simplifier.
		In a cumulative analysis, the errors made during Complex Word Identification are carried onto the following steps of the pipeline.
	
		@param identified: A vector containing one binary value (0 for simple, 1 for complex) for each word in the dataset.
		To produce the vector, one can run a Complex Word Identification approach from LEXenstein over the dataset.
		@param selected: A vector containing the candidates selected for each instance in the dataset.
		To produce the vector, one can pair a Substitution Generation and a Substitution Selection approach from LEXenstein.
		@param ranked: A vector containing the selected candidates ranked in order of simplicity.
		To produce the vector, one can run a Substitution Ranking approach from LEXenstein over the selected candidates provided.
		"""
		
		#Initialize report:
		report = []
		
		#Create CWI gold-standard:
		gold = []
		for line in self.data:
			if line[1] in self.complex:
				gold.append(1)
			else:
				gold.append(0)
				
		#Find errors of type 2:
		error2a = 0
		error2b = 0
		for i in range(0, len(gold)):
			errors = set([])
			g = gold[i]
			p = identified[i]
			if p==0 and g==1:
				error2a += 1
				errors.add('2A')
			elif p==1 and g==0:
				error2b += 1
				errors.add('2B')
			report.append(errors)
				
		#Find errors of type 3:
		error3a = 0
		error3b = 0
		
		goldcands = []
		simplecands = []
		for line in self.data:
				cs = set([cand.strip().split(':')[1].strip() for cand in line[3:]])
				goldcands.append(cs)
				simplecands.append(cs.difference(self.complex))
		
		cands = []
		for vec in selected:
			cands.append(set(vec))
		
		control = []
		for i in range(0, len(self.data)):
			gold_label = gold[i]
			pred_label = identified[i]
			ac = goldcands[i]
			sc = simplecands[i]
			cs = cands[i]
			if gold_label==0:
				sc = set([])
			else:
				if pred_label==0:
					cs = set([])
			ainter = ac.intersection(cs)
			sinter = sc.intersection(cs)

			if gold_label==1:
				if len(ainter)==0:
					error3a += 1
					report[i].add('3A')
					control.append('Error')
				elif len(sinter)==0:
					error3b += 1
					report[i].add('3B')
					control.append('Error')
				else:
					control.append('Ok')
			else:
				control.append('Ignore')
		
		#Find errors of type 4 and 5:
		error4 = 0
		error5 = 0
		noerror = 0
		for i in range(0, len(self.data)):
			gold_label = gold[i]
			pred_label = identified[i]
			ac = goldcands[i]
			sc = simplecands[i]
			cs = ranked[i]
			if gold_label==0:
				sc = set([])
			else:
				if pred_label==0:
					cs = set([])

			sub = ''
			if len(cs)>0:
				sub = cs[0]

			if control[i]=='Ok':
				if sub not in ac:
					error4 += 1
					report[i].add('4')
				elif sub not in sc:
					error5 += 1
					report[i].add('5')
				else:
					noerror += 1
					report[i].add('1')
					
		#Create error count map:
		counts = {}
		counts['2A'] = error2a
		counts['2B'] = error2b
		counts['3A'] = error3a
		counts['3B'] = error3b
		counts['4'] = error4
		counts['5'] = error5
		counts['1'] = noerror
		
		return report, counts
		
	def nonCumulativeAnalysis(self, identified, selected, ranked):
		"""
		Performs the non-cumulative error identification analysis of a simplifier.
		In a non-cumulative analysis, the errors made during Complex Word Identification are not carried onto the following steps of the pipeline.
	
		@param identified: A vector containing one binary value (0 for simple, 1 for complex) for each word in the dataset.
		To produce the vector, one can run a Complex Word Identification approach from LEXenstein over the dataset.
		@param selected: A vector containing the candidates selected for each instance in the dataset.
		To produce the vector, one can pair a Substitution Generation and a Substitution Selection approach from LEXenstein.
		@param ranked: A vector containing the selected candidates ranked in order of simplicity.
		To produce the vector, one can run a Substitution Ranking approach from LEXenstein over the selected candidates provided.
		@return: A report vector containing the errors made in each instance of the dataset, as well as a map containing total error counts for the entire dataset.
		"""
		
		#Initialize report:
		report = []
		
		#Create CWI gold-standard:
		gold = []
		for line in self.data:
			if line[1] in self.complex:
				gold.append(1)
			else:
				gold.append(0)
				
		#Find errors of type 2:
		error2a = 0
		error2b = 0
		for i in range(0, len(gold)):
			errors = set([])
			g = gold[i]
			p = identified[i]
			if p==0 and g==1:
				error2a += 1
				errors.add('2A')
			elif p==1 and g==0:
				error2b += 1
				errors.add('2B')
			report.append(errors)
				
		#Find errors of type 3:
		error3a = 0
		error3b = 0
		
		goldcands = []
		simplecands = []
		for line in self.data:
				cs = set([cand.strip().split(':')[1].strip() for cand in line[3:]])
				goldcands.append(cs)
				simplecands.append(cs.difference(self.complex))
		
		cands = []
		for vec in selected:
			cands.append(set(vec))
		
		for i in range(0, len(self.data)):
			gold_label = gold[i]
			pred_label = identified[i]
			ac = goldcands[i]
			sc = simplecands[i]
			cs = cands[i]
			ainter = ac.intersection(cs)
			sinter = sc.intersection(cs)

			if gold_label==1:
				if len(ainter)==0:
					error3a += 1
					report[i].add('3A')
				elif len(sinter)==0:
					error3b += 1
					report[i].add('3B')
		
		#Find errors of type 4 and 5:
		error4 = 0
		error5 = 0
		noerror = 0
		for i in range(0, len(self.data)):
			gold_label = gold[i]
			pred_label = identified[i]
			ac = goldcands[i]
			sc = simplecands[i]
			cs = ranked[i]

			sub = ''
			if len(cs)>0:
				sub = cs[0]

			if gold_label==1:
				if sub not in ac:
					error4 += 1
					report[i].add('4')
				elif sub not in sc:
					error5 += 1
					report[i].add('5')
				else:
					noerror += 1
					report[i].add('1')
		
		#Create error count map:
		counts = {}
		counts['2A'] = error2a
		counts['2B'] = error2b
		counts['3A'] = error3a
		counts['3B'] = error3b
		counts['4'] = error4
		counts['5'] = error5
		counts['1'] = noerror
		
		return report, counts
