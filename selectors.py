import pywsd
import gensim
from scipy.spatial.distance import cosine
import nltk
import numpy as np

class WordVectorSelector:
	
	def __init__(self, vector_model):
		self.model = gensim.models.word2vec.Word2Vec.load_word2vec_format(vector_model, binary=True)
	
	def selectCandidates(self, substitutions, victor_corpus, proportion=1.0, stop_words_file=None, window=99999, onlyInformative=False, keepTarget=False, onePerWord=False):
		stop_words = set([])
		if stop_words_file != None:
			stop_words = set([word.strip() for word in open(stop_words_file)])
	
		selected_substitutions = []				

		lexf = open(victor_corpus)
		for line in lexf:
			data = line.strip().split('\t')
			sent = data[0].strip()
			target = data[1].strip()
			head = int(data[2].strip())
		
			target_vec = self.getSentVec(sent, head, stop_words, window, onlyInformative, keepTarget, onePerWord)
		
			candidates = []
			if target in substitutions.keys():
				candidates = substitutions[target]
	

			candidate_dists = {}
			for candidate in candidates:
				candidate_vec = self.getWordVec(candidate)
				try:
					candidate_dists[candidate] = cosine(candidate_vec, target_vec)
				except ValueError:
					candidate_dists = candidate_dists

			final_candidates = self.getFinalCandidates(candidate_dists, proportion)

			selected_substitutions.append(final_candidates)
		lexf.close()
		return selected_substitutions
		
	def getSentVec(self, sentence, head, stop_words, window, onlyInformative, keepTarget, onePerWord):
		informative_tags = set([])
		if onlyInformative:
			informative_tags = set(['nn', 'nns', 'jj', 'jjs', 'jjr', 'vb', 'vbd', 'vbg', 'vbn', 'vbp', 'vbz', 'rb', 'rbr', 'rbs'])
		
		tokens = sentence.split(' ')
		pos_tokens = []
		try:
			pos_tokens = nltk.pos_tag(tokens)
		except UnicodeDecodeError:
			informative_tags = set([])
			pos_tokens = []
		
		valid_tokens = []
		if keepTarget:
			valid_tokens.append(tokens[head].strip())
		
		if head>0:
			for i in range(max(0, head-window), head):
				if len(informative_tags)==0 or pos_tokens[i][1].lower().strip() in informative_tags:
					if tokens[i] not in stop_words:
						valid_tokens.append(tokens[i])
		
		if head<len(tokens)-1:
			for i in range(head+1, min(len(tokens), head+1+window)):
				if len(informative_tags)==0 or pos_tokens[i][1].lower().strip() in informative_tags:
					if tokens[i] not in stop_words:
						valid_tokens.append(tokens[i])
						
		if onePerWord:
			valid_tokens = list(set(valid_tokens))
		
		result = []
		for	token in valid_tokens:
			if len(result)==0:
				try:
					result = self.model[token]
				except KeyError:
					try:
						result = self.model[token.lower()]
					except KeyError:
						result = []
			else:
				try:
					result = np.add(result, self.model[token])
				except KeyError:
					try:
						result = np.add(result, self.model[token.lower()])
					except KeyError:
						result = result
		return result
		
	def getWordVec(self, candidate):
		result = []
		try:
			result = self.model[candidate]
		except KeyError:
			try:
				result = self.model[candidate.lower()]
			except KeyError:
				result = result
		return result
				
	def getFinalCandidates(self, candidate_dists, proportion):
		result = sorted(list(candidate_dists.keys()), key=candidate_dists.__getitem__)
		return result[0:max(1, int(proportion*float(len(result))))]

class WSDSelector:

	def __init__(self, method):
		if method == 'lesk':
			self.WSDfunction = self.getLeskSense
		elif method == 'leacho':
			self.WSDfunction = self.getLeaChoSense
		elif method == 'path':
			self.WSDfunction = self.getPathSense
		elif method == 'wupalmer':
			self.WSDfunction = self.getWuPalmerSense
		elif method == 'random':
			self.WSDfunction = self.getRandomSense
		elif method == 'first':
			self.WSDfunction = self.getFirstSense
		else:
			self.WSDfunction = self.getLeskSense
		
	def selectCandidates(self, substitutions, victor_corpus):
		selected_substitutions = []				

		lexf = open(victor_corpus)
		for line in lexf:
			data = line.strip().split('\t')
			sent = data[0].strip()
			target = data[1].strip()
			head = int(data[2].strip())
		
			target_sense = self.WSDfunction.__call__(sent, target)
		
			candidates = []
			if target in substitutions.keys():
				candidates = substitutions[target]
		
			selected_candidates = set([])
			for candidate in candidates:
				candidate_sense = None
				try:
					unic = unicode(candidate)
					candidate_sense = self.WSDfunction.__call__(self.getCandidateSentence(sent, candidate, head), candidate)
				except UnicodeDecodeError:
					candidate_sense = None
				if target_sense or not candidate_sense:
					if not candidate_sense or candidate_sense==target_sense:
						selected_candidates.add(candidate)
			selected_substitutions.append(selected_candidates)
		lexf.close()

		return selected_substitutions

	def getLeskSense(self, sentence, target):
		try:
			result = pywsd.lesk.original_lesk(sentence, target)
			return result
		except IndexError:
			return None

	def getPathSense(self, sentence, target):
		try:
			result = pywsd.similarity.max_similarity(sentence, target, option="path", best=False)
			return result
		except IndexError:
			return None
	
	def getLeaChoSense(self, sentence, target):
		try:
			result = pywsd.similarity.max_similarity(sentence, target, option="lch", best=False)
			return result
		except IndexError:
			return None
			
	def getWuPalmerSense(self, sentence, target):
		try:
			result = pywsd.similarity.max_similarity(sentence, target, option="wup", best=False)
			return result
		except IndexError:
			return None
			
	def getRandomSense(self, sentence, target):
		try:
			result = pywsd.baseline.random_sense(target)
			return result
		except IndexError:
			return None
			
	def getFirstSense(self, sentence, target):
		try:
			result = pywsd.baseline.first_sense(target)
			return result
		except IndexError:
			return None
			
	def getMaxLemmaSense(self, sentence, target):
		try:
			result = pywsd.baseline.max_lemma_count(target)
			return result
		except IndexError:
			return None

	def getCandidateSentence(self, sentence, candidate, head):
		tokens = sentence.strip().split(' ')
		result = ''
		for i in range(0, head):
			result += tokens[i] + ' '
		result += candidate + ' '
		for i in range(head+1, len(tokens)):
			result += tokens[i] + ' '
		return result.strip()
