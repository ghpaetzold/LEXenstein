import pywsd
import gensim
from scipy.spatial.distance import cosine
import nltk
import numpy as np

class POSTagSelector:

	def selectCandidates(self, substitutions, victor_corpus):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: A dictionary linking complex words to a set of candidate substitutions
		Example: substitutions['perched'] = {'sat', 'roosted'}
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		selected_substitutions = []				

		lexf = open(victor_corpus)
		for line in lexf:
			data = line.strip().split('\t')
			sent = data[0].strip()
			target = data[1].strip()
			head = int(data[2].strip())
		
			candidates = []
			if target in substitutions.keys():
				target_POS = self.getTargetPOS(sent, target, head)
				candidates = substitutions[target]
				candidates = self.getCandidatesWithSamePOS(sent.split(' '), head, candidates, target_POS)
		
			selected_substitutions.append(candidates)
		lexf.close()
		return selected_substitutions
	
	def getTargetPOS(self, sent, target, head):
		pos_data = []
		try:
			pos_data = nltk.pos_tag(sent)
			return pos_data[head][1]
		except UnicodeDecodeError:
			try:
				pos_data = nltk.pos_tag(target)
				return pos_data[0][1]
			except UnicodeDecodeError:
				return 'None'
			
		
	def getCandidatesWithSamePOS(self, tokens, head, candidates, pos):
		result = set([])
		pref = ''
		suff = ''
		for i in range(0, head):
			pref += tokens[i] + ' '
		for i in range(head+1, len(tokens)):
			suff += tokens[i] + ' '
		suff = ' ' + suff.strip()
		for candidate in candidates:
			sent = pref + candidate + suff
			candidate_tag = []
			try:
				pos_data = nltk.pos_tag(sent)
				candidate_tag = pos_data[head][1]
			except UnicodeDecodeError:
				try:
					pos_data = nltk.pos_tag(candidate)
					candidate_tag =  pos_data[0][1]
				except UnicodeDecodeError:
					candidate_tag = 'NoneCand'
			if candidate_tag==pos:
				result.add(candidate)
		return result
	
	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()

class VoidSelector:

	def selectCandidates(self, substitutions, victor_corpus):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: A dictionary linking complex words to a set of candidate substitutions
		Example: substitutions['perched'] = {'sat', 'roosted'}
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		selected_substitutions = []				

		lexf = open(victor_corpus)
		for line in lexf:
			data = line.strip().split('\t')
			sent = data[0].strip()
			target = data[1].strip()
		
			candidates = []
			if target in substitutions.keys():
				candidates = substitutions[target]
		
			selected_substitutions.append(candidates)
		lexf.close()
		return selected_substitutions
		
	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()

class BiranSelector:

	def __init__(self, cooc_model):
		"""
		Creates an instance of the BiranSelector class.
	
		@param cooc_model: Path to a word co-occurrence model.
		For instructions on how to create the model, please refer to the LEXenstein Manual.
		"""
		self.model = self.getModel(cooc_model)
		
	def selectCandidates(self, substitutions, victor_corpus, common_distance, candidate_distance):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: A dictionary linking complex words to a set of candidate substitutions
		Example: substitutions['perched'] = {'sat', 'roosted'}
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param common_distance: The cutoff minimum distance from the sentence's co-occurrence vector and the common vector between the target complex word and the candidate.
		We recommend using very small values, such as 0.01, or even 0.0.
		@param candidate_distance: The cutoff maximum distance from the sentence's co-occurrence vector and the candidate vector.
		We recommend using values close to 1.0, such as 0.8, or 0.9.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		selected_substitutions = []				

		lexf = open(victor_corpus)
		for line in lexf:
			data = line.strip().split('\t')
			sent = data[0].strip()
			target = data[1].strip()
			head = int(data[2].strip())
		
			target_vec = self.getSentVec(sent, head)

			candidates = []
			if target in substitutions.keys():
				candidates = substitutions[target]
		
			final_candidates = set([])
			for candidate in candidates:
				candidate_vec = self.getVec(candidate)
				candidate_dist = 1.0
				try:
					candidate_dist = self.getCosine(candidate_vec, target_vec)
				except ValueError:
					candidate_dist = 1.0
		
				common_vec = self.getCommonVec(target, candidate)
				common_dist = 0.0
				try:
					common_dist = self.getCosine(common_vec, target_vec)
				except ValueError:
					common_dist = 0.0
				if common_dist>=common_distance and candidate_dist<=candidate_distance:
					final_candidates.add(candidate)
			selected_substitutions.append(final_candidates)
		lexf.close()
		return selected_substitutions
		
	def getModel(self, path):
		result = {}
		f = open(path)
		for line in f:
			data = line.strip().split('\t')
			target = data[0].strip()
			coocs = data[1:len(data)]
			result[target] = {}
			for cooc in coocs:
				coocd = cooc.strip().split(':')
				word = coocd[0].strip()
				count = int(coocd[1].strip())
				result[target][word] = count
		return result
	
	def getCosine(self, vec1, vec2):
		all_keys = sorted(list(set(vec1.keys()).union(set(vec2.keys()))))
		v1 = []
		v2 = []
		for k in all_keys:
			if k in vec1:
				v1.append(vec1[k])
			else:
				v1.append(0.0)
			if k in vec2:
				v2.append(vec2[k])
			else:
				v2.append(0.0)
		return cosine(v1, v2)
	
	def getCommonVec(self, target, candidate):
		if target not in self.model.keys() or candidate not in self.model.keys():
			return {}
		else:
			result = {}
			common_keys = set(self.model[target].keys()).intersection(set(self.model[candidate].keys()))
			for k in common_keys:
				if self.model[target][k]>self.model[candidate][k]:
					result[k] = self.model[candidate][k]
				else:
					result[k] = self.model[target][k]
			return result
					
	def isNumeral(self, text):
		try:
			num = float(text.strip())
			return True
		except ValueError:
			return False
	
	def getSentVec(self, sent, head):
		coocs = {}
		tokens = sent.strip().split(' ')
		left = max(0, head-5)
		right = min(len(tokens), head+6)
		for j in range(left, right):
			if j!=head:
				cooc = tokens[j]
				if self.isNumeral(cooc):
					cooc = '#NUMERAL#'
				if cooc not in coocs.keys():
					coocs[cooc] = 1
				else:
					coocs[cooc] += 1
		return coocs
	
	def getVec(self, word):
		result = {}
		try:
			result = self.model[word]
		except KeyError:
			try:
				result = self.model[word.lower()]
			except KeyError:
				result = {}
		return result
		
	def getCandidateSentence(self, sentence, candidate, head):
		tokens = sentence.strip().split(' ')
		result = ''
		for i in range(0, head):
			result += tokens[i] + ' '
		result += candidate + ' '
		for i in range(head+1, len(tokens)):
			result += tokens[i] + ' '
		return result.strip()
		
	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()
	
class WordVectorSelector:
	
	def __init__(self, vector_model):
		"""
		Creates an instance of the WordVectorSelector class.
	
		@param vector_model: Path to a word vector model.
		For instructions on how to create the model, please refer to the LEXenstein Manual.
		"""
		self.model = gensim.models.word2vec.Word2Vec.load_word2vec_format(vector_model, binary=True)
	
	def selectCandidates(self, substitutions, victor_corpus, proportion=1.0, stop_words_file=None, window=99999, onlyInformative=False, keepTarget=False, onePerWord=False):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: A dictionary linking complex words to a set of candidate substitutions
		Example: substitutions['perched'] = {'sat', 'roosted'}
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param proportion: Percentage of substitutions to keep.
		@param stop_words_file: Path to the file containing stop words of the desired language.
		The file must contain one stop word per line.
		@param window: Number of tokens around the target complex sentence to consider as its context.
		@param onlyInformative: If True, only content words are considered as part of the complex word's context, such as nouns, verbs, adjectives and adverbs.
		@param keepTarget: If True, the complex target word is also included as part of its context.
		@param onePerWord: If True, a word in the complex word's context can only contribute once to its resulting word vector.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		
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
		
	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()

class WSDSelector:

	def __init__(self, method):
		"""
		Creates an instance of the WSDSelector class.
	
		@param method: Type of Word Sense Disambiguation algorithm to use.
		Options available:
		lesk - Original lesk algorithm.
		path - Path similarity algorithm.
		random - Random sense from WordNet.
		first - First sense from WordNet.
		"""
		
		if method == 'lesk':
			self.WSDfunction = self.getLeskSense
		elif method == 'path':
			self.WSDfunction = self.getPathSense
		elif method == 'random':
			self.WSDfunction = self.getRandomSense
		elif method == 'first':
			self.WSDfunction = self.getFirstSense
		else:
			self.WSDfunction = self.getLeskSense
		
	def selectCandidates(self, substitutions, victor_corpus):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: A dictionary linking complex words to a set of candidate substitutions
		Example: substitutions['perched'] = {'sat', 'roosted'}
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		
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

	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()
