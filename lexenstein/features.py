from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
import kenlm
import math
import gensim
from nltk.tag.stanford import POSTagger
import os
import pickle
from sklearn.preprocessing import normalize

class FeatureEstimator:

	def __init__(self, norm=False):
		"""
		Creates an instance of the FeatureEstimator class.
	
		@param norm: Boolean variable that determines whether or not feature values should be normalized.
		"""
		self.features = []
		self.identifiers = []
		self.norm = norm
		self.resources = {}
		
	def calculateFeatures(self, corpus, format='victor'):
		"""
		Calculate the selected features over the candidates of a VICTOR or CWICTOR corpus.
	
		@param corpus: Path to a corpus in the VICTOR or CWICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param format: Input file format.
		Values available: victor, cwictor
		@return: Returns a MxN matrix, where M is the number of substitutions of all instances in the VICTOR corpus, and N the number of selected features.
		"""
		
		data = []
		if format.strip().lower()=='victor':
			data = [line.strip().split('\t') for line in open(corpus)]
		elif format.strip().lower()=='cwictor':
			f = open(corpus)
			for line in f:
				line_data = line.strip().split('\t')
				data.append([line_data[0].strip(), line_data[1].strip(), line_data[2].strip(), '0:'+line_data[1].strip()])
		else:
			print('Unknown input format during feature estimation!')
			return []
		
		values = []
		for feature in self.features:
			values.append(feature[0].__call__(data, feature[1]))
			
		result = []
		index = 0
		for line in data:
			for i in range(3, len(line)):
				vector = self.generateVector(values, index)
				result.append(vector)
				index += 1
				
		#Normalize if required:
		if self.norm:
			result = normalize(result, axis=0)
			
		return result
		
	def calculateInstanceFeatures(self, sent, target, head, candidate):
		"""
		Calculate the selected features over an instance of a VICTOR corpus.
	
		@param sent: Sentence containing a target complex word.
		@param target: Target complex sentence to be simplified.
		@param head: Position of target complex word in sentence.
		@param candidate: Candidate substitution.
		@return: Returns a vector containing the feature values of VICTOR instance.
		"""
	
		data = [[sent, target, head, '0:'+candidate]]
		
		values = []
		for feature in self.features:
			values.append(feature[0].__call__(data, feature[1]))
		vector = self.generateVector(values, 0)
		return vector
		
	def generateVector(self, feature_vector, index):
		result = []
		for feature in feature_vector:
			if not isinstance(feature[index], list):
				result.append(feature[index])
			else:
				result.extend(feature[index])
		return result
	
	def targetPOSTagProbability(self, data, args):
		model = self.resources[args[0]]
		tagger = self.resources[args[1]]
		result = []
		
		#Get tagged sentences:
		sentences = [l[0].strip().split(' ') for l in data]
		tagged_sents = tagger.tag_sents(sentences)
		
		for i in range(0, len(data)):
			line = data[i]
			target = line[1].strip().lower()
			head = int(line[2].strip())
			target_pos = tagged_sents[i][head][1]
			
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				probability = model[words].prob(target_pos)
				result.append(probability)
		return result
	
	def wordVectorSimilarityFeature(self, data, args):
		model = self.resources[args[0]]
		result = []
		for line in data:
			target = line[1].strip().lower()
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				similarity = 0.0
				cand_size = 0
				for word in words.split(' '):
					cand_size += 1
					try:
						similarity += model.similarity(target, word)
					except KeyError:
						try:
							similarity += model.similarity(target, word.lower())
						except KeyError:
							pass
				similarity /= cand_size
				result.append(similarity)
		return result
	
	def translationProbabilityFeature(self, data, args):
		probabilities = self.resources[args[0]]
		result = []
		for line in data:
			target_probs = {}
			if line[1].strip() in probabilities.keys():
				target_probs = probabilities[line[1].strip()]
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				prob = 1.0
				for word in words.split(' '):
					if word in target_probs.keys():
						prob *= target_probs[word]
					else:
						prob = 0.0
				result.append(prob)
		return result
		
	def lexiconFeature(self, data, args):
		path = args[0]
		result = []
		basics = self.resources[path]
		for line in data:
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				basicCount = 0
				for word in words.split(' '):
					if word.strip() in basics:
						basicCount += 1
				if basicCount==len(words.split(' ')):
					result.append(1.0)
				else:
					result.append(0.0)
		return result
		
	def lengthFeature(self, data, args):
		result = []
		for line in data:
			for subst in line[3:len(line)]:
				word = subst.strip().split(':')[1].strip()
				result.append(len(word))
		return result
		
	def syllableFeature(self, data, args):
		mat = args[0]
		#Create the input for the Java application:
		input = []
		for line in data:
			for subst in line[3:len(line)]:
				word = subst.strip().split(':')[1].strip()
				input.append(word)
	
		#Run the syllable splitter:
		outr = mat.splitSyllables(input)

		#Decode output:
		out = []
		for o in outr:
			out.append(o.decode("latin1").replace(' ', '-'))
	
		#Calculate number of syllables
		result = []
		for instance in out:
			if len(instance.strip())>0:
				result.append(len(instance.split('-')))
		return result
	
	def collocationalFeature(self, data, args):
		lm = args[0]
		spanl = args[1]
		spanr = args[2]
		result = []
		model = self.resources[lm]
		for line in data:
			sent = line[0]
			target = line[1]
			head = int(line[2])
			spanlv = range(0, spanl+1)
			spanrv = range(0, spanr+1)
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				values = []
				for span1 in spanlv:
					for span2 in spanrv:
						ngram, bosv, eosv = self.getNgram(word, sent, head, span1, span2)
						aux = model.score(ngram, bos=bosv, eos=eosv)
						values.append(aux)
				result.append(values)
		return result
		
	def ngramFrequencyFeature(self, data, args):
		lm = args[0]
		spanl = args[1]
		spanr = args[2]
		result = []
		model = self.resources[lm]
		for line in data:
			sent = line[0]
			target = line[1]
			head = int(line[2])
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				ngram, bosv, eosv = self.getNgram(word, sent, head, spanl, spanr)
				prob = model.score(ngram, bos=bosv, eos=eosv)
				result.append(prob)
		return result
	
	def getNgram(self, cand, sent, head, configl, configr):
		if configl==0 and configr==0:
			return cand, False, False
		else:
			result = ''
			tokens = sent.strip().split(' ')
			bosv = False
			if max(0, head-configl)==0:
				bosv = True
			eosv = False
			if min(len(tokens), head+configr+1)==len(tokens):
				eosv = True
			for i in range(max(0, head-configl), head):
				result += tokens[i] + ' '
			result += cand + ' '
			for i in range(head+1, min(len(tokens), head+configr+1)):
				result += tokens[i] + ' '
			return result.strip(), bosv, eosv
			
	def sentenceProbabilityFeature(self, data, args):
		lm = args[0]
		result = []
		model = self.resources[lm]
		for line in data:
			sent = line[0]
			target = line[1]
			head = int(line[2])
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				ngram, bosv, eosv = self.getNgram(word, sent, head, 9999, 9999)
				aux = -1.0*model.score(ngram, bos=bosv, eos=eosv)
				result.append(aux)
		return result
		
	def senseCount(self, data, args):
		resultse = []
		for line in data:
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				sensec = 0
				for word in words.split(' '):
					senses = None
					try:
						senses = wn.synsets(word)
					except UnicodeDecodeError:
						senses = []
					sensec += len(senses)
				resultse.append(sensec)
		return resultse
	
	def synonymCount(self, data, args):
		resultsy = []
		for line in data:
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				syncount = 0
				for word in words.split(' '):
					senses = None
					try:
						senses = wn.synsets(word)
					except UnicodeDecodeError:
						senses = []
					for sense in senses:
						syncount += len(sense.lemmas())
				resultsy.append(syncount)
		return resultsy

	def hypernymCount(self, data, args):
		resulthe = []
		for line in data:
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				hypernyms = set([])
				for word in words.split(' '):
					senses = None
					try:
						senses = wn.synsets(word)
					except UnicodeDecodeError:
						senses = []
					for sense in senses:
						hypernyms.update(sense.hypernyms())
				resulthe.append(len(hypernyms))
		return resulthe
	
	def hyponymCount(self, data, args):
		resultho = []
		for line in data:
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				hyponyms = set([])
				for word in words.split(' '):
					senses = None
					try:
						senses = wn.synsets(word)
					except UnicodeDecodeError:
						senses = []
					for sense in senses:
						hyponyms.update(sense.hyponyms())
				resultho.append(len(hyponyms))
		return resultho
	
	def minDepth(self, data, args):
		resultmi = []
		for line in data:
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				mindepth = 9999999
				for word in words.split(' '):
					senses = None
					try:
						senses = wn.synsets(word)
					except UnicodeDecodeError:
						senses = []
					for sense in senses:
						auxmin = sense.min_depth()
						if auxmin<mindepth:
							mindepth = auxmin
				resultmi.append(mindepth)
		return resultmi
	
	def maxDepth(self, data, args):
		resultma = []
		for line in data:
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				maxdepth = -1
				for word in words.split(' '):
					senses = None
					try:
						senses = wn.synsets(word)
					except UnicodeDecodeError:
						senses = []
					for sense in senses:
						auxmax = sense.max_depth()
						if auxmax>maxdepth:
							maxdepth = auxmax
				resultma.append(maxdepth)
		return resultma
	
	def addTargetPOSTagProbability(self, condprob_model, pos_model, stanford_tagger, java_path, orientation):
		"""
		Adds a target POS tag probability feature to the estimator.
		The value will be the conditional probability between a candidate substitution and the POS tag of a given target word.
	
		@param condprob_model: Path to a binary conditional probability model.
		For instructions on how to create the model, please refer to the LEXenstein Manual.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			os.environ['JAVAHOME'] = java_path
			if pos_model not in self.resources.keys():
				tagger = POSTagger(pos_model, stanford_tagger)
				self.resources[pos_model] = tagger
			if condprob_model not in self.resources.keys():
				m = pickle.load(open(condprob_model, 'rb'))
				self.resources[condprob_model] = m
			
			self.features.append((self.targetPOSTagProbability, [condprob_model, pos_model]))
			self.identifiers.append(('Target POS Tag Probability (Model:'+str(condprob_model)+')', orientation))
	
	def addWordVectorSimilarityFeature(self, model, orientation):
		"""
		Adds a word vector similarity feature to the estimator.
		The value will be the similarity between the word vector of a target complex word and the word vector of a candidate.
	
		@param model: Path to a binary word vector model.
		For instructions on how to create the model, please refer to the LEXenstein Manual.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if model not in self.resources.keys():
				m = gensim.models.word2vec.Word2Vec.load_word2vec_format(model, binary=True)
				self.resources[model] = m
			self.features.append((self.wordVectorSimilarityFeature, [model]))
			self.identifiers.append(('Word Vector Similarity (Model: '+model+')', orientation))
	
	def addTranslationProbabilityFeature(self, translation_probabilities, orientation):
		"""
		Adds a translation probability feature to the estimator.
		The value will be the probability of a target complex word of being translated into a given candidate substitution.
	
		@param translation_probabilities: Path to a file containing the translation probabilities.
		The file must produced by the following command through fast_align:
		fast_align -i <parallel_data> -v -d -o <translation_probabilities_file>
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		path = translation_probabilities
		probabilities = {}
		f = open(path)
		for line in f:
			lined = line.strip().split('\t')
			word1 = lined[0]
			word2 = lined[1]
			prob = math.exp(float(lined[2]))
			if word1 in probabilities.keys():
				probabilities[word1][word2] = prob
			else:
				probabilities[word1] = {word2:prob}
		f.close()
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if translation_probabilities not in self.resources.keys():
				self.resources[translation_probabilities] = probabilities
			self.features.append((self.translationProbabilityFeature, [translation_probabilities]))
			self.identifiers.append(('Translation Probability (File: '+translation_probabilities+')', orientation))
	
	def addLexiconFeature(self, lexicon, orientation):
		"""
		Adds a lexicon feature to the estimator.
		The value will be 1 if a given candidate is in the provided lexicon, and 0 otherwise.
	
		@param lexicon: Path to a file containing the words of the lexicon.
		The file must have one word per line.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if lexicon not in self.resources.keys():
				words = set([w.strip() for w in open(lexicon)])
				self.resources[lexicon] = words
			self.features.append((self.lexiconFeature, [lexicon]))
			self.identifiers.append(('Lexicon Occurrence (Lexicon: '+lexicon+')', orientation))
	
	def addLengthFeature(self, orientation):
		"""
		Adds a word length feature to the estimator.
		The value will be the number of characters in each candidate.
	
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			self.features.append((self.lengthFeature, []))
			self.identifiers.append(('Word Length', orientation))
	
	def addSyllableFeature(self, mat, orientation):
		"""
		Adds a syllable count feature to the estimator.
		The value will be the number of syllables of each candidate.
	
		@param mat: A configured MorphAdornerToolkit object.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			self.features.append((self.syllableFeature, [mat]))
			self.identifiers.append(('Syllable Count', orientation))
		
	def addCollocationalFeature(self, language_model, leftw, rightw, orientation):
		"""
		Adds a set of collocational features to the estimator.
		The values will be the language model probabilities of all collocational features selected.
		Each feature is the probability of an n-gram with 0<=l<=leftw tokens to the left and 0<=r<=rightw tokens to the right.
		This method creates (leftw+1)*(rightw+1) features.
	
		@param language_model: Path to the language model from which to extract probabilities.
		@param leftw: Maximum number of tokens to the left.
		@param rightw: Maximum number of tokens to the right.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if language_model not in self.resources.keys():
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			self.features.append((self.collocationalFeature, [language_model, leftw, rightw]))
			for i in range(0, leftw+1):
				for j in range(0, rightw+1):
					self.identifiers.append(('Collocational Feature ['+str(i)+', '+str(j)+'] (LM: '+language_model+')', orientation))
					
	def addNGramFrequencyFeature(self, language_model, leftw, rightw, orientation):
		"""
		Adds a n-gram frequency feature to the estimator.
		The value is the probability of the n-gram with leftw tokens to the left and rightw tokens to the right.
	
		@param language_model: Path to the language model from which to extract probabilities.
		@param leftw: Number of tokens to the left.
		@param rightw: Number of tokens to the right.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if language_model not in self.resources.keys():
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			self.features.append((self.ngramFrequencyFeature, [language_model, leftw, rightw]))
			self.identifiers.append(('N-Gram Frequency Feature ['+str(leftw)+', '+str(rightw)+'] (LM: '+language_model+')', orientation))
		
	def addSentenceProbabilityFeature(self, language_model, orientation):
		"""
		Adds a sentence probability feature to the estimator.
		The value will be the language model probability of each sentence in the VICTOR corpus with its target complex word replaced by a candidate.
	
		@param language_model: Path to the language model from which to extract probabilities.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if language_model not in self.resources.keys():
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			self.features.append((self.sentenceProbabilityFeature, [language_model]))
			self.identifiers.append(('Sentence Probability (LM: '+language_model+')', orientation))
		
	def addSenseCountFeature(self, orientation):
		"""
		Adds a sense count feature to the estimator.
		Calculates the number of senses registered in WordNet of a candidate.
		
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			self.features.append((self.senseCount ,[]))
			self.identifiers.append(('Sense Count', orientation))
		
	def addSynonymCountFeature(self, orientation):
		"""
		Adds a synonym count feature to the estimator.
		Calculates the number of synonyms registered in WordNet of a candidate.
		
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			self.features.append((self.synonymCount ,[]))
			self.identifiers.append(('Synonym Count', orientation))
		
	def addHypernymCountFeature(self, orientation):
		"""
		Adds a hypernym count feature to the estimator.
		Calculates the number of hypernyms registered in WordNet of a candidate.
		
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			self.features.append((self.hypernymCount ,[]))
			self.identifiers.append(('Hypernym Count', orientation))
		
	def addHyponymCountFeature(self, orientation):
		"""
		Adds a hyponym count feature to the estimator.
		Calculates the number of hyponyms registered in WordNet of a candidate.
		
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			self.features.append((self.hyponymCount ,[]))
			self.identifiers.append(('Hyponym Count', orientation))
		
	def addMinDepthFeature(self, orientation):
		"""
		Adds a minimum sense depth feature to the estimator.
		Calculates the minimum distance between two senses of a given candidate.
		
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			self.features.append((self.minDepth ,[]))
			self.identifiers.append(('Minimal Sense Depth', orientation))
		
	def addMaxDepthFeature(self, orientation):
		"""
		Adds a maximum sense depth feature to the estimator.
		Calculates the maximum distance between two senses of a given candidate.
		
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			self.features.append((self.maxDepth ,[]))
			self.identifiers.append(('Maximal Sense Depth', orientation))
		
	
