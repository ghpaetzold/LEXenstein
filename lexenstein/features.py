from lexenstein.util import *
from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
import kenlm
import math
import gensim
from nltk.tag.stanford import StanfordPOSTagger
from nltk.parse.stanford import StanfordParser
import os
import pickle
from sklearn.preprocessing import normalize
import numpy
import shelve
import urllib2
import json
import re

class FeatureEstimator:

	def __init__(self, norm=False):
		"""
		Creates an instance of the FeatureEstimator class.
	
		@param norm: Boolean variable that determines whether or not feature values should be normalized.
		"""
		#List of features to be calculated:
		self.features = []
		#List of identifiers of features to be calculated:
		self.identifiers = []
		#Normalization parameter:
		self.norm = norm
		#Persistent resource list:
		self.resources = {}
		#One-run resource list:
		self.temp_resources = {}
		
	def calculateFeatures(self, corpus, format='victor', input='file'):
		"""
		Calculate the selected features over the candidates of a VICTOR or CWICTOR corpus.
	
		@param corpus: Path to a corpus in the VICTOR or CWICTOR format.
		For more information about the input's format, refer to the LEXenstein Manual.
		@param format: Input format.
		Values available: victor, cwictor.
		@param input: Type of input provided.
		Values available: file, text.
		@return: Returns a MxN matrix, where M is the number of substitutions of all instances in the VICTOR corpus, and N the number of selected features.
		"""
		data = []
		if format.strip().lower()=='victor':
			if input=='file':
				data = [line.strip().split('\t') for line in open(corpus)]
			elif input=='text':
				data = [line.strip().split('\t') for line in corpus.split('\n')]
			else:
				print('Unrecognized format: must be file or text.')
		elif format.strip().lower()=='cwictor':
			if input=='file':
				f = open(corpus)
				for line in f:
					line_data = line.strip().split('\t')
					data.append([line_data[0].strip(), line_data[1].strip(), line_data[2].strip(), '0:'+line_data[1].strip()])
			elif input=='text':
				for line in corpus.split('\n'):
					line_data = line.strip().split('\t')
					data.append([line_data[0].strip(), line_data[1].strip(), line_data[2].strip(), '0:'+line_data[1].strip()])
			else:
				print('Unrecognized format: must be file or text.')
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
			
		#Clear one-run resources:
		self.temp_resources = {}

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
		tagged_sents = None
		if 'tagged_sents' in self.temp_resources:
			tagged_sents = self.temp_resources['tagged_sents']
		else:
			sentences = [l[0].strip().split(' ') for l in data]
			tagged_sents = tagger.tag_sents(sentences)
			self.temp_resources['tagged_sents'] = tagged_sents
		
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
		
	def taggedWordVectorSimilarityFeature(self, data, args):
		result = []
		
		model = self.resources[args[0]]
		tagger = self.resources[args[1]]
		pos_type = args[2]
		
		#Get tagged sentences:
		tagged_sents = None
		if 'tagged_sents' in self.temp_resources:
			tagged_sents = self.temp_resources['tagged_sents']
		else:
			sentences = [l[0].strip().split(' ') for l in data]
			tagged_sents = tagger.tag_sents(sentences)
			self.temp_resources['tagged_sents'] = tagged_sents
			
		#Transform them to the right format:
		if pos_type=='paetzold':
			transformed = []
			for sent in tagged_sents:
				tokens = []
				for token in sent:
					tokens.append((token[0], getGeneralisedPOS(token[1])))
				transformed.append(tokens)
			tagged_sents = transformed

		for i in range(0, len(data)):
			line = data[i]
			target = line[1].strip().lower()
			head = int(line[2].strip())
			target_pos = tagged_sents[i][head][1]
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				similarity = 0.0
				cand_size = 0
				for word in words.split(' '):
					cand_size += 1
					try:
						similarity += model.similarity(target+'|||'+target_pos, word+'|||'+target_pos)
					except KeyError:
						try:
							similarity += model.similarity(target+'|||'+target_pos, word.lower()+'|||'+target_pos)
						except KeyError:
							pass
				similarity /= cand_size
				result.append(similarity)
		return result
	
	def wordVectorValuesFeature(self, data, args):
		model = self.resources[args[0]]
		size = args[1]
		result = []
		for line in data:
			target = line[1].strip().lower()
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				word_vector = numpy.zeros(size)
				for word in words.split(' '):
					try:
						word_vector = numpy.add(word_vector, model[words])
					except KeyError:
						pass
				result.append(word_vector)
		for i in range(0, len(result)):
			result[i] = result[i].tolist()
		return result
	
	def translationProbabilityFeature(self, data, args):
		probabilities = self.resources[args[0]]
		result = []
		for line in data:
			target = line[1].strip()
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				prob = -9999999999
				for word in words.split(' '):
					if target+'\t'+word in probabilities:
						p = probabilities[target+'\t'+word]
						if p>prob:
							prob = p
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
			sent = line[0].strip().split(' ')
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
						#aux = model.score(ngram)
						values.append(aux)
				result.append(values)
		return result
		
	def frequencyCollocationalFeature(self, data, args):
		ngrams = args[0]
		spanl = args[1]
		spanr = args[2]
		result = []
		counts = self.resources[ngrams]
		for line in data:
			sent = line[0].strip().split(' ')
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
						if ngram in counts:
							values.append(counts[ngram])
						else:
							values.append(0.0)
				result.append(values)
		return result
		
	def taggedFrequencyCollocationalFeature(self, data, args):
		counts = self.resources[args[0]]
		spanl = args[1]
		spanr = args[2]
		tagger = self.resources[args[3]]
		pos_type = args[4]
		
		#Get tagged sentences:
		tagged_sents = None
		if 'tagged_sents' in self.temp_resources:
			tagged_sents = self.temp_resources['tagged_sents']
		else:
			sentences = [l[0].strip().split(' ') for l in data]
			tagged_sents = tagger.tag_sents(sentences)
			self.temp_resources['tagged_sents'] = tagged_sents
			
		#Transform them to the right format:
		if pos_type=='paetzold':
			transformed = []
			for sent in tagged_sents:
				tokens = []
				for token in sent:
					tokens.append((token[0], getGeneralisedPOS(token[1])))
				transformed.append(tokens)
			tagged_sents = transformed
		
		result = []
		for i in range(0, len(data)):
			line = data[i]
			sent = ['<s>'] + [tokendata[1] for tokendata in tagged_sents[i]] + ['</s>']
			target = line[1]
			head = int(line[2])+1
			spanlv = range(0, spanl+1)
			spanrv = range(0, spanr+1)
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				values = []
				for span1 in spanlv:
					for span2 in spanrv:
						ngram, bosv, eosv = self.getNgram(word, sent, head, span1, span2)
						if ngram in counts:
							values.append(counts[ngram])
						else:
							values.append(0.0)
				result.append(values)
		return result
		
	def binaryTaggedFrequencyCollocationalFeature(self, data, args):
		counts = self.resources[args[0]]
		spanl = args[1]
		spanr = args[2]
		tagger = self.resources[args[3]]
		pos_type = args[4]
		
		#Get tagged sentences:
		tagged_sents = None
		if 'tagged_sents' in self.temp_resources:
			tagged_sents = self.temp_resources['tagged_sents']
		else:
			sentences = [l[0].strip().split(' ') for l in data]
			tagged_sents = tagger.tag_sents(sentences)
			self.temp_resources['tagged_sents'] = tagged_sents
			
		#Transform them to the right format:
		if pos_type=='paetzold':
			transformed = []
			for sent in tagged_sents:
				tokens = []
				for token in sent:
					tokens.append((token[0], getGeneralisedPOS(token[1])))
				transformed.append(tokens)
			tagged_sents = transformed
		
		result = []
		for i in range(0, len(data)):
			line = data[i]
			sent = ['<s>'] + [tokendata[1] for tokendata in tagged_sents[i]] + ['</s>']
			target = line[1]
			head = int(line[2])+1
			spanlv = range(0, spanl+1)
			spanrv = range(0, spanr+1)
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				values = []
				for span1 in spanlv:
					for span2 in spanrv:
						ngram, bosv, eosv = self.getNgram(word, sent, head, span1, span2)
						if ngram in counts:
							values.append(1.0)
						else:
							values.append(0.0)
				result.append(values)
		return result
	
	def popCollocationalFeature(self, data, args):
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
						ngrams = self.getPopNgrams(word, sent, head, span1, span2)
						maxscore = -999999
						for ngram in ngrams:
							aux = model.score(ngram[0], bos=ngram[1], eos=ngram[2])
							#aux = model.score(ngram[0])
							if aux>maxscore:
								maxscore = aux
						values.append(maxscore)
				result.append(values)
		return result
		
	def ngramProbabilityFeature(self, data, args):
		lm = args[0]
		spanl = args[1]
		spanr = args[2]
		result = []
		model = self.resources[lm]
		for line in data:
			sent = line[0].strip().split(' ')
			target = line[1]
			head = int(line[2])
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				ngram, bosv, eosv = self.getNgram(word, sent, head, spanl, spanr)
				prob = model.score(ngram, bos=bosv, eos=eosv)
				#prob = model.score(ngram)
				result.append(prob)
		return result
		
	def ngramFrequencyFeature(self, data, args):
		ngrams = args[0]
		spanl = args[1]
		spanr = args[2]
		result = []
		counts = self.resources[ngrams]
		for line in data:
			sent = line[0].strip().split(' ')
			target = line[1]
			head = int(line[2])
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				ngram, bosv, eosv = self.getNgram(word, sent, head, spanl, spanr)
				if ngram in counts:
					result.append(counts[ngram])
				else:
					result.append(0.0)
		return result
		
	def binaryNgramFrequencyFeature(self, data, args):
		ngrams = args[0]
		spanl = args[1]
		spanr = args[2]
		result = []
		counts = self.resources[ngrams]
		for line in data:
			sent = line[0].strip().split(' ')
			target = line[1]
			head = int(line[2])
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				ngram, bosv, eosv = self.getNgram(word, sent, head, spanl, spanr)
				if ngram in counts:
					result.append(1.0)
				else:
					result.append(0.0)
		return result
		
	def popNgramProbabilityFeature(self, data, args):
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
				ngrams = self.getPopNgrams(word, sent, head, spanl, spanl)
				maxscore = -999999
				for ngram in ngrams:
					aux = model.score(ngram[0], bos=ngram[1], eos=ngram[2])
					#aux = model.score(ngram[0])
					if aux>maxscore:
						maxscore = aux
				result.append(maxscore)
		return result
		
	def popNgramFrequencyFeature(self, data, args):
		ngrams = args[0]
		spanl = args[1]
		spanr = args[2]
		result = []
		counts = self.resources[ngrams]
		for line in data:
			sent = line[0].strip()
			target = line[1]
			head = int(line[2])
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				ngrams = self.getPopNgrams(word, sent, head, spanl, spanl)
				maxscore = -999999
				for ngram in ngrams:
					aux = 0.0
					if ngram[0] in counts:
						aux = counts[ngram[0]]
					
					if aux>maxscore:
						maxscore = aux
				result.append(maxscore)
				
		return result
	
	def getNgram(self, cand, tokens, head, configl, configr):
		if configl==0 and configr==0:
			return cand, False, False
		else:
			result = ''
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
			return str(result.strip()), bosv, eosv
	
	def getPopNgrams(self, cand, sent, head, configl, configr):
		if configl==0 and configr==0:
			bos = False
			eos = False
			if head==0:
				bos = True
			if head==len(sent.split(' '))-1:
				eos = True
			return [(cand, bos, eos)]
		else:
			result = set([])
			contexts = self.getPopContexts(sent, head)
			for context in contexts:
				ctokens = context[0]
				chead = context[1]
				bosv = False
				if max(0, chead-configl)==0:
					bosv = True
				eosv = False
				ngram = ''
				if min(len(ctokens), chead+configr+1)==len(ctokens):
					eosv = True
				for i in range(max(0, chead-configl), chead):
					ngram += ctokens[i] + ' '
				ngram += cand + ' '
				for i in range(chead+1, min(len(ctokens), chead+configr+1)):
					ngram += ctokens[i] + ' '
				result.add((ngram.strip(), bosv, eosv))
			return result
			
	def getPopContexts(self, sent, head):
		tokens = sent.strip().split(' ')
		result = []
		check = 0
		if head>0:
			check += 1
			tokens1 = list(tokens)
			tokens1.pop(head-1)
			result.append((tokens1, head-1))
		if head<len(tokens)-1:
			check += 1
			tokens2 = list(tokens)
			tokens2.pop(head+1)
			result.append((tokens2, head))
		if check==2:
			tokens3 = list(tokens)
			tokens3.pop(head+1)
			tokens3.pop(head-1)
			result.append((tokens3, head-1))
		return result
			
	def sentenceProbabilityFeature(self, data, args):
		lm = args[0]
		result = []
		model = self.resources[lm]
		for line in data:
			sent = line[0].strip().split(' ')
			target = line[1]
			head = int(line[2])
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				ngram, bosv, eosv = self.getNgram(word, sent, head, 9999, 9999)
				aux = model.score(ngram, bos=bosv, eos=eosv)
				result.append(aux)
		return result
		
	def reverseSentenceProbabilityFeature(self, data, args):
		lm = args[0]
		result = []
		model = self.resources[lm]
		for line in data:
			sent = line[0].strip().split(' ')
			invsent = []
			for i in range(0, len(sent)):
				invsent.append(sent[len(sent)-1-i])
			target = line[1]
			head = len(sent)-1-int(line[2])
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				ngram, bosv, eosv = self.getNgram(word, invsent, head, 9999, 9999)
				aux = model.score(ngram, bos=bosv, eos=eosv)
				result.append(aux)
		return result
		
	def prefixProbabilityFeature(self, data, args):
		lm = args[0]
		result = []
		model = self.resources[lm]
		for line in data:
			sent = line[0].strip().split(' ')
			target = line[1]
			head = int(line[2])
			sent = sent[0:head+1]
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				ngram, bosv, eosv = self.getNgram(word, sent, head, 9999, 9999)
				aux = model.score(ngram, bos=bosv, eos=eosv)
				result.append(aux)
		return result
		
	def reversePrefixProbabilityFeature(self, data, args):
		lm = args[0]
		result = []
		model = self.resources[lm]
		for line in data:
			sent = line[0].strip().split(' ')
			invsent = []
			for i in range(0, len(sent)):
				invsent.append(sent[len(sent)-1-i])
			target = line[1]
			head = len(sent)-1-int(line[2])
			invsent = invsent[0:head+1]
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				ngram, bosv, eosv = self.getNgram(word, invsent, head, 9999, 9999)
				aux = model.score(ngram, bos=bosv, eos=eosv)
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
		
	def isSynonym(self, data, args):
		resultsy = []
		for line in data:
			target = line[1].strip()
			tgtsenses = set([])
			try:
				tgtsenses = wn.synsets(target)
			except Exception:
				tgtsenses = set([])
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				senses = set([])
				for word in words.split(' '):
					try:
						senses.update(wn.synsets(word))
					except UnicodeDecodeError:
						senses = senses
				if len(tgtsenses)==0 or len(senses.intersection(tgtsenses))>0:
					resultsy.append(1.0)
				else:
					resultsy.append(0.0)
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
		
	def isHypernym(self, data, args):
		resultsy = []
		for line in data:
			target = line[1].strip()
			tgthypernyms = set([])
			try:
				tgtsenses = wn.synsets(target)
				for sense in tgtsenses:
					tgthypernyms.update(sense.hypernyms())
			except Exception:
				tgthypernyms = tgthypernyms
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				senses = set([])
				for word in words.split(' '):
					try:
						senses.update(wn.synsets(word))
					except UnicodeDecodeError:
						senses = senses
				if len(tgthypernyms)==0 or len(senses.intersection(tgthypernyms))>0:
					resultsy.append(1.0)
				else:
					resultsy.append(0.0)
		return resultsy
	
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
		
	def isHyponym(self, data, args):
		resultsy = []
		for line in data:
			target = line[1].strip()
			tgthyponyms = set([])
			try:
				tgtsenses = wn.synsets(target)
				for sense in tgtsenses:
					tgthyponyms.update(sense.hyponyms())
			except Exception:
				tgthyponyms = tgthyponyms
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				senses = set([])
				for word in words.split(' '):
					try:
						senses.update(wn.synsets(word))
					except UnicodeDecodeError:
						senses = senses
				if len(tgthyponyms)==0 or len(senses.intersection(tgthyponyms))>0:
					resultsy.append(1.0)
				else:
					resultsy.append(0.0)
		return resultsy
	
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
		
	def averageDepth(self, data, args):
		resultma = []
		for line in data:
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				avgdepth = 0
				total = 0
				for word in words.split(' '):
					senses = None
					try:
						senses = wn.synsets(word)
					except UnicodeDecodeError:
						senses = []
					for sense in senses:
						auxmax = sense.max_depth()
						avgdepth += auxmax
					total += len(senses)
				try:
					avgdepth /= total
				except Exception:
					avgdepth = 0
				resultma.append(avgdepth)
		return resultma
		
	def subjectDependencyProbabilityFeature(self, data, args):
		model = self.resources[args[0]]
		parser = self.resources[args[1]]
		
		#Get parsed sentences:
		dep_maps = None
		if 'dep_maps' in self.temp_resources:
			dep_maps = self.temp_resources['dep_maps']
		else:
			sentences = [l[0].strip().split(' ') for l in data]
			dep_parsed_sents = None
			if 'dep_parsed_sents' in self.temp_resources:
				dep_parsed_sents = self.temp_resources['dep_parsed_sents']
			else:
				dep_parsed_sents = dependencyParseSentences(parser, sentences)
				self.temp_resources['dep_parsed_sents'] = dep_parsed_sents
			dep_maps = []
			for sent in dep_parsed_sents:
				dep_map = {}
				for parse in sent:
					deplink = str(parse[0])
					subjectindex = int(str(parse[2]))-1
					objectindex = int(str(parse[4]))-1
					if subjectindex not in dep_map:
						dep_map[subjectindex] = {objectindex: set([deplink])}
					elif objectindex not in dep_map[subjectindex]:
						dep_map[subjectindex][objectindex] = set([deplink])
					else:
						dep_map[subjectindex][objectindex].add(deplink)
				dep_maps.append(dep_map)
			self.temp_resources['dep_maps'] = dep_maps

		result = []
		for i in range(0, len(data)):
			line = data[i]
			sent = line[0].strip().split(' ')
			target = line[1].strip().lower()
			head = int(line[2].strip())
			dep_map = dep_maps[i]
			insts = set([])
			if head in dep_map:
				for object in dep_map[head]:
					for dep_link in dep_map[head][object]:
						insts.add((dep_link, sent[object]))
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				total = 0.0
				if len(insts)>0:
					for inst in insts:
						ngram = inst[0] + ' ' + word + ' ' + inst[1]
						prob = math.exp(model.score(ngram, bos=False, eos=False))
						#prob = math.exp(model.score(ngram))
						total += prob
					total /= float(len(insts))
				else:
					total = 1.0
				result.append(total)
		return result
		
	def binarySubjectDependencyFeature(self, data, args):
		model = self.resources[args[0]]
		parser = self.resources[args[1]]
		
		#Get parsed sentences:
		dep_maps = None
		if 'dep_maps' in self.temp_resources:
			dep_maps = self.temp_resources['dep_maps']
		else:
			sentences = [l[0].strip().split(' ') for l in data]
			dep_parsed_sents = None
			if 'dep_parsed_sents' in self.temp_resources:
				dep_parsed_sents = self.temp_resources['dep_parsed_sents']
			else:
				dep_parsed_sents = dependencyParseSentences(parser, sentences)
				self.temp_resources['dep_parsed_sents'] = dep_parsed_sents
			dep_maps = []
			for sent in dep_parsed_sents:
				dep_map = {}
				for parse in sent:
					deplink = str(parse[0])
					subjectindex = int(str(parse[2]))-1
					objectindex = int(str(parse[4]))-1
					if subjectindex not in dep_map:
						dep_map[subjectindex] = {objectindex: set([deplink])}
					elif objectindex not in dep_map[subjectindex]:
						dep_map[subjectindex][objectindex] = set([deplink])
					else:
						dep_map[subjectindex][objectindex].add(deplink)
				dep_maps.append(dep_map)
			self.temp_resources['dep_maps'] = dep_maps

		result = []
		for i in range(0, len(data)):
			line = data[i]
			sent = line[0].strip().split(' ')
			target = line[1].strip().lower()
			head = int(line[2].strip())
			dep_map = dep_maps[i]
			insts = set([])
			if head in dep_map:
				for object in dep_map[head]:
					for dep_link in dep_map[head][object]:
						insts.add((dep_link, sent[object]))
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				total = 1.0
				if len(insts)>0:
					for inst in insts:
						ngram = inst[0] + ' ' + word + ' ' + inst[1]
						if ngram not in model:
							total = 0.0
				else:
					total = 1.0
				result.append(total)
		return result
		
	def subjectDependencyFrequencyFeature(self, data, args):
		model = self.resources[args[0]]
		parser = self.resources[args[1]]
		
		#Get parsed sentences:
		dep_maps = None
		if 'dep_maps' in self.temp_resources:
			dep_maps = self.temp_resources['dep_maps']
		else:
			sentences = [l[0].strip().split(' ') for l in data]
			dep_parsed_sents = None
			if 'dep_parsed_sents' in self.temp_resources:
				dep_parsed_sents = self.temp_resources['dep_parsed_sents']
			else:
				dep_parsed_sents = dependencyParseSentences(parser, sentences)
				self.temp_resources['dep_parsed_sents'] = dep_parsed_sents
			dep_maps = []
			for sent in dep_parsed_sents:
				dep_map = {}
				for parse in sent:
					deplink = str(parse[0])
					subjectindex = int(str(parse[2]))-1
					objectindex = int(str(parse[4]))-1
					if subjectindex not in dep_map:
						dep_map[subjectindex] = {objectindex: set([deplink])}
					elif objectindex not in dep_map[subjectindex]:
						dep_map[subjectindex][objectindex] = set([deplink])
					else:
						dep_map[subjectindex][objectindex].add(deplink)
				dep_maps.append(dep_map)
			self.temp_resources['dep_maps'] = dep_maps

		result = []
		for i in range(0, len(data)):
			line = data[i]
			sent = line[0].strip().split(' ')
			target = line[1].strip().lower()
			head = int(line[2].strip())
			dep_map = dep_maps[i]
			insts = set([])
			if head in dep_map:
				for object in dep_map[head]:
					for dep_link in dep_map[head][object]:
						insts.add((dep_link, sent[object]))
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				total = 0.0
				if len(insts)>0:
					for inst in insts:
						ngram = inst[0] + ' ' + word + ' ' + inst[1]
						if ngram in model:
							total += model[ngram]
					if total>0.0:
						total /= float(len(insts))
				else:
					total = 99999.0
				result.append(total)
		return result
		
	def objectDependencyProbabilityFeature(self, data, args):
		model = self.resources[args[0]]
		parser = self.resources[args[1]]
		
		#Get parsed sentences:
		inv_dep_maps = None
		if 'inv_dep_maps' in self.temp_resources:
			inv_dep_maps = self.temp_resources['inv_dep_maps']
		else:
			dep_maps = None
			if 'dep_maps' in self.temp_resources:
				dep_maps = self.temp_resources['dep_maps']
			else:
				sentences = [l[0].strip().split(' ') for l in data]
				dep_parsed_sents = None
				if 'dep_parsed_sents' in self.temp_resources:
					dep_parsed_sents = self.temp_resources['dep_parsed_sents']
				else:
					dep_parsed_sents = dependencyParseSentences(parser, sentences)
					self.temp_resources['dep_parsed_sents'] = dep_parsed_sents
				dep_maps = []
				for sent in dep_parsed_sents:
					dep_map = {}
					for parse in sent:
						deplink = str(parse[0])
						subjectindex = int(str(parse[2]))-1
						objectindex = int(str(parse[4]))-1
						if subjectindex not in dep_map:
							dep_map[subjectindex] = {objectindex: set([deplink])}
						elif objectindex not in dep_map[subjectindex]:
							dep_map[subjectindex][objectindex] = set([deplink])
						else:
							dep_map[subjectindex][objectindex].add(deplink)
					dep_maps.append(dep_map)
				self.temp_resources['dep_maps'] = dep_maps
				
			inv_dep_maps = []
			for inst in dep_maps:
				inv_dep_map = {}
				for subjectindex in inst:
					for objectindex in inst[subjectindex]:
						if objectindex not in inv_dep_map:
							inv_dep_map[objectindex] = {}
						inv_dep_map[objectindex][subjectindex] = inst[subjectindex][objectindex]
				inv_dep_maps.append(inv_dep_map)
			self.temp_resources['inv_dep_maps'] = inv_dep_maps

		result = []
		for i in range(0, len(data)):
			line = data[i]
			sent = line[0].strip().split(' ')
			target = line[1].strip().lower()
			head = int(line[2].strip())
			inv_dep_map = inv_dep_maps[i]
			insts = set([])
			if head in inv_dep_map:
				for object in inv_dep_map[head]:
					for dep_link in inv_dep_map[head][object]:
						insts.add((dep_link, sent[object]))
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				total = 0.0
				if len(insts)>0:
					for inst in insts:
						ngram = inst[0] + ' ' + inst[1] + ' ' + word
						prob = math.exp(model.score(ngram, bos=False, eos=False))
						#prob = math.exp(model.score(ngram))
						total += prob
					total /= float(len(insts))
				else:
					total = 1.0
				result.append(total)
		return result
		
	def binaryObjectDependencyFeature(self, data, args):
		model = self.resources[args[0]]
		parser = self.resources[args[1]]
		
		#Get parsed sentences:
		inv_dep_maps = None
		if 'inv_dep_maps' in self.temp_resources:
			inv_dep_maps = self.temp_resources['inv_dep_maps']
		else:
			dep_maps = None
			if 'dep_maps' in self.temp_resources:
				dep_maps = self.temp_resources['dep_maps']
			else:
				sentences = [l[0].strip().split(' ') for l in data]
				dep_parsed_sents = None
				if 'dep_parsed_sents' in self.temp_resources:
					dep_parsed_sents = self.temp_resources['dep_parsed_sents']
				else:
					dep_parsed_sents = dependencyParseSentences(parser, sentences)
					self.temp_resources['dep_parsed_sents'] = dep_parsed_sents
				dep_maps = []
				for sent in dep_parsed_sents:
					dep_map = {}
					for parse in sent:
						deplink = str(parse[0])
						subjectindex = int(str(parse[2]))-1
						objectindex = int(str(parse[4]))-1
						if subjectindex not in dep_map:
							dep_map[subjectindex] = {objectindex: set([deplink])}
						elif objectindex not in dep_map[subjectindex]:
							dep_map[subjectindex][objectindex] = set([deplink])
						else:
							dep_map[subjectindex][objectindex].add(deplink)
					dep_maps.append(dep_map)
				self.temp_resources['dep_maps'] = dep_maps
				
			inv_dep_maps = []
			for inst in dep_maps:
				inv_dep_map = {}
				for subjectindex in inst:
					for objectindex in inst[subjectindex]:
						if objectindex not in inv_dep_map:
							inv_dep_map[objectindex] = {}
						inv_dep_map[objectindex][subjectindex] = inst[subjectindex][objectindex]
				inv_dep_maps.append(inv_dep_map)
			self.temp_resources['inv_dep_maps'] = inv_dep_maps

		result = []
		for i in range(0, len(data)):
			line = data[i]
			sent = line[0].strip().split(' ')
			target = line[1].strip().lower()
			head = int(line[2].strip())
			inv_dep_map = inv_dep_maps[i]
			insts = set([])
			if head in inv_dep_map:
				for object in inv_dep_map[head]:
					for dep_link in inv_dep_map[head][object]:
						insts.add((dep_link, sent[object]))
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				total = 1.0
				if len(insts)>0:
					for inst in insts:
						ngram = inst[0] + ' ' + inst[1] + ' ' + word
						if ngram not in model:
							total = 0.0
				else:
					total = 1.0
				result.append(total)
		return result
		
	def objectDependencyFrequencyFeature(self, data, args):
		model = self.resources[args[0]]
		parser = self.resources[args[1]]
		
		#Get parsed sentences:
		inv_dep_maps = None
		if 'inv_dep_maps' in self.temp_resources:
			inv_dep_maps = self.temp_resources['inv_dep_maps']
		else:
			dep_maps = None
			if 'dep_maps' in self.temp_resources:
				dep_maps = self.temp_resources['dep_maps']
			else:
				sentences = [l[0].strip().split(' ') for l in data]
				dep_parsed_sents = None
				if 'dep_parsed_sents' in self.temp_resources:
					dep_parsed_sents = self.temp_resources['dep_parsed_sents']
				else:
					dep_parsed_sents = dependencyParseSentences(parser, sentences)
					self.temp_resources['dep_parsed_sents'] = dep_parsed_sents
				dep_maps = []
				for sent in dep_parsed_sents:
					dep_map = {}
					for parse in sent:
						deplink = str(parse[0])
						subjectindex = int(str(parse[2]))-1
						objectindex = int(str(parse[4]))-1
						if subjectindex not in dep_map:
							dep_map[subjectindex] = {objectindex: set([deplink])}
						elif objectindex not in dep_map[subjectindex]:
							dep_map[subjectindex][objectindex] = set([deplink])
						else:
							dep_map[subjectindex][objectindex].add(deplink)
					dep_maps.append(dep_map)
				self.temp_resources['dep_maps'] = dep_maps
				
			inv_dep_maps = []
			for inst in dep_maps:
				inv_dep_map = {}
				for subjectindex in inst:
					for objectindex in inst[subjectindex]:
						if objectindex not in inv_dep_map:
							inv_dep_map[objectindex] = {}
						inv_dep_map[objectindex][subjectindex] = inst[subjectindex][objectindex]
				inv_dep_maps.append(inv_dep_map)
			self.temp_resources['inv_dep_maps'] = inv_dep_maps

		result = []
		for i in range(0, len(data)):
			line = data[i]
			sent = line[0].strip().split(' ')
			target = line[1].strip().lower()
			head = int(line[2].strip())
			inv_dep_map = inv_dep_maps[i]
			insts = set([])
			if head in inv_dep_map:
				for object in inv_dep_map[head]:
					for dep_link in inv_dep_map[head][object]:
						insts.add((dep_link, sent[object]))
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				total = 0.0
				if len(insts)>0:
					for inst in insts:
						ngram = inst[0] + ' ' + inst[1] + ' ' + word
						if ngram in model:
							total += model[ngram]
					if total>0.0:
						total /= float(len(insts))
				else:
					total = 99999.0
				result.append(total)
		return result
		
	def allDependencyProbabilityFeature(self, data, args):
		model = self.resources[args[0]]
		parser = self.resources[args[1]]
		
		#Get parsed sentences:
		if 'inv_dep_maps' in self.temp_resources:
			inv_dep_maps = self.temp_resources['inv_dep_maps']
		else:
			dep_maps = None
			if 'dep_maps' in self.temp_resources:
				dep_maps = self.temp_resources['dep_maps']
			else:
				sentences = [l[0].strip().split(' ') for l in data]
				dep_parsed_sents = None
				if 'dep_parsed_sents' in self.temp_resources:
					dep_parsed_sents = self.temp_resources['dep_parsed_sents']
				else:
					dep_parsed_sents = dependencyParseSentences(parser, sentences)
					self.temp_resources['dep_parsed_sents'] = dep_parsed_sents
				dep_maps = []
				for sent in dep_parsed_sents:
					dep_map = {}
					for parse in sent:
						deplink = str(parse[0])
						subjectindex = int(str(parse[2]))-1
						objectindex = int(str(parse[4]))-1
						if subjectindex not in dep_map:
							dep_map[subjectindex] = {objectindex: set([deplink])}
						elif objectindex not in dep_map[subjectindex]:
							dep_map[subjectindex][objectindex] = set([deplink])
						else:
							dep_map[subjectindex][objectindex].add(deplink)
					dep_maps.append(dep_map)
				self.temp_resources['dep_maps'] = dep_maps
				
			inv_dep_maps = []
			for inst in dep_maps:
				inv_dep_map = {}
				for subjectindex in inst:
					for objectindex in inst[subjectindex]:
						if objectindex not in inv_dep_map:
							inv_dep_map[objectindex] = {}
						inv_dep_map[objectindex][subjectindex] = inst[subjectindex][objectindex]
				inv_dep_maps.append(inv_dep_map)
			self.temp_resources['inv_dep_maps'] = inv_dep_maps

		dep_maps = self.temp_resources['dep_maps']
		inv_dep_maps = self.temp_resources['inv_dep_maps']
			
		result = []
		for i in range(0, len(data)):
			line = data[i]
			sent = line[0].strip().split(' ')
			target = line[1].strip().lower()
			head = int(line[2].strip())
			
			dep_map = dep_maps[i]
			inv_dep_map = inv_dep_maps[i]
			insts = set([])
			if head in dep_map:
				for object in dep_map[head]:
					for dep_link in dep_map[head][object]:
						insts.add((dep_link, sent[object]))
			insts_inv = set([])
			if head in inv_dep_map:
				for object in inv_dep_map[head]:
					for dep_link in inv_dep_map[head][object]:
						insts_inv.add((dep_link, sent[object]))
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				total = 0.0
				if len(insts)>0 or len(insts_inv)>0:
					for inst in insts:
						ngram = inst[0] + ' ' + word + ' ' + inst[1]
						prob = math.exp(model.score(ngram, bos=False, eos=False))
						#prob = math.exp(model.score(ngram))
						total += prob
					for inst in insts_inv:
						ngram = inst[0] + ' ' + inst[1] + ' ' + word
						prob = math.exp(model.score(ngram, bos=False, eos=False))
						#prob = math.exp(model.score(ngram))
						total += prob
					total /= float(len(insts)+len(insts_inv))
				else:
					total = 1.0
				result.append(total)
		return result
		
	def binaryAllDependencyFeature(self, data, args):
		model = self.resources[args[0]]
		parser = self.resources[args[1]]
		
		#Get parsed sentences:
		if 'inv_dep_maps' in self.temp_resources:
			inv_dep_maps = self.temp_resources['inv_dep_maps']
		else:
			dep_maps = None
			if 'dep_maps' in self.temp_resources:
				dep_maps = self.temp_resources['dep_maps']
			else:
				sentences = [l[0].strip().split(' ') for l in data]
				dep_parsed_sents = None
				if 'dep_parsed_sents' in self.temp_resources:
					dep_parsed_sents = self.temp_resources['dep_parsed_sents']
				else:
					dep_parsed_sents = dependencyParseSentences(parser, sentences)
					self.temp_resources['dep_parsed_sents'] = dep_parsed_sents
				dep_maps = []
				for sent in dep_parsed_sents:
					dep_map = {}
					for parse in sent:
						deplink = str(parse[0])
						subjectindex = int(str(parse[2]))-1
						objectindex = int(str(parse[4]))-1
						if subjectindex not in dep_map:
							dep_map[subjectindex] = {objectindex: set([deplink])}
						elif objectindex not in dep_map[subjectindex]:
							dep_map[subjectindex][objectindex] = set([deplink])
						else:
							dep_map[subjectindex][objectindex].add(deplink)
					dep_maps.append(dep_map)
				self.temp_resources['dep_maps'] = dep_maps
				
			inv_dep_maps = []
			for inst in dep_maps:
				inv_dep_map = {}
				for subjectindex in inst:
					for objectindex in inst[subjectindex]:
						if objectindex not in inv_dep_map:
							inv_dep_map[objectindex] = {}
						inv_dep_map[objectindex][subjectindex] = inst[subjectindex][objectindex]
				inv_dep_maps.append(inv_dep_map)
			self.temp_resources['inv_dep_maps'] = inv_dep_maps

		dep_maps = self.temp_resources['dep_maps']
		inv_dep_maps = self.temp_resources['inv_dep_maps']
			
		result = []
		for i in range(0, len(data)):
			line = data[i]
			sent = line[0].strip().split(' ')
			target = line[1].strip().lower()
			head = int(line[2].strip())
			
			dep_map = dep_maps[i]
			inv_dep_map = inv_dep_maps[i]
			insts = set([])
			if head in dep_map:
				for object in dep_map[head]:
					for dep_link in dep_map[head][object]:
						insts.add((dep_link, sent[object]))
			insts_inv = set([])
			if head in inv_dep_map:
				for object in inv_dep_map[head]:
					for dep_link in inv_dep_map[head][object]:
						insts_inv.add((dep_link, sent[object]))
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				total = 1.0
				if len(insts)>0 or len(insts_inv)>0:
					for inst in insts:
						ngram = inst[0] + ' ' + word + ' ' + inst[1]
						if ngram not in model:
							total = 0.0
					for inst in insts_inv:
						ngram = inst[0] + ' ' + inst[1] + ' ' + word
						if ngram not in model:
							total = 0.0
				else:
					total = 1.0
				result.append(total)
		return result
		
	def allDependencyFrequencyFeature(self, data, args):
		model = self.resources[args[0]]
		parser = self.resources[args[1]]
		
		#Get parsed sentences:
		if 'inv_dep_maps' in self.temp_resources:
			inv_dep_maps = self.temp_resources['inv_dep_maps']
		else:
			dep_maps = None
			if 'dep_maps' in self.temp_resources:
				dep_maps = self.temp_resources['dep_maps']
			else:
				sentences = [l[0].strip().split(' ') for l in data]
				dep_parsed_sents = None
				if 'dep_parsed_sents' in self.temp_resources:
					dep_parsed_sents = self.temp_resources['dep_parsed_sents']
				else:
					dep_parsed_sents = dependencyParseSentences(parser, sentences)
					self.temp_resources['dep_parsed_sents'] = dep_parsed_sents
				dep_maps = []
				for sent in dep_parsed_sents:
					dep_map = {}
					for parse in sent:
						deplink = str(parse[0])
						subjectindex = int(str(parse[2]))-1
						objectindex = int(str(parse[4]))-1
						if subjectindex not in dep_map:
							dep_map[subjectindex] = {objectindex: set([deplink])}
						elif objectindex not in dep_map[subjectindex]:
							dep_map[subjectindex][objectindex] = set([deplink])
						else:
							dep_map[subjectindex][objectindex].add(deplink)
					dep_maps.append(dep_map)
				self.temp_resources['dep_maps'] = dep_maps
				
			inv_dep_maps = []
			for inst in dep_maps:
				inv_dep_map = {}
				for subjectindex in inst:
					for objectindex in inst[subjectindex]:
						if objectindex not in inv_dep_map:
							inv_dep_map[objectindex] = {}
						inv_dep_map[objectindex][subjectindex] = inst[subjectindex][objectindex]
				inv_dep_maps.append(inv_dep_map)
			self.temp_resources['inv_dep_maps'] = inv_dep_maps

		dep_maps = self.temp_resources['dep_maps']
		inv_dep_maps = self.temp_resources['inv_dep_maps']
			
		result = []
		for i in range(0, len(data)):
			line = data[i]
			sent = line[0].strip().split(' ')
			target = line[1].strip().lower()
			head = int(line[2].strip())
			
			dep_map = dep_maps[i]
			inv_dep_map = inv_dep_maps[i]
			insts = set([])
			if head in dep_map:
				for object in dep_map[head]:
					for dep_link in dep_map[head][object]:
						insts.add((dep_link, sent[object]))
			insts_inv = set([])
			if head in inv_dep_map:
				for object in inv_dep_map[head]:
					for dep_link in inv_dep_map[head][object]:
						insts_inv.add((dep_link, sent[object]))
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				total = 0.0
				if len(insts)>0 or len(insts_inv)>0:
					for inst in insts:
						ngram = inst[0] + ' ' + word + ' ' + inst[1]
						if ngram in model:
							total += model[ngram]
					for inst in insts_inv:
						ngram = inst[0] + ' ' + inst[1] + ' ' + word
						if ngram in model:
							total += model[ngram]
					if total>0.0:
						total /= float(len(insts)+len(insts_inv))
				else:
					total = 99999.0
				result.append(total)
		return result
		
	def wordVectorContextSimilarityFeature(self, data, args):
		model = self.resources[args[0]]
		tagger = self.resources[args[1]]
		result = []
		
		#Get tagged sentences:
		tagged_sents = None
		if 'tagged_sents' in self.temp_resources:
			tagged_sents = self.temp_resources['tagged_sents']
		else:
			sentences = [l[0].strip().split(' ') for l in data]
			tagged_sents = tagger.tag_sents(sentences)
			self.temp_resources['tagged_sents'] = tagged_sents
			
		for i in range(0, len(data)):
			line = data[i]
			tokens = line[0].strip().split(' ')
			target = line[1].strip().lower()
			head = int(line[2].strip())
			
			#Get content words in sentence:
			content_words = set([])
			for j in range(0, len(tokens)):
				token = tokens[j]
				tag = tagged_sents[i][j][1]
				if self.isContentWord(token, tag):
					content_words.add(token)
			
			#Produce divisor:
			divisor = float(len(content_words))
			
			for subst in line[3:len(line)]:
				word = subst.strip().split(':')[1].strip()
				similarity = 0.0
				for content_word in content_words:
					try:
						similarity += model.similarity(content_word, word)
					except KeyError:
						try:
							similarity += model.similarity(content_word, word.lower())
						except KeyError:
							pass
				similarity /= divisor
				result.append(similarity)
		return result
		
	def taggedWordVectorContextSimilarityFeature(self, data, args):
		model = self.resources[args[0]]
		tagger = self.resources[args[1]]
		pos_type = args[2]
		result = []
		
		#Get tagged sentences:
		tagged_sents = None
		if 'tagged_sents' in self.temp_resources:
			tagged_sents = self.temp_resources['tagged_sents']
		else:
			sentences = [l[0].strip().split(' ') for l in data]
			tagged_sents = tagger.tag_sents(sentences)
			self.temp_resources['tagged_sents'] = tagged_sents
			
		
		#Produce embeddings vector tags:
		model_tagged_sents = None
		if pos_type=='paetzold':
			transformed = []
			for sent in tagged_sents:
				tokens = []
				for token in sent:
					tokens.append((token[0], getGeneralisedPOS(token[1])))
				transformed.append(tokens)
			model_tagged_sents = transformed
		else:
			model_tagged_sents = tagged_sents
			
		for i in range(0, len(data)):
			line = data[i]
			tokens = line[0].strip().split(' ')
			target = line[1].strip().lower()
			head = int(line[2].strip())
			target_pos = model_tagged_sents[i][head][1]
			
			#Get content words in sentence:
			content_words = set([])
			for j in range(0, len(tokens)):
				token = tokens[j]
				tag = tagged_sents[i][j][1]
				model_tag = model_tagged_sents[i][j][1]
				if self.isContentWord(token, tag):
					content_words.add(token+'|||'+model_tag)
			
			#Produce divisor:
			divisor = float(len(content_words))
			
			for subst in line[3:len(line)]:
				word = subst.strip().split(':')[1].strip()
				similarity = 0.0
				for content_word in content_words:
					try:
						similarity += model.similarity(content_word, word+'|||'+target_pos)
					except KeyError:
						try:
							similarity += model.similarity(content_word, word.lower()+'|||'+target_pos)
						except KeyError:
							pass
				similarity /= divisor
				result.append(similarity)
		return result
		
	def nullLinkNominalFeature(self, data, args):
		parser = self.resources[args[0]]
		
		#Get parsed sentences:
		if 'inv_dep_maps' in self.temp_resources:
			inv_dep_maps = self.temp_resources['inv_dep_maps']
		else:
			dep_maps = None
			if 'dep_maps' in self.temp_resources:
				dep_maps = self.temp_resources['dep_maps']
			else:
				sentences = [l[0].strip().split(' ') for l in data]
				dep_parsed_sents = None
				if 'dep_parsed_sents' in self.temp_resources:
					dep_parsed_sents = self.temp_resources['dep_parsed_sents']
				else:
					dep_parsed_sents = dependencyParseSentences(parser, sentences)
					self.temp_resources['dep_parsed_sents'] = dep_parsed_sents
				dep_maps = []
				for sent in dep_parsed_sents:
					dep_map = {}
					for parse in sent:
						deplink = str(parse[0])
						subjectindex = int(str(parse[2]))-1
						objectindex = int(str(parse[4]))-1
						if subjectindex not in dep_map:
							dep_map[subjectindex] = {objectindex: set([deplink])}
						elif objectindex not in dep_map[subjectindex]:
							dep_map[subjectindex][objectindex] = set([deplink])
						else:
							dep_map[subjectindex][objectindex].add(deplink)
					dep_maps.append(dep_map)
				self.temp_resources['dep_maps'] = dep_maps
				
			inv_dep_maps = []
			for inst in dep_maps:
				inv_dep_map = {}
				for subjectindex in inst:
					for objectindex in inst[subjectindex]:
						if objectindex not in inv_dep_map:
							inv_dep_map[objectindex] = {}
						inv_dep_map[objectindex][subjectindex] = inst[subjectindex][objectindex]
				inv_dep_maps.append(inv_dep_map)
			self.temp_resources['inv_dep_maps'] = inv_dep_maps

		dep_maps = self.temp_resources['dep_maps']
		inv_dep_maps = self.temp_resources['inv_dep_maps']
			
		result = []
		for i in range(0, len(data)):
			line = data[i]
			sent = line[0].strip().split(' ')
			target = line[1].strip().lower()
			head = int(line[2].strip())
			
			dep_map = dep_maps[i]
			inv_dep_map = inv_dep_maps[i]
			value = False
			if head in dep_map or head in inv_dep_map:
				value = True
				
			for subst in line[3:len(line)]:
				result.append(value)
		return result
		
	def backoffBehaviorNominalFeature(self, data, args):
		ngrams = args[0]
		result = []
		counts = self.resources[ngrams]
		for line in data:
			sent = line[0].strip().split(' ')
			target = line[1]
			head = int(line[2])
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				ngram2t, bos2t, eos2t = self.getNgram(word, sent, head, 2, 0)
				ngram1t, bos1t, eos1t = self.getNgram(word, sent, head, 1, 0)
				ngram0t, bos0t, eos0t = self.getNgram(word, sent, head, 0, 0)
				ngram2f, bos2f, eos2f = word, True, False
				ngram1f, bos1f, eos1f = word, True, False
				if head>0:
					ngram2f, bos2f, eos2f = self.getNgram(sent[head-1], sent, head-1, 1, 0)
					ngram1f, bos1f, eos1f = self.getNgram(sent[head-1], sent, head-1, 0, 0)
				
				backoff = -1
				if ngram2t in counts:
					backoff = 7.0
				elif ngram2f in counts and ngram1t in counts:
					backoff = 6.0
				elif ngram1t in counts:
					backoff = 5.0
				elif ngram2f in counts and ngram0t in counts:
					backoff = 4.0
				elif ngram1f in counts and ngram0t in counts:
					backoff = 3.0
				elif ngram0t in counts:
					backoff = 2.0
				else:
					backoff = 1.0
				result.append(backoff)
		return result
		
	def candidateNominalFeature(self, data, args):
		result = []
		for line in data:
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				result.append(words)
		return result
		
	def ngramNominalFeature(self, data, args):
		spanl = args[0]
		spanr = args[1]
		result = []
		for line in data:
			sent = line[0].strip().split(' ')
			target = line[1]
			head = int(line[2])
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				ngram, bosv, eosv = self.getNgram(word, sent, head, spanl, spanr)
				tokens = ngram.split(' ')
				fngram = ''
				for token in tokens:
					fngram += token + '|||'
				result.append(fngram[0:len(fngram)-3])
		return result
		
	def candidatePOSNominalFeature(self, data, args):
		result = []
		
		tagger = self.resources[args[0]]
		pos_type = args[1]
		
		#Get tagged sentences:
		tagged_sents = None
		if 'tagged_sents' in self.temp_resources:
			tagged_sents = self.temp_resources['tagged_sents']
		else:
			sentences = [l[0].strip().split(' ') for l in data]
			tagged_sents = tagger.tag_sents(sentences)
			self.temp_resources['tagged_sents'] = tagged_sents
			
		#Transform them to the right format:
		if pos_type=='paetzold':
			transformed = []
			for sent in tagged_sents:
				tokens = []
				for token in sent:
					tokens.append((token[0], getGeneralisedPOS(token[1])))
				transformed.append(tokens)
			tagged_sents = transformed

		for i in range(0, len(data)):
			line = data[i]
			target = line[1].strip().lower()
			head = int(line[2].strip())
			target_pos = tagged_sents[i][head][1]
			for subst in line[3:len(line)]:
				result.append(target_pos)
		return result
		
	def POSNgramNominalFeature(self, data, args):
		result = []
		
		spanl = args[0]
		spanr = args[1]
		tagger = self.resources[args[2]]
		pos_type = args[3]
		
		#Get tagged sentences:
		tagged_sents = None
		if 'tagged_sents' in self.temp_resources:
			tagged_sents = self.temp_resources['tagged_sents']
		else:
			sentences = [l[0].strip().split(' ') for l in data]
			tagged_sents = tagger.tag_sents(sentences)
			self.temp_resources['tagged_sents'] = tagged_sents
			
		#Transform them to the right format:
		if pos_type=='paetzold':
			transformed = []
			for sent in tagged_sents:
				tokens = []
				for token in sent:
					tokens.append((token[0], getGeneralisedPOS(token[1])))
				transformed.append(tokens)
			tagged_sents = transformed

		for i in range(0, len(data)):
			line = data[i]
			target = line[1].strip().lower()
			head = int(line[2].strip())
			target_pos = tagged_sents[i][head][1]
			POStokens = [posdata[1] for posdata in tagged_sents[i]]
			for subst in line[3:len(line)]:
				ngram, bosv, eosv = self.getNgram(target_pos, POStokens, head, spanl, spanr)
				tokens = ngram.split(' ')
				fngram = ''
				for token in tokens:
					fngram += token + '|||'
				result.append(fngram[0:len(fngram)-3])
		return result
		
	def POSNgramWithCandidateNominalFeature(self, data, args):
		result = []
		
		spanl = args[0]
		spanr = args[1]
		tagger = self.resources[args[2]]
		pos_type = args[3]
		
		#Get tagged sentences:
		tagged_sents = None
		if 'tagged_sents' in self.temp_resources:
			tagged_sents = self.temp_resources['tagged_sents']
		else:
			sentences = [l[0].strip().split(' ') for l in data]
			tagged_sents = tagger.tag_sents(sentences)
			self.temp_resources['tagged_sents'] = tagged_sents
			
		#Transform them to the right format:
		if pos_type=='paetzold':
			transformed = []
			for sent in tagged_sents:
				tokens = []
				for token in sent:
					tokens.append((token[0], getGeneralisedPOS(token[1])))
				transformed.append(tokens)
			tagged_sents = transformed

		for i in range(0, len(data)):
			line = data[i]
			target = line[1].strip().lower()
			head = int(line[2].strip())
			target_pos = tagged_sents[i][head][1]
			POStokens = [posdata[1] for posdata in tagged_sents[i]]
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				ngram, bosv, eosv = self.getNgram(word, POStokens, head, spanl, spanr)
				tokens = ngram.split(' ')
				fngram = ''
				for token in tokens:
					fngram += token + '|||'
				result.append(fngram[0:len(fngram)-3])
		return result
		
	def imageSearchCountFeature(self, data, args):
		result = []
		
		key = args[0]

		for i in range(0, len(data)):
			line = data[i]
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				imagecount = None
				if word not in self.resources['image_counts']:
					imagecount = self.getImageCount(word, key)
					self.resources['image_counts'][word] = imagecount
				else:
					imagecount = self.resources['image_counts'][word]
				result.append(imagecount)
		return result
		
	def webSearchCountFeature(self, data, args):
		result = []

		for i in range(0, len(data)):
			line = data[i]
			for subst in line[3:len(line)]:
				word = subst.split(':')[1].strip()
				pagecount = None
				if word not in self.resources['page_counts']:
					pagecount = self.getPageCount(word)
					self.resources['page_counts'][word] = pagecount
				else:
					pagecount = self.resources['page_counts'][word]
				result.append(pagecount)
		return result
		
	def getImageCount(self, word, key):
		headers = {}
		headers['Api-Key'] = key
		tokens = word.strip().split(' ')
		suffix = ''
		for token in tokens:
			suffix += token + '+'
		suffix = suffix[0:len(suffix)-1]
		
		#Make HTTP request:
		url = 'https://api.gettyimages.com/v3/search/images?fields=id&phrase='+suffix
		req = urllib2.Request(url=url, headers=headers)
		
		#Send request:
		count = None
		try:
			f = urllib2.urlopen(req)
			data = json.loads(f.read())
			count = int(data['result_count'])
		except Exception:
			count = 0
		return count
		
	def getPageCount(self, word):
		tokens = word.strip().split(' ')
		suffix = ''
		for token in tokens:
			suffix += token + '+'
		suffix = suffix[0:len(suffix)-1]
		
		#Make HTTP request:
		exp = re.compile('class=\"sb_count\"[^>]*>([^<]+)<')
		url = 'https://www.bing.com/search?q='+suffix
		req = urllib2.Request(url=url)
		
		#Send request:
		count = None
		try:
			f = urllib2.urlopen(req)
			data = f.read()
			result = exp.findall(data)
			count = int(result[0].strip().split(' ')[0].strip().replace(',', ''))
		except Exception:
			count = 0
		return count
		
	def morphologicalFeature(self, data, args):
		dictionary = args[0]
		result = []
		for line in data:
			for subst in line[3:len(line)]:
				words = subst.strip().split(':')[1].strip()
				if words in dictionary:
					result.append(dictionary[words])
				else:
					result.append(0.0)
		return result
		
	def readNgramFile(self, ngram_file):
		counts = shelve.open(ngram_file, protocol=pickle.HIGHEST_PROTOCOL)
		return counts

	def isContentWord(self, word, tag):
		content_tags = set(['JJ', 'JJS', 'JJR', 'NN', 'NNS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
		if tag in content_tags:
			return True
		else:
			return False
	
	def addWordVectorValues(self, model, size, orientation):
		"""
		Adds all the word vector values of a model to the estimator.
	
		@param model: Path to a binary word vector model.
		For instructions on how to create the model, please refer to the LEXenstein Manual.
		@param size: Number of feature values that represent a word in the model.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if model not in self.resources:
				m = gensim.models.word2vec.Word2Vec.load_word2vec_format(model, binary=True)
				self.resources[model] = m
			self.features.append((self.wordVectorValuesFeature, [model, size]))
			for i in range(0, size):
				self.identifiers.append(('Word Vector Value '+str(i)+' (Model: '+model+')', orientation))
	
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
			if pos_model not in self.resources:
				tagger = StanfordPOSTagger(pos_model, stanford_tagger)
				self.resources[pos_model] = tagger
			if condprob_model not in self.resources:
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
			if model not in self.resources:
				m = gensim.models.word2vec.Word2Vec.load_word2vec_format(model, binary=True)
				self.resources[model] = m
			self.features.append((self.wordVectorSimilarityFeature, [model]))
			self.identifiers.append(('Word Vector Similarity (Model: '+model+')', orientation))
			
	def addTaggedWordVectorSimilarityFeature(self, model, pos_model, stanford_tagger, java_path, pos_type, orientation):
		"""
		Adds a tagged word vector similarity feature to the estimator.
		The value will be the similarity between the word vector of a target complex word and the word vector of a candidate, while accompanied by their POS tags.
		Each entry in the word vector model must be in the following format: <word>|||<tag>
		To create a corpus for such model to be trained, one must tag each word in a corpus, and then concatenate words and tags using the aforementioned convention.
	
		@param model: Path to a binary tagged word vector model.
		For instructions on how to create the model, please refer to the LEXenstein Manual.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param pos_type: The type of POS tags to be used.
		Values supported: treebank, paetzold
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			os.environ['JAVAHOME'] = java_path
			if model not in self.resources:
				m = gensim.models.word2vec.Word2Vec.load_word2vec_format(model, binary=True)
				self.resources[model] = m
			if pos_model not in self.resources:
				tagger = StanfordPOSTagger(pos_model, stanford_tagger)
				self.resources[pos_model] = tagger
			self.features.append((self.taggedWordVectorSimilarityFeature, [model, pos_model, pos_type]))
			self.identifiers.append(('Word Vector Similarity (Model: '+model+') (POS Model: '+pos_model+') (POS Type: '+pos_type+')', orientation))
	
	def addTranslationProbabilityFeature(self, translation_probabilities, orientation):
		"""
		Adds a translation probability feature to the estimator.
		The value will be the probability of a target complex word of being translated into a given candidate substitution.
	
		@param translation_probabilities: Path to a shelve file containing translation probabilities.
		To produce the file, first run the following command through fast_align:
		fast_align -i <parallel_data> -v -d -o <translation_probabilities_file>
		Then, produce a shelve file with the "addTranslationProbabilitiesFileToShelve" function from LEXenstein's "util" module.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		probabilities = self.readNgramFile(translation_probabilities)
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if translation_probabilities not in self.resources:
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
			if lexicon not in self.resources:
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
			if language_model not in self.resources:
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			self.features.append((self.collocationalFeature, [language_model, leftw, rightw]))
			for i in range(0, leftw+1):
				for j in range(0, rightw+1):
					self.identifiers.append(('Collocational Feature ['+str(i)+', '+str(j)+'] (LM: '+language_model+')', orientation))
					
	def addFrequencyCollocationalFeature(self, ngram_file, leftw, rightw, orientation):
		"""
		Adds a set of frequency collocational features to the estimator.
		The values will be the n-gram frequencies of all collocational features selected.
		Each feature is the frequency of an n-gram with 0<=l<=leftw tokens to the left and 0<=r<=rightw tokens to the right.
		This method creates (leftw+1)*(rightw+1) features.
		To produce the ngram counts file, the user must first acquire a large corpus of text.
		In sequence, the user can then use SRILM to produce an ngram counts file with the "-write" option.
		Finally, the user must create a shelve file using the "addNgramCountsFileToShelve" function from the "util" module.
	
		@param ngram_file: Path to a shelve file containing n-gram frequency counts.
		@param leftw: Maximum number of tokens to the left.
		@param rightw: Maximum number of tokens to the right.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if ngram_file not in self.resources:
				counts = self.readNgramFile(ngram_file)
				self.resources[ngram_file] = counts
			self.features.append((self.frequencyCollocationalFeature, [ngram_file, leftw, rightw]))
			for i in range(0, leftw+1):
				for j in range(0, rightw+1):
					self.identifiers.append(('Frequency Collocational Feature ['+str(i)+', '+str(j)+'] (N-Grams File: '+ngram_file+')', orientation))
					
	def addTaggedFrequencyCollocationalFeature(self, ngram_file, leftw, rightw, pos_model, stanford_tagger, java_path, pos_type, orientation):
		"""
		Adds a set of frequency tagged n-gram frequency features to the estimator.
		The values will be the n-gram frequencies of all tagged collocational features selected.
		Each feature is the frequency of an n-gram with 0<=l<=leftw tagged tokens to the left and 0<=r<=rightw tagged tokens to the right.
		This method creates (leftw+1)*(rightw+1) features.
		This function requires for a special type of ngram counts file.
		Each n-gram in the file must be composed of n-1 tags, and exactly 1 word.
		To produce this file, one must first parse a corpus and create a corpus with n-grams in the aforementioned format.
		The user can then use SRILM to produce an ngram counts file with the "-write" option.
		Finally, the user must create a shelve file using the "addNgramCountsFileToShelve" function from the "util" module.
	
		@param ngram_file: Path to a shelve file containing n-gram frequency counts.
		@param leftw: Maximum number of tokens to the left.
		@param rightw: Maximum number of tokens to the right.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param pos_type: The type of POS tags to be used.
		Values supported: treebank, paetzold
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if ngram_file not in self.resources:
				counts = self.readNgramFile(ngram_file)
				self.resources[ngram_file] = counts
			os.environ['JAVAHOME'] = java_path
			if pos_model not in self.resources:
				tagger = StanfordPOSTagger(pos_model, stanford_tagger)
				self.resources[pos_model] = tagger
			self.features.append((self.taggedFrequencyCollocationalFeature, [ngram_file, leftw, rightw, pos_model, pos_type]))
			for i in range(0, leftw+1):
				for j in range(0, rightw+1):
					self.identifiers.append(('Tagged Frequency Collocational Feature ['+str(i)+', '+str(j)+'] (N-Grams File: '+ngram_file+') (POS type: '+pos_type+')', orientation))
	
	def addBinaryTaggedFrequencyCollocationalFeature(self, ngram_file, leftw, rightw, pos_model, stanford_tagger, java_path, pos_type, orientation):
		"""
		Adds a set of binary tagged frequency collocational features to the estimator.
		The values will be the binary n-gram values of all tagged collocational features selected.
		Each feature is the frequency of an n-gram with 0<=l<=leftw tagged tokens to the left and 0<=r<=rightw tagged tokens to the right.
		This method creates (leftw+1)*(rightw+1) features.
		This function requires for a special type of ngram counts file.
		Each n-gram in the file must be composed of n-1 tags, and exactly 1 word.
		To produce this file, one must first parse a corpus and create a corpus with n-grams in the aforementioned format.
		The user can then use SRILM to produce an ngram counts file with the "-write" option.
		Finally, the user must create a shelve file using the "addNgramCountsFileToShelve" function from the "util" module.
	
		@param ngram_file: Path to a shelve file containing n-gram frequency counts.
		@param leftw: Maximum number of tokens to the left.
		@param rightw: Maximum number of tokens to the right.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param pos_type: The type of POS tags to be used.
		Values supported: treebank, paetzold
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if ngram_file not in self.resources:
				counts = self.readNgramFile(ngram_file)
				self.resources[ngram_file] = counts
			os.environ['JAVAHOME'] = java_path
			if pos_model not in self.resources:
				tagger = StanfordPOSTagger(pos_model, stanford_tagger)
				self.resources[pos_model] = tagger
			self.features.append((self.binaryTaggedFrequencyCollocationalFeature, [ngram_file, leftw, rightw, pos_model, pos_type]))
			for i in range(0, leftw+1):
				for j in range(0, rightw+1):
					self.identifiers.append(('Binary Tagged Frequency Collocational Feature ['+str(i)+', '+str(j)+'] (N-Grams File: '+ngram_file+') (POS type: '+pos_type+')', orientation))
	
	def addPopCollocationalFeature(self, language_model, leftw, rightw, orientation):
		"""
		Adds a set of "pop" collocational features to the estimator.
		Each feature is the probability of an n-gram with 0<=l<=leftw tokens to the left and 0<=r<=rightw tokens to the right.
		The value of each feature will be the highest frequency between all "popping" n-gram combinations of one token to the left and right.
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
			if language_model not in self.resources:
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			self.features.append((self.popCollocationalFeature, [language_model, leftw, rightw]))
			for i in range(0, leftw+1):
				for j in range(0, rightw+1):
					self.identifiers.append(('Pop Collocational Feature ['+str(i)+', '+str(j)+'] (LM: '+language_model+')', orientation))
					
	def addNGramProbabilityFeature(self, language_model, leftw, rightw, orientation):
		"""
		Adds a n-gram probability feature to the estimator.
		The value will be the language model probability of the n-gram composed by leftw tokens to the left and rightw tokens to the right of a given word.
	
		@param language_model: Path to the language model from which to extract probabilities.
		@param leftw: Number of tokens to the left.
		@param rightw: Number of tokens to the right.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if language_model not in self.resources:
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			self.features.append((self.ngramProbabilityFeature, [language_model, leftw, rightw]))
			self.identifiers.append(('N-Gram Probability Feature ['+str(leftw)+', '+str(rightw)+'] (LM: '+language_model+')', orientation))
			
	def addNGramFrequencyFeature(self, ngram_file, leftw, rightw, orientation):
		"""
		Adds a n-gram frequency feature to the estimator.
		The value will be the the frequency of the n-gram composed by leftw tokens to the left and rightw tokens to the right of a given word.
		To produce the ngram counts file, the user must first acquire a large corpus of text.
		In sequence, the user can then use SRILM to produce an ngram counts file with the "-write" option.
		Finally, the user must create a shelve file using the "addNgramCountsFileToShelve" function from the "util" module.
	
		@param ngram_file: Path to a shelve file containing n-gram frequency counts.
		@param leftw: Number of tokens to the left.
		@param rightw: Number of tokens to the right.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if ngram_file not in self.resources:
				counts = self.readNgramFile(ngram_file)
				self.resources[ngram_file] = counts
			self.features.append((self.ngramFrequencyFeature, [ngram_file, leftw, rightw]))
			self.identifiers.append(('N-Gram Frequency Feature ['+str(leftw)+', '+str(rightw)+'] (N-grams File: '+ngram_file+')', orientation))
			
	def addBinaryNGramFrequencyFeature(self, ngram_file, leftw, rightw, orientation):
		"""
		Adds a binary n-gram frequency feature to the estimator.
		The value will be 1 if the n-gram composed by leftw tokens to the left and rightw tokens to the right of a given word are in the n-grams file, and 0 otherwise.
		To produce the ngram counts file, the user must first acquire a large corpus of text.
		In sequence, the user can then use SRILM to produce an ngram counts file with the "-write" option.
		Finally, the user must create a shelve file using the "addNgramCountsFileToShelve" function from the "util" module.
	
		@param ngram_file: Path to a shelve file containing n-gram frequency counts.
		@param leftw: Number of tokens to the left.
		@param rightw: Number of tokens to the right.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if ngram_file not in self.resources:
				counts = self.readNgramFile(ngram_file)
				self.resources[ngram_file] = counts
			self.features.append((self.binaryNgramFrequencyFeature, [ngram_file, leftw, rightw]))
			self.identifiers.append(('Binary N-Gram Probability Feature ['+str(leftw)+', '+str(rightw)+'] (N-grams File: '+ngram_file+')', orientation))
			
	def addPopNGramProbabilityFeature(self, language_model, leftw, rightw, orientation):
		"""
		Adds a pop n-gram probability feature to the estimator.
		The value is the highest probability of the n-gram with leftw tokens to the left and rightw tokens to the right, with a popping window of one token to the left and right.
	
		@param language_model: Path to the language model from which to extract probabilities.
		@param leftw: Number of tokens to the left.
		@param rightw: Number of tokens to the right.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if language_model not in self.resources:
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			self.features.append((self.popNgramProbabilityFeature, [language_model, leftw, rightw]))
			self.identifiers.append(('Pop N-Gram Frequency Feature ['+str(leftw)+', '+str(rightw)+'] (LM: '+language_model+')', orientation))
			
	def addPopNGramFrequencyFeature(self, ngram_file, leftw, rightw, orientation):
		"""
		Adds a pop n-gram frequency feature to the estimator.
		The value is the highest raw frequency count of the n-gram with leftw tokens to the left and rightw tokens to the right, with a popping window of one token to the left and right.
		To produce the ngram counts file, the user must first acquire a large corpus of text.
		In sequence, the user can then use SRILM to produce an ngram counts file with the "-write" option.
		Finally, the user must create a shelve file using the "addNgramCountsFileToShelve" function from the "util" module.
	
		@param ngram_file: Path to a shelve file containing n-gram frequency counts.
		@param leftw: Number of tokens to the left.
		@param rightw: Number of tokens to the right.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if ngram_file not in self.resources:
				counts = self.readNgramFile(ngram_file)
				self.resources[ngram_file] = counts
			self.features.append((self.popNgramFrequencyFeature, [ngram_file, leftw, rightw]))
			self.identifiers.append(('Pop N-Gram Frequency Feature ['+str(leftw)+', '+str(rightw)+'] (N-grams File: '+ngram_file+')', orientation))
		
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
			if language_model not in self.resources:
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			self.features.append((self.sentenceProbabilityFeature, [language_model]))
			self.identifiers.append(('Sentence Probability (LM: '+language_model+')', orientation))
			
	def addReverseSentenceProbabilityFeature(self, language_model, orientation):
		"""
		Adds a reverse sentence probability feature to the estimator.
		The value will be the language model probability of each inverted sentence in the VICTOR corpus with its target complex word replaced by a candidate.
	
		@param language_model: Path to the language model from which to extract probabilities.
		This language model must be trained over a corpus composed of inverted sentences (Ex: ". sentence a is This").
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if language_model not in self.resources:
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			self.features.append((self.reverseSentenceProbabilityFeature, [language_model]))
			self.identifiers.append(('Reverse Sentence Probability (LM: '+language_model+')', orientation))
			
	def addPrefixProbabilityFeature(self, language_model, orientation):
		"""
		Adds a prefix probability feature to the estimator.
		The value will be the language model probability of all words in each sentence in the VICTOR corpus until the target complex word, while replaced by a candidate.
	
		@param language_model: Path to the language model from which to extract probabilities.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if language_model not in self.resources:
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			self.features.append((self.prefixProbabilityFeature, [language_model]))
			self.identifiers.append(('Prefix Probability (LM: '+language_model+')', orientation))
			
	def addReversePrefixProbabilityFeature(self, language_model, orientation):
		"""
		Adds a reverse prefix probability feature to the estimator.
		The value will be the language model probability of all words in each inverted sentence in the VICTOR corpus until the target complex word, while replaced by a candidate.
	
		@param language_model: Path to the language model from which to extract probabilities.
		This language model must be trained over a corpus composed of inverted sentences (Ex: ". sentence a is This").
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if language_model not in self.resources:
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			self.features.append((self.reversePrefixProbabilityFeature, [language_model]))
			self.identifiers.append(('Reverse Prefix Probability (LM: '+language_model+')', orientation))
		
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
			
	def addIsSynonymFeature(self, orientation):
		"""
		Adds a synonymy relation feature to the estimator.
		If a candidate substitution is a synonym of the target word, then it returns 1, if not, it returns 0.
		
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			self.features.append((self.isSynonym ,[]))
			self.identifiers.append(('Is Synonym', orientation))
		
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
			
	def addIsHypernymFeature(self, orientation):
		"""
		Adds a hypernymy relation feature to the estimator.
		If a candidate substitution is a hypernym of the target word, then it returns 1, if not, it returns 0.
		
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			self.features.append((self.isHypernym ,[]))
			self.identifiers.append(('Is Hypernym', orientation))
		
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
			
	def addIsHyponymFeature(self, orientation):
		"""
		Adds a hyponymy relation feature to the estimator.
		If a candidate substitution is a hyponym of the target word, then it returns 1, if not, it returns 0.
		
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			self.features.append((self.isHyponym ,[]))
			self.identifiers.append(('Is Hyponym', orientation))
		
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
			
	def addAverageDepthFeature(self, orientation):
		"""
		Adds an average sense depth feature to the estimator.
		Calculates the average distance between two senses of a given candidate.
		
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			self.features.append((self.averageDepth ,[]))
			self.identifiers.append(('Average Sense Depth', orientation))
			
	def addSubjectDependencyProbabilityFeature(self, language_model, stanford_parser, dependency_models, java_path, orientation):
		"""
		Adds a subject dependency probability feature to the estimator.
		The value will be the average language model probability of all dependency links of which the target word is subject, with the target word replaced by a given candidate.
		To train the language model used by this feature, one must first extract dependency links from a large corpora of sentences.
		In sequence, the dependency links must be transformed into the following format: <type_of_dependency_link> <subject_word> <object_word>
		In the format above, each token is space-separated.
		Once transformed, the dependency links can then be placed in a text file, one per line.
		Finally, one can then run any language modelling tool to produce a language model in ARPA format.
	
		@param language_model: Path to the language model from which to extract dependency link probabilities.
		@param stanford_parser: Path to the "stanford-parser.jar" file.
		The parser can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param dependency_models: Path to a JAR file containing parsing models.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if language_model not in self.resources:
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			os.environ['JAVAHOME'] = java_path
			if dependency_models not in self.resources:
				parser = StanfordParser(path_to_jar=stanford_parser, path_to_models_jar=dependency_models)
				self.resources[dependency_models] = parser
			self.features.append((self.subjectDependencyProbabilityFeature, [language_model, dependency_models]))
			self.identifiers.append(('Subject Dependency Probability Feature (Language Model: '+language_model+') (Models: '+dependency_models+')', orientation))
			
	def addBinarySubjectDependencyFeature(self, dep_counts_file, stanford_parser, dependency_models, java_path, orientation):
		"""
		Adds a binary subject dependency feature to the estimator.
		The value will be 1 if all dependency links of which the target word is subject exist for a given candidate, and 0 otherwise.
		To produce the dependency link counts file used by this feature, one must first extract dependency links from a large corpora of sentences.
		In sequence, the dependency links must be transformed into the following format: <type_of_dependency_link> <subject_word> <object_word>
		In the format above, each token is space-separated.
		Once transformed, the dependency links can then be placed in a text file, one per line.
		Finally, one can then run any language modelling tool to produce a language model in ARPA format.
	
		@param dep_counts_file: Path to a shelve file containing dependency link counts.
		@param stanford_parser: Path to the "stanford-parser.jar" file.
		The parser can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param dependency_models: Path to a JAR file containing parsing models.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if dep_counts_file not in self.resources:
				counts = self.readNgramFile(dep_counts_file)
				self.resources[dep_counts_file] = counts
			os.environ['JAVAHOME'] = java_path
			if dependency_models not in self.resources:
				parser = StanfordParser(path_to_jar=stanford_parser, path_to_models_jar=dependency_models)
				self.resources[dependency_models] = parser
			self.features.append((self.binarySubjectDependencyFeature, [dep_counts_file, dependency_models]))
			self.identifiers.append(('Binary Subject Dependency Feature (Dependency Link Counts File: '+dep_counts_file+') (Models: '+dependency_models+')', orientation))
	
	def addSubjectDependencyFrequencyFeature(self, dep_counts_file, stanford_parser, dependency_models, java_path, orientation):
		"""
		Adds a subject dependency frequency feature to the estimator.
		The value will be the average raw frequency of all dependency links of which the target word is subject, with the target word replaced by a given candidate.
		To produce the dependency link counts file used by this feature, one must first extract dependency links from a large corpora of sentences.
		In sequence, the dependency links must be transformed into the following format: <type_of_dependency_link> <subject_word> <object_word>
		In the format above, each token is space-separated.
		Once transformed, the dependency links can then be placed in a text file, one per line.
		Finally, one can then run any language modelling tool to produce a language model in ARPA format.
	
		@param dep_counts_file: Path to a shelve file containing dependency link counts.
		@param stanford_parser: Path to the "stanford-parser.jar" file.
		The parser can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param dependency_models: Path to a JAR file containing parsing models.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if dep_counts_file not in self.resources:
				counts = self.readNgramFile(dep_counts_file)
				self.resources[dep_counts_file] = counts
			os.environ['JAVAHOME'] = java_path
			if dependency_models not in self.resources:
				parser = StanfordParser(path_to_jar=stanford_parser, path_to_models_jar=dependency_models)
				self.resources[dependency_models] = parser
			self.features.append((self.subjectDependencyFrequencyFeature, [dep_counts_file, dependency_models]))
			self.identifiers.append(('Subject Dependency Frequency Feature (Dependency Link Counts File: '+dep_counts_file+') (Models: '+dependency_models+')', orientation))
	
	def addObjectDependencyProbabilityFeature(self, language_model, stanford_parser, dependency_models, java_path, orientation):
		"""
		Adds an object dependency probability feature to the estimator.
		The value will be the average language model probability of all dependency links of which the target word is object, with the target word replaced by a given candidate.
		To train the language model used by this feature, one must first extract dependency links from a large corpora of sentences.
		In sequence, the dependency links must be transformed into the following format: <type_of_dependency_link> <subject_word> <object_word>
		In the format above, each token is space-separated.
		Once transformed, the dependency links can then be placed in a text file, one per line.
		Finally, one can then run any language modelling tool to produce a language model in ARPA format.
	
		@param language_model: Path to the language model from which to extract dependency link probabilities.
		@param stanford_parser: Path to the "stanford-parser.jar" file.
		The parser can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param dependency_models: Path to a JAR file containing parsing models.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if language_model not in self.resources:
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			os.environ['JAVAHOME'] = java_path
			if dependency_models not in self.resources:
				parser = StanfordParser(path_to_jar=stanford_parser, path_to_models_jar=dependency_models)
				self.resources[dependency_models] = parser
			self.features.append((self.objectDependencyProbabilityFeature, [language_model, dependency_models]))
			self.identifiers.append(('Object Dependency Probability Feature (Language Model: '+language_model+') (Models: '+dependency_models+')', orientation))
	
	def addBinaryObjectDependencyFeature(self, dep_counts_file, stanford_parser, dependency_models, java_path, orientation):
		"""
		Adds a binary object dependency feature to the estimator.
		The value will be 1 if all dependency links of which the target word is object exist for a given candidate, and 0 otherwise.
		To produce the dependency link counts file used by this feature, one must first extract dependency links from a large corpora of sentences.
		In sequence, the dependency links must be transformed into the following format: <type_of_dependency_link> <subject_word> <object_word>
		In the format above, each token is space-separated.
		Once transformed, the dependency links can then be placed in a text file, one per line.
		Finally, one can then run any language modelling tool to produce a language model in ARPA format.
	
		@param dep_counts_file: Path to a shelve file containing dependency link counts.
		@param stanford_parser: Path to the "stanford-parser.jar" file.
		The parser can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param dependency_models: Path to a JAR file containing parsing models.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if dep_counts_file not in self.resources:
				counts = self.readNgramFile(dep_counts_file)
				self.resources[dep_counts_file] = counts
			os.environ['JAVAHOME'] = java_path
			if dependency_models not in self.resources:
				parser = StanfordParser(path_to_jar=stanford_parser, path_to_models_jar=dependency_models)
				self.resources[dependency_models] = parser
			self.features.append((self.binaryObjectDependencyFeature, [dep_counts_file, dependency_models]))
			self.identifiers.append(('Binary Object Dependency Feature (Dependency Link Counts File: '+dep_counts_file+') (Models: '+dependency_models+')', orientation))
			
	def addObjectDependencyFrequencyFeature(self, dep_counts_file, stanford_parser, dependency_models, java_path, orientation):
		"""
		Adds an object dependency frequency feature to the estimator.
		The value will be the average raw frequency of all dependency links of which the target word is object, with the target word replaced by a given candidate.
		To produce the dependency link counts file used by this feature, one must first extract dependency links from a large corpora of sentences.
		In sequence, the dependency links must be transformed into the following format: <type_of_dependency_link> <subject_word> <object_word>
		In the format above, each token is space-separated.
		Once transformed, the dependency links can then be placed in a text file, one per line.
		Finally, one can then run any language modelling tool to produce a language model in ARPA format.
	
		@param dep_counts_file: Path to a shelve file containing dependency link counts.
		@param stanford_parser: Path to the "stanford-parser.jar" file.
		The parser can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param dependency_models: Path to a JAR file containing parsing models.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if dep_counts_file not in self.resources:
				counts = self.readNgramFile(dep_counts_file)
				self.resources[dep_counts_file] = counts
			os.environ['JAVAHOME'] = java_path
			if dependency_models not in self.resources:
				parser = StanfordParser(path_to_jar=stanford_parser, path_to_models_jar=dependency_models)
				self.resources[dependency_models] = parser
			self.features.append((self.objectDependencyFrequencyFeature, [dep_counts_file, dependency_models]))
			self.identifiers.append(('Object Dependency Frequency Feature (Dependency Link Counts File: '+dep_counts_file+') (Models: '+dependency_models+')', orientation))
	
	def addAllDependencyProbabilityFeature(self, language_model, stanford_parser, dependency_models, java_path, orientation):
		"""
		Adds a dependency probability feature to the estimator.
		The value will be the average language model probability of all the target word's dependency links, with the target word replaced by a given candidate.
		To train the language model used by this feature, one must first extract dependency links from a large corpora of sentences.
		In sequence, the dependency links must be transformed into the following format: <type_of_dependency_link> <subject_word> <object_word>
		In the format above, each token is space-separated.
		Once transformed, the dependency links can then be placed in a text file, one per line.
		Finally, one can then run any language modelling tool to produce a language model in ARPA format.
	
		@param language_model: Path to the language model from which to extract dependency link probabilities.
		@param stanford_parser: Path to the "stanford-parser.jar" file.
		The parser can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param dependency_models: Path to a JAR file containing parsing models.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if language_model not in self.resources:
				model = kenlm.LanguageModel(language_model)
				self.resources[language_model] = model
			os.environ['JAVAHOME'] = java_path
			if dependency_models not in self.resources:
				parser = StanfordParser(path_to_jar=stanford_parser, path_to_models_jar=dependency_models)
				self.resources[dependency_models] = parser
			self.features.append((self.allDependencyProbabilityFeature, [language_model, dependency_models]))
			self.identifiers.append(('Dependency Probability Feature (Language Model: '+language_model+') (Models: '+dependency_models+')', orientation))

	def addBinaryAllDependencyFeature(self, dep_counts_file, stanford_parser, dependency_models, java_path, orientation):
		"""
		Adds a binary dependency feature to the estimator.
		The value will be 1 if all dependency links of the target word exist for a given candidate, and 0 otherwise.
		To produce the dependency link counts file used by this feature, one must first extract dependency links from a large corpora of sentences.
		In sequence, the dependency links must be transformed into the following format: <type_of_dependency_link> <subject_word> <object_word>
		In the format above, each token is space-separated.
		Once transformed, the dependency links can then be placed in a text file, one per line.
		Finally, one can then run any language modelling tool to produce a language model in ARPA format.
	
		@param dep_counts_file: Path to a shelve file containing dependency link counts.
		@param stanford_parser: Path to the "stanford-parser.jar" file.
		The parser can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param dependency_models: Path to a JAR file containing parsing models.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if dep_counts_file not in self.resources:
				counts = self.readNgramFile(dep_counts_file)
				self.resources[dep_counts_file] = counts
			os.environ['JAVAHOME'] = java_path
			if dependency_models not in self.resources:
				parser = StanfordParser(path_to_jar=stanford_parser, path_to_models_jar=dependency_models)
				self.resources[dependency_models] = parser
			self.features.append((self.binaryAllDependencyFeature, [dep_counts_file, dependency_models]))
			self.identifiers.append(('Binary All Dependency Feature (Dependency Link Counts File: '+dep_counts_file+') (Models: '+dependency_models+')', orientation))
			
	def addAllDependencyFrequencyFeature(self, dep_counts_file, stanford_parser, dependency_models, java_path, orientation):
		"""
		Adds a dependency frequency feature to the estimator.
		The value will be the average raw frequency of all dependency links of the target word, with the target word replaced by a given candidate.
		To produce the dependency link counts file used by this feature, one must first extract dependency links from a large corpora of sentences.
		In sequence, the dependency links must be transformed into the following format: <type_of_dependency_link> <subject_word> <object_word>
		In the format above, each token is space-separated.
		Once transformed, the dependency links can then be placed in a text file, one per line.
		Finally, one can then run any language modelling tool to produce a language model in ARPA format.
	
		@param dep_counts_file: Path to a shelve file containing dependency link counts.
		@param stanford_parser: Path to the "stanford-parser.jar" file.
		The parser can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param dependency_models: Path to a JAR file containing parsing models.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if dep_counts_file not in self.resources:
				counts = self.readNgramFile(dep_counts_file)
				self.resources[dep_counts_file] = counts
			os.environ['JAVAHOME'] = java_path
			if dependency_models not in self.resources:
				parser = StanfordParser(path_to_jar=stanford_parser, path_to_models_jar=dependency_models)
				self.resources[dependency_models] = parser
			self.features.append((self.allDependencyFrequencyFeature, [dep_counts_file, dependency_models]))
			self.identifiers.append(('All Dependency Frequency Feature (Dependency Link Counts File: '+dep_counts_file+') (Models: '+dependency_models+')', orientation))
			
	def addWordVectorContextSimilarityFeature(self, model, pos_model, stanford_tagger, java_path, orientation):
		"""
		Adds a word vector context similarity feature to the estimator.
		The value will be the average similarity between the word vector of a candidate and the vectors of all content word in the target word's context.
	
		@param model: Path to a binary word vector model.
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
			if model not in self.resources:
				m = gensim.models.word2vec.Word2Vec.load_word2vec_format(model, binary=True)
				self.resources[model] = m
			os.environ['JAVAHOME'] = java_path
			if pos_model not in self.resources:
				tagger = StanfordPOSTagger(pos_model, stanford_tagger)
				self.resources[pos_model] = tagger
			self.features.append((self.wordVectorContextSimilarityFeature, [model, pos_model]))
			self.identifiers.append(('Word Vector Context Similarity (Model: '+model+') (POS Model: '+pos_model+')', orientation))

	def addTaggedWordVectorContextSimilarityFeature(self, model, pos_model, stanford_tagger, java_path, pos_type, orientation):
		"""
		Adds a tagged word vector context similarity feature to the estimator.
		The value will be the average similarity between the word vector of a candidate and the vectors of all content word in the target word's context.
		Each entry in the word vector model must be in the following format: <word>|||<tag>
		To create a corpus for such model to be trained, one must tag each word in a corpus, and then concatenate words and tags using the aforementioned convention.
	
		@param model: Path to a binary tagged word vector model.
		For instructions on how to create the model, please refer to the LEXenstein Manual.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param pos_type: The type of POS tags to be used.
		Values supported: treebank, paetzold
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if model not in self.resources:
				m = gensim.models.word2vec.Word2Vec.load_word2vec_format(model, binary=True)
				self.resources[model] = m
			os.environ['JAVAHOME'] = java_path
			if pos_model not in self.resources:
				tagger = StanfordPOSTagger(pos_model, stanford_tagger)
				self.resources[pos_model] = tagger
			self.features.append((self.taggedWordVectorContextSimilarityFeature, [model, pos_model, pos_type]))
			self.identifiers.append(('Tagged Word Vector Context Similarity (Model: '+model+') (POS Model: '+pos_model+') (POS Type: '+pos_type+')', orientation))
			
	def addNullLinkNominalFeature(self, stanford_parser, dependency_models, java_path, orientation):
		"""
		Adds a null link nominal feature to the estimator
		The value will be 1 if there is at least one dependency link of which the candidate is part of, and 0 otherwise.

		@param stanford_parser: Path to the "stanford-parser.jar" file.
		The parser can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param dependency_models: Path to a JAR file containing parsing models.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/lex-parser.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			os.environ['JAVAHOME'] = java_path
			if dependency_models not in self.resources:
				parser = StanfordParser(path_to_jar=stanford_parser, path_to_models_jar=dependency_models)
				self.resources[dependency_models] = parser
				
			self.features.append((self.nullLinkNominalFeature, [dependency_models]))
			self.identifiers.append(('Null Link Nominal Feature (Models: '+dependency_models+')', orientation))
			
	def addBackoffBehaviorNominalFeature(self, ngram_file, orientation):
		"""
		Adds a nominal language model backoff behavior nominal feature to the estimator.
	
		@param ngram_file: Path to a shelve file containing n-gram frequency counts.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if ngram_file not in self.resources:
				counts = self.readNgramFile(ngram_file)
				self.resources[ngram_file] = counts
				
			self.features.append((self.backoffBehaviorNominalFeature, [ngram_file]))
			self.identifiers.append(('N-Gram Nominal Feature (N-Grams File: '+ngram_file+')', orientation))
			
	def addImageSearchCountFeature(self, key, orientation):
		"""
		Adds an image search count feature to the estimator.
		The resulting value will be the number of distinct pictures retrieved by the Getty Images API.
		This feature requires for a free "Connect Embed" key, which gives you access to 5 queries per second, and unlimited queries per day.
		For more information on how to acquire a key, please visit their website at: https://developer.gettyimages.com
	
		@param key: Connect Embed key for the Getty Images API.
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if key not in self.resources:
				self.resources['GettyImagesKey'] = key
			if 'image_counts' not in self.resources:
				self.resources['image_counts'] = {}
				
			self.features.append((self.imageSearchCountFeature, [key]))
			self.identifiers.append(('Image Search Count Feature (Key: '+key+')', orientation))
			
	def addWebSearchCountFeature(self, orientation):
		"""
		Adds a web search count feature to the estimator.
		The resulting value will be the number of websites retrieved by Bing.
	
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:
			if 'page_counts' not in self.resources:
				self.resources['page_counts'] = {}
				
			self.features.append((self.webSearchCountFeature, []))
			self.identifiers.append((' Web Search Count Feature', orientation))
			
	def addMorphologicalFeature(self, dictionary, description, orientation):
		"""
		Adds a generalized morphological feature to the estimator.
		It requires for a dictionary that assigns words to their respective feature values.
		For each word in a dataset, the value of this feature will be the one found in the dictionar provided, or 0 if it is not available.
	
		@param dictionary: A dictionary object assigning words to values.
		Example: dictionary['chair'] = 45.33.
		@param description: Description of the feature.
		Example: "Age of Acquisition".
		@param orientation: Whether the feature is a simplicity of complexity measure.
		Possible values: Complexity, Simplicity.
		"""
		if orientation not in ['Complexity', 'Simplicity']:
			print('Orientation must be Complexity or Simplicity')
		else:				
			self.features.append((self.morphologicalFeature, [dictionary]))
			self.identifiers.append((description, orientation))
			
	# Nominal features:
	
	def addCandidateNominalFeature(self):
		"""
		Adds a candidate nominal feature to the estimator.
		"""
		self.features.append((self.candidateNominalFeature, []))
		self.identifiers.append(('Candidate Nominal Feature', 'Not Applicable'))
	
	def addNgramNominalFeature(self, leftw, rightw):
		"""
		Adds a n-gram nominal feature to the estimator.
	
		@param leftw: Number of tokens to the left.
		@param rightw: Number of tokens to the right.
		"""
		self.features.append((self.ngramNominalFeature, [leftw, rightw]))
		self.identifiers.append(('N-Gram Nominal Feature ['+str(leftw)+', '+str(rightw)+']', 'Not Applicable'))
		
	def addCandidatePOSNominalFeature(self, pos_model, stanford_tagger, java_path, pos_type):
		"""
		Adds a candidate POS tag nominal feature to the estimator.

		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param pos_type: The type of POS tags to be used.
		Values supported: treebank, paetzold
		"""
		os.environ['JAVAHOME'] = java_path
		if pos_model not in self.resources:
			tagger = StanfordPOSTagger(pos_model, stanford_tagger)
			self.resources[pos_model] = tagger
			
		self.features.append((self.candidatePOSNominalFeature, [pos_model, pos_type]))
		self.identifiers.append(('Candidate POS Nominal Feature (POS Model: '+pos_model+') (POS Type: '+pos_type+')', 'Not Applicable'))
		
	def addPOSNgramNominalFeature(self, leftw, rightw, pos_model, stanford_tagger, java_path, pos_type):
		"""
		Adds a POS n-gram nominal feature to the estimator.
		The n-gram will contain the candidate's POS tag surrounded by the POS tags of neighboring words.

		@param leftw: Number of tokens to the left.
		@param rightw: Number of tokens to the right.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param pos_type: The type of POS tags to be used.
		Values supported: treebank, paetzold
		"""
		os.environ['JAVAHOME'] = java_path
		if pos_model not in self.resources:
			tagger = StanfordPOSTagger(pos_model, stanford_tagger)
			self.resources[pos_model] = tagger
			
		self.features.append((self.POSNgramNominalFeature, [leftw, rightw, pos_model, pos_type]))
		self.identifiers.append(('POS N-gram Nominal Feature ['+str(leftw)+', '+str(rightw)+'] (POS Model: '+pos_model+') (POS Type: '+pos_type+')', 'Not Applicable'))
		
	def addPOSNgramWithCandidateNominalFeature(self, leftw, rightw, pos_model, stanford_tagger, java_path, pos_type):
		"""
		Adds a candidate centered POS n-gram nominal feature to the estimator.
		The n-gram will contain the candidate surrounded by the POS tags of neighboring words.

		@param leftw: Number of tokens to the left.
		@param rightw: Number of tokens to the right.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param pos_type: The type of POS tags to be used.
		Values supported: treebank, paetzold
		"""
		os.environ['JAVAHOME'] = java_path
		if pos_model not in self.resources:
			tagger = StanfordPOSTagger(pos_model, stanford_tagger)
			self.resources[pos_model] = tagger
			
		self.features.append((self.POSNgramWithCandidateNominalFeature, [leftw, rightw, pos_model, pos_type]))
		self.identifiers.append(('POS N-gram with Candidate Nominal Feature ['+str(leftw)+', '+str(rightw)+'] (POS Model: '+pos_model+') (POS Type: '+pos_type+')', 'Not Applicable'))
