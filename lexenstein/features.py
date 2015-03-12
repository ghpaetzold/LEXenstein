from nltk.stem.porter import *
from nltk.corpus import wordnet as wn
import kenlm

class FeatureEstimator:

	def __init__(self):
		self.features = []
		
	def calculateFeatures(self, victor_corpus):
		data = [line.strip().split('\t') for line in open(victor_corpus)]
		
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
		return result
		
	def generateVector(self, feature_vector, index):
		result = []
		for feature in feature_vector:
			if not isinstance(feature[index], list):
				result.append(feature[index])
			else:
				result.extend(feature[index])
		return result
		
	def lexiconFeature(self, data, args):
		path = args[0]
		result = []
		basics = [w.strip() for w in open(path)]
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
		model = kenlm.LanguageModel(lm)
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
						aux = -1.0*model.score(ngram, bos=bosv, eos=eosv)
						values.append(aux)
				result.append(values)
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
		model = kenlm.LanguageModel(lm)
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
					senses = wn.synsets(word)
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
					senses = wn.synsets(word)
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
					senses = wn.synsets(word)
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
					senses = wn.synsets(word)
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
					senses = wn.synsets(word)
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
					senses = wn.synsets(word)
					for sense in senses:
						auxmax = sense.max_depth()
						if auxmax>maxdepth:
							maxdepth = auxmax
				resultma.append(maxdepth)
		return resultma
		
	def addLexiconFeature(self, path):
		self.features.append((self.lexiconFeature, [path]))
	
	def addLengthFeature(self):
		self.features.append((self.lengthFeature, []))
	
	def addSyllableFeature(self, mat):
		self.features.append((self.syllableFeature, [mat]))
		
	def addCollocationalFeature(self, lm, leftw, rightw):
		self.features.append((self.collocationalFeature, [lm, leftw, rightw]))
		
	def addSentenceProbabilityFeature(self, lm):
		self.features.append((self.sentenceProbabilityFeature, [lm]))
		
	def addSenseCountFeature(self):
		self.features.append((self.senseCount ,[]))
		
	def addSynonymCountFeature(self):
		self.features.append((self.synonymCount ,[]))
		
	def addHypernymCountFeature(self):
		self.features.append((self.hypernymCount ,[]))
		
	def addHyponymCountFeature(self):
		self.features.append((self.hyponymCount ,[]))
		
	def addMinDepthFeature(self):
		self.features.append((self.minDepth ,[]))
		
	def addMaxDepthFeature(self):
		self.features.append((self.maxDepth ,[]))
		
	
