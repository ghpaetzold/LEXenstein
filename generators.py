import xml.etree.ElementTree as ET
import re
import urllib2 as urllib
from nltk.corpus import wordnet as wn
import subprocess
import nltk
import kenlm

class YamamotoGenerator:

	def __init__(self, mat, dictionary_key):
		self.mat = mat
		self.dictionary_key = dictionary_key

	def getSubstitutions(self, victor_corpus):
		#Get candidate->pos map:
		print('Getting POS map...')
		target_pos = self.getPOSMap(victor_corpus)

		#Get initial set of substitutions:
		print('Getting initial set of substitutions...')
		substitutions_initial = self.getInitialSet(victor_corpus)

		#Get final substitutions:
		print('Inflecting substitutions...')
		substitutions_inflected = self.getInflectedSet(substitutions_initial, target_pos)

		#Return final set:
		print('Finished!')
		return substitutions_inflected

	def getInflectedSet(self, substitutions_initial, target_pos):
		#Create second set of filtered substitutions:
		substitutions_stemmed = {}

		keys = sorted(list(substitutions_initial.keys()))
		nounverbs = []
		cands = set([])
		for key in keys:
			if target_pos[key].startswith('v') or target_pos[key].startswith('n'):
				nounverbs.append(key)
				for cand in substitutions_initial[key]:
					cands.add(cand)
		cands = sorted(list(cands))

		stemk = self.getStems(nounverbs)
		stemc = self.getStems(cands)

		#Create third set of filtered substitutions:
		substitutions_inflected = {}

		singularsk = []
		pluralsk = []
		verbsk = []

		singulars = []
		plurals = []
		verbs = []

		for key in keys:
			poskey = target_pos[key]
			if poskey.startswith('n'):
				singularsk.append(stemk[key])
				for cand in substitutions_initial[key]:
					singulars.append(stemc[cand])
				pluralsk.append(stemk[key])
				for cand in substitutions_initial[key]:
					singulars.append(stemc[cand])
				pluralsk.append(stemk[key])
				for cand in substitutions_initial[key]:
					plurals.append(stemc[cand])
			elif poskey.startswith('v'):
				verbsk.append(stemk[key])
				for candn in substitutions_initial[key]:
					verbs.append(stemc[candn])

		singularskr = self.getPlurals(singularsk)
		pluralskr = self.getSingulars(pluralsk)
		verbskr = self.getInflections(verbsk)

		singularsr = self.getPlurals(singulars)
		pluralsr = self.getSingulars(plurals)
		verbsr = self.getInflections(verbs)

		for key in keys:
			poskey = target_pos[key]
			if poskey.startswith('n'):
				substitutions_inflected[singularskr[stemk[key]]] = set([])
				substitutions_inflected[pluralskr[stemk[key]]] = set([])
				for cand in substitutions_initial[key]:
					substitutions_inflected[singularskr[stemk[key]]].add(singularsr[stemc[cand]])
					substitutions_inflected[pluralskr[stemk[key]]].add(pluralsr[stemc[cand]])
			elif poskey.startswith('v'):
				substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']] = set([])
				substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']] = set([])

				substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']] = set([])
				for candn in substitutions_initial[key]:
					substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PERFECT_PARTICIPLE'])
					substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PARTICIPLE'])
					substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']].add(verbsr[stemc[candn]]['PRESENT_PARTICIPLE'])
		return substitutions_inflected

	def getInflections(self, verbstems):
		data1 = self.mat.conjugateVerbs(verbstems, 'PAST_PERFECT_PARTICIPLE')
		data2 = self.mat.conjugateVerbs(verbstems, 'PAST_PARTICIPLE')
		data3 = self.mat.conjugateVerbs(verbstems, 'PRESENT_PARTICIPLE')

		result = {}
		for i in range(0, len(verbstems)):
			result[verbstems[i]] = {'PAST_PERFECT_PARTICIPLE': data1[i], 'PAST_PARTICIPLE': data2[i], 'PRESENT_PARTICIPLE': data3[i]}
		return result

	def getComplexity(self, word, clm, slm):
		C = (clm.score(word, bos=False, eos=False))/(slm.score(word, bos=False, eos=False))
		L = float(len(word))
		return C*L

	def getSingulars(self, plurstems):
		data = self.mat.inflectNouns(plurstems, 'singular')
		result = {}
		for i in range(0, len(plurstems)):
			result[plurstems[i]] = data[i]
		return result

	def getPlurals(self, singstems):
		data = self.mat.inflectNouns(singstems, 'plural')
		result = {}
		for i in range(0, len(singstems)):
			result[singstems[i]] = data[i]
		return result

	def getStems(self, sings):
		data = self.mat.lemmatizeWords(sings)
		result = {}
		for i in range(0, len(data)):
			stem = data[i]
			sing = sings[i]
			if len(stem.strip())>0:
				result[sing] = stem.strip()
			else:
				result[sing] = sing
		return result

	def getInitialSet(self, victor_corpus):
		substitutions_initial = {}

		lex = open(victor_corpus)
		for line in lex:
			data = line.strip().split('\t')
			target = data[1].strip()
			head = int(data[2].strip())
			targetp = nltk.pos_tag(data[0].strip().split(' '))[head][1]
		
			url = 'http://www.dictionaryapi.com/api/v1/references/collegiate/xml/' + target + '?key=' + self.dictionary_key
			conn = urllib.urlopen(url)
			root = ET.fromstring(conn.read())
		
			newline = target + '\t'
			candidates = set([])
		
			for definition in root.iter('dt'):
				if definition.text!=None:
					text = definition.text.strip()
					text = text[1:len(text)]
					tokens = nltk.word_tokenize(text)
					postags = nltk.pos_tag(tokens)
					for p in postags:
						postag = p[1].strip()
						cand = p[0].strip()
						if postag==targetp:
							candidates.add(cand)
			if len(candidates)>0:
				substitutions_initial[target] = candidates
		lex.close()
		return substitutions_initial

	def cleanLemma(self, lem):
		result = ''
		aux = lem.strip().split('_')
		for word in aux:
			result += word + ' '
		return result.strip()

	def getPOSMap(self, path):
		result = {}
		lex = open(path)
		for line in lex:
			data = line.strip().split('\t')
			sent = data[0].strip().lower().split(' ')
			target = data[1].strip().lower()
			head = int(data[2].strip())

			posd = nltk.pos_tag(sent)
			result[target] = posd[head][1].lower().strip()
		lex.close()
		return result

class MerriamGenerator:

	def __init__(self, mat, thesaurus_key):
		self.mat = mat
		self.thesaurus_key = thesaurus_key

	def getSubstitutions(self, victor_corpus):
		#Get candidate->pos map:
		print('Getting POS map...')
		target_pos = self.getPOSMap(victor_corpus)

		#Get initial set of substitutions:
		print('Getting initial set of substitutions...')
		substitutions_initial = self.getInitialSet(victor_corpus)
		
		#Get final substitutions:
		print('Inflecting substitutions...')
		substitutions_inflected = self.getInflectedSet(substitutions_initial, target_pos)

		#Return final set:
		print('Finished!')
		return substitutions_inflected

	def getInflectedSet(self, substitutions_initial, target_pos):
		#Create second set of filtered substitutions:
		substitutions_stemmed = {}

		keys = sorted(list(substitutions_initial.keys()))
		nounverbs = []
		cands = set([])
		for key in keys:
			if target_pos[key].startswith('v') or target_pos[key].startswith('n'):
				nounverbs.append(key)
				for cand in substitutions_initial[key]:
					cands.add(cand)
		cands = sorted(list(cands))

		stemk = self.getStems(nounverbs)
		stemc = self.getStems(cands)

		#Create third set of filtered substitutions:
		substitutions_inflected = {}

		singularsk = []
		pluralsk = []
		verbsk = []

		singulars = []
		plurals = []
		verbs = []

		for key in keys:
			poskey = target_pos[key]
			if poskey.startswith('n'):
				singularsk.append(stemk[key])
				for cand in substitutions_initial[key]:
					singulars.append(stemc[cand])
				pluralsk.append(stemk[key])
				for cand in substitutions_initial[key]:
					plurals.append(stemc[cand])
			elif poskey.startswith('v'):
				verbsk.append(stemk[key])
				for candn in substitutions_initial[key]:
					verbs.append(stemc[candn])

		singularskr = self.getPlurals(singularsk)
		pluralskr = self.getSingulars(pluralsk)
		verbskr = self.getInflections(verbsk)

		singularsr = self.getPlurals(singulars)
		pluralsr = self.getSingulars(plurals)
		verbsr = self.getInflections(verbs)

		for key in keys:
			poskey = target_pos[key]
			if poskey.startswith('n'):
				substitutions_inflected[singularskr[stemk[key]]] = set([])
				substitutions_inflected[pluralskr[stemk[key]]] = set([])
				for cand in substitutions_initial[key]:
					substitutions_inflected[singularskr[stemk[key]]].add(singularsr[stemc[cand]])
					substitutions_inflected[pluralskr[stemk[key]]].add(pluralsr[stemc[cand]])
			elif poskey.startswith('v'):
				substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']] = set([])
				substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']] = set([])

				substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']] = set([])
				for candn in substitutions_initial[key]:
					substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PERFECT_PARTICIPLE'])
					substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PARTICIPLE'])
					substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']].add(verbsr[stemc[candn]]['PRESENT_PARTICIPLE'])
		return substitutions_inflected

	def getInflections(self, verbstems):
		data1 = self.mat.conjugateVerbs(verbstems, 'PAST_PERFECT_PARTICIPLE')
		data2 = self.mat.conjugateVerbs(verbstems, 'PAST_PARTICIPLE')
		data3 = self.mat.conjugateVerbs(verbstems, 'PRESENT_PARTICIPLE')

		result = {}
		for i in range(0, len(verbstems)):
			result[verbstems[i]] = {'PAST_PERFECT_PARTICIPLE': data1[i], 'PAST_PARTICIPLE': data2[i], 'PRESENT_PARTICIPLE': data3[i]}
		return result

	def getComplexity(self, word, clm, slm):
		C = (clm.score(word, bos=False, eos=False))/(slm.score(word, bos=False, eos=False))
		L = float(len(word))
		return C*L

	def getSingulars(self, plurstems):
		data = self.mat.inflectNouns(plurstems, 'singular')
		result = {}
		for i in range(0, len(plurstems)):
			result[plurstems[i]] = data[i]
		return result

	def getPlurals(self, singstems):
		data = self.mat.inflectNouns(singstems, 'plural')
		result = {}
		for i in range(0, len(singstems)):
			result[singstems[i]] = data[i]
		return result

	def getStems(self, sings):
		data = self.mat.lemmatizeWords(sings)
		result = {}
		for i in range(0, len(data)):
			stem = data[i]
			sing = sings[i]
			if len(stem.strip())>0:
				result[sing] = stem.strip()
			else:
				result[sing] = sing
		return result

	def getInitialSet(self, victor_corpus):
		substitutions_initial = {}

		lex = open(victor_corpus)
		for line in lex:
			data = line.strip().split('\t')
			target = data[1].strip()
			url = 'http://www.dictionaryapi.com/api/v1/references/thesaurus/xml/' + target + '?key=' + self.thesaurus_key
			conn = urllib.urlopen(url)
			root = ET.fromstring(conn.read())
			root = root.findall('entry')
			if len(root)>0:
				root = root[0]
				for sense in root.iter('sens'):
					syn = sense.findall('syn')[0]
				res = ''
				for snip in syn.itertext():
					res += snip + ' '
				finds = re.findall('\([^\)]+\)', res)
				for find in finds:
					res = res.replace(find, '')	

				synonyms = [s.strip() for s in res.split(',')]
				cands = set([])
				for synonym in synonyms:
					if len(synonym.split(' '))==1:
						cands.add(synonym)
			if len(synonyms)>0:
				substitutions_initial[target] = cands
		lex.close()		
		return substitutions_initial

	def cleanLemma(self, lem):
		result = ''
		aux = lem.strip().split('_')
		for word in aux:
			result += word + ' '
		return result.strip()

	def getPOSMap(self, path):
		result = {}
		lex = open(path)
		for line in lex:
			data = line.strip().split('\t')
			sent = data[0].strip().lower().split(' ')
			target = data[1].strip().lower()
			head = int(data[2].strip())

			posd = nltk.pos_tag(sent)
			result[target] = posd[head][1].lower().strip()
		lex.close()
		return result

#Class for the Wordnet Generator
class WordnetGenerator:

	def __init__(self, mat):
		self.mat = mat

	def getSubstitutions(self, victor_corpus):
		#Get candidate->pos map:
		print('Getting POS map...')
		target_pos = self.getPOSMap(victor_corpus)

		#Get initial set of substitutions:
		print('Getting initial set of substitutions...')
		substitutions_initial = self.getInitialSet(victor_corpus)

		#Get final substitutions:
		print('Inflecting substitutions...')
		substitutions_inflected = self.getInflectedSet(substitutions_initial, target_pos)

		#Return final set:
		print('Finished!')
		return substitutions_inflected

	def getInflectedSet(self, substitutions_initial, target_pos):
		#Create second set of filtered substitutions:
		substitutions_stemmed = {}

		keys = sorted(list(substitutions_initial.keys()))
		nounverbs = []
		cands = set([])
		for key in keys:
			if target_pos[key].startswith('v') or target_pos[key].startswith('n'):
				nounverbs.append(key)
				for cand in substitutions_initial[key]:
					cands.add(cand)
		cands = sorted(list(cands))

		stemk = self.getStems(nounverbs)
		stemc = self.getStems(cands)

		#Create third set of filtered substitutions:
		substitutions_inflected = {}

		singularsk = []
		pluralsk = []
		verbsk = []

		singulars = []
		plurals = []
		verbs = []

		for key in keys:
			poskey = target_pos[key]
			if poskey.startswith('n'):
				singularsk.append(stemk[key])
				for cand in substitutions_initial[key]:
					singulars.append(stemc[cand])
				pluralsk.append(stemk[key])
				for cand in substitutions_initial[key]:
					plurals.append(stemc[cand])
			elif poskey.startswith('v'):
				verbsk.append(stemk[key])
				for candn in substitutions_initial[key]:
					verbs.append(stemc[candn])

		singularskr = self.getPlurals(singularsk)
		pluralskr = self.getSingulars(pluralsk)
		verbskr = self.getInflections(verbsk)

		singularsr = self.getPlurals(singulars)
		pluralsr = self.getSingulars(plurals)
		verbsr = self.getInflections(verbs)

		for key in keys:
			poskey = target_pos[key]
			if poskey.startswith('n'):
				substitutions_inflected[singularskr[stemk[key]]] = set([])
				substitutions_inflected[pluralskr[stemk[key]]] = set([])
				for cand in substitutions_initial[key]:
					substitutions_inflected[singularskr[stemk[key]]].add(singularsr[stemc[cand]])
					substitutions_inflected[pluralskr[stemk[key]]].add(pluralsr[stemc[cand]])
			elif poskey.startswith('v'):
				substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']] = set([])
				substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']] = set([])
				substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']] = set([])
				for candn in substitutions_initial[key]:
					substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PERFECT_PARTICIPLE'])
					substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PARTICIPLE'])
					substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']].add(verbsr[stemc[candn]]['PRESENT_PARTICIPLE'])
		return substitutions_inflected

	def getInflections(self, verbstems):
		data1 = self.mat.conjugateVerbs(verbstems, 'PAST_PERFECT_PARTICIPLE')
		data2 = self.mat.conjugateVerbs(verbstems, 'PAST_PARTICIPLE')
		data3 = self.mat.conjugateVerbs(verbstems, 'PRESENT_PARTICIPLE')

		result = {}
		for i in range(0, len(verbstems)):
			result[verbstems[i]] = {'PAST_PERFECT_PARTICIPLE': data1[i], 'PAST_PARTICIPLE': data2[i], 'PRESENT_PARTICIPLE': data3[i]}
		return result

	def getComplexity(self, word, clm, slm):
		C = (clm.score(word, bos=False, eos=False))/(slm.score(word, bos=False, eos=False))
		L = float(len(word))
		return C*L

	def getSingulars(self, plurstems):
		data = self.mat.inflectNouns(plurstems, 'singular')
		result = {}
		for i in range(0, len(plurstems)):
			result[plurstems[i]] = data[i]
		return result

	def getPlurals(self, singstems):
		data = self.mat.inflectNouns(singstems, 'plural')
		result = {}
		for i in range(0, len(singstems)):
			result[singstems[i]] = data[i]
		return result

	def getStems(self, sings):
		data = self.mat.lemmatizeWords(sings)
		result = {}
		for i in range(0, len(data)):
			stem = data[i]
			sing = sings[i]
			if len(stem.strip())>0:
				result[sing] = stem.strip()
			else:
				result[sing] = sing
		return result

	def getInitialSet(self, victor_corpus):
		substitutions_initial = {}
		lex = open(victor_corpus)
		for line in lex:
			data = line.strip().split('\t')
			target = data[1].strip()
			syns = wn.synsets(target)
			newline = target + '\t'
			cands = set([])
			for syn in syns:
				for lem in syn.lemmas():
					candidate = self.cleanLemma(lem.name())
					if len(candidate.split(' '))==1:
						cands.add(candidate)
			if len(cands)>0:
				substitutions_initial[target] = cands
		lex.close()
		return substitutions_initial

	def cleanLemma(self, lem):
		result = ''
		aux = lem.strip().split('_')
		for word in aux:
			result += word + ' '
		return result.strip()

	def getPOSMap(self, path):
		result = {}
		lex = open(path)
		for line in lex:
			data = line.strip().split('\t')
			sent = data[0].strip().lower().split(' ')
			target = data[1].strip().lower()
			head = int(data[2].strip())

			posd = nltk.pos_tag(sent)
			result[target] = posd[head][1].lower().strip()
		lex.close()
		return result

#Class for the Biran Generator:
class BiranGenerator:

	def __init__(self, complex_vocab, simple_vocab, complex_lm, simple_lm, mat):
		self.complex_vocab = self.getVocab(complex_vocab)
		self.simple_vocab = self.getVocab(simple_vocab)
		self.complex_lm = kenlm.LanguageModel(complex_lm)
		self.simple_lm = kenlm.LanguageModel(simple_lm)
		self.mat = mat
		
	def getSubstitutions(self, victor_corpus):
		#Get candidate->pos map:
		print('Getting POS map...')
		target_pos = self.getPOSMap(victor_corpus)

		#Get initial set of substitutions:
		print('Getting initial set of substitutions...')
		substitutions_initial = self.getInitialSet(victor_corpus)

		#Get inflected substitutions:
		print('Inflecting substitutions...')
		substitutions_inflected = self.getInflectedSet(substitutions_initial, target_pos)

		#Get final substitutions:
		print('Filtering simple->complex substitutions...')
		substitutions_final = self.getFinalSet(substitutions_inflected)

		#Return final set:
		print('Finished!')
		return substitutions_final

	def getFinalSet(self, substitutions_inflected):
		#Remove simple->complex substitutions:
		substitutions_final = {}

		for key in substitutions_inflected.keys():
			candidate_list = set([])
			key_score = self.getComplexity(key, self.complex_lm, self.simple_lm)
			for cand in substitutions_inflected[key]:
				cand_score = self.getComplexity(cand, self.complex_lm, self.simple_lm)
				if key_score>=cand_score:
					candidate_list.add(cand)
			if len(candidate_list)>0:
				substitutions_final[key] = candidate_list
		return substitutions_final

	def getInflectedSet(self, substitutions_initial, target_pos):
		#Create second set of filtered substitutions:
		substitutions_stemmed = {}

		keys = sorted(list(substitutions_initial.keys()))
		nounverbs = []
		cands = set([])
		for key in keys:
			if target_pos[key].startswith('v') or target_pos[key].startswith('n'):
				nounverbs.append(key)
				for cand in substitutions_initial[key]:
					cands.add(cand)
		cands = sorted(list(cands))		

		stemk = self.getStems(nounverbs)
		stemc = self.getStems(cands)

		#Create third set of filtered substitutions:
		substitutions_inflected = {}
		
		singularsk = []
		pluralsk = []
		verbsk = []
		
		singulars = []
		plurals = []
		verbs = []
		
		for key in keys:
			poskey = target_pos[key]
			if poskey.startswith('n'):
				singularsk.append(stemk[key])
				for cand in substitutions_initial[key]:
					singulars.append(stemc[cand])
				pluralsk.append(stemk[key])
				for cand in substitutions_initial[key]:
					plurals.append(stemc[cand])
			elif poskey.startswith('v'):
				verbsk.append(stemk[key])
				for candn in substitutions_initial[key]:
					verbs.append(stemc[candn])
		
		singularskr = self.getPlurals(singularsk)
		pluralskr = self.getSingulars(pluralsk)
		verbskr = self.getInflections(verbsk)
		
		singularsr = self.getPlurals(singulars)
		pluralsr = self.getSingulars(plurals)
		verbsr = self.getInflections(verbs)

		for key in keys:
			poskey = target_pos[key]
			if poskey.startswith('n'):
				substitutions_inflected[singularskr[stemk[key]]] = set([])
				substitutions_inflected[pluralskr[stemk[key]]] = set([])
				for cand in substitutions_initial[key]:
					substitutions_inflected[singularskr[stemk[key]]].add(singularsr[stemc[cand]])
					substitutions_inflected[pluralskr[stemk[key]]].add(pluralsr[stemc[cand]])
			elif poskey.startswith('v'):
				substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']] = set([])
				substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']] = set([])
				substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']] = set([])
				for candn in substitutions_initial[key]:
					substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PERFECT_PARTICIPLE'])
					substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PARTICIPLE'])
					substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']].add(verbsr[stemc[candn]]['PRESENT_PARTICIPLE'])
		return substitutions_inflected
	
	def getInflections(self, verbstems):
		data1 = self.mat.conjugateVerbs(verbstems, 'PAST_PERFECT_PARTICIPLE')
		data2 = self.mat.conjugateVerbs(verbstems, 'PAST_PARTICIPLE')
		data3 = self.mat.conjugateVerbs(verbstems, 'PRESENT_PARTICIPLE')

		result = {}
		for i in range(0, len(verbstems)):
			result[verbstems[i]] = {'PAST_PERFECT_PARTICIPLE': data1[i], 'PAST_PARTICIPLE': data2[i], 'PRESENT_PARTICIPLE': data3[i]}
		return result	

	def getComplexity(self, word, clm, slm):
		C = (clm.score(word, bos=False, eos=False))/(slm.score(word, bos=False, eos=False))
		L = float(len(word))
		return C*L

	def getSingulars(self, plurstems):
		data = self.mat.inflectNouns(plurstems, 'singular')
		result = {}
		for i in range(0, len(plurstems)):
			result[plurstems[i]] = data[i]
		return result

	def getPlurals(self, singstems):
		data = self.mat.inflectNouns(singstems, 'plural')
		result = {}
		for i in range(0, len(singstems)):
			result[singstems[i]] = data[i]
		return result

	def getStems(self, sings):
		data = self.mat.lemmatizeWords(sings)
		result = {}
		for i in range(0, len(data)):
			stem = data[i]
			sing = sings[i]
			if len(stem.strip())>0:
				result[sing] = stem.strip()
			else:
				result[sing] = sing
		return result	

	def getInitialSet(self, victor_corpus):
		substitutions_initial = {}
		lex = open(victor_corpus)
		for line in lex:
			data = line.strip().split('\t')
			target = data[1].strip()
			if target in self.complex_vocab:
				syns = wn.synsets(target)
				newline = target + '\t'
				cands = set([])
				for syn in syns:
					for lem in syn.lemmas():
						candidate = self.cleanLemma(lem.name())
						if len(candidate.split(' '))==1 and candidate in self.simple_vocab:
							cands.add(candidate)
					for hyp in syn.hypernyms():
						for lem in hyp.lemmas():
							candidate = self.cleanLemma(lem.name())
							if len(candidate.split(' '))==1 and candidate in self.simple_vocab:
								cands.add(candidate)
				if len(cands)>0:
					substitutions_initial[target] = cands
		lex.close()
		return substitutions_initial

	def getVocab(self, path):
		return set([line.strip() for line in open(path)])

	def cleanLemma(self, lem):
		result = ''
		aux = lem.strip().split('_')
		for word in aux:
			result += word + ' '
		return result.strip()

	def getPOSMap(self, path):
		result = {}
		lex = open(path)
		for line in lex:
			data = line.strip().split('\t')
			sent = data[0].strip().lower().split(' ')
			target = data[1].strip().lower()
			head = int(data[2].strip())

			posd = nltk.pos_tag(sent)
			result[target] = posd[head][1].lower().strip()
		lex.close()
		return result

