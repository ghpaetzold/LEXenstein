import xml.etree.ElementTree as ET
import re
import urllib2 as urllib
from nltk.corpus import wordnet as wn
import subprocess
import nltk
import kenlm

class KauchakGenerator:

	def __init__(self, mat, parallel_pos_file, alignments_file, stop_words):
		self.mat = mat
		self.parallel_pos_file = parallel_pos_file
		self.alignments_file = alignments_file
		self.stop_words = set([word.strip() for word in open(stop_words)])
		
	def getSubstitutions(self, victor_corpus):
		#Get candidate->pos map:
		print('Getting POS map...')
		target_pos = self.getPOSMap(victor_corpus)

		#Get initial set of substitutions:
		print('Getting initial set of substitutions...')
		substitutions_initial = self.getInitialSet(victor_corpus, target_pos)
		
		#Get final substitutions:
		print('Inflecting substitutions...')
		substitutions_inflected = self.getInflectedSet(substitutions_initial)
		
		#Return final set:
		print('Finished!')
		return substitutions_inflected
	
	def getInflectedSet(self, substitutions_initial):
		#Create second set of filtered substitutions:
		substitutions_stemmed = {}

		keys = sorted(list(substitutions_initial.keys()))
		nounverbs = []
		cands = set([])
		for key in keys:
			for key_pos in substitutions_initial[key].keys():
				if key_pos.startswith('v') or key_pos.startswith('n'):
					nounverbs.append(key)
					for cand in substitutions_initial[key][key_pos]:
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
			for poskey in substitutions_initial[key].keys():
				if poskey.startswith('n'):
					singularsk.append(stemk[key])
					for cand in substitutions_initial[key][poskey]:
						singulars.append(stemc[cand])
					pluralsk.append(stemk[key])
					for cand in substitutions_initial[key][poskey]:
						plurals.append(stemc[cand])
				elif poskey.startswith('v'):
					verbsk.append(stemk[key])
					for candn in substitutions_initial[key][poskey]:
						verbs.append(stemc[candn])

		singularskr = self.getPlurals(singularsk)
		pluralskr = self.getSingulars(pluralsk)
		verbskr = self.getInflections(verbsk)

		singularsr = self.getPlurals(singulars)
		pluralsr = self.getSingulars(plurals)
		verbsr = self.getInflections(verbs)

		for key in keys:
			for poskey in substitutions_initial[key].keys():
				if poskey.startswith('n'):
					if singularskr[stemk[key]] not in substitutions_inflected.keys():
						substitutions_inflected[singularskr[stemk[key]]] = set([])
					if pluralskr[stemk[key]] not in substitutions_inflected.keys():
						substitutions_inflected[pluralskr[stemk[key]]] = set([])
					for cand in substitutions_initial[key][poskey]:
						substitutions_inflected[singularskr[stemk[key]]].add(singularsr[stemc[cand]])
						substitutions_inflected[pluralskr[stemk[key]]].add(pluralsr[stemc[cand]])
				elif poskey.startswith('v'):
					if verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']] = set([])
					if verbskr[stemk[key]]['PAST_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']] = set([])
					if verbskr[stemk[key]]['PRESENT_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']] = set([])
					for candn in substitutions_initial[key][poskey]:
						substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PERFECT_PARTICIPLE'])
						substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PARTICIPLE'])
						substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']].add(verbsr[stemc[candn]]['PRESENT_PARTICIPLE'])
				else:
					if key not in substitutions_inflected:
						substitutions_inflected[key] = set([])
					for cand in substitutions_initial[key][poskey]:
						substitutions_inflected[key].add(cand)
		return substitutions_inflected
	
	def getInitialSet(self, victor_corpus, pos_map):
		substitutions_initial = {}
	
		targets = set([line.strip().split('\t')[1].strip() for line in open(victor_corpus)])
	
		fparallel = open(self.parallel_pos_file)
		falignments = open(self.alignments_file)
		
		for line in fparallel:
			data = line.strip().split('\t')
			source = data[0].strip().split(' ')
			target = data[1].strip().split(' ')

			alignments = set(falignments.readline().strip().split(' '))

			for alignment in alignments:
				adata = alignment.strip().split('-')
				left = int(adata[0].strip())
				right = int(adata[1].strip())
				leftraw = source[left].strip()
				leftp = leftraw.split('|||')[1].strip().lower()
				leftw = leftraw.split('|||')[0].strip()
				rightraw = target[right].strip()
				rightp = rightraw.split('|||')[1].strip().lower()
				rightw = rightraw.split('|||')[0].strip()
				
				if leftw in targets and leftp in pos_map[leftw] and len(leftw)>0 and len(rightw)>0 and leftp!='nnp' and rightp!='nnp' and rightp==leftp and leftw not in self.stop_words and rightw not in self.stop_words and leftw!=rightw:
						if leftw in substitutions_initial.keys():
							if leftp in substitutions_initial[leftw].keys():
								substitutions_initial[leftw][leftp].add(rightw)
							else:
								substitutions_initial[leftw][leftp] = set(rightw)
						else:
							substitutions_initial[leftw] = {leftp:set([rightw])}
		fparallel.close()
		falignments.close()
		return substitutions_initial

	def getPOSMap(self, path):
		result = {}
		lex = open(path)
		for line in lex:
			data = line.strip().split('\t')
			sent = data[0].strip().lower().split(' ')
			target = data[1].strip().lower()
			head = int(data[2].strip())

			posd = nltk.pos_tag(sent)
			postarget = posd[head][1].lower().strip()
			if target in result.keys():
				result[target].add(postarget)
			else:
				result[target] = set([postarget])
		lex.close()
		return result	
		
	def getInflections(self, verbstems):
		data1 = self.mat.conjugateVerbs(verbstems, 'PAST_PERFECT_PARTICIPLE')
		data2 = self.mat.conjugateVerbs(verbstems, 'PAST_PARTICIPLE')
		data3 = self.mat.conjugateVerbs(verbstems, 'PRESENT_PARTICIPLE')

		result = {}
		for i in range(0, len(verbstems)):
			result[verbstems[i]] = {'PAST_PERFECT_PARTICIPLE': data1[i], 'PAST_PARTICIPLE': data2[i], 'PRESENT_PARTICIPLE': data3[i]}
		return result

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
		
class YamamotoGenerator:

	def __init__(self, mat, dictionary_key):
		self.mat = mat
		self.dictionary_key = dictionary_key

	def getSubstitutions(self, victor_corpus):
		#Get initial set of substitutions:
		print('Getting initial set of substitutions...')
		substitutions_initial = self.getInitialSet(victor_corpus)

		#Get final substitutions:
		print('Inflecting substitutions...')
		substitutions_inflected = self.getInflectedSet(substitutions_initial)

		#Return final set:
		print('Finished!')
		return substitutions_inflected

	def getInflectedSet(self, substitutions_initial):
		#Create second set of filtered substitutions:
		substitutions_stemmed = {}

		keys = sorted(list(substitutions_initial.keys()))
		nounverbs = []
		cands = set([])
		for key in keys:
			for key_pos in substitutions_initial[key].keys():
				if key_pos.startswith('v') or key_pos.startswith('n'):
					nounverbs.append(key)
					for cand in substitutions_initial[key][key_pos]:
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
			for poskey in substitutions_initial[key].keys():
				if poskey.startswith('n'):
					singularsk.append(stemk[key])
					for cand in substitutions_initial[key][poskey]:
						singulars.append(stemc[cand])
					pluralsk.append(stemk[key])
					for cand in substitutions_initial[key][poskey]:
						plurals.append(stemc[cand])
				elif poskey.startswith('v'):
					verbsk.append(stemk[key])
					for candn in substitutions_initial[key][poskey]:
						verbs.append(stemc[candn])

		singularskr = self.getPlurals(singularsk)
		pluralskr = self.getSingulars(pluralsk)
		verbskr = self.getInflections(verbsk)

		singularsr = self.getPlurals(singulars)
		pluralsr = self.getSingulars(plurals)
		verbsr = self.getInflections(verbs)

		for key in keys:
			for poskey in substitutions_initial[key].keys():
				if poskey.startswith('n'):
					if singularskr[stemk[key]] not in substitutions_inflected.keys():
						substitutions_inflected[singularskr[stemk[key]]] = set([])
					if pluralskr[stemk[key]] not in substitutions_inflected.keys():
						substitutions_inflected[pluralskr[stemk[key]]] = set([])
					for cand in substitutions_initial[key][poskey]:
						substitutions_inflected[singularskr[stemk[key]]].add(singularsr[stemc[cand]])
						substitutions_inflected[pluralskr[stemk[key]]].add(pluralsr[stemc[cand]])
				elif poskey.startswith('v'):
					if verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']] = set([])
					if verbskr[stemk[key]]['PAST_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']] = set([])
					if verbskr[stemk[key]]['PRESENT_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']] = set([])
					for candn in substitutions_initial[key][poskey]:
						substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PERFECT_PARTICIPLE'])
						substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PARTICIPLE'])
						substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']].add(verbsr[stemc[candn]]['PRESENT_PARTICIPLE'])
				else:
					if key not in substitutions_inflected:
						substitutions_inflected[key] = set([])
					for cand in substitutions_initial[key][poskey]:
						substitutions_inflected[key].add(cand)
		return substitutions_inflected

	def getInflections(self, verbstems):
		data1 = self.mat.conjugateVerbs(verbstems, 'PAST_PERFECT_PARTICIPLE')
		data2 = self.mat.conjugateVerbs(verbstems, 'PAST_PARTICIPLE')
		data3 = self.mat.conjugateVerbs(verbstems, 'PRESENT_PARTICIPLE')

		result = {}
		for i in range(0, len(verbstems)):
			result[verbstems[i]] = {'PAST_PERFECT_PARTICIPLE': data1[i], 'PAST_PARTICIPLE': data2[i], 'PRESENT_PARTICIPLE': data3[i]}
		return result

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
		
			url = 'http://www.dictionaryapi.com/api/v1/references/collegiate/xml/' + target + '?key=' + self.dictionary_key
			conn = urllib.urlopen(url)
			root = ET.fromstring(conn.read())
		
			newline = target + '\t'
			cands = {}
		
			entries = root.iter('entry')
			for entry in entries:
				node_pos = root_node.iter('fl')
				if node_pos != None:
					node_pos = node_pos.text.strip()[0].lower()
					if node_pos not in cands.keys():
						cands[node_pos] = set([])
				for definition in entry.iter('dt'):
					if definition.text!=None:
						text = definition.text.strip()
						text = text[1:len(text)]
						tokens = nltk.word_tokenize(text)
						postags = nltk.pos_tag(tokens)
						for p in postags:
							postag = p[1].strip()[0].lower()
							cand = p[0].strip()
							if postag==node_pos:
								cands[node_pos].add(cand)
			if len(cands.keys())>0:
				substitutions_initial[target] = cands
		lex.close()
		return substitutions_initial

class MerriamGenerator:

	def __init__(self, mat, thesaurus_key):
		self.mat = mat
		self.thesaurus_key = thesaurus_key

	def getSubstitutions(self, victor_corpus):
		#Get initial set of substitutions:
		print('Getting initial set of substitutions...')
		substitutions_initial = self.getInitialSet(victor_corpus)
		
		#Get final substitutions:
		print('Inflecting substitutions...')
		substitutions_inflected = self.getInflectedSet(substitutions_initial)

		#Return final set:
		print('Finished!')
		return substitutions_inflected

	def getInflectedSet(self, substitutions_initial):
		#Create second set of filtered substitutions:
		substitutions_stemmed = {}

		keys = sorted(list(substitutions_initial.keys()))
		nounverbs = []
		cands = set([])
		for key in keys:
			for key_pos in substitutions_initial[key].keys():
				if key_pos.startswith('v') or key_pos.startswith('n'):
					nounverbs.append(key)
					for cand in substitutions_initial[key][key_pos]:
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
			for poskey in substitutions_initial[key].keys():
				if poskey.startswith('n'):
					singularsk.append(stemk[key])
					for cand in substitutions_initial[key][poskey]:
						singulars.append(stemc[cand])
					pluralsk.append(stemk[key])
					for cand in substitutions_initial[key][poskey]:
						plurals.append(stemc[cand])
				elif poskey.startswith('v'):
					verbsk.append(stemk[key])
					for candn in substitutions_initial[key][poskey]:
						verbs.append(stemc[candn])

		singularskr = self.getPlurals(singularsk)
		pluralskr = self.getSingulars(pluralsk)
		verbskr = self.getInflections(verbsk)

		singularsr = self.getPlurals(singulars)
		pluralsr = self.getSingulars(plurals)
		verbsr = self.getInflections(verbs)

		for key in keys:
			for poskey in substitutions_initial[key].keys():
				if poskey.startswith('n'):
					if singularskr[stemk[key]] not in substitutions_inflected.keys():
						substitutions_inflected[singularskr[stemk[key]]] = set([])
					if pluralskr[stemk[key]] not in substitutions_inflected.keys():
						substitutions_inflected[pluralskr[stemk[key]]] = set([])
					for cand in substitutions_initial[key][poskey]:
						substitutions_inflected[singularskr[stemk[key]]].add(singularsr[stemc[cand]])
						substitutions_inflected[pluralskr[stemk[key]]].add(pluralsr[stemc[cand]])
				elif poskey.startswith('v'):
					if verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']] = set([])
					if verbskr[stemk[key]]['PAST_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']] = set([])
					if verbskr[stemk[key]]['PRESENT_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']] = set([])
					for candn in substitutions_initial[key][poskey]:
						substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PERFECT_PARTICIPLE'])
						substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PARTICIPLE'])
						substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']].add(verbsr[stemc[candn]]['PRESENT_PARTICIPLE'])
				else:
					if key not in substitutions_inflected:
						substitutions_inflected[key] = set([])
					for cand in substitutions_initial[key][poskey]:
						substitutions_inflected[key].add(cand)
		return substitutions_inflected

	def getInflections(self, verbstems):
		data1 = self.mat.conjugateVerbs(verbstems, 'PAST_PERFECT_PARTICIPLE')
		data2 = self.mat.conjugateVerbs(verbstems, 'PAST_PARTICIPLE')
		data3 = self.mat.conjugateVerbs(verbstems, 'PRESENT_PARTICIPLE')

		result = {}
		for i in range(0, len(verbstems)):
			result[verbstems[i]] = {'PAST_PERFECT_PARTICIPLE': data1[i], 'PAST_PARTICIPLE': data2[i], 'PRESENT_PARTICIPLE': data3[i]}
		return result

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
			
			cands = {}
			if len(root)>0:
				for root_node in root:
					node_pos = root_node.iter('fl')
					if node_pos != None:
						node_pos = node_pos.text.strip()[0].lower()
						if node_pos not in cands.keys():
							cands[node_pos] = set([])
					for sense in root_node.iter('sens'):
						syn = sense.findall('syn')[0]
					res = ''
					for snip in syn.itertext():
						res += snip + ' '
					finds = re.findall('\([^\)]+\)', res)
					for find in finds:
						res = res.replace(find, '')	
	
					synonyms = [s.strip() for s in res.split(',')]
					
					for synonym in synonyms:
						if len(synonym.split(' '))==1:
							cands[node_pos].add(synonym)
			if len(cands.keys())>0:
				substitutions_initial[target] = cands
		lex.close()		
		return substitutions_initial

#Class for the Wordnet Generator
class WordnetGenerator:

	def __init__(self, mat):
		self.mat = mat

	def getSubstitutions(self, victor_corpus):
		#Get initial set of substitutions:
		print('Getting initial set of substitutions...')
		substitutions_initial = self.getInitialSet(victor_corpus)

		#Get final substitutions:
		print('Inflecting substitutions...')
		substitutions_inflected = self.getInflectedSet(substitutions_initial)

		#Return final set:
		print('Finished!')
		return substitutions_inflected

	def getInflectedSet(self, substitutions_initial):
		#Create second set of filtered substitutions:
		substitutions_stemmed = {}

		keys = sorted(list(substitutions_initial.keys()))
		nounverbs = []
		cands = set([])
		for key in keys:
			for key_pos in substitutions_initial[key].keys():
				if key_pos.startswith('v') or key_pos.startswith('n'):
					nounverbs.append(key)
					for cand in substitutions_initial[key][key_pos]:
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
			for poskey in substitutions_initial[key].keys():
				if poskey.startswith('n'):
					singularsk.append(stemk[key])
					for cand in substitutions_initial[key][poskey]:
						singulars.append(stemc[cand])
					pluralsk.append(stemk[key])
					for cand in substitutions_initial[key][poskey]:
						plurals.append(stemc[cand])
				elif poskey.startswith('v'):
					verbsk.append(stemk[key])
					for candn in substitutions_initial[key][poskey]:
						verbs.append(stemc[candn])

		singularskr = self.getPlurals(singularsk)
		pluralskr = self.getSingulars(pluralsk)
		verbskr = self.getInflections(verbsk)

		singularsr = self.getPlurals(singulars)
		pluralsr = self.getSingulars(plurals)
		verbsr = self.getInflections(verbs)

		for key in keys:
			for poskey in substitutions_initial[key].keys():
				if poskey.startswith('n'):
					if singularskr[stemk[key]] not in substitutions_inflected.keys():
						substitutions_inflected[singularskr[stemk[key]]] = set([])
					if pluralskr[stemk[key]] not in substitutions_inflected.keys():
						substitutions_inflected[pluralskr[stemk[key]]] = set([])
					for cand in substitutions_initial[key][poskey]:
						substitutions_inflected[singularskr[stemk[key]]].add(singularsr[stemc[cand]])
						substitutions_inflected[pluralskr[stemk[key]]].add(pluralsr[stemc[cand]])
				elif poskey.startswith('v'):
					if verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']] = set([])
					if verbskr[stemk[key]]['PAST_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']] = set([])
					if verbskr[stemk[key]]['PRESENT_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']] = set([])
					for candn in substitutions_initial[key][poskey]:
						substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PERFECT_PARTICIPLE'])
						substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PARTICIPLE'])
						substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']].add(verbsr[stemc[candn]]['PRESENT_PARTICIPLE'])
				else:
					if key not in substitutions_inflected:
						substitutions_inflected[key] = set([])
					for cand in substitutions_initial[key][poskey]:
						substitutions_inflected[key].add(cand)
		return substitutions_inflected

	def getInflections(self, verbstems):
		data1 = self.mat.conjugateVerbs(verbstems, 'PAST_PERFECT_PARTICIPLE')
		data2 = self.mat.conjugateVerbs(verbstems, 'PAST_PARTICIPLE')
		data3 = self.mat.conjugateVerbs(verbstems, 'PRESENT_PARTICIPLE')

		result = {}
		for i in range(0, len(verbstems)):
			result[verbstems[i]] = {'PAST_PERFECT_PARTICIPLE': data1[i], 'PAST_PARTICIPLE': data2[i], 'PRESENT_PARTICIPLE': data3[i]}
		return result

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
				cands = {}
				for syn in syns:
					synpos = syn.pos()
					if synpos not in cands.keys():
						cands[synpos] = set([])
					for lem in syn.lemmas():
						candidate = self.cleanLemma(lem.name())
						if len(candidate.split(' '))==1 and candidate in self.simple_vocab:
							cands[synpos].add(candidate)
				for pos in cands.keys():
					cands[pos].remove(target)
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

#Class for the Biran Generator:
class BiranGenerator:

	def __init__(self, mat, complex_vocab, simple_vocab, complex_lm, simple_lm):
		self.complex_vocab = self.getVocab(complex_vocab)
		self.simple_vocab = self.getVocab(simple_vocab)
		self.complex_lm = kenlm.LanguageModel(complex_lm)
		self.simple_lm = kenlm.LanguageModel(simple_lm)
		self.mat = mat
		
	def getSubstitutions(self, victor_corpus):
		#Get initial set of substitutions:
		print('Getting initial set of substitutions...')
		substitutions_initial = self.getInitialSet(victor_corpus)

		#Get inflected substitutions:
		print('Inflecting substitutions...')
		substitutions_inflected = self.getInflectedSet(substitutions_initial)

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

	def getInflectedSet(self, substitutions_initial):
		#Create second set of filtered substitutions:
		substitutions_stemmed = {}

		keys = sorted(list(substitutions_initial.keys()))
		nounverbs = []
		cands = set([])
		for key in keys:
			for key_pos in substitutions_initial[key].keys():
				if key_pos.startswith('v') or key_pos.startswith('n'):
					nounverbs.append(key)
					for cand in substitutions_initial[key][key_pos]:
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
			for poskey in substitutions_initial[key].keys():
				if poskey.startswith('n'):
					singularsk.append(stemk[key])
					for cand in substitutions_initial[key][poskey]:
						singulars.append(stemc[cand])
					pluralsk.append(stemk[key])
					for cand in substitutions_initial[key][poskey]:
						plurals.append(stemc[cand])
				elif poskey.startswith('v'):
					verbsk.append(stemk[key])
					for candn in substitutions_initial[key][poskey]:
						verbs.append(stemc[candn])

		singularskr = self.getPlurals(singularsk)
		pluralskr = self.getSingulars(pluralsk)
		verbskr = self.getInflections(verbsk)

		singularsr = self.getPlurals(singulars)
		pluralsr = self.getSingulars(plurals)
		verbsr = self.getInflections(verbs)

		for key in keys:
			for poskey in substitutions_initial[key].keys():
				if poskey.startswith('n'):
					if singularskr[stemk[key]] not in substitutions_inflected.keys():
						substitutions_inflected[singularskr[stemk[key]]] = set([])
					if pluralskr[stemk[key]] not in substitutions_inflected.keys():
						substitutions_inflected[pluralskr[stemk[key]]] = set([])
					for cand in substitutions_initial[key][poskey]:
						substitutions_inflected[singularskr[stemk[key]]].add(singularsr[stemc[cand]])
						substitutions_inflected[pluralskr[stemk[key]]].add(pluralsr[stemc[cand]])
				elif poskey.startswith('v'):
					if verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']] = set([])
					if verbskr[stemk[key]]['PAST_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']] = set([])
					if verbskr[stemk[key]]['PRESENT_PARTICIPLE'] not in substitutions_inflected.keys():
						substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']] = set([])
					for candn in substitutions_initial[key][poskey]:
						substitutions_inflected[verbskr[stemk[key]]['PAST_PERFECT_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PERFECT_PARTICIPLE'])
						substitutions_inflected[verbskr[stemk[key]]['PAST_PARTICIPLE']].add(verbsr[stemc[candn]]['PAST_PARTICIPLE'])
						substitutions_inflected[verbskr[stemk[key]]['PRESENT_PARTICIPLE']].add(verbsr[stemc[candn]]['PRESENT_PARTICIPLE'])
				else:
					if key not in substitutions_inflected:
						substitutions_inflected[key] = set([])
					for cand in substitutions_initial[key][poskey]:
						substitutions_inflected[key].add(cand)
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
				cands = {}
				for syn in syns:
					synpos = syn.pos()
					if synpos not in cands.keys():
						cands[synpos] = set([])
					for lem in syn.lemmas():
						candidate = self.cleanLemma(lem.name())
						if len(candidate.split(' '))==1 and candidate in self.simple_vocab:
							cands[synpos].add(candidate)
					for hyp in syn.hypernyms():
						hyppos = hyp.pos()
						if hyppos not in cands.keys():
							cands[hyppos] = set([])
						for lem in hyp.lemmas():
							candidate = self.cleanLemma(lem.name())
							if len(candidate.split(' '))==1 and candidate in self.simple_vocab:
								cands[synpos].add(candidate)
				for pos in cands.keys():
					cands[pos].remove(target)
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