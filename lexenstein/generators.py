import xml.etree.ElementTree as ET
import re
import urllib2 as urllib
from nltk.corpus import wordnet as wn
import subprocess
import nltk
from nltk.tag.stanford import POSTagger
import kenlm
import codecs

class KauchakGenerator:

	def __init__(self, mat, parallel_pos_file, alignments_file, stop_words, nc, pos_model, stanford_tagger):
		"""
		Creates a KauchakGenerator instance.
	
		@param mat: MorphAdornerToolkit object.
		@param parallel_pos_file: Path to the parsed parallel corpus from which to extract substitutions.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param alignments_file: Path to the alignments for the parsed parallel corpus from which to extract substitutions.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param stop_words: Path to the file containing stop words of the desired language.
		The file must contain one stop word per line.
		@param nc: NorvigCorrector object.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-tagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		"""
		self.mat = mat
		self.parallel_pos_file = parallel_pos_file
		self.alignments_file = alignments_file
		self.stop_words = set([word.strip() for word in open(stop_words)])
		self.nc = nc
		self.tagger = POSTagger(pos_model, stanford_tagger)

	def getSubstitutions(self, victor_corpus):
		"""
		Generates substitutions for the target words of a corpus in VICTOR format.
	
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: A dictionary that assigns target complex words to sets of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		"""
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
				leftp = leftraw.split('|||')[1].strip().upper()
				leftw = leftraw.split('|||')[0].strip()
				rightraw = target[right].strip()
				rightp = rightraw.split('|||')[1].strip().upper()
				rightw = rightraw.split('|||')[0].strip()

				if len(leftw)>0 and len(rightw)>0 and leftp!='NNP' and rightp!='NNP' and rightp==leftp and leftw not in self.stop_words and rightw not in self.stop_words and leftw!=rightw:
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

	def getPOSMap(self, victor_corpus):
		result = {}
		
		lexf = open(victor_corpus)
		sents = []
		targets = []
		heads = []
		for line in lexf:
			data = line.strip().split('\t')
			sent = data[0].strip().split(' ')
			target = data[1].strip()
			head = int(data[2].strip())
			sents.append(sent)
			targets.append(target)
			heads.append(head)
		lexf.close()
		
		tagged_sents = self.tagger.tag_sents(sents)
		
		for i in range(0, len(targets)):
			target = targets[i]
			pos_data = tagged_sents[i]
			head = heads[i]
			
			target_pos = str(pos_data[head][1]).strip()
			
			if target in result.keys():
				result[target].add(target_pos)
			else:
				result[target] = set([target_pos])
		
		return result

	def getInflectedSet(self, subs):
		#Create list of targets:
		targets = []
		
		#Create lists for inflection:
		toNothing = []
		toSingular = []
		toPlural = []
		toPAPEPA = []
		toPA = []
		toPRPA = []
		toPAPA = []
		toPR = []
		toComparative = []
		toSuperlative = []
		toOriginal = []
		
		#Fill lists:
		for target in subs.keys():
			targets.append(target)
			for pos in subs[target].keys():
				#Get cands for a target and tag combination:
				cands = list(subs[target][pos])
				
				#Add candidates to lists:
				if pos == 'NN':
					toSingular.extend(cands)
				elif pos == 'NNS':
					toPlural.extend(cands)
				elif pos == 'VB':
					toPAPEPA.extend(cands)
				elif pos == 'VBD':
					toPA.extend(cands)
					toPAPA.extend(cands)
				elif pos == 'VBG':
					toPRPA.extend(cands)
				elif pos == 'VBN':
					toPA.extend(cands)
					toPAPA.extend(cands)
				elif pos == 'VBP' or pos == 'VBZ':
					toPR.extend(cands)
				elif pos == 'JJR' or pos == 'RBR':
					toComparative.extend(cands)
				elif pos == 'JJS' or pos == 'RBS':
					toSuperlative.extend(cands)
				else:
					toNothing.extend(cands)
					
		#Lemmatize targets:
		targetsL = self.mat.lemmatizeWords(targets)
		
		#Lemmatize words:
		toNothingL = self.correctWords(self.mat.lemmatizeWords(toNothing))
		toSingularL = self.correctWords(self.mat.lemmatizeWords(toSingular))
		toPluralL = self.correctWords(self.mat.lemmatizeWords(toPlural))
		toPAPEPAL = self.correctWords(self.mat.lemmatizeWords(toPAPEPA))
		toPAL = self.correctWords(self.mat.lemmatizeWords(toPA))
		toPRPAL = self.correctWords(self.mat.lemmatizeWords(toPRPA))
		toPAPAL = self.correctWords(self.mat.lemmatizeWords(toPAPA))
		toPRL = self.correctWords(self.mat.lemmatizeWords(toPR))
		toComparativeL = self.correctWords(self.mat.lemmatizeWords(toComparative))
		toSuperlativeL = self.correctWords(self.mat.lemmatizeWords(toSuperlative))
		
		#Inflect nouns:
		singulars = self.correctWords(self.mat.inflectNouns(toSingularL, 'singular'))
		plurals = self.correctWords(self.mat.inflectNouns(toPluralL, 'plural'))
		
		#Inflect verbs:
		papepas = self.correctWords(self.mat.conjugateVerbs(toPAPEPAL, 'PAST_PERFECT_PARTICIPLE'))
		pas = self.correctWords(self.mat.conjugateVerbs(toPAL, 'PAST'))
		prpas = self.correctWords(self.mat.conjugateVerbs(toPRPAL, 'PRESENT_PARTICIPLE'))
		papas = self.correctWords(self.mat.conjugateVerbs(toPAPAL, 'PAST_PARTICIPLE'))
		prs = self.correctWords(self.mat.conjugateVerbs(toPRL, 'PRESENT'))
		
		#Inflect adjectives and adverbs:
		comparatives = self.correctWords(self.mat.inflectAdjectives(toComparativeL, 'comparative'))
		superlatives = self.correctWords(self.mat.inflectAdjectives(toSuperlativeL, 'superlative'))
		
		#Create maps:
		stemM = {}
		singularM = {}
		pluralM = {}
		papepaM = {}
		paM = {}
		prpaM = {}
		papaM = {}
		prM = {}
		comparativeM = {}
		superlativeM = {}

		for i in range(0, len(toNothing)):
			stemM[toNothing[i]] = toNothingL[i]
		for i in range(0, len(targets)):
			stemM[targets[i]] = targetsL[i]
		for i in range(0, len(toSingular)):
			stemM[toSingular[i]] = toSingularL[i]
			singularM[toSingular[i]] = singulars[i]
		for i in range(0, len(toPlural)):
			stemM[toPlural[i]] = toPluralL[i]
			pluralM[toPlural[i]] = plurals[i]
		for i in range(0, len(toPAPEPA)):
			stemM[toPAPEPA[i]] = toPAPEPAL[i]
			papepaM[toPAPEPA[i]] = papepas[i]
		for i in range(0, len(toPA)):
			stemM[toPA[i]] = toPAL[i]
			paM[toPA[i]] = pas[i]
		for i in range(0, len(toPRPA)):
			stemM[toPRPA[i]] = toPRPAL[i]
			prpaM[toPRPA[i]] = prpas[i]
		for i in range(0, len(toPAPA)):
			stemM[toPAPA[i]] = toPAPAL[i]
			papaM[toPAPA[i]] = papas[i]
		for i in range(0, len(toPR)):
			stemM[toPR[i]] = toPRL[i]
			prM[toPR[i]] = prs[i]
		for i in range(0, len(toComparative)):
			stemM[toComparative[i]] = toComparativeL[i]
			comparativeM[toComparative[i]] = comparatives[i]
		for i in range(0, len(toSuperlative)):
			stemM[toSuperlative[i]] = toSuperlativeL[i]
			superlativeM[toSuperlative[i]] = superlatives[i]
			
		#Create final substitutions:
		final_substitutions = {}
		for target in subs.keys():
			#Get lemma of target:
			targetL = stemM[target]
			
			#Create instance in final_substitutions:
			final_substitutions[target] = set([])
			
			#Iterate through pos tags of target:
			for pos in subs[target].keys():
				#Create final cands:
				final_cands = set([])
				
				#Get cands for a target and tag combination:
				cands = list(subs[target][pos])
				
				#Add candidates to lists:
				if pos == 'NN':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(singularM[cand])
							final_cands.add(cand)
				elif pos == 'NNS':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(pluralM[cand])
							final_cands.add(cand)
				elif pos == 'VB':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(papepaM[cand])
				elif pos == 'VBD':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(paM[cand])
							final_cands.add(papaM[cand])
				elif pos == 'VBG':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(prpaM[cand])
				elif pos == 'VBN':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(paM[cand])
							final_cands.add(papaM[cand])
				elif pos == 'VBP' or pos == 'VBZ':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(prM[cand])
				elif pos == 'JJR' or pos == 'RBR':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(comparativeM[cand])
				elif pos == 'JJS' or pos == 'RBS':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(superlativeM[cand])
				else:
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(cand)
					
				#Add final cands to final substitutions:
				final_substitutions[target].update(final_cands)
		return final_substitutions
		
	def correctWords(self, words):
		result = []
		for word in words:
			result.append(self.nc.correct(word))
		return result

class YamamotoGenerator:

	def __init__(self, mat, dictionary_key, nc, pos_model, stanford_tagger):
		"""
		Creates a YamamotoGenerator instance.
	
		@param mat: MorphAdornerToolkit object.
		@param dictionary_key: Key for the Merriam Dictionary.
		For more information on how to get the key for free, please refer to the LEXenstein Manual
		@param nc: NorvigCorrector object.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-tagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		"""
		self.mat = mat
		self.dictionary_key = dictionary_key
		self.nc = nc
		self.tagger = POSTagger(pos_model, stanford_tagger)

	def getSubstitutions(self, victor_corpus):
		"""
		Generates substitutions for the target words of a corpus in VICTOR format.
	
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: A dictionary that assigns target complex words to sets of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		"""
		#Get initial set of substitutions:
		print('Getting initial set of substitutions...')
		substitutions_initial = self.getInitialSet(victor_corpus)

		#Get final substitutions:
		print('Inflecting substitutions...')
		substitutions_inflected = self.getInflectedSet(substitutions_initial)

		#Return final set:
		print('Finished!')
		return substitutions_inflected

	def getInflectedSet(self, subs):
		#Create list of targets:
		targets = []
		
		#Create lists for inflection:
		toNothing = []
		toSingular = []
		toPlural = []
		toPAPEPA = []
		toPA = []
		toPRPA = []
		toPAPA = []
		toPR = []
		toComparative = []
		toSuperlative = []
		toOriginal = []
		
		#Fill lists:
		for target in subs.keys():
			targets.append(target)
			for pos in subs[target].keys():
				#Get cands for a target and tag combination:
				cands = list(subs[target][pos])
				
				#Add candidates to lists:
				if pos == 'NN':
					toSingular.extend(cands)
				elif pos == 'NNS':
					toPlural.extend(cands)
				elif pos == 'VB':
					toPAPEPA.extend(cands)
				elif pos == 'VBD':
					toPA.extend(cands)
					toPAPA.extend(cands)
				elif pos == 'VBG':
					toPRPA.extend(cands)
				elif pos == 'VBN':
					toPA.extend(cands)
					toPAPA.extend(cands)
				elif pos == 'VBP' or pos == 'VBZ':
					toPR.extend(cands)
				elif pos == 'JJR' or pos == 'RBR':
					toComparative.extend(cands)
				elif pos == 'JJS' or pos == 'RBS':
					toSuperlative.extend(cands)
				else:
					toNothing.extend(cands)
					
		#Lemmatize targets:
		targetsL = self.mat.lemmatizeWords(targets)
		
		#Lemmatize words:
		toNothingL = self.correctWords(self.mat.lemmatizeWords(toNothing))
		toSingularL = self.correctWords(self.mat.lemmatizeWords(toSingular))
		toPluralL = self.correctWords(self.mat.lemmatizeWords(toPlural))
		toPAPEPAL = self.correctWords(self.mat.lemmatizeWords(toPAPEPA))
		toPAL = self.correctWords(self.mat.lemmatizeWords(toPA))
		toPRPAL = self.correctWords(self.mat.lemmatizeWords(toPRPA))
		toPAPAL = self.correctWords(self.mat.lemmatizeWords(toPAPA))
		toPRL = self.correctWords(self.mat.lemmatizeWords(toPR))
		toComparativeL = self.correctWords(self.mat.lemmatizeWords(toComparative))
		toSuperlativeL = self.correctWords(self.mat.lemmatizeWords(toSuperlative))
		
		#Inflect nouns:
		singulars = self.correctWords(self.mat.inflectNouns(toSingularL, 'singular'))
		plurals = self.correctWords(self.mat.inflectNouns(toPluralL, 'plural'))
		
		#Inflect verbs:
		papepas = self.correctWords(self.mat.conjugateVerbs(toPAPEPAL, 'PAST_PERFECT_PARTICIPLE'))
		pas = self.correctWords(self.mat.conjugateVerbs(toPAL, 'PAST'))
		prpas = self.correctWords(self.mat.conjugateVerbs(toPRPAL, 'PRESENT_PARTICIPLE'))
		papas = self.correctWords(self.mat.conjugateVerbs(toPAPAL, 'PAST_PARTICIPLE'))
		prs = self.correctWords(self.mat.conjugateVerbs(toPRL, 'PRESENT'))
		
		#Inflect adjectives and adverbs:
		comparatives = self.correctWords(self.mat.inflectAdjectives(toComparativeL, 'comparative'))
		superlatives = self.correctWords(self.mat.inflectAdjectives(toSuperlativeL, 'superlative'))
		
		#Create maps:
		stemM = {}
		singularM = {}
		pluralM = {}
		papepaM = {}
		paM = {}
		prpaM = {}
		papaM = {}
		prM = {}
		comparativeM = {}
		superlativeM = {}

		for i in range(0, len(toNothing)):
			stemM[toNothing[i]] = toNothingL[i]
		for i in range(0, len(targets)):
			stemM[targets[i]] = targetsL[i]
		for i in range(0, len(toSingular)):
			stemM[toSingular[i]] = toSingularL[i]
			singularM[toSingular[i]] = singulars[i]
		for i in range(0, len(toPlural)):
			stemM[toPlural[i]] = toPluralL[i]
			pluralM[toPlural[i]] = plurals[i]
		for i in range(0, len(toPAPEPA)):
			stemM[toPAPEPA[i]] = toPAPEPAL[i]
			papepaM[toPAPEPA[i]] = papepas[i]
		for i in range(0, len(toPA)):
			stemM[toPA[i]] = toPAL[i]
			paM[toPA[i]] = pas[i]
		for i in range(0, len(toPRPA)):
			stemM[toPRPA[i]] = toPRPAL[i]
			prpaM[toPRPA[i]] = prpas[i]
		for i in range(0, len(toPAPA)):
			stemM[toPAPA[i]] = toPAPAL[i]
			papaM[toPAPA[i]] = papas[i]
		for i in range(0, len(toPR)):
			stemM[toPR[i]] = toPRL[i]
			prM[toPR[i]] = prs[i]
		for i in range(0, len(toComparative)):
			stemM[toComparative[i]] = toComparativeL[i]
			comparativeM[toComparative[i]] = comparatives[i]
		for i in range(0, len(toSuperlative)):
			stemM[toSuperlative[i]] = toSuperlativeL[i]
			superlativeM[toSuperlative[i]] = superlatives[i]
			
		#Create final substitutions:
		final_substitutions = {}
		for target in subs.keys():
			#Get lemma of target:
			targetL = stemM[target]
			
			#Create instance in final_substitutions:
			final_substitutions[target] = set([])
			
			#Iterate through pos tags of target:
			for pos in subs[target].keys():
				#Create final cands:
				final_cands = set([])
				
				#Get cands for a target and tag combination:
				cands = list(subs[target][pos])
				
				#Add candidates to lists:
				if pos == 'NN':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(singularM[cand])
							final_cands.add(cand)
				elif pos == 'NNS':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(pluralM[cand])
							final_cands.add(cand)
				elif pos == 'VB':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(papepaM[cand])
				elif pos == 'VBD':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(paM[cand])
							final_cands.add(papaM[cand])
				elif pos == 'VBG':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(prpaM[cand])
				elif pos == 'VBN':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(paM[cand])
							final_cands.add(papaM[cand])
				elif pos == 'VBP' or pos == 'VBZ':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(prM[cand])
				elif pos == 'JJR' or pos == 'RBR':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(comparativeM[cand])
				elif pos == 'JJS' or pos == 'RBS':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(superlativeM[cand])
				else:
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(cand)
					
				#Add final cands to final substitutions:
				final_substitutions[target].update(final_cands)
		return final_substitutions

	def getInitialSet(self, victor_corpus):
		substitutions_initial = {}
		lexf = open(victor_corpus)
		sents = []
		targets = []
		heads = []
		for line in lexf:
			data = line.strip().split('\t')
			sent = data[0].strip().split(' ')
			target = data[1].strip()
			head = int(data[2].strip())
			sents.append(sent)
			targets.append(target)
			heads.append(head)
		lexf.close()
		
		tagged_sents = self.tagger.tag_sents(sents)
		
		for i in range(0, len(sents)):
			target = targets[i]
			head = heads[i]
			target_pos = str(tagged_sents[i][head][1])

			url = 'http://www.dictionaryapi.com/api/v1/references/collegiate/xml/' + target + '?key=' + self.dictionary_key
			conn = urllib.urlopen(url)
			root = ET.fromstring(conn.read())

			cands = set([])

			entries = root.iter('entry')
			for entry in entries:
				for definition in entry.iter('dt'):
					if definition.text!=None:
						text = definition.text.strip()
						text = text[1:len(text)]
						tokens = nltk.word_tokenize(text)
						postags = nltk.pos_tag(tokens)
						for p in postags:
							postag = str(p[1].strip())
							cand = p[0].strip()
							if postag==target_pos:
								try:
									test = codecs.ascii_encode(cand)
									cands.add(cand)
								except UnicodeEncodeError:
									pass
			if len(cands)>0:
				if target in substitutions_initial:
					substitutions_initial[target][target_pos] = cands
				else:
					substitutions_initial[target] = {target_pos:cands}
		return substitutions_initial
		
	def correctWords(self, words):
		result = []
		for word in words:
			result.append(self.nc.correct(word))
		return result

class MerriamGenerator:

	def __init__(self, mat, thesaurus_key, nc, pos_model, stanford_tagger):
		"""
		Creates a MerriamGenerator instance.
	
		@param mat: MorphAdornerToolkit object.
		@param thesaurus_key: Key for the Merriam Thesaurus.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-tagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		"""
		self.mat = mat
		self.thesaurus_key = thesaurus_key
		self.nc = nc
		self.tagger = POSTagger(pos_model, stanford_tagger)

	def getSubstitutions(self, victor_corpus):
		"""
		Generates substitutions for the target words of a corpus in VICTOR format.
	
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: A dictionary that assigns target complex words to sets of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		"""
		#Get initial set of substitutions:
		print('Getting initial set of substitutions...')
		substitutions_initial = self.getInitialSet(victor_corpus)

		#Get final substitutions:
		print('Inflecting substitutions...')
		substitutions_inflected = self.getInflectedSet(substitutions_initial)

		#Return final set:
		print('Finished!')
		return substitutions_inflected

	def getInflectedSet(self, subs):
		#Create list of targets:
		targets = []
		
		#Create lists for inflection:
		toNothing = []
		toSingular = []
		toPlural = []
		toPAPEPA = []
		toPA = []
		toPRPA = []
		toPAPA = []
		toPR = []
		toComparative = []
		toSuperlative = []
		toOriginal = []
		
		#Fill lists:
		for target in subs.keys():
			targets.append(target)
			for pos in subs[target].keys():
				#Get cands for a target and tag combination:
				cands = list(subs[target][pos])
				
				#Add candidates to lists:
				if pos == 'NN':
					toSingular.extend(cands)
				elif pos == 'NNS':
					toPlural.extend(cands)
				elif pos == 'VB':
					toPAPEPA.extend(cands)
				elif pos == 'VBD':
					toPA.extend(cands)
					toPAPA.extend(cands)
				elif pos == 'VBG':
					toPRPA.extend(cands)
				elif pos == 'VBN':
					toPA.extend(cands)
					toPAPA.extend(cands)
				elif pos == 'VBP' or pos == 'VBZ':
					toPR.extend(cands)
				elif pos == 'JJR' or pos == 'RBR':
					toComparative.extend(cands)
				elif pos == 'JJS' or pos == 'RBS':
					toSuperlative.extend(cands)
				else:
					toNothing.extend(cands)
					
		#Lemmatize targets:
		targetsL = self.mat.lemmatizeWords(targets)
		
		#Lemmatize words:
		toNothingL = self.correctWords(self.mat.lemmatizeWords(toNothing))
		toSingularL = self.correctWords(self.mat.lemmatizeWords(toSingular))
		toPluralL = self.correctWords(self.mat.lemmatizeWords(toPlural))
		toPAPEPAL = self.correctWords(self.mat.lemmatizeWords(toPAPEPA))
		toPAL = self.correctWords(self.mat.lemmatizeWords(toPA))
		toPRPAL = self.correctWords(self.mat.lemmatizeWords(toPRPA))
		toPAPAL = self.correctWords(self.mat.lemmatizeWords(toPAPA))
		toPRL = self.correctWords(self.mat.lemmatizeWords(toPR))
		toComparativeL = self.correctWords(self.mat.lemmatizeWords(toComparative))
		toSuperlativeL = self.correctWords(self.mat.lemmatizeWords(toSuperlative))
		
		#Inflect nouns:
		singulars = self.correctWords(self.mat.inflectNouns(toSingularL, 'singular'))
		plurals = self.correctWords(self.mat.inflectNouns(toPluralL, 'plural'))
		
		#Inflect verbs:
		papepas = self.correctWords(self.mat.conjugateVerbs(toPAPEPAL, 'PAST_PERFECT_PARTICIPLE'))
		pas = self.correctWords(self.mat.conjugateVerbs(toPAL, 'PAST'))
		prpas = self.correctWords(self.mat.conjugateVerbs(toPRPAL, 'PRESENT_PARTICIPLE'))
		papas = self.correctWords(self.mat.conjugateVerbs(toPAPAL, 'PAST_PARTICIPLE'))
		prs = self.correctWords(self.mat.conjugateVerbs(toPRL, 'PRESENT'))
		
		#Inflect adjectives and adverbs:
		comparatives = self.correctWords(self.mat.inflectAdjectives(toComparativeL, 'comparative'))
		superlatives = self.correctWords(self.mat.inflectAdjectives(toSuperlativeL, 'superlative'))
		
		#Create maps:
		stemM = {}
		singularM = {}
		pluralM = {}
		papepaM = {}
		paM = {}
		prpaM = {}
		papaM = {}
		prM = {}
		comparativeM = {}
		superlativeM = {}

		for i in range(0, len(toNothing)):
			stemM[toNothing[i]] = toNothingL[i]
		for i in range(0, len(targets)):
			stemM[targets[i]] = targetsL[i]
		for i in range(0, len(toSingular)):
			stemM[toSingular[i]] = toSingularL[i]
			singularM[toSingular[i]] = singulars[i]
		for i in range(0, len(toPlural)):
			stemM[toPlural[i]] = toPluralL[i]
			pluralM[toPlural[i]] = plurals[i]
		for i in range(0, len(toPAPEPA)):
			stemM[toPAPEPA[i]] = toPAPEPAL[i]
			papepaM[toPAPEPA[i]] = papepas[i]
		for i in range(0, len(toPA)):
			stemM[toPA[i]] = toPAL[i]
			paM[toPA[i]] = pas[i]
		for i in range(0, len(toPRPA)):
			stemM[toPRPA[i]] = toPRPAL[i]
			prpaM[toPRPA[i]] = prpas[i]
		for i in range(0, len(toPAPA)):
			stemM[toPAPA[i]] = toPAPAL[i]
			papaM[toPAPA[i]] = papas[i]
		for i in range(0, len(toPR)):
			stemM[toPR[i]] = toPRL[i]
			prM[toPR[i]] = prs[i]
		for i in range(0, len(toComparative)):
			stemM[toComparative[i]] = toComparativeL[i]
			comparativeM[toComparative[i]] = comparatives[i]
		for i in range(0, len(toSuperlative)):
			stemM[toSuperlative[i]] = toSuperlativeL[i]
			superlativeM[toSuperlative[i]] = superlatives[i]
			
		#Create final substitutions:
		final_substitutions = {}
		for target in subs.keys():
			#Get lemma of target:
			targetL = stemM[target]
			
			#Create instance in final_substitutions:
			final_substitutions[target] = set([])
			
			#Iterate through pos tags of target:
			for pos in subs[target].keys():
				#Create final cands:
				final_cands = set([])
				
				#Get cands for a target and tag combination:
				cands = list(subs[target][pos])
				
				#Add candidates to lists:
				if pos == 'NN':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(singularM[cand])
							final_cands.add(cand)
				elif pos == 'NNS':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(pluralM[cand])
							final_cands.add(cand)
				elif pos == 'VB':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(papepaM[cand])
				elif pos == 'VBD':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(paM[cand])
							final_cands.add(papaM[cand])
				elif pos == 'VBG':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(prpaM[cand])
				elif pos == 'VBN':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(paM[cand])
							final_cands.add(papaM[cand])
				elif pos == 'VBP' or pos == 'VBZ':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(prM[cand])
				elif pos == 'JJR' or pos == 'RBR':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(comparativeM[cand])
				elif pos == 'JJS' or pos == 'RBS':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(superlativeM[cand])
				else:
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(cand)
					
				#Add final cands to final substitutions:
				final_substitutions[target].update(final_cands)
		return final_substitutions

	def getInitialSet(self, victor_corpus):
		substitutions_initial = {}
		lexf = open(victor_corpus)
		sents = []
		targets = []
		heads = []
		for line in lexf:
			data = line.strip().split('\t')
			sent = data[0].strip().split(' ')
			target = data[1].strip()
			head = int(data[2].strip())
			sents.append(sent)
			targets.append(target)
			heads.append(head)
		lexf.close()
		
		tagged_sents = self.tagger.tag_sents(sents)
		
		for i in range(0, len(sents)):
			target = targets[i]
			head = heads[i]
			target_pos = str(tagged_sents[i][head][1])
			url = 'http://www.dictionaryapi.com/api/v1/references/thesaurus/xml/' + target + '?key=' + self.thesaurus_key
			conn = urllib.urlopen(url)
			root = ET.fromstring(conn.read())
			root = root.findall('entry')

			cands = set([])
			if len(root)>0:
				for root_node in root:
					syn = None
					for sense in root_node.iter('sens'):
						syndata = sense.findall('syn')
						if len(syndata)>0:
							syn = syndata[0]
					if syn:
						res = ''
						for snip in syn.itertext():
							res += snip + ' '
						finds = re.findall('\([^\)]+\)', res)
						for find in finds:
							res = res.replace(find, '')

						synonyms = [s.strip() for s in res.split(',')]

						for synonym in synonyms:
							if len(synonym.split(' '))==1:
								try:
									test = codecs.ascii_encode(synonym)
									cands.add(synonym)
								except UnicodeEncodeError:
									pass
			if len(cands)>0:
				if target in substitutions_initial:
					substitutions_initial[target][target_pos] = cands
				else:
					substitutions_initial[target] = {target_pos:cands}
		return substitutions_initial
		
	def correctWords(self, words):
		result = []
		for word in words:
			result.append(self.nc.correct(word))
		return result

#Class for the Wordnet Generator
class WordnetGenerator:

	def __init__(self, mat, nc, pos_model, stanford_tagger):
		"""
		Creates a WordnetGenerator instance.
	
		@param mat: MorphAdornerToolkit object.
		@param nc: NorvigCorrector object.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-tagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		"""
		self.mat = mat
		self.nc = nc
		self.tagger = POSTagger(pos_model, stanford_tagger)

	def getSubstitutions(self, victor_corpus):
		"""
		Generates substitutions for the target words of a corpus in VICTOR format.
	
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: A dictionary that assigns target complex words to sets of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		"""
		
		#Get initial set of substitutions:
		print('Getting initial set of substitutions...')
		substitutions_initial = self.getInitialSet(victor_corpus)

		#Get final substitutions:
		print('Inflecting substitutions...')
		substitutions_inflected = self.getInflectedSet(substitutions_initial)

		#Return final set:
		print('Finished!')
		return substitutions_inflected

	def getInflectedSet(self, subs):
		#Create list of targets:
		targets = []
		
		#Create lists for inflection:
		toNothing = []
		toSingular = []
		toPlural = []
		toPAPEPA = []
		toPA = []
		toPRPA = []
		toPAPA = []
		toPR = []
		toComparative = []
		toSuperlative = []
		toOriginal = []
		
		#Fill lists:
		for target in subs.keys():
			targets.append(target)
			for pos in subs[target].keys():
				#Get cands for a target and tag combination:
				cands = list(subs[target][pos])
				
				#Add candidates to lists:
				if pos == 'NN':
					toSingular.extend(cands)
				elif pos == 'NNS':
					toPlural.extend(cands)
				elif pos == 'VB':
					toPAPEPA.extend(cands)
				elif pos == 'VBD':
					toPA.extend(cands)
					toPAPA.extend(cands)
				elif pos == 'VBG':
					toPRPA.extend(cands)
				elif pos == 'VBN':
					toPA.extend(cands)
					toPAPA.extend(cands)
				elif pos == 'VBP' or pos == 'VBZ':
					toPR.extend(cands)
				elif pos == 'JJR' or pos == 'RBR':
					toComparative.extend(cands)
				elif pos == 'JJS' or pos == 'RBS':
					toSuperlative.extend(cands)
				else:
					toNothing.extend(cands)
					
		#Lemmatize targets:
		targetsL = self.mat.lemmatizeWords(targets)
		
		#Lemmatize words:
		toNothingL = self.correctWords(self.mat.lemmatizeWords(toNothing))
		toSingularL = self.correctWords(self.mat.lemmatizeWords(toSingular))
		toPluralL = self.correctWords(self.mat.lemmatizeWords(toPlural))
		toPAPEPAL = self.correctWords(self.mat.lemmatizeWords(toPAPEPA))
		toPAL = self.correctWords(self.mat.lemmatizeWords(toPA))
		toPRPAL = self.correctWords(self.mat.lemmatizeWords(toPRPA))
		toPAPAL = self.correctWords(self.mat.lemmatizeWords(toPAPA))
		toPRL = self.correctWords(self.mat.lemmatizeWords(toPR))
		toComparativeL = self.correctWords(self.mat.lemmatizeWords(toComparative))
		toSuperlativeL = self.correctWords(self.mat.lemmatizeWords(toSuperlative))
		
		#Inflect nouns:
		singulars = self.correctWords(self.mat.inflectNouns(toSingularL, 'singular'))
		plurals = self.correctWords(self.mat.inflectNouns(toPluralL, 'plural'))
		
		#Inflect verbs:
		papepas = self.correctWords(self.mat.conjugateVerbs(toPAPEPAL, 'PAST_PERFECT_PARTICIPLE'))
		pas = self.correctWords(self.mat.conjugateVerbs(toPAL, 'PAST'))
		prpas = self.correctWords(self.mat.conjugateVerbs(toPRPAL, 'PRESENT_PARTICIPLE'))
		papas = self.correctWords(self.mat.conjugateVerbs(toPAPAL, 'PAST_PARTICIPLE'))
		prs = self.correctWords(self.mat.conjugateVerbs(toPRL, 'PRESENT'))
		
		#Inflect adjectives and adverbs:
		comparatives = self.correctWords(self.mat.inflectAdjectives(toComparativeL, 'comparative'))
		superlatives = self.correctWords(self.mat.inflectAdjectives(toSuperlativeL, 'superlative'))
		
		#Create maps:
		stemM = {}
		singularM = {}
		pluralM = {}
		papepaM = {}
		paM = {}
		prpaM = {}
		papaM = {}
		prM = {}
		comparativeM = {}
		superlativeM = {}

		for i in range(0, len(toNothing)):
			stemM[toNothing[i]] = toNothingL[i]
		for i in range(0, len(targets)):
			stemM[targets[i]] = targetsL[i]
		for i in range(0, len(toSingular)):
			stemM[toSingular[i]] = toSingularL[i]
			singularM[toSingular[i]] = singulars[i]
		for i in range(0, len(toPlural)):
			stemM[toPlural[i]] = toPluralL[i]
			pluralM[toPlural[i]] = plurals[i]
		for i in range(0, len(toPAPEPA)):
			stemM[toPAPEPA[i]] = toPAPEPAL[i]
			papepaM[toPAPEPA[i]] = papepas[i]
		for i in range(0, len(toPA)):
			stemM[toPA[i]] = toPAL[i]
			paM[toPA[i]] = pas[i]
		for i in range(0, len(toPRPA)):
			stemM[toPRPA[i]] = toPRPAL[i]
			prpaM[toPRPA[i]] = prpas[i]
		for i in range(0, len(toPAPA)):
			stemM[toPAPA[i]] = toPAPAL[i]
			papaM[toPAPA[i]] = papas[i]
		for i in range(0, len(toPR)):
			stemM[toPR[i]] = toPRL[i]
			prM[toPR[i]] = prs[i]
		for i in range(0, len(toComparative)):
			stemM[toComparative[i]] = toComparativeL[i]
			comparativeM[toComparative[i]] = comparatives[i]
		for i in range(0, len(toSuperlative)):
			stemM[toSuperlative[i]] = toSuperlativeL[i]
			superlativeM[toSuperlative[i]] = superlatives[i]
			
		#Create final substitutions:
		final_substitutions = {}
		for target in subs.keys():
			#Get lemma of target:
			targetL = stemM[target]
			
			#Create instance in final_substitutions:
			final_substitutions[target] = set([])
			
			#Iterate through pos tags of target:
			for pos in subs[target].keys():
				#Create final cands:
				final_cands = set([])
				
				#Get cands for a target and tag combination:
				cands = list(subs[target][pos])
				
				#Add candidates to lists:
				if pos == 'NN':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(singularM[cand])
							final_cands.add(cand)
				elif pos == 'NNS':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(pluralM[cand])
							final_cands.add(cand)
				elif pos == 'VB':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(papepaM[cand])
				elif pos == 'VBD':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(paM[cand])
							final_cands.add(papaM[cand])
				elif pos == 'VBG':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(prpaM[cand])
				elif pos == 'VBN':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(paM[cand])
							final_cands.add(papaM[cand])
				elif pos == 'VBP' or pos == 'VBZ':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(prM[cand])
				elif pos == 'JJR' or pos == 'RBR':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(comparativeM[cand])
				elif pos == 'JJS' or pos == 'RBS':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(superlativeM[cand])
				else:
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(cand)
					
				#Add final cands to final substitutions:
				final_substitutions[target].update(final_cands)
		return final_substitutions

	def getExpandedSet(self, subs):
		#Create lists for inflection:
		nouns = set([])
		verbs = set([])
		adjectives = set([])
		
		#Fill lists:
		for target in subs.keys():
			for pos in subs[target].keys():
				#Get cands for a target and tag combination:
				cands = list(subs[target][pos])
				
				#Add candidates to lists:
				if pos == 'NN' or pos == 'NNS':
					nouns.add(target)
				elif pos.startswith('V'):
					verbs.add(target)
				elif pos.startswith('J') or pos.startswith('RB'):
					adjectives.add(target)
		
		#Transform sets in lists:
		nouns = list(nouns)
		verbs = list(verbs)
		adjectives = list(adjectives)
		
		#Lemmatize words:
		nounsL = self.correctWords(self.mat.lemmatizeWords(nouns))
		verbsL = self.correctWords(self.mat.lemmatizeWords(verbs))
		adjectivesL = self.correctWords(self.mat.lemmatizeWords(adjectives))
		
		#Create lemma maps:
		nounM = {}
		verbM = {}
		adjectiveM = {}
		for i in range(0, len(nouns)):
			nounM[nouns[i]] = nounsL[i]
		for i in range(0, len(verbs)):
			verbM[verbs[i]] = verbsL[i]
		for i in range(0, len(adjectives)):
			adjectiveM[adjectives[i]] = adjectivesL[i]
		
		#Inflect words:
		plurals = self.correctWords(self.mat.inflectNouns(nounsL, 'plural'))
		pas = self.correctWords(self.mat.conjugateVerbs(verbsL, 'PAST'))
		prpas = self.correctWords(self.mat.conjugateVerbs(verbsL, 'PRESENT_PARTICIPLE'))
		papas = self.correctWords(self.mat.conjugateVerbs(verbsL, 'PAST_PARTICIPLE'))
		prs = self.correctWords(self.mat.conjugateVerbs(verbsL, 'PRESENT'))
		comparatives = self.correctWords(self.mat.inflectAdjectives(adjectives, 'comparative'))
		superlatives = self.correctWords(self.mat.inflectAdjectives(adjectives, 'superlative'))
		
		#Create inflected maps:
		pluralM = {}
		paM = {}
		prpaM = {}
		papaM = {}
		prM = {}
		comparativeM = {}
		superlativeM = {}
		for i in range(0, len(nouns)):
			pluralM[nouns[i]] = plurals[i]
		for i in range(0, len(verbs)):
			paM[verbs[i]] = pas[i]
			prpaM[verbs[i]] = prpas[i]
			papaM[verbs[i]] = papas[i]
			prM[verbs[i]] = prs[i]
		for i in range(0, len(adjectives)):
			comparativeM[adjectives[i]] = comparatives[i]
			superlativeM[adjectives[i]] = superlatives[i]
		
		#Create extended substitutions:
		substitutions_extended = {}
		for target in subs.keys():
			for pos in subs[target].keys():
				#Get cands for a target and tag combination:
				cands = list(subs[target][pos])
				
				#Add original to substitution dictionary:
				self.addToExtended(target, pos, cands, substitutions_extended)
				
				#Add candidates to lists:
				if pos == 'NN':
					pluralT = pluralM[target]
					self.addToExtended(pluralT, 'NNS', cands, substitutions_extended)
				elif pos == 'NNS':
					singularT = nounM[target]
					self.addToExtended(singularT, 'NN', cands, substitutions_extended)
				elif pos == 'VB':
					paT = paM[target]
					prpaT = prpaM[target]
					papaT = papaM[target]
					prT = prM[target]
					self.addToExtended(paT, 'VBD', cands, substitutions_extended)
					self.addToExtended(prpaT, 'VBG', cands, substitutions_extended)
					self.addToExtended(papaT, 'VBN', cands, substitutions_extended)
					self.addToExtended(prT, 'VBP', cands, substitutions_extended)
					self.addToExtended(prT, 'VBZ', cands, substitutions_extended)
				elif pos == 'VBD':
					lemmaT = verbM[target]
					prpaT = prpaM[target]
					papaT = papaM[target]
					prT = prM[target]
					self.addToExtended(lemmaT, 'VB', cands, substitutions_extended)
					self.addToExtended(prpaT, 'VBG', cands, substitutions_extended)
					self.addToExtended(papaT, 'VBN', cands, substitutions_extended)
					self.addToExtended(prT, 'VBP', cands, substitutions_extended)
					self.addToExtended(prT, 'VBZ', cands, substitutions_extended)
				elif pos == 'VBG':
					lemmaT = verbM[target]
					paT = paM[target]
					papaT = papaM[target]
					prT = prM[target]
					self.addToExtended(lemmaT, 'VB', cands, substitutions_extended)
					self.addToExtended(paT, 'VBD', cands, substitutions_extended)
					self.addToExtended(papaT, 'VBN', cands, substitutions_extended)
					self.addToExtended(prT, 'VBP', cands, substitutions_extended)
					self.addToExtended(prT, 'VBZ', cands, substitutions_extended)
				elif pos == 'VBN':
					lemmaT = verbM[target]
					paT = paM[target]
					prpaT = prpaM[target]
					prT = prM[target]
					self.addToExtended(lemmaT, 'VB', cands, substitutions_extended)
					self.addToExtended(paT, 'VBD', cands, substitutions_extended)
					self.addToExtended(prpaT, 'VBG', cands, substitutions_extended)
					self.addToExtended(prT, 'VBP', cands, substitutions_extended)
					self.addToExtended(prT, 'VBZ', cands, substitutions_extended)
				elif pos == 'VBP':
					lemmaT = verbM[target]
					paT = paM[target]
					prpaT = prpaM[target]
					papaT = prM[target]
					self.addToExtended(target, 'VBZ', cands, substitutions_extended)
					self.addToExtended(lemmaT, 'VB', cands, substitutions_extended)
					self.addToExtended(paT, 'VBD', cands, substitutions_extended)
					self.addToExtended(prpaT, 'VBG', cands, substitutions_extended)
					self.addToExtended(papaT, 'VBN', cands, substitutions_extended)
				elif pos == 'VBZ':
					lemmaT = verbM[target]
					paT = paM[target]
					prpaT = prpaM[target]
					papaT = prM[target]
					self.addToExtended(target, 'VBP', cands, substitutions_extended)
					self.addToExtended(lemmaT, 'VB', cands, substitutions_extended)
					self.addToExtended(paT, 'VBD', cands, substitutions_extended)
					self.addToExtended(prpaT, 'VBG', cands, substitutions_extended)
					self.addToExtended(papaT, 'VBN', cands, substitutions_extended)
				elif pos == 'JJ':
					comparativeT = comparativeM[target]
					superlativeT = superlativeM[target]
					self.addToExtended(comparativeT, 'JJR', cands, substitutions_extended)
					self.addToExtended(superlativeT, 'JJS', cands, substitutions_extended)
				elif pos == 'JJR':
					lemmaT = adjectiveM[target]
					superlativeT = superlativeM[target]
					self.addToExtended(lemmaT, 'JJ', cands, substitutions_extended)
					self.addToExtended(superlativeT, 'JJS', cands, substitutions_extended)
				elif pos == 'JJS':
					lemmaT = adjectiveM[target]
					comparativeT = comparativeM[target]
					self.addToExtended(lemmaT, 'JJ', cands, substitutions_extended)
					self.addToExtended(comparativeT, 'JJR', cands, substitutions_extended)
				elif pos == 'RB':
					comparativeT = comparativeM[target]
					superlativeT = superlativeM[target]
					self.addToExtended(comparativeT, 'RBR', cands, substitutions_extended)
					self.addToExtended(superlativeT, 'RBS', cands, substitutions_extended)
				elif pos == 'RBR':
					lemmaT = adjectiveM[target]
					superlativeT = superlativeM[target]
					self.addToExtended(lemmaT, 'RB', cands, substitutions_extended)
					self.addToExtended(superlativeT, 'RBS', cands, substitutions_extended)
				elif pos == 'RBS':
					lemmaT = adjectiveM[target]
					comparativeT = comparativeM[target]
					self.addToExtended(lemmaT, 'RB', cands, substitutions_extended)
					self.addToExtended(comparativeT, 'RBR', cands, substitutions_extended)
		return substitutions_extended
		
	def getInitialSet(self, victor_corpus):
		substitutions_initial = {}
		lexf = open(victor_corpus)
		sents = []
		targets = []
		heads = []
		for line in lexf:
			data = line.strip().split('\t')
			sent = data[0].strip().split(' ')
			target = data[1].strip()
			head = int(data[2].strip())
			sents.append(sent)
			targets.append(target)
			heads.append(head)
		lexf.close()
		
		tagged_sents = self.tagger.tag_sents(sents)
		
		for i in range(0, len(sents)):
			target = targets[i]
			head = heads[i]
			target_pos = str(tagged_sents[i][head][1])
			target_wnpos = self.getWordnetPOS(target_pos)
			
			syns = wn.synsets(target)

			cands = set([])
			for syn in syns:
				for lem in syn.lemmas():
					candidate = self.cleanLemma(lem.name())
					if len(candidate.split(' '))==1:
						cands.add(candidate)
			if len(cands)>0:
				if target in substitutions_initial:
					substitutions_initial[target][target_pos] = cands
				else:
					substitutions_initial[target] = {target_pos:cands}
		return substitutions_initial

	def addToExtended(self, target, tag, cands, subs):
		if target not in subs.keys():
			subs[target] = {tag:cands}
		else:
			if tag not in subs[target].keys():
				subs[target][tag] = cands
			else:
				subs[target][tag].extend(cands)
		
	def correctWords(self, words):
		result = []
		for word in words:
			result.append(self.nc.correct(word))
		return result

	def cleanLemma(self, lem):
		result = ''
		aux = lem.strip().split('_')
		for word in aux:
			result += word + ' '
		return result.strip()
		
	def getWordnetPOS(self, pos):
		if pos[0] == 'N' or pos[0] == 'V' or pos == 'RBR' or pos == 'RBS':
			return pos[0].lower()
		elif pos[0] == 'J':
			return 'a'
		else:
			return None

#Class for the Biran Generator:
class BiranGenerator:

	def __init__(self, mat, complex_vocab, simple_vocab, complex_lm, simple_lm, nc, pos_model, stanford_tagger):
		"""
		Creates a BiranGenerator instance.
	
		@param mat: MorphAdornerToolkit object.
		@param complex_vocab: Path to a vocabulary of complex words.
		For more information on how to create the file, refer to the LEXenstein Manual.
		@param simple_vocab: Path to a vocabulary of simple words.
		For more information on how to create the file, refer to the LEXenstein Manual.
		@param complex_lm: Path to a language model built over complex text.
		For more information on how to create the file, refer to the LEXenstein Manual.
		@param simple_lm: Path to a language model built over simple text.
		For more information on how to create the file, refer to the LEXenstein Manual.
		@param nc: NorvigCorrector object.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-tagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		"""

		self.complex_vocab = self.getVocab(complex_vocab)
		self.simple_vocab = self.getVocab(simple_vocab)
		self.complex_lm = kenlm.LanguageModel(complex_lm)
		self.simple_lm = kenlm.LanguageModel(simple_lm)
		self.mat = mat
		self.nc = nc
		self.tagger = POSTagger(pos_model, stanford_tagger)

	def getSubstitutions(self, victor_corpus):
		"""
		Generates substitutions for the target words of a corpus in VICTOR format.
	
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: A dictionary that assigns target complex words to sets of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		"""
		
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

	def getInflectedSet(self, subs):
		#Create list of targets:
		targets = []
		
		#Create lists for inflection:
		toNothing = []
		toSingular = []
		toPlural = []
		toPAPEPA = []
		toPA = []
		toPRPA = []
		toPAPA = []
		toPR = []
		toComparative = []
		toSuperlative = []
		toOriginal = []
		
		#Fill lists:
		for target in subs.keys():
			targets.append(target)
			for pos in subs[target].keys():
				#Get cands for a target and tag combination:
				cands = list(subs[target][pos])
				
				#Add candidates to lists:
				if pos == 'NN':
					toSingular.extend(cands)
				elif pos == 'NNS':
					toPlural.extend(cands)
				elif pos == 'VB':
					toPAPEPA.extend(cands)
				elif pos == 'VBD':
					toPA.extend(cands)
					toPAPA.extend(cands)
				elif pos == 'VBG':
					toPRPA.extend(cands)
				elif pos == 'VBN':
					toPA.extend(cands)
					toPAPA.extend(cands)
				elif pos == 'VBP' or pos == 'VBZ':
					toPR.extend(cands)
				elif pos == 'JJR' or pos == 'RBR':
					toComparative.extend(cands)
				elif pos == 'JJS' or pos == 'RBS':
					toSuperlative.extend(cands)
				else:
					toNothing.extend(cands)
					
		#Lemmatize targets:
		targetsL = self.mat.lemmatizeWords(targets)
		
		#Lemmatize words:
		toNothingL = self.correctWords(self.mat.lemmatizeWords(toNothing))
		toSingularL = self.correctWords(self.mat.lemmatizeWords(toSingular))
		toPluralL = self.correctWords(self.mat.lemmatizeWords(toPlural))
		toPAPEPAL = self.correctWords(self.mat.lemmatizeWords(toPAPEPA))
		toPAL = self.correctWords(self.mat.lemmatizeWords(toPA))
		toPRPAL = self.correctWords(self.mat.lemmatizeWords(toPRPA))
		toPAPAL = self.correctWords(self.mat.lemmatizeWords(toPAPA))
		toPRL = self.correctWords(self.mat.lemmatizeWords(toPR))
		toComparativeL = self.correctWords(self.mat.lemmatizeWords(toComparative))
		toSuperlativeL = self.correctWords(self.mat.lemmatizeWords(toSuperlative))
		
		#Inflect nouns:
		singulars = self.correctWords(self.mat.inflectNouns(toSingularL, 'singular'))
		plurals = self.correctWords(self.mat.inflectNouns(toPluralL, 'plural'))
		
		#Inflect verbs:
		papepas = self.correctWords(self.mat.conjugateVerbs(toPAPEPAL, 'PAST_PERFECT_PARTICIPLE'))
		pas = self.correctWords(self.mat.conjugateVerbs(toPAL, 'PAST'))
		prpas = self.correctWords(self.mat.conjugateVerbs(toPRPAL, 'PRESENT_PARTICIPLE'))
		papas = self.correctWords(self.mat.conjugateVerbs(toPAPAL, 'PAST_PARTICIPLE'))
		prs = self.correctWords(self.mat.conjugateVerbs(toPRL, 'PRESENT'))
		
		#Inflect adjectives and adverbs:
		comparatives = self.correctWords(self.mat.inflectAdjectives(toComparativeL, 'comparative'))
		superlatives = self.correctWords(self.mat.inflectAdjectives(toSuperlativeL, 'superlative'))
		
		#Create maps:
		stemM = {}
		singularM = {}
		pluralM = {}
		papepaM = {}
		paM = {}
		prpaM = {}
		papaM = {}
		prM = {}
		comparativeM = {}
		superlativeM = {}

		for i in range(0, len(toNothing)):
			stemM[toNothing[i]] = toNothingL[i]
		for i in range(0, len(targets)):
			stemM[targets[i]] = targetsL[i]
		for i in range(0, len(toSingular)):
			stemM[toSingular[i]] = toSingularL[i]
			singularM[toSingular[i]] = singulars[i]
		for i in range(0, len(toPlural)):
			stemM[toPlural[i]] = toPluralL[i]
			pluralM[toPlural[i]] = plurals[i]
		for i in range(0, len(toPAPEPA)):
			stemM[toPAPEPA[i]] = toPAPEPAL[i]
			papepaM[toPAPEPA[i]] = papepas[i]
		for i in range(0, len(toPA)):
			stemM[toPA[i]] = toPAL[i]
			paM[toPA[i]] = pas[i]
		for i in range(0, len(toPRPA)):
			stemM[toPRPA[i]] = toPRPAL[i]
			prpaM[toPRPA[i]] = prpas[i]
		for i in range(0, len(toPAPA)):
			stemM[toPAPA[i]] = toPAPAL[i]
			papaM[toPAPA[i]] = papas[i]
		for i in range(0, len(toPR)):
			stemM[toPR[i]] = toPRL[i]
			prM[toPR[i]] = prs[i]
		for i in range(0, len(toComparative)):
			stemM[toComparative[i]] = toComparativeL[i]
			comparativeM[toComparative[i]] = comparatives[i]
		for i in range(0, len(toSuperlative)):
			stemM[toSuperlative[i]] = toSuperlativeL[i]
			superlativeM[toSuperlative[i]] = superlatives[i]
			
		#Create final substitutions:
		final_substitutions = {}
		for target in subs.keys():
			#Get lemma of target:
			targetL = stemM[target]
			
			#Create instance in final_substitutions:
			final_substitutions[target] = set([])
			
			#Iterate through pos tags of target:
			for pos in subs[target].keys():
				#Create final cands:
				final_cands = set([])
				
				#Get cands for a target and tag combination:
				cands = list(subs[target][pos])
				
				#Add candidates to lists:
				if pos == 'NN':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(singularM[cand])
							final_cands.add(cand)
				elif pos == 'NNS':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(pluralM[cand])
							final_cands.add(cand)
				elif pos == 'VB':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(papepaM[cand])
				elif pos == 'VBD':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(paM[cand])
							final_cands.add(papaM[cand])
				elif pos == 'VBG':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(prpaM[cand])
				elif pos == 'VBN':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(paM[cand])
							final_cands.add(papaM[cand])
				elif pos == 'VBP' or pos == 'VBZ':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(prM[cand])
				elif pos == 'JJR' or pos == 'RBR':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(comparativeM[cand])
				elif pos == 'JJS' or pos == 'RBS':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(superlativeM[cand])
				else:
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(cand)
					
				#Add final cands to final substitutions:
				final_substitutions[target].update(final_cands)
		return final_substitutions
		
	def getInitialSet(self, victor_corpus):
		substitutions_initial = {}
		lexf = open(victor_corpus)
		sents = []
		targets = []
		heads = []
		for line in lexf:
			data = line.strip().split('\t')
			sent = data[0].strip().split(' ')
			target = data[1].strip()
			head = int(data[2].strip())
			sents.append(sent)
			targets.append(target)
			heads.append(head)
		lexf.close()
		
		tagged_sents = self.tagger.tag_sents(sents)
		
		for i in range(0, len(sents)):
			target = targets[i]
			head = heads[i]
			target_pos = str(tagged_sents[i][head][1])
			target_wnpos = self.getWordnetPOS(target_pos)

			if target in self.complex_vocab:
				syns = wn.synsets(target)
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
				if target in cands:
					cands.remove(target)
				if len(cands)>0:
					if target in substitutions_initial:
						substitutions_initial[target][target_pos] = cands
					else:
						substitutions_initial[target] = {target_pos:cands}
		return substitutions_initial
		
	def getComplexity(self, word, clm, slm):
		C = (clm.score(word, bos=False, eos=False))/(slm.score(word, bos=False, eos=False))
		L = float(len(word))
		return C*L

	def getVocab(self, path):
		return set([line.strip() for line in open(path)])

	def cleanLemma(self, lem):
		result = ''
		aux = lem.strip().split('_')
		for word in aux:
			result += word + ' '
		return result.strip()

	def getWordnetPOS(self, pos):
		if pos[0] == 'N' or pos[0] == 'V' or pos == 'RBR' or pos == 'RBS':
			return pos[0].lower()
		elif pos[0] == 'J':
			return 'a'
		else:
			return None

	def correctWords(self, words):
		result = []
		for word in words:
			result.append(self.nc.correct(word))
		return result
