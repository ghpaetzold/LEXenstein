import xml.etree.ElementTree as ET
import re
import urllib2 as urllib
from nltk.corpus import wordnet as wn
import subprocess
import nltk
from nltk.tag.stanford import StanfordPOSTagger
import kenlm
import codecs
import os
import gensim
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

class PaetzoldGenerator:

	def __init__(self, posw2vmodel, nc, pos_model, stanford_tagger, java_path):
		"""
		Creates a PaetzoldGenerator instance.
	
		@param posw2vmodel: Binary parsed word vector model.
		For more information on how to produce the model, please refer to the LEXenstein Manual.
		@param nc: NorvigCorrector object.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		"""
		self.lemmatizer = WordNetLemmatizer()
		self.stemmer = PorterStemmer()
		self.model = gensim.models.KeyedVectors.load_word2vec_format(posw2vmodel, binary=True)
		self.nc = nc
		os.environ['JAVAHOME'] = java_path
		self.tagger = StanfordPOSTagger(pos_model, stanford_tagger)

	def getSubstitutions(self, victor_corpus, amount):
		"""
		Generates substitutions for the target words of a corpus in VICTOR format.
	
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: A dictionary that assigns target complex words to sets of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		"""
		#Get candidate->pos map:
		tagged_sents = self.getParsedSentences(victor_corpus)

		#Get initial set of substitutions:
		substitutions = self.getInitialSet(victor_corpus, tagged_sents, amount)
		return substitutions
		
	def getParsedSentences(self, victor_corpus):
		lexf = open(victor_corpus)
		sents = []
		for line in lexf:
			data = line.strip().split('\t')
			sent = data[0].strip().split(' ')
			sents.append(sent)
		lexf.close()
		
		tagged_sents = self.tagger.tag_sents(sents)
		return tagged_sents

	def getInitialSet(self, victor_corpus, tsents, amount):
		lexf = open(victor_corpus)
		data = []
		for line in lexf:
			d = line.strip().split('\t')
			data.append(d)
		lexf.close()
		
		trgs = []
		trgsc = []
		trgsstems = []
		trgslemmas = []
		trgscstems = []
		trgsclemmas = []
		for i in range(0, len(data)):
			d = data[i]
			tags = tsents[i]
			target = d[1].strip().lower()
			head = int(d[2].strip())
			tag = self.getClass(tags[head][1])
			targetc = self.nc.correct(target)
			trgs.append(target)
			trgsc.append(targetc)
		trgslemmas = self.lemmatizeWords(trgs)
		trgsclemmas = self.lemmatizeWords(trgsc)
		trgsstems = self.stemWords(trgs)
		trgscstems = self.stemWords(trgsc)
		trgmap = {}
		for i in range(0, len(trgslemmas)):
			target = data[i][1].strip().lower()
			head = int(data[i][2].strip())
			tag = self.getClass(tsents[i][head][1])
			lemma = trgslemmas[i]
			stem = trgsstems[i]
			trgmap[target] = (lemma, stem)
	
		subs = []
		cands = set([])
		for i in range(0, len(data)):
			d = data[i]

			t = trgs[i]
			tstem = trgsstems[i]
			tlemma = trgslemmas[i] 
			tc = trgsc[i]
			tcstem = trgscstems[i]
			tclemma = trgsclemmas[i]

			tags = tsents[i]
			head = int(d[2].strip())
			tag = tags[head][1]

			word = t+'|||'+self.getClass(tag)
			wordc = tc+'|||'+self.getClass(tag)

			most_sim = []
			try:
				most_sim = self.model.most_similar(positive=[word], topn=50)
			except KeyError:
				try:
					most_sim = self.model.most_similar(positive=[wordc], topn=50)
				except KeyError:
					most_sim = []

			subs.append([word[0] for word in most_sim])
			
		subsr = subs
		subs = []
		for l in subsr:
			lr = []
			for inst in l:
				cand = inst.split('|||')[0].strip()
				encc = None
				try:
					encc = cand.encode('ascii')
				except Exception:
					encc = None
				if encc:
					cands.add(cand)
					lr.append(inst)
			subs.append(lr)
			
		cands = list(cands)
		candslemmas = self.lemmatizeWords(cands)
		candsstems = self.stemWords(cands)
		candmap = {}
		for i in range(0, len(cands)):
			cand = cands[i]
			lemma = candslemmas[i]
			stem = candsstems[i]
			candmap[cand] = (lemma, stem)
		
		subs_filtered = self.filterSubs(data, tsents, subs, candmap, trgs, trgsc, trgsstems, trgscstems, trgslemmas, trgsclemmas)
		
		final_cands = {}
		for i in range(0, len(data)):
			target = data[i][1]
			cands = subs_filtered[i][0:min(amount, subs_filtered[i])]
			cands = [str(word.split('|||')[0].strip()) for word in cands]
			if target not in final_cands:
				final_cands[target] = set([])
			final_cands[target].update(set(cands))
		
		return final_cands
		
	def lemmatizeWords(self, words):
		result = []
		for word in words:
			result.append(self.lemmatizer.lemmatize(word))
		return result
		
	def stemWords(self, words):
		result = []
		for word in words:
			result.append(self.stemmer.stem(word))
		return result
	
	def filterSubs(self, data, tsents, subs, candmap, trgs, trgsc, trgsstems, trgscstems, trgslemmas, trgsclemmas):
		result = []
		for i in range(0, len(data)):
			d = data[i]

			t = trgs[i]
			tstem = trgsstems[i]
			tlemma = trgslemmas[i]
			tc = trgsc[i]
			tcstem = trgscstems[i]
			tclemma = trgsclemmas[i]

			tags = tsents[i]
			head = int(d[2].strip())
			tag = self.getClass(tags[head][1])

			word = t+'|||'+self.getClass(tag)
			wordc = tc+'|||'+self.getClass(tag)

			most_sim = subs[i]
			most_simf = []

			for cand in most_sim:
				candd = cand.split('|||')
				cword = candd[0].strip()
				ctag = candd[1].strip()
				clemma = candmap[cword][0]
				cstem = candmap[cword][1]

				if ctag==tag:
					if clemma!=tlemma and clemma!=tclemma and cstem!=tstem and cstem!=tcstem:
						if cword not in t and cword not in tc and t not in cword and tc not in cword:
							most_simf.append(cand)
			
			class_filtered = []
			for cand in most_simf:
				candd = cand.split('|||')
				cword = candd[0].strip()
				ctag = candd[1].strip()
				clemma = candmap[cword][0]
				cstem = candmap[cword][1]

				if tag=='V':
					if (t.endswith('ing') or tc.endswith('ing')) and cword.endswith('ing'):
						class_filtered.append(cand)
					elif (t.endswith('d') or tc.endswith('d')) and cword.endswith('d'):
						class_filtered.append(cand)
				else:
					class_filtered.append(cand)

			result.append(most_simf)
		return result
		
	def getClass(self, tag):
		result = None
		if tag.startswith('N'):
			result = 'N'
		elif tag.startswith('V'):
			result = 'V'
		elif tag.startswith('RB'):
			result = 'A'
		elif tag.startswith('J'):
			result = 'J'
		elif tag.startswith('W'):
			result = 'W'
		elif tag.startswith('PRP'):
			result = 'P'
		else:
			result = tag.strip()
		return result

class GlavasGenerator:

	def __init__(self, w2vmodel):
		"""
		Creates a GlavasGenerator instance.
	
		@param w2vmodel: Binary parsed word vector model.
		For more information on how to produce the model, please refer to the LEXenstein Manual.
		"""
		self.lemmatizer = WordNetLemmatizer()
		self.stemmer = PorterStemmer()
		self.model = gensim.models.KeyedVectors.load_word2vec_format(w2vmodel, binary=True)

	def getSubstitutions(self, victor_corpus, amount):
		"""
		Generates substitutions for the target words of a corpus in VICTOR format.
	
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: A dictionary that assigns target complex words to sets of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		"""

		#Get initial set of substitutions:
		substitutions = self.getInitialSet(victor_corpus, amount)
		return substitutions

	def getInitialSet(self, victor_corpus, amount):
		lexf = open(victor_corpus)
		data = []
		for line in lexf:
			d = line.strip().split('\t')
			data.append(d)
		lexf.close()
		
		trgs = []
		trgsstems = []
		trgslemmas = []
		for i in range(0, len(data)):
			d = data[i]
			target = d[1].strip().lower()
			head = int(d[2].strip())
			trgs.append(target)
		trgslemmas = self.lemmatizeWords(trgs)
		trgsstems = self.stemWords(trgs)
		
		trgmap = {}
		for i in range(0, len(trgslemmas)):
			target = data[i][1].strip().lower()
			head = int(data[i][2].strip())
			lemma = trgslemmas[i]
			stem = trgsstems[i]
			trgmap[target] = (lemma, stem)
	
		subs = []
		cands = set([])
		for i in range(0, len(data)):
			d = data[i]

			t = trgs[i]
			tstem = trgsstems[i]
			tlemma = trgslemmas[i]

			word = t

			most_sim = []
			try:
				most_sim = self.model.most_similar(positive=[word], topn=50)
			except KeyError:
				most_sim = []

			subs.append([word[0] for word in most_sim])
			
		subsr = subs
		subs = []
		for l in subsr:
			lr = []
			for inst in l:
				cand = inst.split('|||')[0].strip()
				encc = None
				try:
					encc = cand.encode('ascii')
				except Exception:
					encc = None
				if encc:
					cands.add(cand)
					lr.append(inst)
			subs.append(lr)
			
		cands = list(cands)
		candslemmas = self.lemmatizeWords(cands)
		candsstems = self.stemWords(cands)
		candmap = {}
		for i in range(0, len(cands)):
			cand = cands[i]
			lemma = candslemmas[i]
			stem = candsstems[i]
			candmap[cand] = (lemma, stem)
		
		subs_filtered = self.filterSubs(data, subs, candmap, trgs, trgsstems, trgslemmas)
		
		final_cands = {}
		for i in range(0, len(data)):
			target = data[i][1]
			cands = subs_filtered[i][0:min(amount, subs_filtered[i])]
			cands = [str(word.split('|||')[0].strip()) for word in cands]
			if target not in final_cands:
				final_cands[target] = set([])
			final_cands[target].update(set(cands))
		
		return final_cands
		
	def lemmatizeWords(self, words):
		result = []
		for word in words:
			result.append(self.lemmatizer.lemmatize(word))
		return result
		
	def stemWords(self, words):
		result = []
		for word in words:
			result.append(self.stemmer.stem(word))
		return result
	
	def filterSubs(self, data, subs, candmap, trgs, trgsstems, trgslemmas):
		result = []
		for i in range(0, len(data)):
			d = data[i]

			t = trgs[i]
			tstem = trgsstems[i]
			tlemma = trgslemmas[i]

			word = t

			most_sim = subs[i]
			most_simf = []

			for cand in most_sim:
				cword = cand
				clemma = candmap[cword][0]
				cstem = candmap[cword][1]

				if clemma!=tlemma and cstem!=tstem:
					most_simf.append(cand)

			result.append(most_simf)
		return result

class KauchakGenerator:

	def __init__(self, mat, parallel_pos_file, alignments_file, stop_words, nc):
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
		"""
		self.mat = mat
		self.parallel_pos_file = parallel_pos_file
		self.alignments_file = alignments_file
		self.stop_words = set([word.strip() for word in open(stop_words)])
		self.nc = nc

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
				leftp = leftraw.split('|||')[1].strip().lower()
				leftw = leftraw.split('|||')[0].strip()
				rightraw = target[right].strip()
				rightp = rightraw.split('|||')[1].strip().lower()
				rightw = rightraw.split('|||')[0].strip()

				if len(leftw)>0 and len(rightw)>0 and leftp!='nnp' and rightp!='nnp' and rightp==leftp and leftw not in self.stop_words and rightw not in self.stop_words and leftw!=rightw:
						if leftw in substitutions_initial:
							if leftp in substitutions_initial[leftw]:
								substitutions_initial[leftw][leftp].add(rightw)
							else:
								substitutions_initial[leftw][leftp] = set([rightw])
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
			if target in result:
				result[target].add(postarget)
			else:
				result[target] = set([postarget])
		lex.close()
		return result

	def getInflectedSet(self, result):
		final_substitutions = {}

		#Get inflections:
		allkeys = sorted(list(result.keys()))

		singulars = {}
		plurals = {}
		verbs = {}

		singularsk = {}
		pluralsk = {}
		verbsk = {}

		for i in range(0, len(allkeys)):
			key = allkeys[i]
			leftw = key

			for leftp in result[leftw]:
				if leftp.startswith('n'):
					if leftp=='nns':
						pluralsk[leftw] = set([])
						for subst in result[key][leftp]:
							plurals[subst] = set([])
					else:
						singularsk[leftw] = set([])
						for subst in result[key][leftp]:
							singulars[subst] = set([])
				elif leftp.startswith('v'):
					verbsk[leftw] = {}
					for subst in result[key][leftp]:
						verbs[subst] = {}

		#------------------------------------------------------------------------------------------------

		#Generate keys input:
		singkeys = sorted(list(singularsk.keys()))
		plurkeys = sorted(list(pluralsk.keys()))
		verbkeys = sorted(list(verbsk.keys()))

		#Get stems:
		singstems, plurstems, verbstems = self.getStems(singkeys, plurkeys, verbkeys)

		#Get plurals:
		singres = self.getPlurals(singstems)

		#Get singulars:
		plurres = self.getSingulars(plurstems)

		#Get verb inflections:
		verbres1, verbres2, verbres3, verbres4, verbres5 = self.getInflections(verbstems)

		#Add information to dictionaries:
		for i in range(0, len(singkeys)):
			k = singkeys[i]
			singre = singres[i]
			singularsk[k] = singre
		for i in range(0, len(plurkeys)):
			k = plurkeys[i]
			plurre = plurres[i]
			pluralsk[k] = plurre
		for i in range(0, len(verbkeys)):
			k = verbkeys[i]
			verbre1 = verbres1[i]
			verbre2 = verbres2[i]
			verbre3 = verbres3[i]
			verbre4 = verbres4[i]
			verbre5 = verbres5[i]
			verbsk[k] = {'PAST_PERFECT_PARTICIPLE': verbre1, 'PAST_PARTICIPLE': verbre2, 'PRESENT_PARTICIPLE': verbre3, 'PRESENT': verbre4, 'PAST': verbre5}

		#------------------------------------------------------------------------------------------------

		#Generate substs input:
		singkeys = sorted(list(singulars.keys()))
		plurkeys = sorted(list(plurals.keys()))
		verbkeys = sorted(list(verbs.keys()))

		#Get stems:
		singstems, plurstems, verbstems = self.getStems(singkeys, plurkeys, verbkeys)

		#Get plurals:
		singres = self.getPlurals(singstems)

		#Get singulars:
		plurres = self.getSingulars(plurstems)

		#Get verb inflections:
		verbres1, verbres2, verbres3, verbres4, verbres5 = self.getInflections(verbstems)

		#Add information to dictionaries:
		for i in range(0, len(singkeys)):
			k = singkeys[i]
			singre = singres[i]
			singulars[k] = singre
		for i in range(0, len(plurkeys)):
			k = plurkeys[i]
			plurre = plurres[i]
			plurals[k] = plurre
		for i in range(0, len(verbkeys)):
			k = verbkeys[i]
			verbre1 = verbres1[i]
			verbre2 = verbres2[i]
			verbre3 = verbres3[i]
			verbre4 = verbres4[i]
			verbre5 = verbres5[i]
			verbs[k] = {'PAST_PERFECT_PARTICIPLE': verbre1, 'PAST_PARTICIPLE': verbre2, 'PRESENT_PARTICIPLE': verbre3, 'PRESENT': verbre4, 'PAST': verbre5}

		#------------------------------------------------------------------------------------------------

		#Generate final substitution list:
		for i in range(0, len(allkeys)):
			key = allkeys[i]
			leftw = key
			for leftp in result[leftw]:			

				#Add final version to candidates:
				if leftw not in final_substitutions:
					final_substitutions[leftw] = result[key][leftp]
				else:
					final_substitutions[leftw] = final_substitutions[leftw].union(result[key][leftp])
				#If left is a noun:
				if leftp.startswith('n'):
					#If it is a plural:
					if leftp=='nns':
						plurl = pluralsk[leftw]
						newcands = set([])
						for candidate in result[key][leftp]:
							candplurl = plurals[candidate]
							newcands.add(candplurl)
						if plurl not in final_substitutions:
							final_substitutions[plurl] = newcands
						else:
							final_substitutions[plurl] = final_substitutions[plurl].union(newcands)
					#If it is singular:
					else:
						singl = singularsk[leftw]
						newcands = set([])
						for candidate in result[key][leftp]:
							candsingl = singulars[candidate]
							newcands.add(candsingl)
						if singl not in final_substitutions:
							final_substitutions[singl] = newcands
						else:
							final_substitutions[singl] = final_substitutions[singl].union(newcands)
				#If left is a verb:
				elif leftp.startswith('v'):
					for verb_tense in ['PAST_PERFECT_PARTICIPLE', 'PAST_PARTICIPLE', 'PRESENT_PARTICIPLE', 'PRESENT', 'PAST']:
						tensedl = verbsk[leftw][verb_tense]
						newcands = set([])
						for candidate in result[key][leftp]:
							candtensedl = verbs[candidate][verb_tense]
							newcands.add(candtensedl)
						if tensedl not in final_substitutions:
							final_substitutions[tensedl] = newcands
						else:
							final_substitutions[tensedl] = final_substitutions[tensedl].union(newcands)
		return final_substitutions
		
	def getInflections(self, verbstems):
		data1 = self.mat.conjugateVerbs(verbstems, 'PAST_PERFECT_PARTICIPLE', 'FIRST_PERSON_SINGULAR')
		data2 = self.mat.conjugateVerbs(verbstems, 'PAST_PARTICIPLE', 'FIRST_PERSON_SINGULAR')
		data3 = self.mat.conjugateVerbs(verbstems, 'PRESENT_PARTICIPLE', 'FIRST_PERSON_SINGULAR')
		data4 = self.mat.conjugateVerbs(verbstems, 'PRESENT', 'FIRST_PERSON_SINGULAR')
		data5 = self.mat.conjugateVerbs(verbstems, 'PAST', 'FIRST_PERSON_SINGULAR')
		return self.correctWords(data1), self.correctWords(data2), self.correctWords(data3), self.correctWords(data4), self.correctWords(data5)

	def getSingulars(self, plurstems):
		data = self.mat.inflectNouns(plurstems, 'singular')
		return self.correctWords(data)
		
	def getPlurals(self, singstems):
		data = self.mat.inflectNouns(singstems, 'plural')
		return self.correctWords(data)

	def getStems(self, sings, plurs, verbs):
		data = self.mat.lemmatizeWords(sings+plurs+verbs)
		rsings = []
		rplurs = []
		rverbs = []
		c = -1
		for sing in sings:
			c += 1
			if len(data[c])>0:
				rsings.append(data[c])
			else:
				rsings.append(sing)
		for plur in plurs:
			c += 1
			if len(data[c])>0:
				rplurs.append(data[c])
			else:
				rplurs.append(plur)
		for verb in verbs:
			c += 1
			if len(data[c])>0:
				rverbs.append(data[c])
			else:
				rverbs.append(verb)
		return self.correctWords(rsings), self.correctWords(rplurs), self.correctWords(rverbs)
		
	def correctWords(self, words):
		result = []
		for word in words:
			result.append(self.nc.correct(word))
		return result

class YamamotoGenerator:

	def __init__(self, mat, dictionary_key, nc):
		"""
		Creates a YamamotoGenerator instance.
	
		@param mat: MorphAdornerToolkit object.
		@param dictionary_key: Key for the Merriam Dictionary.
		@param nc: NorvigCorrector object.
		For more information on how to get the key for free, please refer to the LEXenstein Manual
		"""
		self.mat = mat
		self.dictionary_key = dictionary_key
		self.nc = nc

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

	def getInflectedSet(self, result):
		final_substitutions = {}

		#Get inflections:
		allkeys = sorted(list(result.keys()))

		singulars = {}
		plurals = {}
		verbs = {}

		singularsk = {}
		pluralsk = {}
		verbsk = {}

		for i in range(0, len(allkeys)):
			key = allkeys[i]
			leftw = key

			for leftp in result[leftw]:
				if leftp.startswith('n'):
					if leftp=='nns':
						pluralsk[leftw] = set([])
						for subst in result[key][leftp]:
							plurals[subst] = set([])
					else:
						singularsk[leftw] = set([])
						for subst in result[key][leftp]:
							singulars[subst] = set([])
				elif leftp.startswith('v'):
					verbsk[leftw] = {}
					for subst in result[key][leftp]:
						verbs[subst] = {}

		#------------------------------------------------------------------------------------------------

		#Generate keys input:
		singkeys = sorted(list(singularsk.keys()))
		plurkeys = sorted(list(pluralsk.keys()))
		verbkeys = sorted(list(verbsk.keys()))

		#Get stems:
		singstems, plurstems, verbstems = self.getStems(singkeys, plurkeys, verbkeys)

		#Get plurals:
		singres = self.getPlurals(singstems)

		#Get singulars:
		plurres = self.getSingulars(plurstems)

		#Get verb inflections:
		verbres1, verbres2, verbres3, verbres4, verbres5 = self.getInflections(verbstems)

		#Add information to dictionaries:
		for i in range(0, len(singkeys)):
			k = singkeys[i]
			singre = singres[i]
			singularsk[k] = singre
		for i in range(0, len(plurkeys)):
			k = plurkeys[i]
			plurre = plurres[i]
			pluralsk[k] = plurre
		for i in range(0, len(verbkeys)):
			k = verbkeys[i]
			verbre1 = verbres1[i]
			verbre2 = verbres2[i]
			verbre3 = verbres3[i]
			verbre4 = verbres4[i]
			verbre5 = verbres5[i]
			verbsk[k] = {'PAST_PERFECT_PARTICIPLE': verbre1, 'PAST_PARTICIPLE': verbre2, 'PRESENT_PARTICIPLE': verbre3, 'PRESENT': verbre4, 'PAST': verbre5}

		#------------------------------------------------------------------------------------------------

		#Generate substs input:
		singkeys = sorted(list(singulars.keys()))
		plurkeys = sorted(list(plurals.keys()))
		verbkeys = sorted(list(verbs.keys()))

		#Get stems:
		singstems, plurstems, verbstems = self.getStems(singkeys, plurkeys, verbkeys)

		#Get plurals:
		singres = self.getPlurals(singstems)

		#Get singulars:
		plurres = self.getSingulars(plurstems)

		#Get verb inflections:
		verbres1, verbres2, verbres3, verbres4, verbres5 = self.getInflections(verbstems)

		#Add information to dictionaries:
		for i in range(0, len(singkeys)):
			k = singkeys[i]
			singre = singres[i]
			singulars[k] = singre
		for i in range(0, len(plurkeys)):
			k = plurkeys[i]
			plurre = plurres[i]
			plurals[k] = plurre
		for i in range(0, len(verbkeys)):
			k = verbkeys[i]
			verbre1 = verbres1[i]
			verbre2 = verbres2[i]
			verbre3 = verbres3[i]
			verbre4 = verbres4[i]
			verbre5 = verbres5[i]
			verbs[k] = {'PAST_PERFECT_PARTICIPLE': verbre1, 'PAST_PARTICIPLE': verbre2, 'PRESENT_PARTICIPLE': verbre3, 'PRESENT': verbre4, 'PAST': verbre5}

		#------------------------------------------------------------------------------------------------

		#Generate final substitution list:
		for i in range(0, len(allkeys)):
			key = allkeys[i]
			leftw = key
			for leftp in result[leftw]:			

				#Add final version to candidates:
				if leftw not in final_substitutions:
					final_substitutions[leftw] = result[key][leftp]
				else:
					final_substitutions[leftw] = final_substitutions[leftw].union(result[key][leftp])
				#If left is a noun:
				if leftp.startswith('n'):
					#If it is a plural:
					if leftp=='nns':
						plurl = pluralsk[leftw]
						newcands = set([])
						for candidate in result[key][leftp]:
							candplurl = plurals[candidate]
							newcands.add(candplurl)
						if plurl not in final_substitutions:
							final_substitutions[plurl] = newcands
						else:
							final_substitutions[plurl] = final_substitutions[plurl].union(newcands)
					#If it is singular:
					else:
						singl = singularsk[leftw]
						newcands = set([])
						for candidate in result[key][leftp]:
							candsingl = singulars[candidate]
							newcands.add(candsingl)
						if singl not in final_substitutions:
							final_substitutions[singl] = newcands
						else:
							final_substitutions[singl] = final_substitutions[singl].union(newcands)
				#If left is a verb:
				elif leftp.startswith('v'):
					for verb_tense in ['PAST_PERFECT_PARTICIPLE', 'PAST_PARTICIPLE', 'PRESENT_PARTICIPLE', 'PRESENT', 'PAST']:
						tensedl = verbsk[leftw][verb_tense]
						newcands = set([])
						for candidate in result[key][leftp]:
							candtensedl = verbs[candidate][verb_tense]
							newcands.add(candtensedl)
						if tensedl not in final_substitutions:
							final_substitutions[tensedl] = newcands
						else:
							final_substitutions[tensedl] = final_substitutions[tensedl].union(newcands)
		return final_substitutions
		
	def getInflections(self, verbstems):
		data1 = self.mat.conjugateVerbs(verbstems, 'PAST_PERFECT_PARTICIPLE', 'FIRST_PERSON_SINGULAR')
		data2 = self.mat.conjugateVerbs(verbstems, 'PAST_PARTICIPLE', 'FIRST_PERSON_SINGULAR')
		data3 = self.mat.conjugateVerbs(verbstems, 'PRESENT_PARTICIPLE', 'FIRST_PERSON_SINGULAR')
		data4 = self.mat.conjugateVerbs(verbstems, 'PRESENT', 'FIRST_PERSON_SINGULAR')
		data5 = self.mat.conjugateVerbs(verbstems, 'PAST', 'FIRST_PERSON_SINGULAR')
		return self.correctWords(data1), self.correctWords(data2), self.correctWords(data3), self.correctWords(data4), self.correctWords(data5)

	def getSingulars(self, plurstems):
		data = self.mat.inflectNouns(plurstems, 'singular')
		return self.correctWords(data)
		
	def getPlurals(self, singstems):
		data = self.mat.inflectNouns(singstems, 'plural')
		return self.correctWords(data)

	def getStems(self, sings, plurs, verbs):
		data = self.mat.lemmatizeWords(sings+plurs+verbs)
		rsings = []
		rplurs = []
		rverbs = []
		c = -1
		for sing in sings:
			c += 1
			if len(data[c])>0:
				rsings.append(data[c])
			else:
				rsings.append(sing)
		for plur in plurs:
			c += 1
			if len(data[c])>0:
				rplurs.append(data[c])
			else:
				rplurs.append(plur)
		for verb in verbs:
			c += 1
			if len(data[c])>0:
				rverbs.append(data[c])
			else:
				rverbs.append(verb)
		return self.correctWords(rsings), self.correctWords(rplurs), self.correctWords(rverbs)

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
				node_pos = entry.find('fl')
				if node_pos != None:
					node_pos = node_pos.text.strip()[0].lower()
					if node_pos not in cands:
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
			for pos in cands:
				if target in cands[pos]:
					cands[pos].remove(target)
			if len(cands.keys())>0:
				substitutions_initial[target] = cands
		lex.close()
		return substitutions_initial

	def correctWords(self, words):
		result = []
		for word in words:
			result.append(self.nc.correct(word))
		return result

class MerriamGenerator:

	def __init__(self, mat, thesaurus_key, nc):
		"""
		Creates a MerriamGenerator instance.
	
		@param mat: MorphAdornerToolkit object.
		@param thesaurus_key: Key for the Merriam Thesaurus.
		For more information on how to get the key for free, please refer to the LEXenstein Manual
		@param nc: NorvigCorrector object.
		"""
		self.mat = mat
		self.thesaurus_key = thesaurus_key
		self.nc = nc

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

	def getInflectedSet(self, result):
		final_substitutions = {}

		#Get inflections:
		allkeys = sorted(list(result.keys()))

		singulars = {}
		plurals = {}
		verbs = {}

		singularsk = {}
		pluralsk = {}
		verbsk = {}

		for i in range(0, len(allkeys)):
			key = allkeys[i]
			leftw = key

			for leftp in result[leftw]:
				if leftp.startswith('n'):
					if leftp=='nns':
						pluralsk[leftw] = set([])
						for subst in result[key][leftp]:
							plurals[subst] = set([])
					else:
						singularsk[leftw] = set([])
						for subst in result[key][leftp]:
							singulars[subst] = set([])
				elif leftp.startswith('v'):
					verbsk[leftw] = {}
					for subst in result[key][leftp]:
						verbs[subst] = {}

		#------------------------------------------------------------------------------------------------

		#Generate keys input:
		singkeys = sorted(list(singularsk.keys()))
		plurkeys = sorted(list(pluralsk.keys()))
		verbkeys = sorted(list(verbsk.keys()))

		#Get stems:
		singstems, plurstems, verbstems = self.getStems(singkeys, plurkeys, verbkeys)

		#Get plurals:
		singres = self.getPlurals(singstems)

		#Get singulars:
		plurres = self.getSingulars(plurstems)

		#Get verb inflections:
		verbres1, verbres2, verbres3, verbres4, verbres5 = self.getInflections(verbstems)

		#Add information to dictionaries:
		for i in range(0, len(singkeys)):
			k = singkeys[i]
			singre = singres[i]
			singularsk[k] = singre
		for i in range(0, len(plurkeys)):
			k = plurkeys[i]
			plurre = plurres[i]
			pluralsk[k] = plurre
		for i in range(0, len(verbkeys)):
			k = verbkeys[i]
			verbre1 = verbres1[i]
			verbre2 = verbres2[i]
			verbre3 = verbres3[i]
			verbre4 = verbres4[i]
			verbre5 = verbres5[i]
			verbsk[k] = {'PAST_PERFECT_PARTICIPLE': verbre1, 'PAST_PARTICIPLE': verbre2, 'PRESENT_PARTICIPLE': verbre3, 'PRESENT': verbre4, 'PAST': verbre5}

		#------------------------------------------------------------------------------------------------

		#Generate substs input:
		singkeys = sorted(list(singulars.keys()))
		plurkeys = sorted(list(plurals.keys()))
		verbkeys = sorted(list(verbs.keys()))

		#Get stems:
		singstems, plurstems, verbstems = self.getStems(singkeys, plurkeys, verbkeys)

		#Get plurals:
		singres = self.getPlurals(singstems)

		#Get singulars:
		plurres = self.getSingulars(plurstems)

		#Get verb inflections:
		verbres1, verbres2, verbres3, verbres4, verbres5 = self.getInflections(verbstems)

		#Add information to dictionaries:
		for i in range(0, len(singkeys)):
			k = singkeys[i]
			singre = singres[i]
			singulars[k] = singre
		for i in range(0, len(plurkeys)):
			k = plurkeys[i]
			plurre = plurres[i]
			plurals[k] = plurre
		for i in range(0, len(verbkeys)):
			k = verbkeys[i]
			verbre1 = verbres1[i]
			verbre2 = verbres2[i]
			verbre3 = verbres3[i]
			verbre4 = verbres4[i]
			verbre5 = verbres5[i]
			verbs[k] = {'PAST_PERFECT_PARTICIPLE': verbre1, 'PAST_PARTICIPLE': verbre2, 'PRESENT_PARTICIPLE': verbre3, 'PRESENT': verbre4, 'PAST': verbre5}

		#------------------------------------------------------------------------------------------------

		#Generate final substitution list:
		for i in range(0, len(allkeys)):
			key = allkeys[i]
			leftw = key
			for leftp in result[leftw]:			

				#Add final version to candidates:
				if leftw not in final_substitutions:
					final_substitutions[leftw] = result[key][leftp]
				else:
					final_substitutions[leftw] = final_substitutions[leftw].union(result[key][leftp])
				#If left is a noun:
				if leftp.startswith('n'):
					#If it is a plural:
					if leftp=='nns':
						plurl = pluralsk[leftw]
						newcands = set([])
						for candidate in result[key][leftp]:
							candplurl = plurals[candidate]
							newcands.add(candplurl)
						if plurl not in final_substitutions:
							final_substitutions[plurl] = newcands
						else:
							final_substitutions[plurl] = final_substitutions[plurl].union(newcands)
					#If it is singular:
					else:
						singl = singularsk[leftw]
						newcands = set([])
						for candidate in result[key][leftp]:
							candsingl = singulars[candidate]
							newcands.add(candsingl)
						if singl not in final_substitutions:
							final_substitutions[singl] = newcands
						else:
							final_substitutions[singl] = final_substitutions[singl].union(newcands)
				#If left is a verb:
				elif leftp.startswith('v'):
					for verb_tense in ['PAST_PERFECT_PARTICIPLE', 'PAST_PARTICIPLE', 'PRESENT_PARTICIPLE', 'PRESENT', 'PAST']:
						tensedl = verbsk[leftw][verb_tense]
						newcands = set([])
						for candidate in result[key][leftp]:
							candtensedl = verbs[candidate][verb_tense]
							newcands.add(candtensedl)
						if tensedl not in final_substitutions:
							final_substitutions[tensedl] = newcands
						else:
							final_substitutions[tensedl] = final_substitutions[tensedl].union(newcands)
		return final_substitutions
		
	def getInflections(self, verbstems):
		data1 = self.mat.conjugateVerbs(verbstems, 'PAST_PERFECT_PARTICIPLE', 'FIRST_PERSON_SINGULAR')
		data2 = self.mat.conjugateVerbs(verbstems, 'PAST_PARTICIPLE', 'FIRST_PERSON_SINGULAR')
		data3 = self.mat.conjugateVerbs(verbstems, 'PRESENT_PARTICIPLE', 'FIRST_PERSON_SINGULAR')
		data4 = self.mat.conjugateVerbs(verbstems, 'PRESENT', 'FIRST_PERSON_SINGULAR')
		data5 = self.mat.conjugateVerbs(verbstems, 'PAST', 'FIRST_PERSON_SINGULAR')
		return self.correctWords(data1), self.correctWords(data2), self.correctWords(data3), self.correctWords(data4), self.correctWords(data5)

	def getSingulars(self, plurstems):
		data = self.mat.inflectNouns(plurstems, 'singular')
		return self.correctWords(data)
		
	def getPlurals(self, singstems):
		data = self.mat.inflectNouns(singstems, 'plural')
		return self.correctWords(data)

	def getStems(self, sings, plurs, verbs):
		data = self.mat.lemmatizeWords(sings+plurs+verbs)
		rsings = []
		rplurs = []
		rverbs = []
		c = -1
		for sing in sings:
			c += 1
			if len(data[c])>0:
				rsings.append(data[c])
			else:
				rsings.append(sing)
		for plur in plurs:
			c += 1
			if len(data[c])>0:
				rplurs.append(data[c])
			else:
				rplurs.append(plur)
		for verb in verbs:
			c += 1
			if len(data[c])>0:
				rverbs.append(data[c])
			else:
				rverbs.append(verb)
		return self.correctWords(rsings), self.correctWords(rplurs), self.correctWords(rverbs)

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
					node_pos = root_node.find('fl')
					if node_pos != None:
						node_pos = node_pos.text.strip()[0].lower()
						if node_pos not in cands:
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
							try:
								test = codecs.ascii_encode(synonym)
								cands[node_pos].add(synonym)
							except UnicodeEncodeError:
								cands = cands
			for pos in cands:
				if target in cands[pos]:
					cands[pos].remove(target)
			if len(cands.keys())>0:
				substitutions_initial[target] = cands
		lex.close()
		return substitutions_initial

	def correctWords(self, words):
		result = []
		for word in words:
			result.append(self.nc.correct(word))
		return result

#Class for the Wordnet Generator
class WordnetGenerator:

	def __init__(self, mat, nc, pos_model, stanford_tagger, java_path):
		"""
		Creates a WordnetGenerator instance.
	
		@param mat: MorphAdornerToolkit object.
		@param nc: NorvigCorrector object.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		"""
		self.mat = mat
		self.nc = nc
		os.environ['JAVAHOME'] = java_path
		self.tagger = StanfordPOSTagger(pos_model, stanford_tagger)

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
		toPE = []
		toPR = []
		toComparative = []
		toSuperlative = []
		toOriginal = []
		
		#Fill lists:
		for target in subs:
			targets.append(target)
			for pos in subs[target]:
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
				elif pos == 'VBP':
					toPE.extend(cands)
				elif pos == 'VBZ':
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
		toPEL = self.correctWords(self.mat.lemmatizeWords(toPE))
		toPRL = self.correctWords(self.mat.lemmatizeWords(toPR))
		toComparativeL = self.correctWords(self.mat.lemmatizeWords(toComparative))
		toSuperlativeL = self.correctWords(self.mat.lemmatizeWords(toSuperlative))
		
		#Inflect nouns:
		singulars = self.correctWords(self.mat.inflectNouns(toSingularL, 'singular'))
		plurals = self.correctWords(self.mat.inflectNouns(toPluralL, 'plural'))
		
		#Inflect verbs:
		papepas = self.correctWords(self.mat.conjugateVerbs(toPAPEPAL, 'PAST_PERFECT_PARTICIPLE', 'FIRST_PERSON_SINGULAR'))
		pas = self.correctWords(self.mat.conjugateVerbs(toPAL, 'PAST', 'FIRST_PERSON_SINGULAR'))
		prpas = self.correctWords(self.mat.conjugateVerbs(toPRPAL, 'PRESENT_PARTICIPLE', 'FIRST_PERSON_SINGULAR'))
		papas = self.correctWords(self.mat.conjugateVerbs(toPAPAL, 'PAST_PARTICIPLE', 'FIRST_PERSON_SINGULAR'))
		pes = self.correctWords(self.mat.conjugateVerbs(toPEL, 'PERFECT', 'FIRST_PERSON_SINGULAR'))
		prs = self.correctWords(self.mat.conjugateVerbs(toPRL, 'PRESENT', 'THIRD_PERSON_SINGULAR'))
		
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
		peM = {}
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
		for i in range(0, len(toPE)):
			stemM[toPE[i]] = toPEL[i]
			peM[toPE[i]] = pes[i]
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
		for target in subs:
			#Get lemma of target:
			targetL = stemM[target]
			
			#Create instance in final_substitutions:
			final_substitutions[target] = set([])
			
			#Iterate through pos tags of target:
			for pos in subs[target]:
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
				elif pos == 'VBP':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(peM[cand])
				elif pos == 'VBZ':
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
		for target in subs:
			for pos in subs[target]:
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
		for target in subs:
			for pos in subs[target]:
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
		if target not in subs:
			subs[target] = {tag:cands}
		else:
			if tag not in subs[target]:
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

	def __init__(self, mat, complex_vocab, simple_vocab, complex_lm, simple_lm, nc, pos_model, stanford_tagger, java_path):
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
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		"""

		self.complex_vocab = self.getVocab(complex_vocab)
		self.simple_vocab = self.getVocab(simple_vocab)
		self.complex_lm = kenlm.LanguageModel(complex_lm)
		self.simple_lm = kenlm.LanguageModel(simple_lm)
		self.mat = mat
		self.nc = nc
		os.environ['JAVAHOME'] = java_path
		self.tagger = StanfordPOSTagger(pos_model, stanford_tagger)

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

		for key in substitutions_inflected:
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
		toPE = []
		toPR = []
		toComparative = []
		toSuperlative = []
		toOriginal = []
		
		#Fill lists:
		for target in subs:
			targets.append(target)
			for pos in subs[target]:
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
				elif pos == 'VBP':
					toPE.extend(cands)
				elif pos == 'VBZ':
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
		toPEL = self.correctWords(self.mat.lemmatizeWords(toPE))
		toPRL = self.correctWords(self.mat.lemmatizeWords(toPR))
		toComparativeL = self.correctWords(self.mat.lemmatizeWords(toComparative))
		toSuperlativeL = self.correctWords(self.mat.lemmatizeWords(toSuperlative))
		
		#Inflect nouns:
		singulars = self.correctWords(self.mat.inflectNouns(toSingularL, 'singular'))
		plurals = self.correctWords(self.mat.inflectNouns(toPluralL, 'plural'))
		
		#Inflect verbs:
		papepas = self.correctWords(self.mat.conjugateVerbs(toPAPEPAL, 'PAST_PERFECT_PARTICIPLE', 'FIRST_PERSON_SINGULAR'))
		pas = self.correctWords(self.mat.conjugateVerbs(toPAL, 'PAST', 'FIRST_PERSON_SINGULAR'))
		prpas = self.correctWords(self.mat.conjugateVerbs(toPRPAL, 'PRESENT_PARTICIPLE', 'FIRST_PERSON_SINGULAR'))
		papas = self.correctWords(self.mat.conjugateVerbs(toPAPAL, 'PAST_PARTICIPLE', 'FIRST_PERSON_SINGULAR'))
		pes = self.correctWords(self.mat.conjugateVerbs(toPEL, 'PERFECT', 'FIRST_PERSON_SINGULAR'))
		prs = self.correctWords(self.mat.conjugateVerbs(toPRL, 'PRESENT', 'THIRD_PERSON_SINGULAR'))
		
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
		peM = {}
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
		for i in range(0, len(toPE)):
			stemM[toPE[i]] = toPEL[i]
			peM[toPE[i]] = pes[i]
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
		for target in subs:
			#Get lemma of target:
			targetL = stemM[target]
			
			#Create instance in final_substitutions:
			final_substitutions[target] = set([])
			
			#Iterate through pos tags of target:
			for pos in subs[target]:
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
				elif pos == 'VBP':
					for cand in cands:
						if targetL!=stemM[cand]:
							final_cands.add(peM[cand])
				elif pos == 'VBZ':
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
		#C = (clm.score(word)/(slm.score(word))
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
