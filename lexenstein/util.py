import nltk
import pickle
import shelve
import re

def dependencyParseSentences(parser, sentences):
	"""
	Use StanfordParser to parse multiple sentences.
	Takes multiple sentences as a list where each sentence is a list of words.
	Each sentence will be automatically tagged with this StanfordParser instance's tagger.
	If whitespaces exists inside a token, then the token will be treated as separate tokens.
	This method is an adaptation of the code provided by NLTK.

	@param parser: An instance of the nltk.parse.stanford.StanfordParser class.
	@param sentences: Input sentences to parse.
	Each sentence must be a list of tokens.
	@return A list of the dependency links of each sentence.
	Each dependency link is composed by the relation type, the source word, its position in the sentence, the target word, and its position in the sentence.
	"""
	cmd = [
	    'edu.stanford.nlp.parser.lexparser.LexicalizedParser',
	    '-model', parser.model_path,
	    '-sentences', 'newline',
	    '-outputFormat', 'typedDependencies',
	    '-tokenized',
	    '-escaper', 'edu.stanford.nlp.process.PTBEscapingProcessor',
	]

	output=parser._execute(cmd, '\n'.join(' '.join(sentence) for sentence in sentences), False)

	depexp = re.compile("([^\\(]+)\\(([^\\,]+)\\,\s([^\\)]+)\\)")

	res = []
	cur_lines = []
	for line in output.splitlines(False):
	    if line == '':
			res.append(cur_lines)
			cur_lines = []
	    else:
			depdata = re.findall(depexp, line)
			if len(depdata)>0:
				link = depdata[0]
				subjecth = link[1].rfind('-')
				objecth = link[2].rfind('-')
				subjectindex = link[1][subjecth+1:len(link[1])]
				if subjectindex.endswith(r"'"):
					subjectindex = subjectindex[0:len(subjectindex)-1]
				objectindex = link[2][objecth+1:len(link[2])]
				if objectindex.endswith(r"'"):
					objectindex = objectindex[0:len(objectindex)-1]
				clean_link = (link[0], link[1][0:subjecth], subjectindex, link[2][0:objecth], objectindex)
				try:
					a = int(subjectindex)
					b = int(objectindex)
					cur_lines.append(clean_link)
				except Exception:
					pass
	return res

def getGeneralisedPOS(tag):
	"""
	Returns a generalised version of a POS tag in Treebank format.

	@param tag: POS tag in Treebank format.
	@return A generalised POS tag.
	"""
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
	
def createTaggedNgramsFile(ngrams_file, tagged_ngrams_file):
	"""
	Creates a tagged version of an annotated n-gram counts file.
	
	@param ngrams_file: File containing POS tag annotated n-gram counts.
	The file must be in the format produced by the "-write" option of SRILM.
	Each word in the corpus used must be in the following format: <word>|||<tag>
	@param tagged_ngrams_file: File with tagged n-gram counts.
	"""
	o = open(tagged_ngrams_file, 'w')
	
	print('Opening input n-gram counts file...')
	c = 0
	f = open(ngrams_file)
	for line in f:
		c += 1
		if c % 1000000 == 0:
			print(str(c) + ' n-grams processed.')
		data = line.strip().split('\t')
		tokens = [t.split('|||') for t in data[0].split(' ')]
		if len(tokens)==2:
			o.write(tokens[0][0] + ' ' + tokens[1][min(1, len(tokens[1])-1)] + '\t' + data[1] + '\n')
			o.write(tokens[0][min(1, len(tokens[0])-1)] + ' ' + tokens[1][0] + '\t' + data[1] + '\n')
		elif len(tokens)==3:
			o.write(tokens[0][0] + ' ' + tokens[1][min(1, len(tokens[1])-1)] + ' ' + tokens[2][min(1, len(tokens[2])-1)] + '\t' + data[1] + '\n')
			o.write(tokens[0][min(1, len(tokens[0])-1)] + ' ' + tokens[1][0] + ' ' + tokens[2][min(1, len(tokens[2])-1)] + '\t' + data[1] + '\n')
			o.write(tokens[0][min(1, len(tokens[0])-1)] + ' ' + tokens[1][min(1, len(tokens[1])-1)] + ' ' + tokens[2][0] + '\t' + data[1] + '\n')
		elif len(tokens)==4:
			o.write(tokens[0][min(1, len(tokens[0])-1)] + ' ' + tokens[1][min(1, len(tokens[1])-1)] + ' ' + tokens[2][0] + ' ' + tokens[3][min(1, len(tokens[3])-1)] + '\t' + data[1] + '\n')
			o.write(tokens[0][min(1, len(tokens[0])-1)] + ' ' + tokens[1][0] + ' ' + tokens[2][min(1, len(tokens[2])-1)] + ' ' + tokens[3][min(1, len(tokens[3])-1)] + '\t' + data[1] + '\n')
		elif len(tokens)==5:
			o.write(tokens[0][min(1, len(tokens[0])-1)] + ' ' + tokens[1][min(1, len(tokens[1])-1)] + ' ' + tokens[2][0] + ' ' + tokens[3][min(1, len(tokens[3])-1)] + ' ' + tokens[4][min(1, len(tokens[4])-1)] + '\t' + data[1] + '\n')
	f.close()
	print('N-grams file read!')
	
	print('Saving model...')
	o.close()
	print('Finished!')

def removeUnkFromNgramsFile(ngrams_file, output):
	"""
	Removes n-grams with "<unk>" tokens from an SRILM n-grams file.
	
	@param ngrams_file: Input n-grams file.
	@param output: Filtered n-grams file.
	"""
	f = open(ngrams_file)
	o = open(output, 'w')
	c = 0
	for line in f:
		c += 1
		if c % 1000000==0:
			print(str(c) + ' tokens filtered.')
		if '<unk>' not in line:
			o.write(line)
	f.close()
	o.close()

def getVocabularyFromDataset(dataset, vocab_file, leftw, rightw, format='victor'):
	"""
	Extracts the vocabulary from a dataset in VICTOR or CWICTOR format.
	This vocabularies can be used along with SRILM in order for smaller n-gram count files to be produced.
	
	@param dataset: Dataset from which to extract the vocabulary.
	@param vocab_file: File in which to save the vocabulary.
	@param leftw: Window to consider from the left of the target word.
	@param rightw: Window to consider from the right of the target word.
	@param format: Format of the dataset.
	Values accepted: victor, cwictor
	"""
	#Obtain vocabulary:
	vocab = set([])
	if format=='victor':
		f = open(dataset)
		for line in f:
			data = line.strip().split('\t')
			sent = data[0].strip().split(' ')
			head = int(data[2].strip())
			for i in range(max(0, head-leftw), head):
				vocab.add(sent[i])
			for i in range(head, min(len(sent), head+rightw+1)):
				vocab.add(sent[i])
			target = data[1].strip()
			vocab.add(target)
			for sub in data[3:len(data)]:
				words = sub.strip().split(':')[1].strip().split(' ')
				for word in words:
					vocab.add(word.strip())
		f.close()
	elif format=='cwictor':
		f = open(dataset)
		for line in f:
			data = line.strip().split('\t')
			sent = data[0].strip().split(' ')
			head = int(data[2].strip())
			for i in range(max(0, head-leftw), head):
				vocab.add(sent[i])
			for i in range(head, min(len(sent), head+rightw+1)):
				vocab.add(sent[i])
			target = data[1].strip()
			vocab.update(sent)
			vocab.add(target)
		f.close()
	
	#Save vocabulary:
	f = open(vocab_file, 'w')
	for word in vocab:
		if len(word.strip())>0:
			f.write(word.strip() + '\n')
	f.close()

def addTranslationProbabilitiesFileToShelve(transprob_file, model_file):
	"""
	Adds a translation probabilities file to an either new, or existing shelve dictionary.
	The shelve file can then be used for the calculation of features.
	To produce the translation probabilities file, first run the following command through fast_align:
	fast_align -i <parallel_data> -v -d -o <transprob_file>
	
	@param transprob_file: File containing translation probabilities.
	@param model_file: Shelve file in which to save the translation probabilities.
	"""
	print('Opening shelve file...')
	d = shelve.open(model_file, protocol=pickle.HIGHEST_PROTOCOL)
	print('Shelve file open!')
	
	print('Reading translation probabilities file...')
	c = 0
	f = open(transprob_file)
	for line in f:
		c += 1
		if c % 1000000 == 0:
			print(str(c) + ' translation probabilities read.')
		data = line.strip().split('\t')
		key = data[0] + '\t' + data[1]
		value = float(data[2])
		if key not in d:
			d[key] = value
		else:
			d[key] += value
	f.close()
	print('Translation probabilities file read!')
	
	print('Saving model...')
	d.close()
	print('Finished!')
	
def addNgramCountsFileToShelve(ngrams_file, model_file):
	"""
	Adds a n-gram counts file to an either new, or existing shelve dictionary.
	The shelve file can then be used for the calculation of several features.
	The file must be in the format produced by the "-write" option of SRILM ngram-count application.
	
	@param ngrams_file: File containing n-gram counts.
	@param model_file: Shelve file in which to save the n-gram counts file.
	"""
	print('Opening shelve file...')
	d = shelve.open(model_file, protocol=pickle.HIGHEST_PROTOCOL)
	print('Shelve file open!')
	
	print('Reading n-grams file...')
	c = 0
	f = open(ngrams_file)
	for line in f:
		c += 1
		if c % 1000000 == 0:
			print(str(c) + ' n-grams read.')
		data = line.strip().split('\t')
		if data[0] not in d:
			d[data[0]] = int(data[1])
		else:
			d[data[0]] += int(data[1])
	f.close()
	print('N-grams file read!')
	
	print('Saving model...')
	d.close()
	print('Finished!')

def createConditionalProbabilityModel(folder, fileids, model, sep='/', encoding='utf8'):
	"""
	Creates an tagging probability model to be used along with the FeatureEstimator object.
	Files of tagged data must contain one sentence per line, and each line must follow the following format:
	<word_1><separator><tag_1> <word_2><separator><tag_2> ... <word_n-1><separator><tag_n-1> <word_n><separator><tag_n>
	
	@param folder: Folder containing files of tagged sentences.
	@param fileids: A list or regular expressions specifying the file names with tagged data in "folder".
	@param model: File in which to save the trained model.
	@param sep: Separator between words and tags in the files with tagged data.
	@param encoding: Encoding of the files with tagged data.
	"""
	print('Reading files...')
	tcr = nltk.corpus.reader.tagged.TaggedCorpusReader(folder, fileids, sep=sep, encoding=encoding)
	
	print('Extracting tagged data...')
	data = tcr.tagged_words()
	
	print('Creating conditional probability maps...')
	cfd_tagwords = nltk.ConditionalFreqDist(data)
	cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)
	
	print('Saving model...')
	pickle.dump(cpd_tagwords, open(model, "wb"))
	print('Finished!')

def fitTranslationProbabilityFileToCorpus(translation_probabilities, corpus, output):
	"""
	Creates a translation probabilities file that has only translations pertaining to the target complex words of a given VICTOR or CWICTOR corpus.
	
	@param translation_probabilities: Path to a file containing the translation probabilities.
	The file must produced by the following command through fast_align:
	fast_align -i <parallel_data> -v -d -o <translation_probabilities>
	@param corpus: Path to a corpus in the VICTOR or CWICTOR format.
	For more information about the file's format, refer to the LEXenstein Manual.
	@param output: Path in which to save the filtered translation probabilities file.
	"""
	targets = set([])
	f = open(corpus)
	for line in f:
		data = line.strip().split('\t')
		target = data[1].strip()
		targets.add(target)
	f.close()
	
	o = open(output, 'w')
	f = open(translation_probabilities)
	for line in f:
		data = line.strip().split('\t')
		word = data[0].strip()
		if word in targets:
			o.write(line.strip() + '\n')
	f.close()
	o.close()
	
def addTargetAsFirstToVictorCorpus(self, victor_corpus, output):
	"""
	Creates a modified version of an input VICTOR corpus in which the target complex word is ranked first.
	Can be very useful for the training of Substitution Selection Models
	
	@param victor_corpus: Path to a corpus in the VICTOR format.
	For more information about the file's format, refer to the LEXenstein Manual.
	@param output: Path in which to save the modified VICTOR corpus.
	"""
	f = open(victor_corpus)
	o = open(output, 'w')
	for line in f:
		data = line.strip().split('\t')
		newline = data[0].strip() + '\t' + data[1].strip() + '\t' + data[2].strip() + '\t' + '1:'+data[1].strip() + '\t'
		for subst in data[3:len(data)]:
			substd = subst.strip().split(':')
			rank = int(substd[0].strip())
			word = substd[1].strip()
			newline += str(rank+1)+':'+word + '\t'
		o.write(newline.strip() + '\n')
	f.close()
	o.close()

def produceWordCooccurrenceModel(text_file, window, model_file):
	"""
	Creates a co-occurrence model from a text file.
	These models can be used by certain classes in LEXenstein, such as the Yamamoto Ranker and the Biran Selector.
	
	@param text_file: Text from which to estimate the word co-occurrence model.
	@param window: Number of tokens to the left and right of a word to be included as a co-occurring word.
	@param model_file: Path in which to save the word co-occurrence model.
	"""
	inp = open(text_file)

	coocs = {}

	c = 0
	for line in inp:
		c += 1
		print('At line: ' + str(c))
		tokens = line.strip().lower().split(' ')
		for i in range(0, len(tokens)):
			target = tokens[i]
			if target not in coocs.keys():
				coocs[target] = {}
			left = max(0, i-window)
			right = min(len(tokens), i+window+1)
			for j in range(left, right):
				if j!=i:
					cooc = tokens[j]
					if cooc not in coocs[target].keys():
						coocs[target][cooc] = 1
					else:
						coocs[target][cooc] += 1
	inp.close()

	targets = sorted(coocs.keys())

	out = open(model_file, 'w')
	for target in targets:
		newline = target + '\t'
		words = sorted(coocs[target].keys())
		for word in words:
			newline += word + ':' + str(coocs[target][word]) + '\t'
		out.write(newline.strip() + '\n')
	out.close()
