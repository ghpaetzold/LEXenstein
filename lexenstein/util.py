import nltk
import pickle
import shelve

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

def addNgramCountsFileToShelve(ngrams_file, model_file):
	"""
	Adds a n-gram counts file to an either new, or existing shelve dictionary.
	The shelve file can then be used for the calculation of several features.
	The file must be in the format produced by the "-write" option of SRILM.
	
	@param ngrams_file: File containing n-gram counts.
	@param model_file: File in which to save the frequency model.
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
