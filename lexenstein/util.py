import nltk
import pickle

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
