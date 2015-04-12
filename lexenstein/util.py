
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
