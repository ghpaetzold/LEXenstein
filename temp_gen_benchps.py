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
		
		subs = []
		cands = set([])
		for i in range(0, len(data)):
			d = data[i]
			print d

			word = d[1].replace(' ', '_')

			most_sim = []
			try:
				most_sim = self.model.most_similar(positive=[word], topn=50)
			except KeyError:
				most_sim = []

			subs.append([w[0] for w in most_sim])
			
		subs_filtered = self.filterSubs(data, subs)
		
		final_cands = {}
		for i in range(0, len(data)):
			target = data[i][1]
			cands = subs_filtered[i][0:min(amount, subs_filtered[i])]
#			cands = [str(word.split('|||')[0].strip()) for word in cands]
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
	
	def filterSubs(self, data, subs):
		result = []

		prohibited_edges = set([line.strip() for line in open('../../corpora/prohibited_edges.txt')])
		prohibited_chars = set([line.strip() for line in open('../../corpora/prohibited_chars.txt')])
		vowels = set('aeiouyw')
		consonants = set('bcdfghjklmnpqrstvxz')

		for i in range(0, len(data)):
			d = data[i]
			
			sent = d[0].split(' ')
			index = int(d[2])
			if index==0:
				prevtgt = 'NULL'
			else:
				prevtgt = sent[index-1]
			if index==len(sent)-1:
				proxtgt = 'NULL'
			else:
				proxtgt = sent[index+1]

			target = d[1]
			targett = target.split(' ')
			firsttgt = targett[0]
		        lasttgt = targett[-1]

			most_sim = subs[i]
			most_simf = []

			for cand in most_sim:
				c = cand.replace('_', ' ')
				if '|||' in c:
					c = c.split('|||')[0]
				tokens = c.split(' ')
				first = tokens[0]
				last = tokens[-1]
				cchars = set(c)
				edges = set([first, last])
				inter_edge = edges.intersection(prohibited_edges)
	                        inter_chars = cchars.intersection(prohibited_chars)
				if c not in target and target not in c and first!=prevtgt and last!=proxtgt:
					if len(inter_edge)==0 and len(inter_chars)==0:
						if (firsttgt=='most' and first!='more') or (firsttgt=='more' and first!='most') or (firsttgt!='more' and firsttgt!='most'):
							if (prevtgt=='an' and c[0] in vowels) or (prevtgt=='a' and c[0] in consonants) or (prevtgt!='an' and prevtgt!='a'):
								most_simf.append(c)

			result.append(most_simf)
		return result


victor_corpus = sys.argv[1].strip()

w2v = '/export/data/ghpaetzold/word2vecvectors/models/word_vectors_mweall_generalized_1300_cbow_retrofitted.bin'

kg = GlavasGenerator(w2v)
subs = kg.getSubstitutions(victor_corpus, 11)

