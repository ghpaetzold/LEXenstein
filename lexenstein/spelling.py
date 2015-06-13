import re, collections, pickle

class NorvigCorrector:

	def __init__(self, model_file, format='text'):
		"""
		Creates an instance of the NorvigCorrector class.
	
		@param model_file: Path to a file containing either raw, untokenized text, or a binary spelling correction model.
		If "model_file" is the path to a text file, then the value of "format" must be "text".
		If "model_file" is the path to a binary spelling correction model, then the value of "format" must be "bin".
		@param format: Indicator of the type of input provided.
		Possible values: "text", "bin".
		"""
		
		#If input is text, then train a model:
		if format=='text':
			#Read text file:
			file = open(model_file)
			text = file.read()
			file.close()
			
			#Create model:
			self.model = self.getSpellingModel(re.findall('[a-z]+', text))
		#If input is binary, then load the model:
		elif format=='bin':
			self.model = pickle.load(open(model_file, 'rb'))
		else:
			self.model = None
			print('Input format \"' + format + '\" no supported, see documentation for available formats.')
			
		#Create alphabet:
		self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
	
	def correct(self, word):
		"""
		Returns the spell-corrected version of a word.
		If the model determines that the word has no spelling errors, it returns the word itself.
	
		@param word: Word to be spell-corrected.
		"""
		
		candidates = self.getKnown([word]) or self.getKnown(self.getEdits(word)) or self.getKnownEdits(word) or [word]
		return max(candidates, key=self.model.get)
		
	def saveBinaryModel(self, model_path):
		"""
		Saves the spelling correction model in binary format.
		The saved model can then be loaded with the "bin" format during the creation of a NorvigCorrector.
	
		@param model_path: Path in which to save the model.
		"""
		
		pickle.dump(self.model, open(model_path, 'wb'))
	
	def getSpellingModel(self, words):
		model = collections.defaultdict(int)
		for f in words:
			model[f] += 1
		return model

	def getEdits(self, word):
		splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
		deletes = [a + b[1:] for a, b in splits if b]
		transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
		replaces = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
		inserts = [a + c + b for a, b in splits for c in self.alphabet]
		return set(deletes + transposes + replaces + inserts)

	def getKnownEdits(self, word):
		return set(e2 for e1 in self.getEdits(word) for e2 in self.getEdits(e1) if e2 in self.model)

	def getKnown(self, words):
		return set(w for w in words if w in self.model)
