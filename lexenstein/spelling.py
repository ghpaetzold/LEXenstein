import re, collections

class NorvigCorrector:

	def __init__(self, text_file):
		"""
		Creates an instance of the NorvigCorrector class.
	
		@param text_file: Path to a file containing raw, untokenized text.
		"""
		
		#Read text file:
		file = open(text_file)
		text = file.read()
		file.close()
		
		#Create model:
		self.model = self.getSpellingModel(re.findall('[a-z]+', text))

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
	
	def getSpellingModel(self, words):
		model = collections.defaultdict(lambda: 1)
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
