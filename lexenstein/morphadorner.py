import subprocess

class MorphAdornerToolkit:

	def __init__(self, path):
		self.root = path
		if not self.root.endswith('/'):
			self.root += '/'
		self.lemmatizer = self.root + 'WordLemmatizer/WordLemmatizer.jar'
		self.stemmer = self.root + 'WordStemmer/WordStemmer.jar'
		self.conjugator = self.root + 'VerbConjugator/VerbConjugator.jar'
		self.inflector = self.root + 'NounInflector/NounInflector.jar'
		self.tenser = self.root + 'VerbTenser/VerbTenser.jar'
		self.syllabler = self.root + 'SyllableSplitter/SyllableSplitter.jar'
	
	def lemmatizeWords(self, words):
		input = ''
		for word in words:
			input += word + '\n'
		input += '\n'

		args = ['java', '-jar', self.lemmatizer]
		proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
		(out, err) = proc.communicate(input)

		result = out.strip().split('\n')
		return result
		
	def stemWords(self, words):
		input = ''
		for word in words:
			input += word + '\n'
		input += '\n'

		args = ['java', '-jar', self.stemmer]
		proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
		(out, err) = proc.communicate(input)

		result = out.strip().split('\n')
		return result

	def conjugateVerbs(self, lemmas, tense):
		input = ''
		for lemma in lemmas:
			input += lemma + ' ' + tense +  '\n'
		input += '\n'

		args = ['java', '-jar', self.conjugator]
		proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
		(out, err) = proc.communicate(input)

		result = out.strip().split('\n')
		return result


	def inflectNouns(self, lemmas, number):
		input = ''
		for lemma in lemmas:
			input += lemma + ' ' + number +  '\n'
		input += '\n'

		args = ['java', '-jar', self.inflector]
		proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
		(out, err) = proc.communicate(input)

		result = out.strip().split('\n')
		return result

	def tenseVerbs(self, lemmas, verbs):
		input = ''
		for i in range(0, len(lemmas)):
			input += lemmas[i] + ' ' + verbs[i] +  '\n'
		input += '\n'

		args = ['java', '-jar', self.tenser]
		proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
		(out, err) = proc.communicate(input)

		result = out.strip().split('\n')
		return result


	def splitSyllables(self, words):
		input = ''
		for word in words:
			input += word + '\n'
		input += '\n'

		args = ['java', '-jar', self.syllabler]
		proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
		(out, err) = proc.communicate(input)
		
		out = out.replace('\xc2\xad', '-')
		result = out.strip().split('\n')
		return result
