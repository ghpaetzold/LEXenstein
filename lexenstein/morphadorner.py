import subprocess

class MorphAdornerToolkit:

	def __init__(self, path):
		"""
		Creates an instance of the MorphAdornerToolkit class.
	
		@param path: Path to the root installation folder of Morph Adorner Toolkit.
		"""
		
		self.root = path
		if not self.root.endswith('/'):
			self.root += '/'
		self.lemmatizer = self.root + 'WordLemmatizer/WordLemmatizer.jar'
		self.stemmer = self.root + 'WordStemmer/WordStemmer.jar'
		self.conjugator = self.root + 'VerbConjugator/VerbConjugator.jar'
		self.inflector = self.root + 'NounInflector/NounInflector.jar'
		self.tenser = self.root + 'VerbTenser/VerbTenser.jar'
		self.syllabler = self.root + 'SyllableSplitter/SyllableSplitter.jar'
		self.adjinflector = self.root + 'AdjectiveInflector/AdjectiveInflector.jar'
	
	def lemmatizeWords(self, words):
		"""
		Lemmatizes a set of words.
	
		@param words: List of words to be lemmatized.
		@return: List of the lemmas of the words passed as input.
		"""
		
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
		"""
		Porter stems a set of words.
	
		@param words: List of words to be Porter stemmed.
		@return: List of the Porter stems of the words passed as input.
		"""
	
		input = ''
		for word in words:
			input += word + '\n'
		input += '\n'

		args = ['java', '-jar', self.stemmer]
		proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
		(out, err) = proc.communicate(input)

		result = out.strip().split('\n')
		return result

	def conjugateVerbs(self, lemmas, tense, person):
		"""
		Conjugate a set of verbs in a given tense.
	
		@param lemmas: Lemmas of verbs to be conjugated.
		@param tense: Tense in which to conjugate the verbs.
		Tenses available: PAST, PAST_PARTICIPLE, PAST_PERFECT, PAST_PERFECT_PARTICIPLE, PERFECT, PRESENT, PRESENT_PARTICIPLE.
		@param person: Person in which to conjugate the verbs.
		Tenses available: FIRST_PERSON_SINGULAR, FIRST_PERSON_PLURAL, SECOND_PERSON_SINGULAR, SECOND_PERSON_PLURAL, THIRD_PERSON_SINGULAR, THIRD_PERSON_PLURAL.
		@return: List of the conjugated versions of the verb lemmas passed as input.
		"""
		
		input = ''
		for lemma in lemmas:
			input += lemma + ' ' + tense +  ' ' + person + '\n'
		input += '\n'

		args = ['java', '-jar', self.conjugator]
		proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
		(out, err) = proc.communicate(input)

		result = out.strip().split('\n')
		return result


	def inflectNouns(self, lemmas, number):
		"""
		Inflect a list of nouns to its singular or plural form.
	
		@param lemmas: Lemmas of nouns to be inflected.
		@param number: Form in which to inflect the lemmas.
		Forms available: singular, plural.
		@return: List of the inflected versions of the noun lemmas passed as input.
		"""
		
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
		"""
		Retrieve the tense of a given set of verbs.
	
		@param lemmas: Lemmas of verbs to be tensed.
		@param verbs: Verbs in their original forms.
		@return: List of the tenses and persons of the verb passed as input.
		Tenses available: PAST, PAST_PARTICIPLE, PAST_PERFECT, PAST_PERFECT_PARTICIPLE, PERFECT, PRESENT, PRESENT_PARTICIPLE.
		Persons available: FIRST_PERSON_SINGULAR, FIRST_PERSON_PLURAL, SECOND_PERSON_SINGULAR, SECOND_PERSON_PLURAL, THIRD_PERSON_SINGULAR, THIRD_PERSON_PLURAL.
		"""
		
		input = ''
		for i in range(0, len(lemmas)):
			input += lemmas[i] + ' ' + verbs[i] +  '\n'
		input += '\n'

		args = ['java', '-jar', self.tenser]
		proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
		(out, err) = proc.communicate(input)

		result = [line.strip().split(' ') for line in out.strip().split('\n')]
		return result


	def splitSyllables(self, words):
		"""
		Splits a set of words in syllables.
	
		@param words: List of words to be lemmatized.
		@return: List of words with their syllables separated by hyphen markers.
		"""
		
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
		
	def inflectAdjectives(self, lemmas, form):
		"""
		Inflect a list of adjectives/adverbs to its singular or plural form.
	
		@param lemmas: Lemmas of adjectives/adverbs to be inflected.
		@param form: Form in which to inflect the lemmas.
		Forms available: comparative, superlative.
		@return: List of the inflected versions of the adjective/adverb lemmas passed as input.
		"""
		
		input = ''
		for lemma in lemmas:
			input += lemma + ' ' + form +  '\n'
		input += '\n'

		args = ['java', '-jar', self.adjinflector]
		proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
		(out, err) = proc.communicate(input)

		result = out.strip().split('\n')
		return result
