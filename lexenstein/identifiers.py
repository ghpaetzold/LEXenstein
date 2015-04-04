class LexiconIdentifier:

	def __init__(self, lexicon, type):
		"""
		Creates a LexiconIdentifier instance.
	
		@param lexicon: Lexicon containing simple or complex, one word per line.
		@param type: Type of lexicon.
		Values: 'complex', 'simple'
		"""
		self.lexicon = set([line.strip() for line in open(lexicon)])
		self.type = type
		self.feature_index = None

	def identifyComplexWords(self, corpus):
		"""
		Judge if the target words of a corpus in VICTOR or CWICTOR format are complex or not
	
		@param corpus: Path to a corpus in the VICTOR or CWICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: A list of binary values, one per line, with value 1 if a target word is complex, and 0 otherwise.
		"""
		result = []
		f = open(corpus)
		for line in f:
			data = line.strip().split('\t')
			target = data[1].strip()
			if target in self.lexicon:
				if self.type=='simple':
					result.append(0)
				else:
					result.append(1)
			else:
				if self.type=='simple':
					result.append(1)
				else:
					result.append(0)
		return result
		
class ThresholdIdentifier:

	def __init__(self, fe):
		"""
		Creates a ThresholdIdentifier instance.
	
		@param fe: FeatureEstimator object.
		"""
		self.fe = fe

	def calculateTrainingFeatures(self, training_corpus):
		"""
		Calculate features of a corpus in CWICTOR format.
	
		@param training_corpus: Path to a corpus in the CWICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		"""
		f = open(training_corpus)
		self.Xtr = []
		self.Ytr = []
		for line in f:
			data = line.strip().split('\t')
			x = self.fe.calculateInstanceFeatures(data[0], data[1], data[2], '0:'+data[1])
			y = int(data[3].strip())
			self.Xtr.append(x)
			self.Ytr.append(y)
			
	def calculateTestingFeatures(self, testing_corpus):
		"""
		Calculate testing features of a corpus in VICTOR or CWICTOR format.
	
		@param testing_corpus: Path to a corpus in the VICTOR or CWICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		"""
		f = open(testing_corpus)
		self.Xte = []
		self.Yte = []
		for line in f:
			data = line.strip().split('\t')
			x = self.fe.calculateInstanceFeatures(data[0], data[1], data[2], '0:'+data[1])
			y = int(data[3].strip())
			self.Xte.append(x)
			self.Yte.append(y)
			
	def trainIdentifierBruteForce(self, feature_index, step=None):
		"""
		Trains the threshold identifier with respect to a certain feature through brute force.

		@param feature_index: Index of the feature to be used in training.
		"""
		#Save feature index:
		self.feature_index = feature_index
		
		#Estimate min and max:
		self.minX, self.maxX = self.getMinMax()

		#Set initial min, max and pivot:
		min = float(self.minX)
		max = float(self.maxX)

		#Set step:
		if step==None:
			step = (max-min)/1000

		#Find best threshold:
		best = -1
		bestIndex = None
		i = step
		while i<max:
			score = self.getScore(i)
			if score>best:
				best=score
				bestIndex = i
			i += step

		#Set threshold and score:
		self.threshold = bestIndex
		
	def trainIdentifierBinarySearch(self, feature_index, step=None):
		"""
		Trains the threshold identifier with respect to a certain feature through binary search.

		@param feature_index: Index of the feature to be used in training.
		"""
		#Save feature index:
		self.feature_index = feature_index
		
		#Estimate min and max:
		self.minX, self.maxX = self.getMinMax()
		
		#Set initial min, max and pivot:
		min = float(self.minX)
		max = float(self.maxX)

		#Define difference threshold:
		if diff==None:
			diff = (max-min)/1000

		#Define order:
		if order==None or order<1:
			order = 1

		#Estimate best threshold:
		best = -1
		bestIndex = None
		divisor = float(2**order)
		step = (max-min)/divisor
		for i in range(1, int(divisor)):
			pivot = i*step
			index, score = self.findMaxBinary(min, max, pivot, diff)
			if score>best:
				best = score
				bestIndex = index

		#Set threshold and score:
		self.threshold = bestIndex
		
	def findMaxBinary(self, min, max, pivot, diff):
		#Estimate best threshold:
		best = -1
		bestIndex = None
		while (max-min)>diff:
			left = (min+pivot)/2.0
			right = (pivot+max)/2.0
			scoreL = self.getScore(left)
			scoreR = self.getScore(right)
			if scoreL>scoreR:
				max = pivot
				pivot = left
				if scoreL>best:
					best = scoreL
					bestIndex = left
			else:
				min = pivot
				pivot = right
				if scoreR>best:
					best = scoreR
					bestIndex = right

		#Set threshold and score:
		return bestIndex, best
		
	def identifyComplexWords(self):
		"""
		Judge if the target words of the testing instances are complex or not.

		@return: A list of binary values, one per line, with value 1 if a target word is complex, and 0 otherwise.
		"""
		corrects = 0
		for i in range(0, len(self.Xte)):
			x = self.Xte[i][self.feature_index]
			y = self.Yte[i]
			if self.fe.identifiers[feature_index][1]=='Complexity':
				if (x>threshold and y==1) or (x<threshold and y==0):
					corrects += 1
			else:
				if (x<threshold and y==1) or (x>threshold and y==0):
					corrects += 1
		return float(corrects)/float(len(self.Xte))
		
		
	def getMinMax(self):
		return np.min(self.Xtr[:,self.feature_index]), np.max(self.Xtr[:,self.feature_index])
		
	def getScore(self, threshold):
		corrects = 0
		for i in range(0, len(self.Xtr)):
			x = self.Xtr[i][self.feature_index]
			y = self.Ytr[i]
			if self.fe.identifiers[self.feature_index][1]=='Complexity':
				if (x>threshold and y==1) or (x<threshold and y==0):
					corrects += 1
			else:
				if (x<threshold and y==1) or (x>threshold and y==0):
					corrects += 1
		return float(corrects)/float(len(self.Xtr))