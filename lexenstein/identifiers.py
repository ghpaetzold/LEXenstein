import numpy as np
from sklearn import svm
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import normalize

class MachineLearningIdentifier:

	def __init__(self, fe):
		"""
		Creates a MachineLearningIdentifier instance.
	
		@param fe: FeatureEstimator object.
		"""
		self.fe = fe
		self.classifier = None
	
	def calculateTrainingFeatures(self, training_corpus):
		"""
		Calculate features of a corpus in CWICTOR format.
	
		@param training_corpus: Path to a corpus in the CWICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		"""
		self.Xtr = self.fe.calculateFeatures(training_corpus, format='cwictor')
		self.Ytr = []
		f = open(training_corpus)
		for line in f:
			data = line.strip().split('\t')
			y = int(data[3].strip())
			self.Ytr.append(y)
		f.close()
		
	def calculateTestingFeatures(self, testing_corpus):
		"""
		Calculate testing features of a corpus in VICTOR or CWICTOR format.
	
		@param testing_corpus: Path to a corpus in the VICTOR or CWICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		"""
		self.Xte = self.fe.calculateFeatures(testing_corpus, format='cwictor')
		
	def selectKBestFeatures(self, k='all'):
		"""
		Selects the k best features through univariate feature selection.
	
		@param k: Number of features to be selected.
		"""
		feature_selector = SelectKBest(f_classif, k=k)
		feature_selector.fit(self.Xtr, self.Ytr)
		self.Xtr = feature_selector.transform(self.Xtr)
		self.Xte = feature_selector.transform(self.Xte)
		
	def trainSVM(self, C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, class_weight={0:1.0, 1:1.0}):
		"""
		Trains an SVM classifier. To know more about the meaning of each parameter,
		please refer to http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
		"""
		self.classifier = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, class_weight=class_weight)
		self.classifier.fit(self.Xtr, self.Ytr)
		
	def trainPassiveAggressiveClassifier(self, C=1.0, loss='hinge'):
		"""
		Trains a Passive Agressive classifier. To know more about the meaning of each parameter,
		please refer to http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html
		"""
		self.classifier = PassiveAggressiveClassifier(C=C, loss=loss)
		self.classifier.fit(self.Xtr, self.Ytr)
		
	def trainSGDClassifier(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, epsilon=0.001, class_weight={0:1.0, 1:1.0}):
		"""
		Trains a Stochastic Gradient Descent classifier. To know more about the meaning of each parameter,
		please refer to http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
		"""
		self.classifier = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, epsilon=epsilon, class_weight=class_weight)
		self.classifier.fit(self.Xtr, self.Ytr)
		
	def trainDecisionTreeClassifier(self, criterion='gini', splitter='best', max_features=None, max_depth=None):
		"""
		Trains a Decision Tree classifier. To know more about the meaning of each parameter,
		please refer to http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
		"""
		self.classifier = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_features=max_features, max_depth=max_depth)
		self.classifier.fit(self.Xtr, self.Ytr)
	
	def trainAdaBoostClassifier(self, n_estimators=50, learning_rate=1, algorithm='SAMME.R'):
		"""
		Trains an Ada Boost Classifier. To know more about the meaning of each parameter,
		please refer to http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
		"""
		self.classifier = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm)
		self.classifier.fit(self.Xtr, self.Ytr)
		
	def trainGradientBoostClassifier(self, loss='deviance', n_estimators=50, learning_rate=1, max_features=None):
		"""
		Trains an Gradient Boost Classifier. To know more about the meaning of each parameter,
		please refer to http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
		"""
		self.classifier = GradientBoostingClassifier(loss=loss, n_estimators=n_estimators, learning_rate=learning_rate, max_features=max_features)
		self.classifier.fit(self.Xtr, self.Ytr)
		
	def trainExtraTreesClassifier(self, n_estimators=50, criterion='gini', max_features=None):
		"""
		Trains an Extra Trees Classifier. To know more about the meaning of each parameter,
		please refer to http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
		"""
		self.classifier = ExtraTreesClassifier(n_estimators=n_estimators, criterion=criterion, max_features=max_features)
		self.classifier.fit(self.Xtr, self.Ytr)
		
	def trainRandomForestClassifier(self, n_estimators=50, criterion='gini', max_features=None):
		"""
		Trains an Random Trees Classifier. To know more about the meaning of each parameter,
		please refer to http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
		"""
		self.classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_features=max_features)
		self.classifier.fit(self.Xtr, self.Ytr)
		
	def identifyComplexWords(self):
		return self.classifier.predict(self.Xte)

class SimplifyAllIdentifier:

	def identifyComplexWords(self, corpus):
		"""
		Assign label 1 (complex) to all target words in the VICTOR or CWICTOR corpus.
	
		@param corpus: Path to a corpus in the VICTOR or CWICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: A list of binary values, one per line, with value 1.
		"""
		result = []
		f = open(corpus)
		for line in f:
			result.append(1)
		f.close()
		return result
		
class SimplifyNoneIdentifier:

	def identifyComplexWords(self, corpus):
		"""
		Assign label 0 (simple) to all target words in the VICTOR or CWICTOR corpus.
	
		@param corpus: Path to a corpus in the VICTOR or CWICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: A list of binary values, one per line, with value 0.
		"""
		result = []
		f = open(corpus)
		for line in f:
			result.append(0)
		f.close()
		return result
		
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
		f.close()
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
		self.Xtr = self.fe.calculateFeatures(training_corpus, format='cwictor')
		self.Ytr = []
		f = open(training_corpus)
		for line in f:
			data = line.strip().split('\t')
			y = int(data[3].strip())
			self.Ytr.append(y)
		f.close()
			
	def calculateTestingFeatures(self, testing_corpus):
		"""
		Calculate testing features of a corpus in VICTOR or CWICTOR format.
	
		@param testing_corpus: Path to a corpus in the VICTOR or CWICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		"""
		self.Xte = self.fe.calculateFeatures(testing_corpus, format='cwictor')
			
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
		i = min+step
		while i<max:
			score = self.getScore(i)
			if score>best:
				best=score
				bestIndex = i
			i += step

		#Set threshold and score:
		self.threshold = bestIndex
		
	def trainIdentifierBinarySearch(self, feature_index, diff=None, order=None):
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
		result = []
		for i in range(0, len(self.Xte)):
			x = self.Xte[i][self.feature_index]
			if self.fe.identifiers[self.feature_index][1]=='Complexity':
				if x>self.threshold:
					result.append(1)
				else:
					result.append(0)
			else:
				if x<self.threshold:
					result.append(1)
				else:
					result.append(0)
		return result
		
		
	def getMinMax(self):
		min = 99999
		max = -99999
		for i in range(0, len(self.Xtr)):
			value = self.Xtr[i][self.feature_index]
			if value>max:
				max = value
			if value<min:
				min = value
		return min, max
		
	def getScore(self, threshold):
		precisionc = 0
		precisiont = 0
		recallc = 0
		recallt = 0
		for i in range(0, len(self.Xtr)):
			x = self.Xtr[i][self.feature_index]
			y = self.Ytr[i]
			if self.fe.identifiers[self.feature_index][1]=='Complexity':
				if (x>threshold and y==1) or (x<threshold and y==0):
					precisionc += 1
					if y==1:
						recallc += 1
			else:
				if (x<threshold and y==1) or (x>threshold and y==0):
					precisionc += 1
					if y==1:
						recallc += 1
			precisiont += 1
			if y==1:
				recallt += 1
				
		precision = float(precisionc)/float(precisiont)
		recall = float(recallc)/float(recallt)
		fmean = 0.0
		if precision==0.0 and recall==0.0:
			fmean = 0.0
		else:
			fmean = 2*(precision*recall)/(precision+recall)
			
		#Return F-Measure:
		return fmean
