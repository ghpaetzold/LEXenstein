

class MetricRanker:

	def __init__(self, fe):
		self.fe = fe
		self.feature_values = None
		
	def getRankings(self, victor_corpus, featureIndex):
		#If feature values are not available, then estimate them:
		if self.feature_values == None:
			self.feature_values = self.fe.calculateFeatures(victor_corpus)
		
		#Create object for results:
		result = []
		
		#Read feature values for each candidate in victor corpus:
		f = open(victor_corpus)
		index = 0
		for line in f:
			#Get all substitutions in ranking instance:
			data = line.strip().split('\t')
			substitutions = data[3:len(data)]
			
			#Create dictionary of substitution to feature value:
			scores = {}
			for substitution in substitutions:
				word = substitution.strip().split(':')[1].strip()
				scores[word] = self.feature_values[index][featureIndex]
				index += 1
			
			#Check if feature is simplicity or complexity measure:
			rev = False
			if self.fe.identifiers[featureIndex][1]=='Simplicity':
				rev = True
			
			#Sort substitutions:
			sorted_substitutions = sorted(scores.keys(), key=scores.__getitem__, reverse=rev)
		
			#Add them to result:
			result.append(sorted_substitutions)
		f.close()
		
		#Return result:
		return result
		
	def size(self):
		return len(self.fe.identifiers)