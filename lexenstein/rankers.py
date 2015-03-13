import os
from sklearn.preprocessing import normalize

class SVMRanker:

	def __init__(self, fe, svmrank_path):
		self.fe = fe
		self.svmrank = svmrank_path
		if not self.svmrank.endswith('/'):
			self.svmrank += '/'
	
	def getFeaturesFile(self, victor_corpus, output_file):
		#Read victor corpus:
		data = []
		f = open(victor_corpus)
		for line in f:
			data.append(line.strip().split('\t'))
		
		#Get feature values:
		feature_train = self.fe.calculateFeatures(victor_corpus)
		features_train = normalize(features_train, axis=0)
		
		#Save training file:
		out = open(output_file, 'w')
		index = 0
		for i in range(0, len(data)):
			inst = data[i]
			for subst in inst[3:len(inst)]:
				rank = subst.strip().split(':')[0].strip()
				word = subst.strip().split(':')[1].strip()
				newline = rank + ' qid:' + str(i+1) + ' '
				feature_values = features_train[index]
				index += 1
				for j in range(0, len(feature_values)):
					newline += str(j+1) + ':' + str(feature_values[j]) + ' '
				newline += '# ' + word + '\n'
				out.write(newline)
		out.close()
	
	def getTrainingModel(self, features_file, c, epsilon, kernel, output_file):
		print('Training...')
		comm = self.svmrank+'svm_rank_learn -c '+str(c)+' -e '+str(epsilon)+' -t '+str(kernel)+' '+features_file+' '+output_file
		os.system(comm)
		print('Trained!')
	
	def getScoresFile(self, features_file, model_file, output_file):
		print('Scoring...')
		comm = self.svmrank+'svm_rank_classify '+features_file+' '+model_file+' '+output_file
		os.system(comm)
		print('Scored!')
	
	def getRankings(self, features_file, scores_file):
		#Read features file:
		f = open(features_file)
		data = []
		for line in f:
			data.append(line.strip().split(' '))
		f.close()
		
		#Read scores file:
		f = open(scores_file)
		scores = []
		for line in f:
			scores.append(float(line.strip()))
		f.close()
		
		#Combine data:
		ranking_data = {}
		index = 0
		for line in data:
			id = int(line[1].strip().split(':')[1].strip())
			word = line[len(line)-1]
			score = scores[index]
			index += 1
			if id in ranking_data.keys():
				ranking_data[id][word] = score
			else:
				ranking_data[id] = {word:score}
		
		#Produce rankings:
		result = []
		for id in sorted(ranking_data.keys()):
			candidates = ranking_data[id].keys()
			candidates = sorted(candidates, key=ranking_data[id].__getitem__, reverse=False)
			result.append(candidates)
			
		#Return rankings:
		return result
	
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