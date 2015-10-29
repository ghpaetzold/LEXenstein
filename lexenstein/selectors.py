from lexenstein.util import *
import pywsd
import gensim
from scipy.spatial.distance import cosine
import nltk
from nltk.tag.stanford import StanfordPOSTagger
import numpy as np
import os
import pickle

class SVMRankSelector:

	def __init__(self, svm_ranker):
		"""
		Creates an instance of the SVMRankSelector class.
	
		@param svm_ranker: An instance of the SVMRanker class.
		"""
		self.ranker = svm_ranker
		
	def trainSelector(self, tr_victor_corpus, tr_features_file, model_file, c, epsilon, kernel):
		"""
		Trains a SVM Ranker according to the parameters provided.
	
		@param tr_victor_corpus: Path to a training corpus in VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param tr_features_file: File in which to save the training features file.
		@param model_file: File in which to save the trained model.
		@param c: Trade-off between training error and margin.
		Recommended values: 0.001, 0.01
		@param epsilon: Acceptable error margin.
		Recommended values: 0.00001, 0.0001
		@param kernel: ID for the kernel to be used.
		Kernels available:
		0 - Linear
		1 - Polynomial
		2 - Radial Basis Function
		3 - Sigmoid
		"""
		self.ranker.getFeaturesFile(tr_victor_corpus, tr_features_file)
		self.ranker.getTrainingModel(tr_features_file, c, epsilon, kernel, model_file)
		self.model = model_file
	
	def trainSelectorWithCrossValidation(self, victor_corpus, features_file, model_file, folds, test_size, temp_folder, temp_id, Cs=['0.01', '0.001'], epsilons=[0.0001, 0.001], kernels=['0', '2', '3']):
		"""
		Trains a SVM Selector while maximizing hyper-parameters through cross-validation.
		It uses the TRank-at-1 as an optimization metric.
	
		@param victor_corpus: Path to a training corpus in VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param features_file: File in which to save the training features file.
		@param model_file: File in which to save the trained model.
		@param folds: Number of folds to be used in cross-validation.
		@param test_size: Percentage of the dataset to be used in testing.
		Recommended values: 0.2, 0.25, 0.33
		@param temp_folder: Folder in which to save temporary files.
		@param temp_id: ID to be used in the identification of temporary files.
		@param Cs: Trade-offs between training error and margin.
		Recommended values: 0.001, 0.01
		@param epsilons: Acceptable error margins.
		Recommended values: 0.00001, 0.0001
		@param kernels: ID for the kernels to be considered.
		Kernels available:
		0 - Linear
		1 - Polynomial
		2 - Radial Basis Function
		3 - Sigmoid
		"""
		parameters = self.ranker.trainRankerWithCrossValidation(victor_corpus, folds, test_size, temp_folder, temp_id, Cs=Cs, epsilons=epsilons, kernels=kernels)
		self.ranker.getFeaturesFile(victor_corpus, features_file)
		self.ranker.getTrainingModel(features_file, parameters[0], parameters[2], parameters[1], model_file)
		self.model = model_file
		
	def selectCandidates(self, substitutions, victor_corpus, features_file, scores_file, temp_file, proportion, proportion_type='percentage'):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: Candidate substitutions to be filtered.
		It can be in two formats:
		A dictionary produced by a Substitution Generator linking complex words to a set of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		A list of candidate substitutions selected for the "victor_corpus" dataset by a Substitution Selector.
		Example: [['sat', 'roosted'], ['easy', 'uncomplicated']]
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param features_file: File in which to save the testing features file.
		@param scores_file: File in which to save the scores file.
		User must have the privilege to delete such file without administrator privileges.
		@param temp_file: File in which to save a temporary victor corpus.
		The file is removed after the algorithm is concluded.
		@param proportion: Proportion of substitutions to keep.
		If proportion_type is set to "percentage", then this parameter must be a floating point number between 0 and 1.
		If proportion_type is set to "integer", then this parameter must be an integer number.
		@param proportion_type: Type of proportion to be kept.
		Values supported: percentage, integer.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		void = VoidSelector()
		selected_void = void.selectCandidates(substitutions, victor_corpus)
		void.toVictorFormat(victor_corpus, selected_void, temp_file)
		
		self.ranker.getFeaturesFile(temp_file, features_file)
		self.ranker.getScoresFile(features_file, self.model, scores_file)
		rankings = self.getRankings(temp_file, features_file, scores_file)
		
		selected_substitutions = []				

		lexf = open(victor_corpus)
		index = -1
		for line in lexf:
			index += 1
		
			selected_candidates = None
			if proportion_type == 'percentage':
				toselect = None
				if proportion > 1.0:
					toselect = 1.0
				else:
					toselect = proportion
				selected_candidates = rankings[index][0:max(1, int(toselect*float(len(rankings[index]))))]
			else:
				toselect = None
				if proportion < 1:
					toselect = 1
				elif proportion > len(rankings[index]):
					toselect = len(rankings[index])
				else:
					toselect = proportion
				selected_candidates = rankings[index][0:toselect]
		
			selected_substitutions.append(selected_candidates)
		lexf.close()
		
		#Delete temp_file:
		os.system('rm ' + temp_file)
		return selected_substitutions
		
	def getRankings(self, victor_corpus, features_file, scores_file):		
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
			starti = 0
			while line[starti]!='#':
				starti += 1
			word = ''
			for i in range(starti+1, len(line)):
				word += line[i] + ' '
			word = word.strip()
			score = scores[index]
			index += 1
			if id in ranking_data:
				ranking_data[id][word] = score
			else:
				ranking_data[id] = {word:score}
		
		#Produce rankings:
		result = []
		f = open(victor_corpus)
		id = 0
		for line in f:
			id += 1
			candidates = []
			if id in ranking_data:
				candidates = ranking_data[id].keys()
				candidates = sorted(candidates, key=ranking_data[id].__getitem__, reverse=False)
			result.append(candidates)
			
		#Return rankings:
		return result
		
	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()

class SVMBoundarySelector:

	def __init__(self, svm_boundary_ranker):
		"""
		Creates an instance of the SVMBoundarySelector class.
	
		@param svm_boundary_ranker: An instance of the BoundaryRanker class.
		"""
		self.ranker = svm_boundary_ranker
		
	def trainSelector(self, victor_corpus, positive_range, C, kernel, degree, gamma, coef0, k='all'):
		"""
		Trains a Boundary Ranker according to the parameters provided.
	
		@param victor_corpus: Path to a training corpus in VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param positive_range: Maximum rank to which label 1 is assigned in the binary classification setup.
		Recommended value: 1.
		@param C: Penalty parameter.
		Recommended values: 0.1, 1, 10.
		@param kernel: Kernel function to be used.
		Supported values: 'linear', 'poly', 'rbf', 'sigmoid'.
		@param degree: Degree of the polynomial kernel.
		Recommended values: 2, 3.
		@param gamma: Kernel coefficient.
		Recommended values: 0.01, 0.1, 1.
		@param coef0: Independent term value.
		Recommended values: 0, 1.
		@param k: Number of best features to be selected through univariate feature selection.
		If k='all', then no feature selection is performed.
		"""
		self.ranker.trainRanker(victor_corpus, positive_range, C, kernel, degree, gamma, coef0, k=k)
	
	def trainSelectorWithCrossValidation(self, victor_corpus, positive_range, folds, test_size, Cs=[0.1, 1, 10], kernels=['linear', 'rbf', 'poly', 'sigmoid'], degrees=[2], gammas=[0.01, 0.1, 1], coef0s=[0, 1], k='all'):
		"""
		Trains a Boundary Selector while maximizing hyper-parameters through cross-validation.
		It uses the TRank-at-1 as an optimization metric.
	
		@param victor_corpus: Path to a training corpus in VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param positive_range: Maximum rank to which label 1 is assigned in the binary classification setup.
		Recommended value: 1.
		@param folds: Number of folds to be used in cross-validation.
		@param test_size: Percentage of the dataset to be used in testing.
		Recommended values: 0.2, 0.25, 0.33
		@param Cs: Penalty parameters.
		Recommended values: 0.1, 1, 10.
		@param kernels: Kernel functions to be used.
		Supported values: 'linear', 'poly', 'rbf', 'sigmoid'.
		@param degrees: Degrees of the polynomial kernel.
		Recommended values: 2, 3.
		@param gammas: Kernel coefficients.
		Recommended values: 0.01, 0.1, 1.
		@param coef0s: Independent term values.
		Recommended values: 0, 1.
		@param k: Number of best features to be selected through univariate feature selection.
		If k='all', then no feature selection is performed.
		"""
		self.ranker.trainRankerWithCrossValidation(victor_corpus, positive_range, folds, test_size, Cs=Cs, kernels=kernels, degrees=degrees, gammas=gammas, coef0s=coef0s, k=k)
		
	def selectCandidates(self, substitutions, victor_corpus, temp_file, proportion, proportion_type='percentage'):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: Candidate substitutions to be filtered.
		It can be in two formats:
		A dictionary produced by a Substitution Generator linking complex words to a set of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		A list of candidate substitutions selected for the "victor_corpus" dataset by a Substitution Selector.
		Example: [['sat', 'roosted'], ['easy', 'uncomplicated']]
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		User must have the privilege to delete such file without administrator privileges.
		@param temp_file: File in which to save a temporary victor corpus.
		The file is removed after the algorithm is concluded.
		@param proportion: Proportion of substitutions to keep.
		If proportion_type is set to "percentage", then this parameter must be a floating point number between 0 and 1.
		If proportion_type is set to "integer", then this parameter must be an integer number.
		@param proportion_type: Type of proportion to be kept.
		Values supported: percentage, integer.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		void = VoidSelector()
		selected_void = void.selectCandidates(substitutions, victor_corpus)
		void.toVictorFormat(victor_corpus, selected_void, temp_file)
		
		rankings = self.ranker.getRankings(temp_file)
		
		selected_substitutions = []				

		lexf = open(victor_corpus)
		index = -1
		for line in lexf:
			index += 1
		
			selected_candidates = None
			if proportion_type == 'percentage':
				toselect = None
				if proportion > 1.0:
					toselect = 1.0
				else:
					toselect = proportion
				selected_candidates = rankings[index][0:max(1, int(toselect*float(len(rankings[index]))))]
			else:
				toselect = None
				if proportion < 1:
					toselect = 1
				elif proportion > len(rankings[index]):
					toselect = len(rankings[index])
				else:
					toselect = proportion
				selected_candidates = rankings[index][0:toselect]
		
			selected_substitutions.append(selected_candidates)
		lexf.close()
		
		#Delete temp_file:
		os.system('rm ' + temp_file)
		return selected_substitutions
		
	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()
		
class BoundarySelector:

	def __init__(self, boundary_ranker):
		"""
		Creates an instance of the BoundarySelector class.
	
		@param boundary_ranker: An instance of the BoundaryRanker class.
		"""
		self.ranker = boundary_ranker
		
	def trainSelector(self, victor_corpus, positive_range, loss, penalty, alpha, l1_ratio, epsilon, k='all'):
		"""
		Trains a Boundary Ranker according to the parameters provided.
	
		@param victor_corpus: Path to a training corpus in VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param positive_range: Maximum rank to which label 1 is assigned in the binary classification setup.
		Recommended value: 1.
		@param loss: Loss function to be used.
		Values available: hinge, log, modified_huber, squared_hinge, perceptron.
		@param penalty: Regularization term to be used.
		Values available: l2, l1, elasticnet.
		@param alpha: Constant that multiplies the regularization term.
		Recommended values: 0.0001, 0.001, 0.01, 0.1
		@param l1_ratio: Elastic net mixing parameter.
		Recommended values: 0.05, 0.10, 0.15
		@param epsilon: Acceptable error margin.
		Recommended values: 0.0001, 0.001
		@param k: Number of best features to be selected through univariate feature selection.
		If k='all', then no feature selection is performed.
		"""
		self.ranker.trainRanker(victor_corpus, positive_range, loss, penalty, alpha, l1_ratio, epsilon, k=k)
	
	def trainSelectorWithCrossValidation(self, victor_corpus, positive_range, folds, test_size, losses=['hinge', 'modified_huber'], penalties=['elasticnet'], alphas=[0.0001, 0.001, 0.01], l1_ratios=[0.0, 0.15, 0.25, 0.5, 0.75, 1.0], k='all'):
		"""
		Trains a Boundary Selector while maximizing hyper-parameters through cross-validation.
		It uses the TRank-at-1 as an optimization metric.
	
		@param victor_corpus: Path to a training corpus in VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param positive_range: Maximum rank to which label 1 is assigned in the binary classification setup.
		Recommended value: 1.
		@param folds: Number of folds to be used in cross-validation.
		@param test_size: Percentage of the dataset to be used in testing.
		Recommended values: 0.2, 0.25, 0.33
		@param losses: Loss functions to be considered.
		Values available: hinge, log, modified_huber, squared_hinge, perceptron.
		@param penalties: Regularization terms to be considered.
		Values available: l2, l1, elasticnet.
		@param alphas: Constants that multiplies the regularization term.
		Recommended values: 0.0001, 0.001, 0.01, 0.1
		@param l1_ratios: Elastic net mixing parameters.
		Recommended values: 0.05, 0.10, 0.15
		@param k: Number of best features to be selected through univariate feature selection.
		If k='all', then no feature selection is performed.
		"""
		self.ranker.trainRankerWithCrossValidation(victor_corpus, positive_range, folds, test_size, losses=losses, penalties=penalties, alphas=alphas, l1_ratios=l1_ratios, k=k)
		
	def selectCandidates(self, substitutions, victor_corpus, temp_file, proportion, proportion_type='percentage'):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: Candidate substitutions to be filtered.
		It can be in two formats:
		A dictionary produced by a Substitution Generator linking complex words to a set of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		A list of candidate substitutions selected for the "victor_corpus" dataset by a Substitution Selector.
		Example: [['sat', 'roosted'], ['easy', 'uncomplicated']]
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		User must have the privilege to delete such file without administrator privileges.
		@param temp_file: File in which to save a temporary victor corpus.
		The file is removed after the algorithm is concluded.
		@param proportion: Proportion of substitutions to keep.
		If proportion_type is set to "percentage", then this parameter must be a floating point number between 0 and 1.
		If proportion_type is set to "integer", then this parameter must be an integer number.
		@param proportion_type: Type of proportion to be kept.
		Values supported: percentage, integer.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		void = VoidSelector()
		selected_void = void.selectCandidates(substitutions, victor_corpus)
		void.toVictorFormat(victor_corpus, selected_void, temp_file)
		
		rankings = self.ranker.getRankings(temp_file)
		
		selected_substitutions = []				

		lexf = open(victor_corpus)
		index = -1
		for line in lexf:
			index += 1
		
			selected_candidates = None
			if proportion_type == 'percentage':
				toselect = None
				if proportion > 1.0:
					toselect = 1.0
				else:
					toselect = proportion
				selected_candidates = rankings[index][0:max(1, int(toselect*float(len(rankings[index]))))]
			else:
				toselect = None
				if proportion < 1:
					toselect = 1
				elif proportion > len(rankings[index]):
					toselect = len(rankings[index])
				else:
					toselect = proportion
				selected_candidates = rankings[index][0:toselect]
		
			selected_substitutions.append(selected_candidates)
		lexf.close()
		
		#Delete temp_file:
		os.system('rm ' + temp_file)
		return selected_substitutions
		
	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()

class BelderSelector:

	def __init__(self, clusters):
		"""
		Creates an instance of the BelderSelector class.
	
		@param clusters: Path to a file containing clusters of words.
		For instructions on how to create the file, please refer to the LEXenstein Manual.
		"""
		self.clusters_to_words, self.words_to_clusters = self.getClusterData(clusters)

	def selectCandidates(self, substitutions, victor_corpus):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: Candidate substitutions to be filtered.
		It can be in two formats:
		A dictionary produced by a Substitution Generator linking complex words to a set of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		A list of candidate substitutions selected for the "victor_corpus" dataset by a Substitution Selector.
		Example: [['sat', 'roosted'], ['easy', 'uncomplicated']]
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		selected_substitutions = []

		substitution_candidates = []
		if isinstance(substitutions, list):
			substitution_candidates = substitutions
		elif isinstance(substitutions, dict):
			void = VoidSelector()
			substitution_candidates = void.selectCandidates(substitutions, victor_corpus)
		else:
			print('ERROR: Substitutions are neither a dictionary or a list!')
			return selected_substitutions

		c = -1
		lexf = open(victor_corpus)
		for line in lexf:
			c += 1
			data = line.strip().split('\t')
			sent = data[0].strip()
			target = data[1].strip()
		
			selected_candidates = set([])
			if target in self.words_to_clusters:	
				cluster = self.words_to_clusters[target]
				candidates = set(substitution_candidates[c])
				selected_candidates = candidates.intersection(self.clusters_to_words[cluster])
		
			selected_substitutions.append(selected_candidates)
		lexf.close()
		return selected_substitutions
		
	def getClusterData(self, clusters):
		cw = {}
		wc = {}
		f = open(clusters)
		for line in f:
			data = line.strip().split('\t')
			cluster = data[0].strip()
			word = data[1].strip()
			
			if cluster in cw:
				cw[cluster].add(word)
			else:
				cw[cluster] = set([word])
			
			wc[word] = cluster
		f.close()
		return cw, wc
		
	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()

class POSProbSelector:

	def __init__(self, condprob_model, pos_model, stanford_tagger, java_path):
		"""
		Creates a POSProbSelector instance.
		It selects only the candidate substitutions of which the most likely POS tag is that of the target word.
	
		@param condprob_model: Path to a binary conditional probability model.
		For instructions on how to create the model, please refer to the LEXenstein Manual.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		"""
		os.environ['JAVAHOME'] = java_path
		self.tagger = StanfordPOSTagger(pos_model, stanford_tagger)
		self.model = pickle.load(open(condprob_model, 'rb'))

	def selectCandidates(self, substitutions, victor_corpus):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: Candidate substitutions to be filtered.
		It can be in two formats:
		A dictionary produced by a Substitution Generator linking complex words to a set of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		A list of candidate substitutions selected for the "victor_corpus" dataset by a Substitution Selector.
		Example: [['sat', 'roosted'], ['easy', 'uncomplicated']]
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		selected_substitutions = []

		substitution_candidates = []
		if isinstance(substitutions, list):
			substitution_candidates = substitutions
		elif isinstance(substitutions, dict):
			void = VoidSelector()
			substitution_candidates = void.selectCandidates(substitutions, victor_corpus)
		else:
			print('ERROR: Substitutions are neither a dictionary or a list!')
			return selected_substitutions
		
		#Read VICTOR corpus:
		lexf = open(victor_corpus)
		sents = []
		targets = []
		heads = []
		c = -1
		for line in lexf:
			c += 1
			data = line.strip().split('\t')
			sent = data[0].strip().split(' ')
			target = data[1].strip()
			head = int(data[2].strip())
			sents.append(sent)
			targets.append(target)
			heads.append(head)
		lexf.close()
		
		#Tag sentences:
		tagged_sents = self.tagger.tag_sents(sents)
		
		for i in range(0, len(sents)):
			target = targets[i]
			head = heads[i]
			target_pos = str(tagged_sents[i][head][1])
		
			candidates = []
			candidates = set(substitution_candidates[i])
			candidates = self.getCandidatesWithSamePOS(candidates, target_pos)
		
			selected_substitutions.append(candidates)
		lexf.close()
		return selected_substitutions
	
	def getTargetPOS(self, sent, target, head):
		pos_data = []
		try:
			pos_data = nltk.pos_tag(sent)
			return pos_data[head][1]
		except UnicodeDecodeError:
			try:
				pos_data = nltk.pos_tag(target)
				return pos_data[0][1]
			except UnicodeDecodeError:
				return 'None'
			
	def getCandidatesWithSamePOS(self, candidates, target_pos):
		result = set([])
		for candidate in candidates:
			cand_tag = None
			try:
				cand_tag = self.model[candidate].max()
			except Exception:
				pass
			if cand_tag and cand_tag==target_pos:
				result.add(candidate)
		return result
	
	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()
		
class AluisioSelector:

	def __init__(self, condprob_model, pos_model, stanford_tagger, java_path):
		"""
		Creates an AluisioSelector instance.
		It selects only candidate substitutions that can assume the same POS tag of the target word.
	
		@param condprob_model: Path to a binary conditional probability model.
		For instructions on how to create the model, please refer to the LEXenstein Manual.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		"""
		os.environ['JAVAHOME'] = java_path
		self.tagger = StanfordPOSTagger(pos_model, stanford_tagger)
		self.model = pickle.load(open(condprob_model, 'rb'))

	def selectCandidates(self, substitutions, victor_corpus):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: Candidate substitutions to be filtered.
		It can be in two formats:
		A dictionary produced by a Substitution Generator linking complex words to a set of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		A list of candidate substitutions selected for the "victor_corpus" dataset by a Substitution Selector.
		Example: [['sat', 'roosted'], ['easy', 'uncomplicated']]
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		selected_substitutions = []

		substitution_candidates = []
		if isinstance(substitutions, list):
			substitution_candidates = substitutions
		elif isinstance(substitutions, dict):
			void = VoidSelector()
			substitution_candidates = void.selectCandidates(substitutions, victor_corpus)
		else:
			print('ERROR: Substitutions are neither a dictionary or a list!')
			return selected_substitutions
		
		#Read VICTOR corpus:
		lexf = open(victor_corpus)
		sents = []
		targets = []
		heads = []
		c = -1
		for line in lexf:
			c += 1
			data = line.strip().split('\t')
			sent = data[0].strip().split(' ')
			target = data[1].strip()
			head = int(data[2].strip())
			sents.append(sent)
			targets.append(target)
			heads.append(head)
		lexf.close()
		
		#Tag sentences:
		tagged_sents = self.tagger.tag_sents(sents)
		
		for i in range(0, len(sents)):
			target = targets[i]
			head = heads[i]
			target_pos = str(tagged_sents[i][head][1])
		
			candidates = []
			candidates = set(substitution_candidates[i])
			candidates = self.getCandidatesWithSamePOS(candidates, target_pos)
		
			selected_substitutions.append(candidates)
		lexf.close()
		return selected_substitutions
	
	def getTargetPOS(self, sent, target, head):
		pos_data = []
		try:
			pos_data = nltk.pos_tag(sent)
			return pos_data[head][1]
		except UnicodeDecodeError:
			try:
				pos_data = nltk.pos_tag(target)
				return pos_data[0][1]
			except UnicodeDecodeError:
				return 'None'
			
	def getCandidatesWithSamePOS(self, candidates, target_pos):
		result = set([])
		for candidate in candidates:
			tag_freq = 0
			try:
				tag_freq = self.model[candidate].prob(target_pos)
			except Exception:
				pass
			if tag_freq>0:
				result.add(candidate)
		return result
	
	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()

class VoidSelector:

	def selectCandidates(self, substitutions, victor_corpus):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: Candidate substitutions to be filtered.
		It can be in two formats:
		A dictionary produced by a Substitution Generator linking complex words to a set of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		A list of candidate substitutions selected for the "victor_corpus" dataset by a Substitution Selector.
		Example: [['sat', 'roosted'], ['easy', 'uncomplicated']]
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		selected_substitutions = []

		if isinstance(substitutions, list):
			return substitutions	

		lexf = open(victor_corpus)
		for line in lexf:
			data = line.strip().split('\t')
			sent = data[0].strip()
			target = data[1].strip()
		
			candidates = []
			if target in substitutions:
				candidates = substitutions[target]
		
			selected_substitutions.append(candidates)
		lexf.close()
		return selected_substitutions
		
	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()

class BiranSelector:

	def __init__(self, cooc_model):
		"""
		Creates an instance of the BiranSelector class.
	
		@param cooc_model: Path to a word co-occurrence model.
		For instructions on how to create the model, please refer to the LEXenstein Manual.
		"""
		self.model = self.getModel(cooc_model)
		
	def selectCandidates(self, substitutions, victor_corpus, common_distance=0.01, candidate_distance=0.9):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: Candidate substitutions to be filtered.
		It can be in two formats:
		A dictionary produced by a Substitution Generator linking complex words to a set of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		A list of candidate substitutions selected for the "victor_corpus" dataset by a Substitution Selector.
		Example: [['sat', 'roosted'], ['easy', 'uncomplicated']]
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param common_distance: The cutoff minimum distance from the sentence's co-occurrence vector and the common vector between the target complex word and the candidate.
		We recommend using very small values, such as 0.01, or even 0.0.
		@param candidate_distance: The cutoff maximum distance from the sentence's co-occurrence vector and the candidate vector.
		We recommend using values close to 1.0, such as 0.8, or 0.9.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		selected_substitutions = []

		substitution_candidates = []
		if isinstance(substitutions, list):
			substitution_candidates = substitutions
		elif isinstance(substitutions, dict):
			void = VoidSelector()
			substitution_candidates = void.selectCandidates(substitutions, victor_corpus)
		else:
			print('ERROR: Substitutions are neither a dictionary or a list!')
			return selected_substitutions			

		c = -1
		lexf = open(victor_corpus)
		for line in lexf:
			c += 1
			data = line.strip().split('\t')
			sent = data[0].strip()
			target = data[1].strip()
			head = int(data[2].strip())
		
			target_vec = self.getSentVec(sent, head)

			candidates = set(substitution_candidates[c])
		
			final_candidates = set([])
			for candidate_raw in candidates:
				candidate = str(candidate_raw)
				candidate_vec = self.getVec(candidate)
				candidate_dist = 1.0
				try:
					candidate_dist = self.getCosine(candidate_vec, target_vec)
				except ValueError:
					candidate_dist = 1.0
		
				common_vec = self.getCommonVec(target, candidate)
				common_dist = 0.0
				try:
					common_dist = self.getCosine(common_vec, target_vec)
				except ValueError:
					common_dist = 0.0
				if common_dist>=common_distance and candidate_dist<=candidate_distance:
					final_candidates.add(candidate)
			selected_substitutions.append(final_candidates)
		lexf.close()
		return selected_substitutions
		
	def getModel(self, path):
		result = {}
		f = open(path)
		for line in f:
			data = line.strip().split('\t')
			target = data[0].strip()
			coocs = data[1:len(data)]
			result[target] = {}
			for cooc in coocs:
				coocd = cooc.strip().split(':')
				word = coocd[0].strip()
				count = int(coocd[1].strip())
				result[target][word] = count
		return result
	
	def getCosine(self, vec1, vec2):
		all_keys = sorted(list(set(vec1.keys()).union(set(vec2.keys()))))
		v1 = []
		v2 = []
		for k in all_keys:
			if k in vec1:
				v1.append(vec1[k])
			else:
				v1.append(0.0)
			if k in vec2:
				v2.append(vec2[k])
			else:
				v2.append(0.0)
		return cosine(v1, v2)
	
	def getCommonVec(self, target, candidate):
		if target not in self.model.keys() or candidate not in self.model:
			return {}
		else:
			result = {}
			common_keys = set(self.model[target].keys()).intersection(set(self.model[candidate].keys()))
			for k in common_keys:
				if self.model[target][k]>self.model[candidate][k]:
					result[k] = self.model[candidate][k]
				else:
					result[k] = self.model[target][k]
			return result
					
	def isNumeral(self, text):
		try:
			num = float(text.strip())
			return True
		except ValueError:
			return False
	
	def getSentVec(self, sent, head):
		coocs = {}
		tokens = sent.strip().split(' ')
		left = max(0, head-5)
		right = min(len(tokens), head+6)
		for j in range(left, right):
			if j!=head:
				cooc = tokens[j]
				if self.isNumeral(cooc):
					cooc = '#NUMERAL#'
				if cooc not in coocs:
					coocs[cooc] = 1
				else:
					coocs[cooc] += 1
		return coocs
	
	def getVec(self, word):
		result = {}
		try:
			result = self.model[word]
		except KeyError:
			try:
				result = self.model[word.lower()]
			except KeyError:
				result = {}
		return result
		
	def getCandidateSentence(self, sentence, candidate, head):
		tokens = sentence.strip().split(' ')
		result = ''
		for i in range(0, head):
			result += tokens[i] + ' '
		result += candidate + ' '
		for i in range(head+1, len(tokens)):
			result += tokens[i] + ' '
		return result.strip()
		
	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()
	
class WordVectorSelector:
	
	def __init__(self, vector_model, pos_model, stanford_tagger, java_path, pos_type='none'):
		"""
		Creates an instance of the WordVectorSelector class.
	
		@param vector_model: Path to a binary word vector model.
		For instructions on how to create the model, please refer to the LEXenstein Manual.
		@param pos_model: Path to a POS tagging model for the Stanford POS Tagger.
		The models can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param stanford_tagger: Path to the "stanford-postagger.jar" file.
		The tagger can be downloaded from the following link: http://nlp.stanford.edu/software/tagger.shtml
		@param java_path: Path to the system's "java" executable.
		Can be commonly found in "/usr/bin/java" in Unix/Linux systems, or in "C:/Program Files/Java/jdk_version/java.exe" in Windows systems.
		@param pos_type: The type of POS tags with which the model's words are annotated, if any.
		Values supported: none, treebank, paetzold
		"""
		self.model = gensim.models.word2vec.Word2Vec.load_word2vec_format(vector_model, binary=True)
		self.pos_type = pos_type
		os.environ['JAVAHOME'] = java_path
		self.tagger = StanfordPOSTagger(pos_model, stanford_tagger)
	
	def selectCandidates(self, substitutions, victor_corpus, proportion=1.0, proportion_type='percentage', stop_words_file=None, window=99999, onlyInformative=False, keepTarget=False, onePerWord=False):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: Candidate substitutions to be filtered.
		It can be in two formats:
		A dictionary produced by a Substitution Generator linking complex words to a set of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		A list of candidate substitutions selected for the "victor_corpus" dataset by a Substitution Selector.
		Example: [['sat', 'roosted'], ['easy', 'uncomplicated']]
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@param proportion: Percentage of substitutions to keep.
		If proportion_type is set to "percentage", then this parameter must be a floating point number between 0 and 1.
		If proportion_type is set to "integer", then this parameter must be an integer number.
		@param proportion_type: Type of proportion to be kept.
		Values supported: percentage, integer.
		@param stop_words_file: Path to the file containing stop words of the desired language.
		The file must contain one stop word per line.
		@param window: Number of tokens around the target complex sentence to consider as its context.
		@param onlyInformative: If True, only content words are considered as part of the complex word's context, such as nouns, verbs, adjectives and adverbs.
		@param keepTarget: If True, the complex target word is also included as part of its context.
		@param onePerWord: If True, a word in the complex word's context can only contribute once to its resulting word vector.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		#Initialize selected substitutions:
		selected_substitutions = []
		
		#Read stop words:
		stop_words = set([])
		if stop_words_file != None:
			stop_words = set([word.strip() for word in open(stop_words_file)])

		#Configure input:
		substitution_candidates = []
		if isinstance(substitutions, list):
			substitution_candidates = substitutions
		elif isinstance(substitutions, dict):
			void = VoidSelector()
			substitution_candidates = void.selectCandidates(substitutions, victor_corpus)
		else:
			print('ERROR: Substitutions are neither a dictionary or a list!')
			return selected_substitutions		

		#Parse sentences:
		lexf = open(victor_corpus)
		sents = [line.strip().split('\t')[0].strip().split(' ') for line in lexf]
		lexf.close()
		tagged_sents = self.tagger.tag_sents(sents)
		
		#Transform them to the right format:
		if self.pos_type=='paetzold':
			transformed = []
			for sent in tagged_sents:
				tokens = []
				for token in sent:
					tokens.append((token[0], getGeneralisedPOS(token[1])))
				transformed.append(tokens)
			tagged_sents = transformed
		
		#Rank candidates:
		c = -1
		lexf = open(victor_corpus)
		for line in lexf:
			c += 1
			data = line.strip().split('\t')
			sent = data[0].strip()
			target = data[1].strip()
			head = int(data[2].strip())
			pos_tags = tagged_sents[c]
			target_pos = pos_tags[head][1]
		
			target_vec = self.getSentVec(sent, head, stop_words, window, onlyInformative, keepTarget, onePerWord, pos_tags)
			candidates = substitution_candidates[c]

			candidate_dists = {}
			for candidate in candidates:
				candidate_vec = self.getWordVec(candidate, target_pos)
				try:
					candidate_dists[candidate] = cosine(candidate_vec, target_vec)
				except ValueError:
					candidate_dists = candidate_dists

			final_candidates = self.getFinalCandidates(candidate_dists, proportion, proportion_type)

			selected_substitutions.append(final_candidates)
		lexf.close()
		return selected_substitutions
		
	def getSentVec(self, sentence, head, stop_words, window, onlyInformative, keepTarget, onePerWord, pos_tokens):
		informative_tags = set([])
		if onlyInformative:
			if self.pos_type=='treebank':
				informative_tags = set(['NN', 'NNS', 'JJ', 'JJS', 'JJR', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'RB', 'RBR', 'RBS'])
			if self.pos_type=='paetzold':
				informative_tags = set(['N', 'V', 'J', 'R'])
		
		tokens = sentence.split(' ')
		
		valid_tokens = []
		if keepTarget:
			valid = tokens[head].strip()
			if self.pos_type!='none':
				valid += '|||' + pos_tokens[head][1]
			valid_tokens.append(valid)
		
		if head>0:
			for i in range(max(0, head-window), head):
				if len(informative_tags)==0 or pos_tokens[i][1].lower().strip() in informative_tags:
					if tokens[i] not in stop_words:
						valid = tokens[i]
						if self.pos_type!='none':
							valid += '|||' + pos_tokens[i][1]
						valid_tokens.append(valid)
		
		if head<len(tokens)-1:
			for i in range(head+1, min(len(tokens), head+1+window)):
				if len(informative_tags)==0 or pos_tokens[i][1].lower().strip() in informative_tags:
					if tokens[i] not in stop_words:
						valid = tokens[i]
						if self.pos_type!='none':
							valid += '|||' + pos_tokens[i][1]
						valid_tokens.append(valid)
						
		if onePerWord:
			valid_tokens = list(set(valid_tokens))
		
		result = np.array([])
		for	token in valid_tokens:
			if len(result)==0:
				try:
					result = self.model[token]
				except Exception:
					pass
			else:
				try:
					result = np.add(result, self.model[token])
				except Exception:
					pass
		result = result/float(len(valid_tokens))
		return result
		
	def getWordVec(self, candidate, target_pos):
		cand = None
		if self.pos_type!='none':
			cand = candidate + '|||' + target_pos
		else:
			cand = candidate

		result = np.array([])
		try:
			result = self.model[cand]
		except Exception:
			pass
		return result
				
	def getFinalCandidates(self, candidate_dists, proportion, proportion_type):
		result = sorted(list(candidate_dists.keys()), key=candidate_dists.__getitem__)
		if proportion_type=='percentage':
			return result[0:max(1, int(proportion*float(len(result))))]
		elif proportion_type=='integer':
			if proportion>=len(result):
				return result
			else:
				return result[0:max(1, int(proportion))]
		else:
			print('Unrecognized proportion type.')
			return result
		
	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()

class WSDSelector:

	def __init__(self, method):
		"""
		Creates an instance of the WSDSelector class.
	
		@param method: Type of Word Sense Disambiguation algorithm to use.
		Options available:
		lesk - Original lesk algorithm.
		path - Path similarity algorithm.
		random - Random sense from WordNet.
		first - First sense from WordNet.
		"""
		
		if method == 'lesk':
			self.WSDfunction = self.getLeskSense
		elif method == 'path':
			self.WSDfunction = self.getPathSense
		elif method == 'random':
			self.WSDfunction = self.getRandomSense
		elif method == 'first':
			self.WSDfunction = self.getFirstSense
		else:
			self.WSDfunction = self.getLeskSense
		
	def selectCandidates(self, substitutions, victor_corpus):
		"""
		Selects which candidates can replace the target complex words in each instance of a VICTOR corpus.
	
		@param substitutions: Candidate substitutions to be filtered.
		It can be in two formats:
		A dictionary produced by a Substitution Generator linking complex words to a set of candidate substitutions.
		Example: substitutions['perched'] = {'sat', 'roosted'}
		A list of candidate substitutions selected for the "victor_corpus" dataset by a Substitution Selector.
		Example: [['sat', 'roosted'], ['easy', 'uncomplicated']]
		@param victor_corpus: Path to a corpus in the VICTOR format.
		For more information about the file's format, refer to the LEXenstein Manual.
		@return: Returns a vector of size N, containing a set of selected substitutions for each instance in the VICTOR corpus.
		"""
		
		selected_substitutions = []

		substitution_candidates = []
		if isinstance(substitutions, list):
			substitution_candidates = substitutions
		elif isinstance(substitutions, dict):
			void = VoidSelector()
			substitution_candidates = void.selectCandidates(substitutions, victor_corpus)
		else:
			print('ERROR: Substitutions are neither a dictionary or a list!')
			return selected_substitutions					

		c = -1
		lexf = open(victor_corpus)
		for line in lexf:
			c += 1
			data = line.strip().split('\t')
			sent = data[0].strip()
			target = data[1].strip()
			head = int(data[2].strip())
		
			target_sense = self.WSDfunction.__call__(sent, target)
		
			candidates = substitution_candidates[c]
		
			selected_candidates = set([])
			for candidate in candidates:
				candidate_sense = None
				try:
					unic = unicode(candidate)
					candidate_sense = self.WSDfunction.__call__(self.getCandidateSentence(sent, candidate, head), candidate)
				except UnicodeDecodeError:
					candidate_sense = None
				if target_sense or not candidate_sense:
					if not candidate_sense or candidate_sense==target_sense:
						selected_candidates.add(candidate)
			selected_substitutions.append(selected_candidates)
		lexf.close()
		return selected_substitutions

	def getLeskSense(self, sentence, target):
		try:
			result = pywsd.lesk.original_lesk(sentence, target)
			return result
		except IndexError:
			return None

	def getPathSense(self, sentence, target):
		try:
			result = pywsd.similarity.max_similarity(sentence, target, option="path", best=False)
			return result
		except IndexError:
			return None
			
	def getRandomSense(self, sentence, target):
		try:
			result = pywsd.baseline.random_sense(target)
			return result
		except IndexError:
			return None
			
	def getFirstSense(self, sentence, target):
		try:
			result = pywsd.baseline.first_sense(target)
			return result
		except IndexError:
			return None
			
	def getMaxLemmaSense(self, sentence, target):
		try:
			result = pywsd.baseline.max_lemma_count(target)
			return result
		except IndexError:
			return None

	def getCandidateSentence(self, sentence, candidate, head):
		tokens = sentence.strip().split(' ')
		result = ''
		for i in range(0, head):
			result += tokens[i] + ' '
		result += candidate + ' '
		for i in range(head+1, len(tokens)):
			result += tokens[i] + ' '
		return result.strip()

	def toVictorFormat(self, victor_corpus, substitutions, output_path, addTargetAsCandidate=False):
		"""
		Saves a set of selected substitutions in a file in VICTOR format.
	
		@param victor_corpus: Path to the corpus in the VICTOR format to which the substitutions were selected.
		@param substitutions: The vector of substitutions selected for the VICTOR corpus.
		@param output_path: The path in which to save the resulting VICTOR corpus.
		@param addTargetAsCandidate: If True, adds the target complex word of each instance as a candidate substitution.
		"""
		o = open(output_path, 'w')
		f = open(victor_corpus)
		for subs in substitutions:
			data = f.readline().strip().split('\t')
			sentence = data[0].strip()
			target = data[1].strip()
			head = data[2].strip()
			
			newline = sentence + '\t' + target + '\t' + head + '\t'
			for sub in subs:
				newline += '0:'+sub + '\t'
			o.write(newline.strip() + '\n')
		f.close()
		o.close()
