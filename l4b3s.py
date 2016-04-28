#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import mode #handling empty values

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


"""
	Read the csv data and returns dataframes objs
"""
def read_data(train_file, test_file):

	directory = os.getcwd()

	directory = directory + '/inputs/'

	#read the files
	train   = pd.read_csv(directory + train_file)
	test    = pd.read_csv(directory + test_file)

	#return the files as pandas dataframes
	return train, test



"""
	Split train data into train and validation to check the model
"""
def cross_validation_data(train):

	#create crossvalidation dataset using just the train.csv
	cross_train = train.sample(frac = 0.7, random_state = 1)	   #split it in 70% of train
	cross_valid  = train.loc[~train.index.isin(cross_train.index)]  #and 30%

	return cross_train, cross_valid




"""
	Given a dataframe isolate the target and return the dataframe without the targert attribute
"""
def get_target(dataframe):

	#isolate the target from cross_train and cross_valid datasets
	target = dataframe.target

	#delete the target information from the tables
	dataframe = dataframe.drop(['target'],axis=1)
	
	#return updated dataframe and target info
	return dataframe, target




"""
	Get train and test dataframes and make some pre_processing to the data
"""
def pre_processing(train, test, columns_attr):
	
	#duplicated
	#test  = test.drop(['v107'],axis=1)
	#train = train.drop(['v107'],axis=1)

	#get columns 
	columns = list(train.columns.values)

	#categorical atributes
	cat_cols = ['v3','v22','v24','v30','v31','v47','v52','v56','v66','v71','v74','v75','v79','v91','v107','v110','v112','v113','v125']

	for c in columns:

		#print 'handling column: ', c

		#avoid column target  processing
		if (not c in cat_cols) and (c != 'target'):

			#get the average and replace empty entrys
			#print 'Average: ', train[c].mean()
			mean = train[c].mean() 
			train[c].fillna(mean, inplace=True)
			test[c].fillna(mean, inplace=True)

		#avoid column target  processing
		elif (c != 'target'):

			#get the moda and replace empty entrys
			#print 'Moda: ',mode(train[c])[0][0]
			moda = mode(train[c])[0][0]
			train[c].fillna(moda, inplace=True)
			test[c].fillna(moda, inplace=True)


	#handling categorical columns.. 
	for cat in cat_cols:

		#possible values
		values = list(train[cat].unique())

		dict_cols = {}
		
		for i in range(0, len(values)):
			
			#extra columns names
			dict_cols[values[i]] = i

		
		#define the max value of columns atributtes that will be binarized	
		if len(dict_cols.keys()) <= columns_attr:

			#train[cat] = coding(train[cat], dict_cols)
			#test[cat] = coding(test[cat], dict_cols)

			dummies = pd.get_dummies(train[cat])

			#print dummies
			
			#train = train.append(dummies)
			#test  = test.append(dummies)

		#after that remove the categorical attribute
		test  = test.drop([cat],axis=1)
		train = train.drop([cat],axis=1)


	#return categorized train and test
	return train, test



"""
	Create a submission file in kaggle model
"""
def create_submission(filename, probability):

	directory = os.getcwd()

	#get the samples as example
	sample_path = '/inputs/sample_submission.csv'

	directory = directory + sample_path

	#create dataframe using sample as example
	submission = pd.read_csv(directory)
	
	#define index column
	submission.index = submission.ID
	
	#set the values to predictedProb column
	submission.PredictedProb=probability[:,1]

	#create a csv file
	result = open(filename,'wb')
	
	#convert dataframe to csv
	submission.to_csv(result, index=False)
	
	#plot a histo..
	submission.PredictedProb.hist(bins=30)

	print 'Finish the pre_processing..'


"""
	This method is used to verify the acuracy of the model
"""
def cross_validation_score(alg, train):

	#create the data to make crossvalidation
	cross_train, cross_valid = cross_validation_data(train)

	cross_train, cross_train_target = get_target(cross_train)
	cross_valid, cross_valid_target = get_target(cross_valid)

	#fit the model to the algorithm
	alg.fit(cross_train,cross_train_target) #trainnig

	#predict
	alg.predict(cross_valid)

	#check the score
	score = alg.score(cross_valid, cross_valid_target)

	return score


def valid_model(alg, train, train_target, test):

	#fit the model to the algorithm
	alg.fit(train,target) #trainnig

	#predict
	alg.predict(test)

	#check the score
	#print alg.score(cross_test, cross_valid_target)

	alg_probability = alg.predict_proba(test)

	#return the prediction of the algorithm after the trainnig section
	return alg_probability	


"""
	This method will use the base model to create the neighboorhood
"""
def init_function(current_model, candidates):

	print 'Candidates in this iteration: ', candidates
	print 'Size of the list: ', len(candidates)

	neighbors = []

	for c in candidates:

		try:
			neighbor = current_model.drop([c],axis=1)

			#create 
			neighbors.append([neighbor, c])
		except Exception, e:
			pass
		

	return neighbors



"""
	Simple Hill climb implementation..
"""
def hillclimb(alg, init_function, current_model, cross_validation_score, max_evaluations):

	#possible attributes to be deleted
	candidates = ["v46", "v11", "v76", "v63", "v64", "v54", "v89", "v105", "v60",
				"v96", "v83", "v114", "v116", "v106", "v89", "v100", "v76", "v115",
				"v121", "v95", "v118", "v128"]

	removed = []

	neighbor_list = init_function(current_model, candidates)  #original
	
	best_score = cross_validation_score(alg, current_model) #get the best score to original model

	num_evaluations = 1

	while num_evaluations < max_evaluations:

		info = 'I: ============== Running Hill Climbing. Evaluation: %d ============== \n' % (num_evaluations)

		print info

		# examine moves around our current position
		move_made = False

		i = 0 #index

		while i < len(neighbor_list):

			print 'I: Iteration: ', i

			#for neighbor in neighbor_list:
			neighbor  = neighbor_list[i][0]

			#update cadidate attributes to be deleted
			candidate = neighbor_list[i][1]
			
			print 'I: Neighbor generated using attr: ', candidate	

			if num_evaluations >= max_evaluations:
				break

			# see if this move is better than the current
			next_score = cross_validation_score(alg,neighbor)

			print 'I: Score produced: %f \n' % next_score

			if next_score > best_score:


				#Update the value of the best solution
				current_model = neighbor
				best_score = next_score
				move_made = True


				#remove the candidate that generates a better solution
				candidates.remove(candidate)


				removed.append(candidate)

				print 'I: Find a better solution!'

				print 'I: Number of evaluations until the moment: ', num_evaluations
				print 'I: Best score: ', best_score
				print 'I: Removed: ', removed
				print 'I: Remain: ', candidates

				break # depth first search

			i = i + 1 #increase the loop index

		#Increase the number of evaluations
		num_evaluations += 1

		#Run a new search to get the newest neighborhood
		neighbor_list = init_function(current_model, candidates)

	#return number of evaluations, the best score and the beste model solution
	return num_evaluations, best_score, current_model, removed


"""
	Remove the better combination found on hillclimb
"""
def best_choice(train, test):
	
	test  = test.drop(['v11'],axis=1)
	train = train.drop(['v11'],axis=1)

	test  = test.drop(['v46'],axis=1)
	train = train.drop(['v46'],axis=1)

	test  = test.drop(['v76'],axis=1)
	train = train.drop(['v76'],axis=1)

	return train, test



if __name__ == '__main__':
	

	#classifiers
	knn = KNeighborsClassifier()
	tree = DecisionTreeClassifier()
	gnb = GaussianNB()
	rf = RandomForestClassifier(n_estimators=100)

	#file names
	test_file  = 'test.csv'
	train_file = 'train2.csv'

	#get dataframe objects
	train, test = read_data(train_file, test_file)


	#pre_processing
	columns_attr = 5 #number of max values in a categorical attribute to be converted in binary
	train, test = pre_processing(train, test, columns_attr)

	#crossvalidation
	#score = cross_validation_score(rf, train)

	#print score

	#hillclimbing
	#max_evaluations = 25
	#num_evaluations, best_score, current_model, removed = hillclimb(rf, init_function, train, cross_validation_score, max_evaluations)

	#print 'End of hillclimb algorithm'

	#print 'Number of evaluations: ', num_evaluations
	#print 'Best score: ', best_score
	#print 'Removed: ', removed

	#get targets from dataframes
	train, target = get_target(train)

	#best choice
	train, test = best_choice(train, test)

	#get probability
	prob = valid_model(rf, train, target, test)

	#name of the result file
	result = 'submission.csv'

	#create csv to submit
	create_submission(result, prob)
