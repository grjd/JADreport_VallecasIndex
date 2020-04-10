#######################################################
# Python program name	: 
# Description	: scirep_sub_v2.py 
# Args          : Code for "Selecting the most important self-assessed 
#				features for predicting conversion to Mild Cognitive                                                                                      
#				Impairment with Random Forest and Permutation-based methods"
# Author       	: Jaime Gomez-Ramirez                                               
# Email         : jd.gomezramirez@gmail.com 
# Repo          : https://github.com/grjd/VallecasIndex
#######################################################
# -*- coding: utf-8 -*-
import os, sys, pdb, operator
import time
import numpy as np
import pandas as pd
import importlib
import sys
import statsmodels.api as sm
import time
import random
import pickle
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from pprint import pprint
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score,\
confusion_matrix,roc_auc_score, roc_curve, auc, classification_report,precision_recall_curve,\
make_scorer, average_precision_score


# Set upo directory for saving figurt
figures_dir = ''

def redirect_to_file(text, file):
	"""
	"""
	original = sys.stdout
	sys.stdout = open(file, 'a')
	#print('This is your redirected text:')
	print(text)
	sys.stdout = original
    #print('This string goes to stdout, NOT the file!')

def select_rows_all_visits(dataframe, visits):
	"""select_rows_all_visits
	Args:
	Output: df with the rows of subjects with all visits
	"""
	
	#df2,df3,df4,df5,df6 = dataframe[['tpo1.2']].notnull(),dataframe[['tpo1.3']].notnull(),dataframe[['tpo1.4']].notnull(),dataframe[['tpo1.5']].notnull(),dataframe[['tpo1.6']].notnull()
	#print('Visits per year 2..6:',np.sum(df2)[0], np.sum(df3)[0],np.sum(df4)[0],np.sum(df5)[0],np.sum(df6)[0])
	df_loyals = dataframe[visits].notnull()
	#y4 749 visits[:-2] y5 668 visits[:-1]

	rows_loyals = df_loyals.all(axis=1)
	return dataframe[rows_loyals]

def remove_features_lowvariance(df):
	"""remove_features_lowvariance: Feature selector that removes all low-variance features.
	https://scikit-learn.org/stable/modules/feature_selection.html
	Output return index list of features with low variance
	""" 
	from sklearn.feature_selection import VarianceThreshold
	#remove all features that are either one or zero (on or off) in more than 80% of the samples.
	sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	sel.fit_transform(df)
	selected_features = df.columns[sel.get_support()]
	removed_features = df.columns[~sel.get_support()]
	print('REMOVED Features for Low variance: ', removed_features)
	return removed_features


def plot_wrapper_fig(rfecv, figname, estimator_name=None):
	"""
	"""
	figures_dir = '/Users/jaime/github/papers/JADreport_VallecasIndex/figures/'
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score (nb of correct classifications)")
	plt.title('Feature importance Wrapper:' + estimator_name)
	plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	plt.savefig(os.path.join(figures_dir, figname), dpi=240,bbox_inches='tight')


def feature_selection_wrapping(X, y, nboffeats=None):
	"""feature_selection_wrapping select important features RFE using 
	machine learnign and a evaluation metric.
	Create the RFE object and compute a cross-validated score.
	"""
	#from iterimport tools import product
	import itertools as it
	from sklearn.svm import SVC
	from sklearn.feature_selection import RFECV
	from sklearn.linear_model import LogisticRegression

	#figures_dir = '/Users/jaime/github/papers/JADr_vallecasindex/figures'
	# C smaller values specify stronger regularization
	model_dict = {'SVC': {'C': np.array([ 0.01, 0.1, 1, 10]), 'kernel':['linear'], \
	'scoring':['accuracy', 'recall', 'f1', 'roc_auc']}, \
	'LogisticRegression':{'C': np.array([ 0.01, 0.1, 1, 10]),\
	'class_weight' : [None], 'penalty': ['l1', 'l2'], 'solver':['liblinear'], \
	'scoring': ['accuracy', 'recall', 'f1', 'roc_auc']}} 
	#Logistic Regression supports only penalties in ['l1', 'l2'], not elasticnet

	keys = model_dict.keys()
	for keymodel in keys:
		dictio = model_dict[keymodel]
		allNames = sorted(dictio)
		combinations = list(it.product(*(dictio[Name] for Name in allNames)))
		print(combinations)
		if keymodel == 'SVC':
			for hyperpar in combinations:
				print('Building SVC model for hyperparameters', hyperpar)
				C = hyperpar[0]; kernel = hyperpar[1]; scoring = hyperpar[2]
				svc = SVC(C=C, kernel=kernel)
				print('Calling to RFECV for Feature ranking with recursive feature elimination and cross-validated selection of the best number of features . \n')
				rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring=scoring, n_jobs=5, verbose=0)
				print('Estimator x-validation built. Fitting the estimator....\n')
				start = time.time()
				rfecv.fit(X, y)
				end = time.time()
				print('RFE Fitting time was=', end - start)
				print("\n Optimal number of features : %d" % rfecv.n_features_)
				print("\n Support of features : %s" % X.columns[rfecv.support_])
				print("\n Ranking of features :%s  %s" % (X.columns, rfecv.ranking_))
				for index, item in enumerate(X.columns):
					print("Feature: %s, Ranked in importance %d:" %(item,rfecv.ranking_[index]))
				# Plot number of features VS. cross-validation scores
				figname = 'Wrapper_scv' + 'C_' + str(C) + 'kernel_' + str(kernel) + '_' + scoring + '.png'
				plot_wrapper_fig(rfecv, figname, rfecv.estimator.__class__.__name__)

		elif keymodel == 'LogisticRegression':
			for hyperpar in combinations:
				print('Building LogReg model for hyperparameters', hyperpar)
				C = hyperpar[0]; class_weight = hyperpar[1]; penalty = hyperpar[2]; \
				scoring = hyperpar[3]; solver = hyperpar[4]; 
				logreg = LogisticRegression(penalty=penalty, C= C, class_weight=class_weight, random_state=0, solver=solver)
				rfecv = RFECV(estimator=logreg, step=1, cv=StratifiedKFold(3), scoring=scoring, n_jobs=6, verbose=0)
				rfecv.fit(X, y)
				print("\n Optimal number of features : %d" % rfecv.n_features_)
				print("\n Support of features : %s" % X.columns[rfecv.support_])
				print("\n Ranking of features :%s  %s" % (X.columns, rfecv.ranking_))
				for index, item in enumerate(X.columns):
					print("Feature: %s, Ranked in importance %d:" %(item,rfecv.ranking_[index]))
				# Plot number of features VS. cross-validation scores
				if class_weight is None:
					class_weight = 'None'
				figname = 'Wrapper_Log' + '_class_weight_' + class_weight + '_pena_' + penalty + '_C_' + str(C) + '_solver_' + str(solver) + '_' + scoring + '.png'
				plot_wrapper_fig(rfecv, figname, rfecv.estimator.__class__.__name__)
		pdb.set_trace()	


def feature_selection_embedding_lasso(X,y):
	"""Select features with Lasso regularization
	"""
	from sklearn.feature_selection import SelectFromModel
	from sklearn.linear_model import LassoCV
	from sklearn.svm import LinearSVC

	print('Regularization -Lasso- for Feature Selection \n\n')
	# Use the base estimator LassoCV, the L1 norm promotes sparsity of features.
	clf = LassoCV(cv=5)
	# Set a minimum threshold of 0.25
	threshold = 0.25
	# SelectFromModel is a meta-transformer that can be used along 
	# with any estimator that has a coef_ or feature_importances_ attribute after fitting
	sfm = SelectFromModel(clf, threshold=threshold)
	sfm.fit(X, y)
	n_features = sfm.transform(X).shape[1]
	print("Number of features == %d " % (n_features))
	
	lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, verbose=1).fit(X, y)
	#lsvc = LogisticRegression(C=0.01, penalty="l1", dual=False, verbose=1).fit(X, y)
	model = SelectFromModel(lsvc, prefit=True, max_features=14)
	X_new = model.transform(X)
	n_features = X_new.shape[1]
	feature_idx = model.get_support()
	feature_name = X.columns[feature_idx]
	#With SVMs and logistic-regression, the parameter C controls the sparsity: 
	#the smaller C the fewer features selected. 
	#With Lasso, the higher the alpha parameter, the fewer features selected.
	print('LinearCSV L1 removed features :', X.shape,' to ', X_new.shape)
	#print('Feature Importance::',lsvc.feature_importances_)
	#print('Coef Importance::',lsvc.coef)
	print('Estimator parameters ::', model.get_params())
	print('Feature names selected ::', feature_name)
	return {'n_features': n_features, 'feature_name': feature_name, 'feature_idx':feature_idx}
	#plot_Lasso_fig(rfecv, figname, rfecv.estimator.__class__.__name__)


def plot_rf_importance(rf, X, y, image_name=None):
	"""https://github.com/parrt/random-forest-importances/blob/master/notebooks/pimp_plots.ipynb
	"""
	#figures_dir = '/Users/jaime/github/papers/JADreport_VallecasIndex/figures/nonrandomAPOE/'
	Imp = importances(rf, X, y)
	viz = plot_importances(Imp)
	viz.save(os.path.join(figures_dir, 'Vanilla_importances_' + image_name + '.svg'))
	viz = stemplot_importances(Imp, vscale=.7)
	viz.save(os.path.join(figures_dir, 'Stem_importances.svg'))
	
	Imp_df = pd.DataFrame()
	Imp_df['Feature'] = X.columns
	Imp_df['Importance'] = rf.feature_importances_
	Imp_df = Imp_df.sort_values('Importance', ascending=False)
	Imp_df = Imp_df.set_index('Feature')
	viz = plot_importances(Imp_df,title="Feature importance via average gini/variance drop (sklearn)")
	viz.save(os.path.join(figures_dir, 'Gini_importances_' + image_name + '.svg'))
	#width=6,color='#FDDB7D',bgcolor='#F1F8FE'
	
	# Permutation Importance
	Imp_drop = oob_dropcol_importances(rf, X, y)
	viz = plot_importances(Imp_drop, title="Drop column importance using OOB score")
	viz.save(os.path.join(figures_dir, 'Permuta_Drop_OOB_importances_' + image_name + '.svg'))


def visualize_DTfromRF(forest, feature_names, target_names):
	"""
	"""
	from sklearn.tree import export_graphviz
	from subprocess import call
	import pydot
	#figures_dir = '/Users/jaime/github/papers/JADr_vallecasindex/figures/'
	# Extract single tree
	estimator = forest.estimators_[6]
	# Export as dot file
	export_graphviz(estimator, out_file='tree.dot',feature_names = feature_names, class_names = target_names, rounded = True, proportion = False, precision = 2, filled = True)
	# Convert to png using system command (requires Graphviz)
	print('Saving tree.dot....')
	#call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
	# Use dot file to create a graph
	(graph, ) = pydot.graph_from_dot_file('tree.dot')

	graph.write_png(os.path.join(figures_dir,'tree_allfeatures.png'))


def plot_rf_score(X,forest, nboffeats):
	""" plot_rf_score Plot the feature importances of the forest
	OUTDATED. REMOVE FUNCTION
	"""

	#figures_dir = '/Users/jaime/github/papers/JADr_vallecasindex/figures/'

	criterion = forest.get_params()['criterion']
	n_estimators = forest.get_params()['n_estimators']
	class_weight = forest.get_params()['class_weight']
	if class_weight is None: class_weight = 'None'
	class_name = forest.__class__.__name__
	importances = forest.feature_importances_ 
	std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
	#####
	importances = importances*std
	#####
	indices = np.argsort(importances)[::-1]
	indices_imp = indices[:nboffeats]
	columns_imp = X.columns[indices_imp].tolist()
	importances_imp = importances[indices_imp]
	print('Most important features :', columns_imp) 
	# Plot most important features
	figname = class_name + '_criterion_' + criterion + '_N_' + str(n_estimators) + '_weight_'+ class_weight +'.png'
	plt.figure(figsize=(9,8))
	plt.title("Feature importances:" + class_name)
	#plt.bar(range(X.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
	#plt.bar(range(X[columns_imp].shape[1]), importances_imp, color="r", yerr=std[indices][indices_imp], align="center")
	plt.bar(range(X[columns_imp].shape[1]), importances_imp, color="r", align="center")
	plt.xticks(range(X[columns_imp].shape[1]), columns_imp, fontsize=10)
	plt.xlim([-1, X[columns_imp].shape[1]])
	plt.xticks(rotation=90)
	plt.savefig(os.path.join(figures_dir, figname), dpi=240,bbox_inches='tight')
	#plt.yticks(rotation=0)

	#plt.savefig(os.path.join(figures_dir, figname), dpi=240,bbox_inches='tight')
	return [importances, indices_imp, columns_imp, importances_imp, std]


def plot_dummy_vs_classifier(model_scores, dummy_scores):
	"""model_scores dict {'RF':score} dummy_scores = dict{'':, '':,...}
	"""

	fig, ax = plt.subplots()
	figname = 'dummies_vs_rf.png'
	labels = list(model_scores.keys()) + list(dummy_scores.keys())
	scores = list(model_scores.values()) + list(dummy_scores.values())
	x = np.arange(len(scores))
	barlist = plt.bar(x,scores)
	barlist[0].set_color('r') 
	plt.xticks(x, labels)
	plt.ylabel('score')
	plt.ylabel('RF accuracy test score vs dummy estimators')
	plt.savefig(os.path.join(figures_dir, figname), dpi=240,bbox_inches='tight')


def build_dummy_scores(X_train, y_train, X_test, y_test):
	""" build_dummy_scores: When doing supervised learning, a simple sanity check consists of 
	comparing one's estimator against simple rules of thumb. 
	DummyClassifier implements such strategies(stratified most_frequent, prior, uniform, constant). Used for imbalanced datasets
	Args: X_train, X_test, y_train, y_test
	Outputs: dict of dummy estimators"""
	from sklearn.dummy import DummyClassifier
	dummy_strategy = ['uniform', 'constant']
	dummies =[]; dummy_scores =[]
	for strat in dummy_strategy:
		if strat is 'constant':
			for const in range(0,2):
				estimator_dummy = DummyClassifier(strategy='constant', random_state=0, constant=const)
				estimator_dummy = estimator_dummy.fit(X_train, y_train)
				dummies.append(estimator_dummy)
				dscore = estimator_dummy.score(X_test, y_test)
				dummy_scores.append(dscore)
				print("Score of Dummy {}={} estimator={}".format(strat, const,dscore ))
		else:
			estimator_dummy = DummyClassifier(strategy=strat, random_state=0)
			estimator_dummy = estimator_dummy.fit(X_train, y_train)
			dummies.append(estimator_dummy)
			dscore = estimator_dummy.score(X_test, y_test)
			dummy_scores.append(dscore)
			print("Score of Dummy {} estimator={}".format(strat, dscore))
	dict_dummy_scores = {'uniform':dummy_scores[0] , 'constant0':dummy_scores[1],'constant1':dummy_scores[2]}
	return dict_dummy_scores

def feature_selection_embedding(X, y, nboffeats):
	"""feature_selection_embedding: select features wiuth random forest |Lasso
	"""
	from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
	import itertools as it
	# Regularization
	################
	dict_Lasso = feature_selection_embedding_lasso(X, y)
	print('\n\n\n', 'Number of features=', dict_Lasso['n_features'], 'Feature Names:',\
		dict_Lasso['feature_name'], ' Feature indices:',dict_Lasso['feature_idx'], '\n\n\n')

	# Random Forest
	###############
	# Add  random  feature
	#X['random'] = np.random.random(size=len(X))
	#https://stackoverflow.com/questions/22409855/randomforestclassifier-vs-extratreesclassifier-in-scikit-learn
	print('\n\n RF for Feature Selection::ExtraTreesClassifier or RandomForestClassifier')
	model_dict = {'ExtraTreesClassifier':{'criterion':['gini', 'entropy'], \
	'n_estimators':[100,1000,10000,100000], 'class_weight':[None, 'balanced']},\
	'RandomForestClassifier':{'criterion':[ 'gini', 'entropy'], 'n_estimators':[100,1000,10000,100000,1000000],\
	'class_weight':[None, 'balanced']}}
	model_dict = {'RandomForestClassifier':{'criterion':[ 'gini'], 'n_estimators':[1000],\
	'class_weight':['balanced']}}
	keys = model_dict.keys()
	for keymodel in keys:
		dictio = model_dict[keymodel]
		allNames = sorted(dictio)
		combinations = list(it.product(*(dictio[Name] for Name in allNames)))
		print(combinations)
		if keymodel == 'ExtraTreesClassifier':
			for hyperpar in combinations:
				print('Building ExtraTreesClassifier model for hyperparameters', hyperpar)
		elif keymodel == 'RandomForestClassifier':
			for hyperpar in combinations:
				print('Building RandomForestClassifier model for hyperparameters', hyperpar)
				n_estimators = hyperpar[2]; criterion = hyperpar[1]; class_weight = hyperpar[0];
				max_depth = hyperpar[-1]
				forest = RandomForestClassifier(n_estimators=n_estimators,max_depth =8, bootstrap=True,\
					oob_score=True,criterion=criterion, class_weight=class_weight, random_state=0, n_jobs=6,verbose=2)
				forest = forest.fit(X, y)
				# rf score for importance/std
				[importances, indices_imp, columns_imp, importances_imp, std] = plot_rf_score(X,y,forest, nboffeats)
				forest_redux = important_forest(X[columns_imp[0:2]], y,n_estimators,criterion,class_weight)
				#visualize_DTfromRF(forest_redux,X[columns_imp[0:3]].columns ,y.name)
				

				pdb.set_trace()
				#oob_c = permutation_rf(X,y,forest)
				if class_weight is None:
					class_weight='None'
				image_name = str(n_estimators) + criterion + class_weight
				#imp = permutation_importances(forest, X, y, oob_classifier_accuracy, image_name)
				print('Visualizing DT from RF...')
				forest_redux = important_forest(X[columns_imp], y,n_estimators,criterion,class_weight)
				visualize_DTfromRF(forest_redux,X[columns_imp].columns ,y.name)
				pdb.set_trace()
	
def ROC_Curve(rf, auc,X_train,X_test,y_train,y_test):
	from sklearn.preprocessing import OneHotEncoder

	#figures_dir = '/Users/jaime/github/papers/JADr_vallecasindex/figures/figurescv5/'
	one_hot_encoder = OneHotEncoder()
	rf_fit = rf.fit(X_train, y_train)
	#fit = one_hot_encoder.fit(rf.apply(X_train))
	y_predicted = rf_fit.predict_proba(X_test)[:, 1]
	false_positive, true_positive, _ = roc_curve(y_test, y_predicted)

	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(false_positive, true_positive, color='darkorange', label='Random Forest')
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	#plt.title('ROC curve (area = %0.2f)' % auc)
	plt.legend(loc='best')
	figname='roc_curve.png'
	plt.savefig(os.path.join(figures_dir, figname), dpi=240,bbox_inches='tight')
	plt.show()


def important_forest(X, y,n_estimators,criterion,class_weight):
	"""
	"""	

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)
	forest = RandomForestClassifier(n_estimators=n_estimators, max_depth =4, bootstrap=True,oob_score=True,criterion=criterion, class_weight=class_weight,random_state=0, n_jobs=6,verbose=2)
	forest = forest.fit(X_train, y_train)

	y_pred = forest.predict(X_test)
	print(len(X_test))
	print(accuracy_score(y_test,y_pred))
	print(recall_score(y_test,y_pred))
	print(confusion_matrix(y_test,y_pred))
	#Plot ROC
	best_model = forest[90]
	print('\nbest_model:\n', best_model)
	y_predicted = best_model.predict(X_train)
	y_predicted_train = best_model.predict(X_train)
	cm = confusion_matrix(y_train, y_predicted_train)
	auc = roc_auc_score(y_train, y_predicted_train)
	ROC_Curve(best_model, auc, X_train,X_test,y_train,y_test)
	pdb.set_trace()
	return forest

def shap_explanations(model, X_test):
	"""shap_explanations SHAP (SHapley Additive exPlanations) to explain the output of machine learning model.
	 SHAP values represent a feature's responsibility for a change in the model output
	 Args: model, X
	 Output: shap values and figures
	"""
	import shap
	from itertools import combinations 

	features = list(X_test.columns)
	row_to_show = 2
	data_for_prediction = X_test.iloc[row_to_show] 
	data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
	model.predict_proba(data_for_prediction_array)

	plt.figure()
	# shap.initjs()
	explainer = shap.TreeExplainer(model)
	shap_values = explainer.shap_values(X_test)
	#shap_values[0];shap_values[0]
	fig0 = shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], matplotlib=True, link="logit")
	plt.savefig(os.path.join(figures_dir, 'shap_vals_force.png'))
	# visualize the training set predictions
	#plt.figure()
	#shap.force_plot(explainer.expected_value, shap_values, X_test)
	#plt.savefig(os.path.join(figures_dir, 'shap_vals_force_training.png'))
	
	#plt.figure()
	#fig00 = shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, link="logit")
	#plt.savefig(os.path.join(figures_dir, 'shap_vals_force_allins.png'), dpi=240,bbox_inches='tight')

	# Summary plots .shap_values[1] to get the SHAP values for the prediction of "True" or MCIconverter.
	plt.figure()
	fig = shap.summary_plot(shap_values[1], X_test)
	plt.savefig(os.path.join(figures_dir, 'shap_vals_mostimp.png'), dpi=240,bbox_inches='tight')
	#fig3 = shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
	plt.figure()
	fig3 = shap.summary_plot(shap_values, X_test)
	plt.savefig(os.path.join(figures_dir, 'shap_vals_bar.png'), dpi=240,bbox_inches='tight')
	plt.figure()
	fig34 = shap.summary_plot(shap_values, X_test, plot_type="bar")
	plt.savefig(os.path.join(figures_dir, 'shap_vals_bar2.png'), dpi=240,bbox_inches='tight')
	
	# SHAP Dependence Contribution Plots, Delve into single features Plot dependence of 2 features 0 and -1
	pairs = combinations(np.arange(0,len(features)),2)
	for pair in pairs:
		# fig2 = shap.dependence_plot(features[pair[0]], shap_values[1],X_test,interaction_index=features[pair[1]])
		# figname = 'shap_feat_' + str(pair[0]) + str(pair[1]) + '_vals.png'
		# plt.savefig(os.path.join(figures_dir, figname), bbox_inches='tight')
		fig2 = shap.dependence_plot(features[pair[0]], shap_values[1],X_test,interaction_index=features[pair[1]])
		figname = 'shap_feat_' + str(pair[0]) + str(pair[1]) + '_vals.png'
		plt.savefig(os.path.join(figures_dir, figname), dpi=240,bbox_inches='tight')
	return shap_values

def partial_dependence_plots(model, X_train):
	"""partial_dependence_plots creates pdp_plot and pdp_plot_pairs .pngs
	"""
	from pdpbox import pdp, get_dataset, info_plots
	from itertools import combinations
	from sklearn import tree
	import graphviz


	feature_names = list(X_train.columns)
	feature_names_inpairs = list(combinations(feature_names,2))
	# Plot a tree  https://stackoverflow.com/questions/43372723/how-to-open-dot-on-mac
	#dot -Tpng DocName.dot -o DocName.png
	#visualize_DTfromRF(model, feature_names, label_names)
	tree_graph = tree.export_graphviz(model.estimators_[7], out_file=os.path.join(figures_dir, 'tree_example1.dot'), feature_names=feature_names,rounded = True, proportion = False, precision = 2, filled = True)
	graphviz.Source(tree_graph)
	tree_graph = tree.export_graphviz(model.estimators_[8], out_file=os.path.join(figures_dir, 'tree_example2.dot'), feature_names=feature_names,rounded = True, proportion = False, precision = 2, filled = True)
	graphviz.Source(tree_graph)
	tree_graph = tree.export_graphviz(model.estimators_[9], out_file=os.path.join(figures_dir, 'tree_example3.dot'), feature_names=feature_names,rounded = True, proportion = False, precision = 2, filled = True)
	graphviz.Source(tree_graph)
	#  Partial Dependence Plot using the PDPBox library
	for feature in feature_names:
		figname = 'pdp_plot_' + str(feature) + '.png'
		print('pdp.pdp_isolate for:%s' %feature)
		
		pdp_goals = pdp.pdp_isolate(model=model, dataset=X_train, model_features=feature_names, feature=feature)
		fig, axes = pdp.pdp_plot(pdp_goals, feature)
		fig.savefig(os.path.join(figures_dir, figname), dpi=240,bbox_inches='tight')

	#Commented because pdp_interact_plot 2D plots has a bug  Bug here https://github.com/SauceCat/PDPbox/issues/37	
	for pair in feature_names_inpairs:
		figname = 'pdp_plot_pair_' + str(pair) + '.png'
		inter1  =  pdp.pdp_interact(model=model, dataset=X_train, model_features=feature_names, features=pair)
		fig, axes = pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=pair, plot_type='contour', plot_pdp=True)
		fig.savefig(os.path.join(figures_dir, figname), dpi=240,bbox_inches='tight')


def permutation_importance_eli5(model,X_test, y_test ):
	"""feature importance with permutation using eli5 library
	create permutimp_allfeats.html
	Args: model, X,y 
	Output: sklearn.PermutationImportance object
	"""
	import eli5
	from eli5.sklearn import PermutationImportance
	from eli5.permutation_importance import get_score_importances
	from IPython.display import display, HTML

	predictions = model.predict(X_test)
	score_func = metrics.accuracy_score(y_test.tolist(), predictions.tolist())
	# base_score, score_decreases = get_score_importances(accuracy(y_test, predictions), X_test, y_test)
	# feature_importances = np.mean(score_decreases, axis=0)

	figname = 'permutimp_allfeats.html'
	perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)
	image = eli5.show_weights(perm, feature_names = X_test.columns.tolist())
	html = image.data
	with open(os.path.join(figures_dir, figname), 'w') as f:
		f.write(html)
	print('HTML with permutation for fitted model at figname:%s' %figname)
	return perm

def permutation_importance_eli5_tofit(model,X_train, y_train, X_test, y_test ):
	"""permutation_importance_eli5_tofit PermutationImportance without previous Fit
	Args: model, X,y 2x train and test 
	Output: sklearn.PermutationImportance object
	"""
	# Feature Imnportance without previous Fit
	import eli5
	from eli5.sklearn import PermutationImportance
	from sklearn.feature_selection import SelectFromModel
	perm = PermutationImportance(model, cv=7, random_state=12)
	perm.fit(X_train, y_train)
	image = eli5.show_weights(perm,feature_names = X_test.columns.tolist())
	html = image.data
	figname = 'permutimp_allfeats_prefit.html'
	with open(os.path.join(figures_dir, figname), 'w') as f:
		f.write(html)
	print('HTML with permutation for fitted model at figname:%s' %figname)
	print('Important Features are:')
	pprint(perm.feature_importances_ )
	# Select features with increase in accuracy at least 0.05
	sel = SelectFromModel(perm, threshold=0.001, prefit=True)
	feature_idx = sel.get_support()
	feature_name = X_train.columns[feature_idx]
	print('Important features above thr::%s' % feature_name)
	X_trans = sel.transform(X_train)
	return sel

def permutation_importances(rf, X_train, y_train, metric, image_name):
	"""feature importance with permutation
	https://explained.ai/rf-importance/index.html
	does not normalize the importance values, such as dividing by the standard deviation
	https://github.com/parrt/random-forest-importances
	"""

	baseline = metric(rf, X_train, y_train)
	print('The baseline for permutation is == %.5f' % (baseline))
	imp = []
	for col in X_train.columns:
		save = X_train[col].copy()
		X_train[col] = np.random.permutation(X_train[col])
		m = metric(rf, X_train, y_train)
		X_train[col] = save
		imp.append(baseline - m)
		print('Feature %s improved by %.5f' %(col, baseline - m))
	print('Calling to plot importance rfpimp Library ')
	plot_rf_importance(rf, X_train, y_train, image_name)	
	return np.array(imp) 


 
def feature_selection_filtering(X, y, nboffeats):
	"""feature_selection_filtering: SlectKBest features
	Args:X,y, nboffeats number of features
	Output:selector returned object SelectKBest and plot 2 figures in figures_dir
	"""
	from sklearn import preprocessing
	#from sklearn.preprocessing import Imputer
	predictors = X.columns.tolist()
	#['f_classif', 'mutual_info_classif', 'chi2', 'f_regression', 'mutual_info_regression']
	selector = SelectKBest(chi2, k=nboffeats).fit(X, y)
	# scores_ : array-like, shape=(n_features,) pvalues_ : array-like, shape=(n_features,)
	top_indices = np.nan_to_num(selector.scores_).argsort()[-nboffeats:][::-1]
	print("Selector {} scores:",nboffeats, selector.scores_[top_indices])
	print("Top features:\n", X.columns[top_indices])

	#Plot heatmaps
	# figures_dir = '/Users/jaime/github/papers/JADr_vallecasindex/figures/nonrandomAPOE'
	f_name = selector.get_params()['score_func'].__name__
	figname = 'ENG_HeatmapF_' + f_name + '.png'
	plt.figure(figsize=(18, 16))
	#corr = dataframe[predictors].corr()
	#corr =dataframe[predictors].join(y).corr()
	#corr = X.join(y).corr()
	corr = X.join(y,how='left', lsuffix='_left', rsuffix='_right').corr()
	predictors.append('conversionmci')
	sns.heatmap(corr[(corr >= 0.1) | (corr <= -0.1)], cmap='RdYlGn', vmax=1.10, vmin=-1.10, xticklabels=predictors, yticklabels=predictors, linewidths=0.1,annot=True,annot_kws={"size": 8}, square=True);
	plt.title(f_name + ' heatmap Pearson\'s corr')
	plt.xticks(rotation=90)
	plt.yticks(rotation=0)
	plt.savefig(os.path.join(figures_dir, figname), dpi=360,bbox_inches='tight')
	
	figname = 'ENG_HeatmapTopF_' + f_name + '.png'
	topi = X.columns[top_indices]; 
	#topi = topi + ['conversionmci']
	plt.figure(figsize=(8, 4))
	#corr = dataframe[topi].corr()
	#corr =dataframe[topi].join(y).corr()
	corr = X[topi].corr()
	corr = X[topi].join(y,how='left', lsuffix='_left', rsuffix='_right').corr()
	xylabels = topi.values.tolist()
	xylabels.append('conversionmci')
	sns.heatmap(corr[(corr >= 0.10) | (corr <= -0.10)], cmap='RdYlGn', vmax=1.10, vmin=-1.10, xticklabels=xylabels, yticklabels=xylabels, linewidths=0.1,annot=True,annot_kws={"size": 5.5}, square=True);
	plt.title(f_name + ' Top indices heatmap Pearson\'s corr')
	plt.xticks(rotation=90)
	plt.yticks(rotation=0)
	plt.savefig(os.path.join(figures_dir, figname), dpi=360,bbox_inches='tight')
	return [selector, corr]


def split_training_test_sets(df, target_label,test_size=None):
	"""
	"""
	# Split data into features and target
	labels = np.array(df[target_label])

	# Remove the labels from the features
	features= df.drop(target_label, axis = 1)
	feature_list = list(features.columns)
	features_transform_np = False
	if features_transform_np is True:
		features = np.array(features)
	
	#Split data into training and testing sets
	if test_size is None:
		test_size = 0.25

	train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=test_size, stratify=labels, random_state=1)
	print('Train Features Shape:', train_features.shape, '. Train Labels Shape:', train_labels.shape)
	print('Test Features Shape:', test_features.shape, '. Test Labels Shape:', test_labels.shape, '\n')
	print('Train_labels class distribution')
	y = np.bincount(train_labels)
	ii = np.nonzero(y)[0]
	print(np.vstack((ii,y[ii])).T)
	print('Train labels ratio of 1/0s: %.4f' %(y[ii][-1]/y[ii][0]))

	print('Test_labels class distribution')
	y = np.bincount(test_labels)
	ii = np.nonzero(y)[0]
	print(np.vstack((ii,y[ii])).T)
	print('Test labels ratio of 1/0s: %.4f' %(y[ii][-1]/y[ii][0]))
	train_labels = train_labels.astype('int')
	test_labels = test_labels.astype('int')
	#X_train = X_train.astype('int')
	#X_test = X_test.astype('int')
	return train_features, test_features, train_labels, test_labels


def evaluate_model(model, X_test, y_test):
	"""
	"""	
	predictions = model.predict(X_test)

	accuracy = metrics.accuracy_score(y_test.tolist(), predictions.tolist())
	precision = metrics.precision_score(y_test.tolist(), predictions.tolist())
	recall = metrics.recall_score(y_test.tolist(), predictions.tolist())
	f1 = metrics.f1_score(y_test.tolist(), predictions.tolist())
	print('The RF accuracies :', accuracy)
	print('The RF precision:', precision)
	print('The RF recalls :', recall)
	print('The RF f1s :', f1)


def plot_roc_curve(y_test,y_scores,fit_type=None):
	"""
	"""
	# method I: plt

	matplotlib.use('tkagg')
	
	fpr, tpr, auc_thresholds = roc_curve(y_test, y_scores)
	roc_auc = auc(fpr, tpr)
	print('roc_auc = %.4f' %(roc_auc)) # AUC of ROC
	
	#figures_dir = '/Users/jaime/github/papers/JADr_vallecasindex/figures/figurescv5/'
	figname = 'roc_curve_' + fit_type + '.png'
	plt.figure(figsize=(8,8))
	plt.title('ROC Curve')
	plt.plot(fpr, tpr, linewidth=2, label= 'AUC = %0.2f' % roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.axis([-0.005, 1, 0, 1.005])
	plt.xticks(np.arange(0,1, 0.05), rotation=90)
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate (Recall)")
	plt.legend(loc='best')
	plt.savefig(os.path.join(figures_dir, figname), dpi=480,bbox_inches='tight')
	plt.show(block=False)
	plt.pause(3)
	plt.close()
	
def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    """
    return [1 if y >= t else 0 for y in y_scores]

def print_classsifier_parameters(clf):
	"""
	"""

	# Look at parameters used by our current forest
	print('Parameters currently in use:\n')
	pprint(clf.get_params())

def create_dict_hyperparameters():
	"""
	"""
	param_grid = [{'n_estimators':[1000, 1000, 10000], 'max_depth': [3, 6, 10, 20],\
	'min_samples_split':[2,4,6], 'min_samples_leaf':[2, 4,8, 16], \
	'max_features': ['auto'], \
	'random_state':[42], \
	'class_weight':['balanced']}]
	#'warm_start': [True, False],'oob_score': [True, False]
	#Out of bag estimation only available if bootstrap=True
	# recommended Breiman m = sqrt(X.shape) 'n_estimators':[1000,10000,100000]
	pprint(param_grid)
	return param_grid

def create_dict_scorers():
	"""
	"""
	# Look at parameters used by our current forest

	scorers = {
	'precision_score': make_scorer(precision_score),
	'recall_score': make_scorer(recall_score),
	'accuracy_score': make_scorer(accuracy_score),
	'f1_score': make_scorer(f1_score),
	'AUC': 'roc_auc'
	}
	pprint(scorers)
	return scorers


def train_RF(X_train, y_train, X_test, y_test, rf=None):
	"""
	"""
	from sklearn.ensemble import RandomForestClassifier
	if rf is None:
		# To create and to fit
		n_estimators = 1000
		# rf = RandomForestClassifier(n_estimators = n_estimators,random_state=1)
		rf = RandomForestClassifier(n_estimators = n_estimators, class_weight='balanced_subsample',min_samples_leaf=5, min_samples_split=10, bootstrap=True, max_features = 2, max_depth=3, verbose=1, n_jobs=3, random_state = 42)
	else:
		print('Model passed only to Fit with new dimensional data')
	print_classsifier_parameters(rf) 
	rf.fit(X_train, y_train)
	return rf 

def select_important_features_fitted_RF(rf, X, topN=None):
	"""select_important_features_fitted_RF
	Args:fitted RF X for colum names and topN = 3, 3 top features 
	Output: list topN feature names
	"""
	if topN is None:
		topN = 6
	# Select important features
	importances = list(rf.feature_importances_)
	# List of tuples with variable and importance
	std = np.std([tree.feature_importances_ for tree in rf.estimators_],axis=0)
	feature_importances = [(feature, round(importance, 3), round(stdev,3)) for feature, importance, stdev in zip(X.columns, importances,std)]
	feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
	[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
	suma1 = [t[1] for t in feature_importances]; np.sum(suma1) ==1
	# Important features above a threshold
	feature_importances_abv = [t for t in feature_importances if t[1] > 0.02]
	[print('\nVariable thr: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_abv]
	topfeats_thr = sorted(feature_importances_abv, key=lambda t: feature_importances_abv[1], reverse=True)[:topN]
	print('Top N features:'); pprint(topfeats_thr)
	
	feature_top_names = [t[0] for t in feature_importances][:topN]
	plot_feature_importance(rf, [t for t in feature_importances][:topN])
	#plot_rf_score(X[feature_top_names],rf, 6)
	return feature_top_names

def plot_feature_importance(rf, feature_importances_abv):
	"""plot_feature_importance plot important features saved in figures/dateformat.png
	"""
	import datetime
	DT = datetime.datetime.now()
	figname = 'importance_' + str(DT.year) + str(DT.month) + str(DT.day) + str(DT.hour) + str(DT.minute) + str(DT.second) + '.png'

	plt.title('Feature Importances')
	importances = [t[1] for t in feature_importances_abv]

	# ax.barh(range(len(feature_importances_abv)), importances, color='b', align='center')
	# plt.bar(range(len(feature_importances_abv)), [t[1] for t in feature_importances_abv],yerr=[t[2] for t in feature_importances_abv], align="center")
	plt.bar(range(len(feature_importances_abv)), [t[1] for t in feature_importances_abv], align="center")
	#ax.set_yticks(range(len(feature_importances_abv)), [t[0] for t in feature_importances_abv])
	plt.xticks(range(len(feature_importances_abv)), ([t[0] for t in feature_importances_abv]))
	plt.xlabel('Relative Importance')
	plt.xticks(rotation=90, size=8)
	plt.savefig(os.path.join(figures_dir, figname), dpi=240,bbox_inches='tight')
	return

# Utility function to report best scores
def report(results, n_top=3):
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
		for candidate in candidates:
			print("Model with rank: {0}".format(i))
			print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results['mean_test_score'][candidate],results['std_test_score'][candidate]))
			print("Parameters: {0}".format(results['params'][candidate]))
			print("")

def Run_RandomizedSearchCV(X_train, y_train):
	"""
	"""
	from time import time
	from datetime import datetime
	from scipy.stats import randint as sp_randint
	# build a classifier
	clf = RandomForestClassifier(n_estimators=20, n_jobs=6)
	# specify parameters and distributions to sample from
	# param_grid = [{'class_weight':['balanced'],'n_estimators':[1000,10000]}, {'class_weight':['balanced'],'min_samples_leaf':[4,6],'max_depth': [2,3], 'max_features': [2,4]},  {'class_weight':['balanced'],'min_samples_leaf':[4,6,8],'max_depth': [2,3,4], 'max_features': [2,4,6],'n_estimators':[100,1000,10000],'min_weight_fraction_leaf':[0.0, 0.2, 0.4],'min_impurity_decrease':[0.0,0.1,0.2]}]
	param_dist = {"class_weight":['balanced'], 'n_estimators':[10,20,50,100], "max_depth": sp_randint(3, 6), "max_features": sp_randint(1, 10), "min_samples_split": sp_randint(2, 8), "min_samples_leaf": sp_randint(1, 8), "bootstrap": [True, False]}
	n_iter_search = 100
	refit_score = 'AUC'
	random_search = RandomizedSearchCV(clf, scoring=create_dict_scorers(), refit=refit_score, param_distributions=param_dist, n_iter=n_iter_search, cv=5, iid=False, random_state=1, verbose=2)
	start = time()
	random_search.fit(X_train, y_train)
	print("RandomizedSearchCV took %.2f seconds for %d candidates"" parameter settings." % ((time() - start), n_iter_search))
	pprint(random_search.cv_results_)
	filereport = os.path.join(figures_dir, 'reports/') + datetime.now().strftime('%Y%m%d-%H%M%S_') + str(refit_score) + '.txt' 
	redirect_to_file(random_search.cv_results_, filereport)
	# report(random_search.cv_results_)
	return random_search

def Run_GridSearchCV(X_train, y_train):
	"""Run_GridSearchCV run grid search
	Args: X,y
	Output: grid_search fitted object and creates a reportf in figures_diur/reports
	"""
	from time import time
	from datetime import datetime
	from sklearn.ensemble import RandomForestClassifier
	# scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
	scoring = create_dict_scorers()
	# Setting refit='AUC', refits an estimator on the whole dataset with the
	# parameter setting that has the best cross-validated AUC score.
	refit_scores = ['precision_score', 'recall_score', 'f1_score', 'accuracy_score','AUC']
	refit_scores = refit_scores[-2]
	clf = RandomForestClassifier(n_estimators=10, n_jobs=6)
	param_grid= create_dict_hyperparameters()
	grid_search = GridSearchCV(clf, param_grid, refit=refit_scores, scoring=scoring, cv=5, return_train_score=True, verbose=2, iid=False)
	# grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, iid=False)
	start = time()
	grid_search.fit(X_train, y_train)
	print("GridSearchCV took %.2f seconds for %d candidate parameter settings."% (time() - start, len(grid_search.cv_results_['params'])))
	pprint(grid_search.cv_results_)
	filereport = os.path.join(figures_dir, 'reports/') + datetime.now().strftime('%Y%m%d-%H%M%S_') + str(refit_scores) + '.txt' 
	redirect_to_file(grid_search.cv_results_, filereport)
	# report(grid_search.cv_results_)
	return grid_search

def normalized_data(X):
	from sklearn import preprocessing
	x = X.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	standard_scaler = preprocessing.StandardScaler()
	x_mm = min_max_scaler.fit_transform(x)
	x_std = standard_scaler.fit_transform(x)
	dfmm = pd.DataFrame(x_mm)
	dfstd = pd.DataFrame(x_std)
	return dfmm, dfstd


##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
def main():
	np.random.seed(42)
	#importlib.reload(JADr_paper);import JADr_paper; JADr_paper.main()
	print('Code for paper: Selecting the most important self-assessed features for predicting conversion to Mild Cognitive Impairment with Random Forest and Permutation-based methods\n')
	# open csv with pv databse
	plt.close('all')
	csv_path =  'pvix_trunv2c.csv'
	csv_path =  'pix.csv'
	dataframe = pd.read_csv(csv_path)
	dataframe = dataframe.drop(dataframe.columns[0], axis=1)
	# Generate random column of mcis at ratio .9 .1
	#mcis = np.random.choice([0, 1], size=dataframe.shape[0], p=[.1, .9])
	#y = dataframe[target_label] = mcis

	target_label = 'conversionmci'
	y = dataframe[target_label]
	X = dataframe.loc[:, dataframe.columns != 'conversionmci']
	
	df_pv = X.copy()
	df_pv[target_label] = y
	
	oneintenrule = 10
	selector, covmatrix = feature_selection_filtering(rename_esp_eng_pv_columns(X), y, oneintenrule)
	test_size = 0.20
	
	[X_train, X_test, y_train, y_test] = split_training_test_sets(df_pv, target_label, test_size)

	# truncated df_trunc = df_pv.sample(frac=0.2).reset_index(drop=True)
	dummy_scores = build_dummy_scores(X_train, y_train, X_test, y_test)

	grid_search = Run_GridSearchCV(X_train, y_train)
	grid_search.get_params()
	params_ = grid_search.best_params_
	rf = grid_search.best_estimator_

	# # Compare with Dummies
	# rf = train_RF(X_train, y_train, X_test, y_test)
	model_scores = {'RF':rf.score(X_test, y_test)}
	# Call after hyperparameter tunning
	plot_dummy_vs_classifier(model_scores, dummy_scores)

	feature_top_names = select_important_features_fitted_RF(rf,X_train, oneintenrule)
	# Datafreme of important dataframe
	Ximp = X[feature_top_names]
	#Ximp[y.name] = y
	Ximp.insert(0, y.name,y,allow_duplicates=False)
	
	# Train again with the best estimator from RFCV, rf
	[X_train, X_test, y_train, y_test] = split_training_test_sets(Ximp, target_label, test_size)
	
	# If rf passed as last parameter only fit already existing Rf, if nothing passed, created and fit the RF
	# Use with rf passed when GridSearchCV called before
	rf = train_RF(X_train, y_train, X_test, y_test, rf)
	# y_pred = rf.predict(X_test)
	# y_pred_train = rf.predict(X_train)
	# plot_roc_curve(y_test, y_pred, 'test')
	# plot_roc_curve(y_train, y_pred_train, 'train')
	# print(confusion_matrix(y_test,y_pred))
	# print(confusion_matrix(y_pred_train,y_train))
	# evaluate_model(rf, X_test, y_test)
	# evaluate_model(rf, X_train, y_train)

	print('Calling to permutation_importance_eli5 ...')
	permutation_importance_eli5(rf, X_test, y_test)
	permutation_importance_eli5_tofit(rf,X_train, y_train, X_test, y_test)
	partial_dependence_plots(rf, X_train)
	shap_values = shap_explanations(rf, X_test)

	print('\n\n\n Program ENDED!')	 
if __name__ == "__name__":
	
	main()