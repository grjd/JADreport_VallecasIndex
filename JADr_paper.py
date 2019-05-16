#######################################################
# Python program name	: 
#Description	: JADr_paper.py
#Args           : Code for JADr paper:Feature Selection to Build the Vallecas Index                                                                                     
#Author       	: Jaime Gomez-Ramirez                                               
#Email         	: jd.gomezramirez@gmail.com 
#REMEMBER to Activate Keras source ~/github/code/tensorflow/bin/activate
#pyenv install 3.7.0
#pyenv local 3.7.0
#python3 -V
# To use ipython3 debes unset esta var pq contien old version
#PYTHONPATH=/usr/local/lib/python2.7/site-packages
#unset PYTHONPATH
# $ipython3
# To use ipython2 /usr/local/bin/ipython2
#/Library/Frameworks/Python.framework/Versions/3.7/bin/ipython3
#pip install rfpimp. (only for python 3)
#######################################################
# -*- coding: utf-8 -*-
import os, sys, pdb, operator
import datetime
import time
import numpy as np
import pandas as pd
import importlib
#importlib.reload(module)
import sys
import statsmodels.api as sm
import time
import importlib
#import rfpimp
from rfpimp import *
#sys.path.append('/Users/jaime/github/code/tensorflow/production')
#import descriptive_stats as pv
#sys.path.append('/Users/jaime/github/papers/EDA_pv/code')
import warnings
from subprocess import check_output
#import area_under_curve 
import matplotlib
matplotlib.use('Agg')
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2

def vallecas_features_dictionary(dataframe):
	"""vallecas_features_dictionary: builds a dictionary with the feature clusters of PV
	NOTE: hardcoded y1 yo year 6, same function in descriptive_stats.py to year7. 
	YS: do a btter version of this function not hardcoded, with number of years as option
	Args: None
	Output: cluster_dict tpe is dict
	""" 
	# REMOVED 'nivelrenta','educrenta'
	cluster_dict = {'Demographics':['edad_visita1','edad_visita2', 'edad_visita3', 'edad_visita4', 'edad_visita5', \
	'edad_visita6', 'edad_visita7', 'edadinicio_visita1', 'edadinicio_visita2', 'edadinicio_visita3',\
	'edadinicio_visita4', 'edadinicio_visita5', 'edadinicio_visita6','edad_ultimodx'],'Demographics_s':\
	['renta', 'numero_barrio','numero_distrito','sexo','nivel_educativo',\
	'anos_escolaridad','sdestciv','sdhijos', 'numhij','sdvive','sdocupac', 'sdresid', \
	'sdtrabaja','sdeconom','sdatrb', 'edad_visita1'],'SCD':['scd_visita1', \
	'scd_visita2', 'scd_visita3', 'scd_visita4', 'scd_visita5', 'scd_visita6', \
	'scdgroups_visita1', 'scdgroups_visita2', 'scdgroups_visita3', 'scdgroups_visita4', \
	'scdgroups_visita5', 'scdgroups_visita6', 'peorotros_visita1', \
	'peorotros_visita2', 'peorotros_visita3', 'peorotros_visita4', 'peorotros_visita5', \
	'peorotros_visita6', 'preocupacion_visita1', 'preocupacion_visita2',\
	'preocupacion_visita3', 'preocupacion_visita4', 'preocupacion_visita5', 'preocupacion_visita6',\
	'eqm06_visita1', 'eqm06_visita2', 'eqm06_visita3', 'eqm06_visita4', \
	'eqm06_visita5', 'eqm06_visita6', 'eqm07_visita1', 'eqm07_visita2', \
	'eqm07_visita3', 'eqm07_visita4', 'eqm07_visita5','eqm81_visita1', 'eqm81_visita2', \
	'eqm81_visita3', 'eqm81_visita4', 'eqm81_visita5', 'eqm82_visita1', 'eqm82_visita2', \
	'eqm82_visita3', 'eqm82_visita4', 'eqm82_visita5', 'eqm83_visita1', 'eqm83_visita2', \
	'eqm83_visita3', 'eqm83_visita4', 'eqm83_visita5', 'eqm84_visita1', 'eqm84_visita2', \
	'eqm84_visita3', 'eqm84_visita4', 'eqm84_visita5', 'eqm85_visita1', 'eqm85_visita2', \
	'eqm85_visita3', 'eqm85_visita4', 'eqm85_visita5', 'eqm86_visita1', 'eqm86_visita2', \
	'eqm86_visita3', 'eqm86_visita4', 'eqm86_visita5','eqm09_visita1', 'eqm09_visita2', \
	'eqm09_visita3', 'eqm09_visita4', 'eqm09_visita5', 'eqm10_visita1', 'eqm10_visita2',\
	'eqm10_visita3', 'eqm10_visita4', 'eqm10_visita5', 'eqm10_visita6', \
	'act_aten_visita1', 'act_aten_visita2', 'act_aten_visita3', 'act_aten_visita4', \
	'act_aten_visita5', 'act_aten_visita6','act_orie_visita1',\
	'act_orie_visita2', 'act_orie_visita3', 'act_orie_visita4', 'act_orie_visita5',\
	'act_orie_visita6','act_mrec_visita1', 'act_mrec_visita2', \
	'act_mrec_visita3', 'act_mrec_visita4', 'act_mrec_visita5', 'act_mrec_visita6',\
	'act_expr_visita1', 'act_expr_visita2', 'act_expr_visita3', \
	'act_expr_visita4', 'act_expr_visita5', 'act_expr_visita6', \
	'act_memt_visita1', 'act_memt_visita2', 'act_memt_visita3', 'act_memt_visita4', \
	'act_memt_visita5', 'act_memt_visita6', 'act_prax_visita1', \
	'act_prax_visita2', 'act_prax_visita3', 'act_prax_visita4', 'act_prax_visita5', \
	'act_prax_visita6','act_ejec_visita1', 'act_ejec_visita2', \
	'act_ejec_visita3', 'act_ejec_visita4', 'act_ejec_visita5', 'act_ejec_visita6',\
	'act_comp_visita1', 'act_comp_visita2', 'act_comp_visita3', \
	'act_comp_visita4', 'act_comp_visita5', 'act_comp_visita6', \
	'act_visu_visita1', 'act_visu_visita2', 'act_visu_visita3', 'act_visu_visita4', \
	'act_visu_visita5', 'act_visu_visita6'],'Neuropsychiatric':\
	['act_ansi_visita1', 'act_ansi_visita2', 'act_ansi_visita3', 'act_ansi_visita4',\
	'act_ansi_visita5', 'act_ansi_visita6','act_apat_visita1',\
	'act_apat_visita2', 'act_apat_visita3', 'act_apat_visita4', 'act_apat_visita5', \
	'act_apat_visita6','act_depre_visita1', 'act_depre_visita2',\
	'act_depre_visita3', 'act_depre_visita4', 'act_depre_visita5', 'act_depre_visita6',\
	'gds_visita1', 'gds_visita2', 'gds_visita3', 'gds_visita4', \
	'gds_visita5', 'gds_visita6', 'stai_visita1', 'stai_visita2', \
	'stai_visita3', 'stai_visita4', 'stai_visita5', 'stai_visita6'],\
	'CognitivePerformance':['animales_visita1', 'animales_visita2', 'animales_visita3', \
	'animales_visita4','animales_visita5','animales_visita6',\
	'p_visita1', 'p_visita2', 'p_visita3', 'p_visita4','p_visita5','p_visita6',\
	'mmse_visita1', 'mmse_visita2', 'mmse_visita3', 'mmse_visita4','mmse_visita5', 'mmse_visita6', \
	'reloj_visita1', 'reloj_visita2','reloj_visita3', 'reloj_visita4', 'reloj_visita5', 'reloj_visita6', \
	#'faq_visita1', 'faq_visita2', 'faq_visita3', 'faq_visita4', 'faq_visita5', 'faq_visita6','faq_visita7',\
	'fcsrtlibdem_visita1', 'fcsrtlibdem_visita2', 'fcsrtlibdem_visita3', \
	'fcsrtlibdem_visita4', 'fcsrtlibdem_visita5', 'fcsrtlibdem_visita6', \
	'fcsrtrl1_visita1', 'fcsrtrl1_visita2', 'fcsrtrl1_visita3', 'fcsrtrl1_visita4', 'fcsrtrl1_visita5',\
	'fcsrtrl1_visita6', 'fcsrtrl2_visita1', 'fcsrtrl2_visita2', 'fcsrtrl2_visita3',\
	'fcsrtrl2_visita4', 'fcsrtrl2_visita5', 'fcsrtrl2_visita6', 'fcsrtrl3_visita1', \
	'fcsrtrl3_visita2', 'fcsrtrl3_visita3', 'fcsrtrl3_visita4', 'fcsrtrl3_visita5', 'fcsrtrl3_visita6', \
	'cn_visita1', 'cn_visita2', 'cn_visita3', 'cn_visita4','cn_visita5', 'cn_visita6',\
	#'cdrsum_visita1', 'cdrsum_visita2', 'cdrsum_visita3', 'cdrsum_visita4', 'cdrsum_visita5','cdrsum_visita6', 'cdrsum_visita7'
	],'QualityOfLife':['eq5dmov_visita1', 'eq5dmov_visita2', 'eq5dmov_visita3',\
	'eq5dmov_visita4', 'eq5dmov_visita5', 'eq5dmov_visita6','eq5dcp_visita1', 'eq5dcp_visita2',\
	'eq5dcp_visita3', 'eq5dcp_visita4', 'eq5dcp_visita5', 'eq5dcp_visita6','eq5dact_visita1',\
	'eq5dact_visita2', 'eq5dact_visita3', 'eq5dact_visita4', 'eq5dact_visita5', 'eq5dact_visita6',\
	'eq5ddol_visita1', 'eq5ddol_visita2', 'eq5ddol_visita3', 'eq5ddol_visita4', 'eq5ddol_visita5', 'eq5ddol_visita6', \
	'eq5dans_visita1', 'eq5dans_visita2', 'eq5dans_visita3', 'eq5dans_visita4', 'eq5dans_visita5',\
	'eq5dans_visita6',  'eq5dsalud_visita1', 'eq5dsalud_visita2', 'eq5dsalud_visita3', \
	'eq5dsalud_visita4', 'eq5dsalud_visita5', 'eq5dsalud_visita6', 'eq5deva_visita1', \
	'eq5deva_visita2', 'eq5deva_visita3', 'eq5deva_visita4', 'eq5deva_visita5', 'eq5deva_visita6', \
	'valcvida2_visita1', 'valcvida2_visita2', 'valcvida2_visita3', 'valcvida2_visita4',\
	'valcvida2_visita6', 'valsatvid2_visita1', 'valsatvid2_visita2', 'valsatvid2_visita3',\
	'valsatvid2_visita4', 'valsatvid2_visita5', 'valsatvid2_visita6', 'valfelc2_visita1',\
	'valfelc2_visita2', 'valfelc2_visita3', 'valfelc2_visita4', 'valfelc2_visita5', 'valfelc2_visita6' \
	],'SocialEngagement_s':['relafami', 'relaamigo','relaocio_visita1','rsoled_visita1'],'PhysicalExercise_s':['ejfisicototal'], 'Diet_s':['alaceit', 'alaves', 'alcar', \
	'aldulc', 'alemb', 'alfrut', 'alhuev', 'allact', 'alleg', 'alpan', 'alpast', 'alpesblan', 'alpeszul', \
	'alverd','dietaglucemica', 'dietagrasa', 'dietaproteica', 'dietasaludable'],'EngagementExternalWorld_s':\
	['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12', 'a13', 'a14'],\
	'Cardiovascular_s':['hta', 'hta_ini','glu','lipid','tabac', 'tabac_cant', 'tabac_fin', 'tabac_ini',\
	'sp', 'cor','cor_ini','arri','arri_ini','card','card_ini','tir','ictus','ictus_num','ictus_ini','ictus_secu'],\
	'PsychiatricHistory_s':['depre', 'depre_ini', 'depre_num', 'depre_trat','ansi', 'ansi_ini', 'ansi_num', 'ansi_trat'],\
	'TraumaticBrainInjury_s':['tce', 'tce_con', 'tce_ini', 'tce_num', 'tce_secu'],'Sleep_s':['sue_con', 'sue_dia', 'sue_hor',\
	'sue_man', 'sue_mov', 'sue_noc', 'sue_pro', 'sue_rec', 'sue_ron', 'sue_rui', 'sue_suf'],'Anthropometric_s':['lat_manual',\
	'pabd','peso','talla','audi','visu', 'imc'],'Genetics_s':['apoe','apoe2niv', 'familial_ad'],'Diagnoses':['conversionmci','dx_corto_visita1', \
	'dx_corto_visita2', 'dx_corto_visita3', 'dx_corto_visita4', 'dx_corto_visita5', 'dx_corto_visita6', 'dx_corto_visita7',\
	'dx_largo_visita1', 'dx_largo_visita2', 'dx_largo_visita3', 'dx_largo_visita4', 'dx_largo_visita5', 'dx_largo_visita6',\
	'dx_largo_visita7', 'dx_visita1', 'dx_visita2', 'dx_visita3', 'dx_visita4', 'dx_visita5', 'dx_visita6', 'dx_visita7',\
	'ultimodx','edad_conversionmci', 'edad_ultimodx','tpo1.2', 'tpo1.3', 'tpo1.4', 'tpo1.5', 'tpo1.6', 'tpo1.7', \
	'tpoevol_visita1', 'tpoevol_visita2', 'tpoevol_visita3', 'tpoevol_visita4', 'tpoevol_visita5', 'tpoevol_visita6',\
	'tpoevol_visita7','tiempodementia', 'tiempomci']}
	#check thatthe dict  exist in the dataset
	for key,val in cluster_dict.items():
		print('Checking if {} exists in the dataframe',key)
		if set(val).issubset(dataframe.columns) is False:
			print('ERROR!! some of the dictionary:{} are not column names!! \n', key)
			print(dataframe[val])
			#return None
		else:
			print('		Found the cluster key {} and its values {} in the dataframe columns\n',key, val)
	# remove features do not need
	#dataset['PhysicalExercise'] = dataset['ejfre']*dataset['ejminut']
	#list_feature_to_remove = []
	#dataframe.drop([list_feature_to_remove], axis=1,  inplace=True)
	return cluster_dict

def encode_in_quartiles(df,float_f=None):
	"""encode_in_quartiles: ecnode Real features  ['renta', 'pabd', 'talla', 'imc', 'peso']
	into quartiles (4 bins)
	Args: df
	Output:df
	"""
	#from sklearn.preprocessing import KBinsDiscretizer
	#strategies = ['uniform', 'quantile', 'kmeans']
	# get float type of datframe
	df_cut = df.copy()
	series = df.dtypes # series[ss] == np.float64
	num_bins = 4
	if float_f is None:float_f = ['renta', 'pabd', 'talla', 'imc', 'peso', 'ejfisicototal','edad_visita1']
	# float_f = ['renta', 'pabd', 'talla', 'imc', 'peso', 'ejfisicototal',\
	# 'scd_visita1', 'sdatrb', 'anos_escolaridad', 'sue_noc', 'sue_dia', 'sdeconom',\
	# 'sdocupac','numhij','relaamigo','sdvive','edad_visita1']
	for ff in float_f:
		print('Encoding in quartiles %s' %ff)
		#enc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy=strategies[1])	
		#del df_cut[ff]
		#bins = pd.qcut(df[ff],num_bins)
		
		bins = pd.qcut(df[ff],num_bins, duplicates='drop')
		labels, uniques= pd.factorize(bins, sort=True)
		df_cut[ff] = labels
	return df_cut

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

def remove_features_redundant(df):
	"""remove_features_redundant
	Output: dataframe
	"""

	redundant = ['nivelrenta', 'educrenta']
	return df.drop(redundant, axis=1)



def plot_wrapper_fig(rfecv, figname, estimator_name=None):
	"""
	"""
	figures_dir = '/Users/jaime/github/papers/JADr_vallecasindex/figures/edad'
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
	from sklearn.model_selection import StratifiedKFold
	from sklearn.feature_selection import RFECV
	from sklearn.linear_model import LogisticRegression

	#figures_dir = '/Users/jaime/github/papers/JADr_vallecasindex/figures'
	# C smaller values specify stronger regularization
	model_dict = {'SVC': {'C': np.array([ 0.01, 0.1, 1, 10]), 'kernel':['linear'], \
	'scoring':['accuracy', 'recall', 'f1', 'roc_auc']}, \
	'LogisticRegression':{'C': np.array([ 0.01, 0.1, 1, 10]),\
	'class_weight' : [None, 'balanced'], 'penalty': ['l1', 'l2'], 'solver':['liblinear'], \
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
				rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring=scoring, n_jobs=14, verbose=0)
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
				rfecv = RFECV(estimator=logreg, step=1, cv=StratifiedKFold(3), scoring=scoring, n_jobs=14, verbose=0)
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
	figures_dir = '/Users/jaime/github/papers/JADr_vallecasindex/figures/edad/'
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

def plot_rf_score(X,y,forest, nboffeats):
	""" plot_rf_score Plot the feature importances of the forest
	"""

	figures_dir = '/Users/jaime/github/papers/JADr_vallecasindex/figures/quartiles/test'
	criterion = forest.get_params()['criterion']
	n_estimators = forest.get_params()['n_estimators']
	class_weight = forest.get_params()['class_weight']
	if class_weight is None: class_weight = 'None'
	class_name = forest.__class__.__name__
	importances = forest.feature_importances_ 
	std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
	indices = np.argsort(importances)[::-1]
	indices_imp = indices[:nboffeats]
	columns_imp = X.columns[indices_imp].tolist()
	importances_imp = importances[indices_imp]
	print('Most important features :', columns_imp) 
	# Print the feature ranking
	for f in range(X.shape[1]):
		print(" %d. feature %d (%f)" % ( f + 1, indices[f], importances[indices[f]]))
	# Score with training dataset
	print('Score: %f' % forest.score(X, y))
	# Score Out of the bag (quasi test): evaluating our instances in the training set 
	# using only the trees for which they were omitted.
	print('OOB Score %f' % forest.oob_score_) 
	
	# Plot most important features
	figname = class_name + '_criterion_' + criterion + '_N_' + str(n_estimators) + '_weight_'+ class_weight +'.png'
	plt.figure(figsize=(9,8))
	plt.title("Feature importances:" + class_name)
	#plt.bar(range(X.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
	plt.bar(range(X[columns_imp].shape[1]), importances_imp, color="r", yerr=std[indices][indices_imp], align="center")
	plt.xticks(range(X[columns_imp].shape[1]), columns_imp, fontsize=10)
	plt.xlim([-1, X[columns_imp].shape[1]])
	plt.xticks(rotation=90)
	#plt.yticks(rotation=0)
	plt.savefig(os.path.join(figures_dir, figname), dpi=240,bbox_inches='tight')


# def permutation_rf(X,y,forest):
# 	"""
# 	""" 
	
# 	oob_c = rfpimp.oob_classifier_accuracy(forest, X, y)
# 	print('oob_classifier_accuracy == %.5f' % (oob_c))
# 	n_samples = 1000
# 	I = rfpimp.importances(forest, X, y, n_samples=n_samples)
# 	print(I)
# 	viz = rfpimp.plot_importances(I)
# 	figname = 'RF_selector_Permutation_' + criterion + '_N_' + str(n_estimators) + '_samples_' + str(n_samples) + '.png'
# 	viz.save(os.path.join(figures_dir, figname))
# 	return oob_c
# 	print('Calling to rf_with_permutation ....')
# 	#rf_with_permutation(forest, X, y, rfpimp.oob_classifier_accuracy)
# 	pdb.set_trace()
# 	print('DONE!!')

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
	X['random'] = np.random.random(size=len(X))
	#https://stackoverflow.com/questions/22409855/randomforestclassifier-vs-extratreesclassifier-in-scikit-learn
	print('\n\n RF for Feature Selection::ExtraTreesClassifier or RandomForestClassifier')
	model_dict = {'ExtraTreesClassifier':{'criterion':['gini', 'entropy'], \
	'n_estimators':[100,1000,10000,100000], 'class_weight':[None, 'balanced']},\
	'RandomForestClassifier':{'criterion':[ 'gini', 'entropy'], 'n_estimators':[100,1000,10000,100000,1000000],\
	'class_weight':[None, 'balanced']}}
	model_dict = {'RandomForestClassifier':{'criterion':[ 'gini', 'entropy'], 'n_estimators':[10,100,1000,10000],\
	'class_weight':[None, 'balanced']}}
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
				forest = RandomForestClassifier(n_estimators=n_estimators, bootstrap=True,\
					oob_score=True,criterion=criterion, class_weight=class_weight, random_state=0, n_jobs=12,verbose=2)
				forest = forest.fit(X, y)
				#plot_rf_score(X,y,forest, nboffeats)
				#oob_c = permutation_rf(X,y,forest)
				if class_weight is None:
					class_weight='None'
				image_name = str(n_estimators) + criterion + class_weight
				imp = permutation_importances(forest, X, y, oob_classifier_accuracy, image_name)


def permutation_importances(rf, X_train, y_train, metric, image_name):
	"""feature impprtance with permutation
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

# def rf_with_permutation(rf, X_train, y_train, oob_c):
# 	"""rf_with_permutation: 
# 	rf must be pre-trained 
# 	"""
# 	#import sys
# 	#!{sys.executable} -m pip install rfpimp
# 	#metric for regression oob_regression_r2_score
# 	imp = permutation_importances(rf, X_train, y_train, oob_c)
# 	print('Important features by permutation DONE:', imp)
# 	pdb.set_trace()

 
def feature_selection_filtering(X, y, nboffeats):
	"""feature_selection_filtering: SlectKBeast features
	Args:X,y, nbof features
	Output:None
	"""
	from sklearn import preprocessing
	from sklearn.preprocessing import Imputer
	predictors = X.columns.tolist()
	#['f_classif', 'mutual_info_classif', 'chi2', 'f_regression', 'mutual_info_regression']
	selector = SelectKBest(chi2, k=nboffeats).fit(X, y)
	# scores_ : array-like, shape=(n_features,) pvalues_ : array-like, shape=(n_features,)
	top_indices = np.nan_to_num(selector.scores_).argsort()[-nboffeats:][::-1]
	print("Selector {} scores:",nboffeats, selector.scores_[top_indices])
	print("Top features:\n", X.columns[top_indices])

	#Plot heatmaps
	figures_dir = '/Users/jaime/github/papers/JADr_vallecasindex/figures/edad'
	f_name = selector.get_params()['score_func'].__name__
	figname = 'HeatmapF_' + f_name + '.png'
	plt.figure(figsize=(18, 16))
	#corr = dataframe[predictors].corr()
	#corr =dataframe[predictors].join(y).corr()
	corr = X.join(y).corr()
	predictors.append('conversionmci')
	sns.heatmap(corr[(corr >= 0.1) | (corr <= -0.1)], cmap='RdYlGn', vmax=1.10, vmin=-1.10, xticklabels=predictors, yticklabels=predictors, linewidths=0.1,annot=True,annot_kws={"size": 8}, square=True);
	plt.title(f_name + ' heatmap Pearson\'s corr')
	plt.xticks(rotation=90)
	plt.yticks(rotation=0)
	plt.savefig(os.path.join(figures_dir, figname), dpi=240,bbox_inches='tight')
	
	figname = 'HeatmapTopF_' + f_name + '.png'
	topi = X.columns[top_indices]; 
	#topi = topi + ['conversionmci']
	plt.figure(figsize=(8, 6))
	#corr = dataframe[topi].corr()
	#corr =dataframe[topi].join(y).corr()
	corr = X[topi].corr()
	corr = X[topi].join(y).corr()
	xylabels = topi.values.tolist()
	xylabels.append('conversionmci')
	sns.heatmap(corr[(corr >= 0.10) | (corr <= -0.10)], cmap='RdYlGn', vmax=1.10, vmin=-1.10, xticklabels=xylabels, yticklabels=xylabels, linewidths=0.1,annot=True,annot_kws={"size": 8}, square=True);
	plt.title(f_name + ' Top indices heatmap Pearson\'s corr')
	plt.xticks(rotation=90)
	plt.yticks(rotation=0)
	plt.savefig(os.path.join(figures_dir, figname), dpi=240,bbox_inches='tight')


def prepare_df(dataframe, pv_dict):
	"""prepare_df
	Args: Dataframe and dictionary
	Output: Input X and traget y
	"""
	# Predictors are static feature ['Genetics_s', 'Cardiovascular_s', 'PhysicalExercise_s', 'PsychiatricHistory_s', 'Sleep_s', \
	#'Anthropometric_s', 'Diet_s', 'SocialEngagement_s', 'TraumaticBrainInjury_s', 'Demographics_s', \
	#'EngagementExternalWorld_s']
	# Select Features by group
	genetics = pv_dict['Genetics_s']; genetics.remove('apoe2niv')
	print(np.sum(dataframe[genetics].isnull()==True))
	demog = pv_dict['Demographics_s']; 
	demog = list(filter(lambda x: x not in [ 'numero_barrio', 'numero_distrito', 'tce_num', 'tce_secu'], demog))
	#demog = ['renta', 'nivelrenta', 'educrenta','sexo', 'nivel_educativo', 'anos_escolaridad', 'sdestciv', 'sdhijos', 'numhij', 'sdvive', 'sdocupac', 'sdresid', 'sdtrabaja', 'sdeconom', 'sdatrb']
	print(np.sum(dataframe[demog].isnull()==True)) 
	# nivel_educativo anos_escolaridad educrenta (2) 
	antr = pv_dict['Anthropometric_s']
	print(np.sum(dataframe[antr].isnull()==True)) #lat_manual 41
	sleep = pv_dict['Sleep_s']; print(np.sum(dataframe[sleep].isnull()==True))
	social = pv_dict['SocialEngagement_s']; print(np.sum(dataframe[social].isnull()==True))
	engagement = pv_dict['EngagementExternalWorld_s']; print(np.sum(dataframe[engagement].isnull()==True))
	diet = pv_dict['Diet_s']
	diet = list(filter(lambda x: x not in ['alaceit', 'dietaglucemica', 'dietagrasa', 'dietaproteica', 'dietasaludable'], diet))
	print(np.sum(dataframe[diet].isnull()==True))	
	phys = pv_dict['PhysicalExercise_s']; print(np.sum(dataframe[social].isnull()==True))
	cardio = pv_dict['Cardiovascular_s']
	cardio = list(filter(lambda x: x not in ['hta_ini', 'sp', 'tabac_fin', 'tabac_cant', 'tabac_ini', 'cor_ini','arri_ini', 'card_ini','ictus_ini','ictus_num','ictus_secu'], cardio))
	print(np.sum(dataframe[cardio].isnull()==True))
	tbi = pv_dict['TraumaticBrainInjury_s'] 
	tbi = list(filter(lambda x: x not in [ 'tce_con', 'tce_ini', 'tce_num', 'tce_secu'], tbi))
	print(np.sum(dataframe[tbi].isnull()==True))
	psy = pv_dict['PsychiatricHistory_s']
	psy = list(filter(lambda x: x not in [ 'depre_ini', 'depre_num', 'depre_trat', 'ansi_ini', 'ansi_num', 'ansi_trat'], psy))
	print(np.sum(dataframe[psy].isnull()==True))
	# pick only year 1 scd = pv_dict['SCD'] qol =  pv_dict['QualityOfLife']
	scd = ['scd_visita1','act_aten_visita1', 'peorotros_visita1','act_orie_visita1', \
	'act_mrec_visita1','act_expr_visita1','act_prax_visita1','act_ejec_visita1','act_comp_visita1',\
	'act_visu_visita1','eqm06_visita1','eqm07_visita1','eqm81_visita1','eqm82_visita1','eqm83_visita1',\
	'eqm84_visita1','eqm85_visita1','eqm86_visita1','eqm09_visita1','eqm10_visita1']
	qol =  ['eq5dmov_visita1', 'eq5ddol_visita1', 'valfelc2_visita1','eq5dsalud_visita1']
	#remove NaNs
	dataframe.fillna(method='ffill', inplace=True)
	#values = {'apoe': 0, 'educrenta': 3, 'nivel_educativo': 2, 'anos_escolaridad': 8}
	#dataframe.fillna(value=values)

	#Select SelectKBest features
	target = 'conversionmci'
	predictors = genetics + demog + antr + sleep + social + engagement + diet + phys + cardio + tbi + psy + scd + qol
	print(dataframe[predictors].info())
	atleast2visits = dataframe['conversionmci'].notna()
	# remove features with low variance
	lowvar_features = remove_features_lowvariance(dataframe[predictors])
	predictors = [elem for elem in predictors if elem not in lowvar_features]

	X = dataframe[predictors][atleast2visits]
	y = dataframe[target][atleast2visits]
	#for sklearn to recognize y type (not object)
	y = y.astype('int')
	return X, y

def	change_DkDa_features(df):
	"""change_DkDa_features: replace 9 by distribution with proportion
	"""
	featureswithnines =['tce','arri','cor', 'ictus', 'lipid', 'tir',\
	'sue_ron', 'sue_rui', 'sue_hor','sue_mov','sue_pro','sue_suf','sue_rec']
	dfo = df.copy()
	for feat in featureswithnines:
		print(df[feat].value_counts())
		zeros = sum(df[feat] == 0); ones = sum(df[feat] == 1);twos = sum(df[feat] == 2);
		threes = sum(df[feat] == 3);nines = sum(df[feat] == 9)
		if twos > 0:
			listofvalues = [0,1,2]
			p0 = float(zeros/(zeros+ones+twos)); p1 = float(ones/(zeros+ones+twos))
			listofps = [p0, p1, 1 - p0 - p1]
		else:
			listofvalues = [0,1]
			p0 = float(zeros/(zeros+ones)); p1 = float(ones/(zeros+ones))
			listofps = [p0, 1 - p0]
		if threes > 0:
			listofvalues = [0,1,2,3]
			p0 = float(zeros/(zeros+ones+twos+threes)); p1 = float(ones/(zeros+ones+twos+threes));
			p2 =  float(twos/(zeros+ones+twos+threes))
			listofps = [p0, p1, p2, 1 - p0 -p1 -p2]
		replaces = np.random.choice(listofvalues, nines , listofps)
		mask = df[feat] == 9 
		df.loc[mask, feat] = replaces
		print(df[feat].value_counts())
	return df

def castdf_to_int(dataframe):

	dataframe['conversionmci'] = pd.Series(dataframe['conversionmci'],dtype='Int64')
	dataframe['apoe'] = pd.Series(dataframe['apoe'],dtype='Int64')
	#dataframe['educrenta'] = pd.Series(dataframe['educrenta'],dtype='Int64')
	dataframe['numero_barrio'] = pd.Series(dataframe['numero_barrio'],dtype='Int64')
	dataframe['numero_distrito'] = pd.Series(dataframe['numero_distrito'],dtype='Int64')
	dataframe['nivel_educativo'] = pd.Series(dataframe['nivel_educativo'],dtype='Int64')
	dataframe['anos_escolaridad'] = pd.Series(dataframe['anos_escolaridad'],dtype='Int64')
	dataframe['lat_manual'] = pd.Series(dataframe['lat_manual'],dtype='Int64')
	dataframe['edad_ultimodx'] = pd.Series(dataframe['edad_ultimodx'],dtype='Int64')
	dataframe['edad_visita1'] = pd.Series(dataframe['edad_visita1'],dtype='Int64')

	return dataframe

##################################################################################
##################################################################################
##################################################################################
##################################################################################
##################################################################################
def main():
	#importlib.reload(JADr_paper);import JADr_paper; JADr_paper.main()
	print('Code for JADr paper on Feature Selection to Build the Vallecas Index\n')
	# open csv with pv databse
	plt.close('all')
	csv_path = '~/vallecas/data/BBDD_vallecas/Vallecas_Index-Vols134567-22April2019.csv'
	figures_path = '/Users/jaime/github/papers/EDA_pv/figures/'
	dataframe = pd.read_csv(csv_path)
	# Copy dataframe with the cosmetic changes e.g. Tiempo is now tiempo
	dataframe_orig = dataframe.copy()
	print('Build dictionary with features ontology and check the features are in the dataframe\n') 
	
	# remove redundant features ['nivelrenta', 'educrenta']
	# replace features dont know dont answer 9
	dataframe = change_DkDa_features(dataframe)
	dataframe = remove_features_redundant(dataframe)
	# cast to int some features eg conversionmci
	dataframe = castdf_to_int(dataframe)

	#features_dict is the list of clusters, the actual features to plot are hardcoded
	features_dict = vallecas_features_dictionary(dataframe)
	# select rows with 5 visits
	visits=['tpo1.2', 'tpo1.3','tpo1.4', 'tpo1.5','tpo1.6']
	df_loyals = select_rows_all_visits(dataframe, visits)

	#Feature Selection using three different methodoloies: Filtering (intrinsic) 
	#and model based : Wrapping (RFEF, GA) and Embedded(RF, Regularization)
	##########################
	# Filtering ###
	##########################
	X, y = prepare_df(dataframe, features_dict)
	# test for correct transformation of Real features ['renta', 'pabd', 'talla', 'imc', 'peso']
	toq = ['renta', 'pabd', 'talla', 'imc', 'peso', 'ejfisicototal','edad_visita1']
	X = encode_in_quartiles(X, toq)

	print('Filtering for Feature Selection')
	#10 to 1 rule
	nboffeats = int(round((np.sum(dataframe['conversionmci']==1)/10)))
	print("The minority class has %d members" % (np.sum(dataframe['conversionmci']==1)))
	print("The number of parameters for the 10:1 rule is %d" % (nboffeats))
	#remove apoe
	#X = X.drop('apoe', axis=1)
	feature_selection_filtering(X, y, nboffeats)
	##########################
	# Wrapping (REF) ###
	##########################
	#remove apoe
	X = X.drop('apoe', axis=1)
	print('Wrapping methods (SVC LogReg), for Feature Selection \n')
	feature_selection_wrapping(X, y, nboffeats)
	
	##########################
	# Embedded (RF) ###
	##########################

	print('Embeded method (RandomForest) for Feature Selection \n')
	feature_selection_embedding(X, y, nboffeats)
	print('END Embeded method (RandomForest) for Feature Selection \n')
	pdb.set_trace()
	##########################
	#testing here cut paste###
	##########################


	# estiamte parameters y = betai xi xi =  sum + 12 + 23
	print('\n\n\n END!!!!')	 
if __name__ == "__name__":
	
	main()