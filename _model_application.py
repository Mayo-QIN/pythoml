#!/usr/bin/env python

'''
This is only to be used for the demo not a true classifier it is just a simulation since we did not have data to properly train a classifier

'''
import matplotlib as mpl
mpl.use('pdf')
from time import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import argparse
import nibabel as nib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from sklearn.linear_model import RandomizedLasso
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import calibration_curve
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from  openpyxl import Workbook
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from _analyticscalc import analyticscalc
from sklearn.learning_curve import learning_curve
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.learning_curve import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import glob
import shutil
import os
import zipfile
import uuid
from sklearn import svm
np.random.seed(42)


def machinelearningpipeline(dataset,output='output.zip'):

	subdir = str(uuid.uuid4())

	# Create a folder for all the temporary stuff and remove at the end
	directory='/tmp/'+subdir+'/'
	if not os.path.exists(directory):
		os.makedirs(directory)
	print ('I am saving temporary things in: %s'%(directory))
	# The input is a zip file: Unzip it in a temp folder and load the csv file as pandas
	ziptoproces=glob.glob("*.zip")
	zip_ref = zipfile.ZipFile(ziptoproces[0], 'r')
	zip_ref.extractall(directory)
	zip_ref.close()
	# Load CSV
	data=pd.read_csv(directory+'/3D.feature.csv', index_col=0, header=None).T
	h = .02  # step size in the mesh
	name =data['aa_info.patient.FamilyName'].values[0]
	volume=data['size.volume(mm^3)'].values[0]
	print volume
	volume=float(volume)/1000
	sphericity=float(data['sphericity.value'].values[0])
	print sphericity
	class1=np.random.random((20, 2))
	class1[:,0]=0.7*class1[:,0]
	class2=np.random.random((20, 2))
	class2[:,0]=class2[:,0]
	X=np.append(class1,class2,axis=0)
	y=np.append(np.zeros((len(class1))),np.ones((len(class2))),axis=0)
	C = 1.0  # SVM regularization parameter
	svc = svm.SVC(kernel='rbf', gamma=1, C=C).fit(X, y)
	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))
	# title for the plots
	titles = ['SVC with linear kernel']
	f=plt.figure()
	clf=svc
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
	# Plot also the training points
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
	plt.xlabel('Volume')
	plt.ylabel('sphericity.value')
	plt.plot(volume,sphericity,'r*', markersize=20)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.xticks(())
	plt.yticks(())
	noduletype=clf.predict([volume,sphericity])
	if noduletype==0.:
		plt.title('The classification for this case is benign')
	else:
		plt.title('The classification for this case is benign')
	f.savefig(directory+'/output.pdf')
	types = ('*.pdf') # the tuple of file types
	files_grabbed = []
	for files in types:
		files_grabbed.extend(glob.glob(files))
	for file in files_grabbed:
		if os.path.isfile(file):
			shutil.copy2(file, directory)
	shutil.make_archive(output[:-4], 'zip', directory)
	return 0


def main(argv):
	machinelearningpipeline(argv.dataset, argv.output)
	return 0

if __name__ == "__main__":
	parser = argparse.ArgumentParser( description='Apply a trained model')
	parser.add_argument ("-i", "--dataset",  help="unknowdata these data have to be in the format that standford feature calculator outputs" , required=True)
	parser.add_argument ("-o", "--output",  help="output name of zip file" , required=True)
	parser.add_argument('--version', action='version', version='%(prog)s 0.1')
	parser.add_argument("-q", "--quiet",
						  action="store_false", dest="verbose",
						  default=True,
						  help="don't print status messages to stdout")
	args = parser.parse_args()
	main(args)
