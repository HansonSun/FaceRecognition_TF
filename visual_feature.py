from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


def draw_2d_features(features,labels):
	f = plt.figure(figsize=(16,9))
	c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', 
		 '#ff00ff', '#990000', '#999900', '#009900', '#009999']
	for index,f in enumerate(features):
		plt.plot(f[0],f[1], '.', c=c[int(labels[index])] ) 
	plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
	plt.grid()
	plt.show()

def draw_3d_features(features,labels):
	fig=plt.figure(figsize=(16,9))
	ax = fig.add_subplot(111, projection = '3d')  
	c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', 
		 '#ff00ff', '#990000', '#999900', '#009900', '#009999']
	for index,f in enumerate(features):
		ax.scatter(f[0],f[1],f[2] , '.', c=c[int(labels[index])] ) 
	plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
	plt.grid()
	plt.show()

def draw_pca3d_features(features,labels):
	pca = decomposition.PCA(n_components=3)
	new_X = pca.fit_transform(features)
	fig = plt.figure(figsize=(16,9))
	ax = fig.gca(projection='3d')
	ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=labels, cmap=plt.cm.spectral)
	plt.show()

def draw_pca2d_features(features,labels):
	pca = decomposition.PCA(n_components=2)
	new_X = pca.fit_transform(features)
	fig = plt.figure(figsize=(16,9))
	#ax = fig.gca(projection='3d')
	plt.scatter(new_X[:, 0], new_X[:, 1], c=labels)
	plt.show()

def draw_tsne2d_features(features,labels):
	X_tsne = TSNE(n_components=2,learning_rate=100).fit_transform(features)
	print("finishe!")
	plt.figure(figsize=(12, 6))
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels)
	plt.colorbar()
	plt.show()

def draw_tsne3d_features(features,labels):
	X_tsne = TSNE(n_components=3,learning_rate=100).fit_transform(features)
	print("finishe!")
	plt.figure(figsize=(12, 6))
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1],X_tsne[:, 2], c=labels)
	plt.colorbar()
	plt.show() 

if __name__ == "__main__":
	mnist = datasets.load_digits()
	x = mnist.data
	y = mnist.target
	draw_tsne3d_feature(x,y )

