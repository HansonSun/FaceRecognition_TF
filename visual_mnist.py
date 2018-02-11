from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def draw_2d_mnist_featuremap(features,labels):
	f = plt.figure(figsize=(16,9))
	c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', 
		 '#ff00ff', '#990000', '#999900', '#009900', '#009999']
	for index,f in enumerate(features):
		plt.plot(int(f[0]),int(f[1]), '.', c=c[int(labels[index])] ) 
	plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
	plt.grid()
	plt.show()

def draw_3d_mnist_featuremap(features,labels):
	fig=plt.figure(figsize=(16,9))
	ax = fig.add_subplot(111, projection = '3d')  
	c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', 
		 '#ff00ff', '#990000', '#999900', '#009900', '#009999']
	for index,f in enumerate(features):
		ax.scatter(int(f[0]),int(f[1]),int(f[2]) , '.', c=c[int(labels[index])] ) 
	plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
	plt.grid()
	plt.show()

def draw_pca_mnist_featuremap(features,labels):
	pca = decomposition.PCA(n_components=3)
	new_X = pca.fit_transform(features)
	fig = plt.figure(figsize=(16,9))
	ax = fig.gca(projection='3d')
	ax.scatter(new_X[:, 0], new_X[:, 1], new_X[:, 2], c=labels, cmap=plt.cm.spectral)
	plt.show()

def draw_tsne_mnist_feature_featuremap():
	pass


if __name__ == "__main__":
	mnist = datasets.load_digits()
	x = mnist.data
	y = mnist.target
	draw_pca_mnist_featuremap(x,y )

