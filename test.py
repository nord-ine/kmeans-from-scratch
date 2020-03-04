import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from kmeans import kmeans

k=5
sigma =0.6
n =50
x,y = make_blobs(n_samples=n,centers=k,random_state=0,cluster_std=sigma)

plt.scatter(x[:,0],x[:,1],c=y)

model = kmeans(5,"random",100)

model.fit(x)