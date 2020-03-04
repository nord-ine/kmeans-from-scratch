import numpy as np
import random 

class kmeans : 

    def __init__(self,nb_clusters,initialisation,max_iteration = 200):
        self.k = nb_clusters
        self.init = initialisation # random or kmeans++
        self.max_iter = max_iteration
        self.clusters = []
        self.centroids = []
    
    def fit(self,X):
        nb_features = np.shape(X)[1]
        centroids_prime=np.random.rand(self.k,nb_features) # centroid n
        self.centroids = self._init_centroids(X) # centroid n+1
        # print(self.centroids)
        iteration=0
        while(not (np.all(self.centroids==centroids_prime))):   # the algorithm srops when centroid n != centroid n+1
            if(iteration>self.max_iter): # or here
                break
            centroids_prime = np.copy(self.centroids)      
            self.assignto_cluster(X)
            self.update_centroids(X)
            iteration+=1
        for j in self.clusters:
            print(X[j])

     

    def assignto_cluster(self,X):
        self.clusters = [[] for i in range(self.k)]
        Cm= 0
        for i,xi in enumerate(X) : 
            Cm = np.argmin([np.linalg.norm(xi-c) for c in self.centroids])
            self.clusters[Cm].append(i)
    
    def update_centroids(self,X):
        for i in range(self.k):
            # print(X[self.clusters[i]])
            if(self.clusters[i]==[]):
                self.centroids[i] = [0,0]
            else :
                self.centroids[i] = np.mean(X[self.clusters[i]] ,axis=0)

     
    def _init_centroids(self,X): 
        if(self.init == "random"):
            return self._random_init(X)

        elif (self.init == "kmeans++"):
            return self._kmeanspp_init(X) 
        else :
            print("incorrect initialisation name") # chan,ge this to an exception 
        

    def _random_init(self,X):

        indeces = np.random.choice(len(X),self.k)
        return X[indeces]


    def _kmeanspp_init(self,X):

        pass