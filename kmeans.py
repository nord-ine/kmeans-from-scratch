import numpy as np
import random 

class kmeans : 

    def __init__(self,nb_clusters,initialisation,max_iteration = 200):
        self.k = nb_clusters
        self.init = initialisation # random or kmeans++
        self.max_iter = max_iteration # the max number of iteration for the algorithm to converge 
        self.clusters = [] # the list of the clusters , and each list contains the indexes of the points in the cluster 
        self.centroids = [] # list of the centroids 
    
    def fit(self,X):
        nb_features = np.shape(X)[1] 
        centroids_prime=np.random.rand(self.k,nb_features) # list of centroids at  (n) iteration , here they are initialised randomly
        self.centroids = self._init_centroids(X) # list of centroids at  (n+1) iteration , intialised randomly or by using "kmeans++"
        iteration=0
        while(not (np.all(self.centroids==centroids_prime))):   # the algorithm stops when centroids (n) = centroids (n+1)
            if(iteration>self.max_iter): # or here
                break
            centroids_prime = np.copy(self.centroids)     # centroids (n) = centroids (n+1)
            self.assignto_cluster(X)  
            self.update_centroids(X) # 
            iteration+=1


     

    def assignto_cluster(self,X):
        """
        assigns every data point to the nearst cluster 
        by adding its index in the list X to the closest cluster 
        
        """
        self.clusters = [[] for i in range(self.k)] # empty the clusters at each call of the function 
        Cm= 0 # the index of the closest cluster to the  point
        for i,xi in enumerate(X) : 
            Cm = np.argmin([np.linalg.norm(xi-c) for c in self.centroids]) 
            self.clusters[Cm].append(i) 
    
    def update_centroids(self,X):
        """
        updates the values of the centroids 

        """
        for i in range(self.k):
            if(self.clusters[i]==[]): # if the cluster is empty then the coresponding centroid is set to zero 
                self.centroids[i] = [0,0]
            else :
                self.centroids[i] = np.mean(X[self.clusters[i]] ,axis=0)

     
    def _init_centroids(self,X): 
        """
        method to init the values of the centroid 
        
        Returns:
            a list of centroids 
        """
        if(self.init == "random"):
            return self._random_init(X)

        elif (self.init == "kmeans++"):
            return self._kmeanspp_init(X) 
        else :
            print("incorrect initialisation name") # change this to an exception 
        

    def _random_init(self,X):
        """
        
        method to randomly initialise the centroids 
        """
        indeces = np.random.choice(len(X),self.k)
        return X[indeces]


    def _kmeanspp_init(self,X):
        """
        method to init the centroids using the kmeans++ algorithm 

        """
        index = np.random.choice(len(X),1)
        first_centroid = X[index] # first step of the algorithm is to pick a random point from X
        initial_centroids = [first_centroid] # the list of the initial centroids that is returned, the first centroid is added
        """
        we pick the rest of the centtroids such that the distance between them is as long as possible 
     
        """
        jth_centroid = 0
        Distances = []
        sum_Dx = 0
        for _ in range(self.k-1):
            Distances = []
            for xi in X:
                Distances.append(np.min([np.linalg.norm(xi-c) for c in initial_centroids]))
            sum_Dx = np.sum(Distances)
            jth_centroid = np.argmax([ np.square(d)/sum_Dx for d in Distances])
            initial_centroids.append(X[jth_centroid])


        return initial_centroids
    


    def predict(self,X) : 
        """
        method to predict the closest cluster of the new instances of data 
        
        """
        predictions = []
        for xi in X :
            predictions.append(np.argmin([np.linalg.norm(xi-c) for c in self.centroids]))
        return predictions

       