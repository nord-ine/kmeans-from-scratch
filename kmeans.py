import numpy as np


class kmeans : 

    def __init__(self,nb_clusters,initialisation,max_iteration = 200):
        self.k = nb_clusters
        self.init = initialisation
        self.max_iter = max_iteration
        self.clusters = []
    
    def fit(self,X):


        pass




    def assignto_cluster(self,X,cent):
        Cm= 0
        for i,xi in enumerate(X) : 
            Cm = np.argmin([np.linalog.norm(xi-c) for c in cent])
            clusters[Cm].append(i)





     
    def init_centroids(self,X): 
        if(self.init == "random"):
            return self.random_init(X)

        elif (self.init == "kmeans++"):
            return self.kmeanspp_init(X) 
        else :
            print("incorrect initialisation name") # chan,ge this to an exception 
        

    def random_init(self,X):
        pass


    def kmeanspp_init(self,X):
        pass