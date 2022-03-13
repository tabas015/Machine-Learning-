#import libraries
import numpy as np

class Kmeans:
    def __init__(self,k=8): # k is number of clusters
        self.num_cluster = k
        self.center = None # centers for different clusters
        self.cluster_label = np.zeros([k,]) # class labels for different clusters
        self.error_history = []

    def fit(self, X, y):
        # initialize the centers of clutsers as a set of pre-selected samples
        init_idx = [1, 200, 500, 1000, 1001, 1500, 2000, 2005]
         
        self.centroids = []
        
        for i in init_idx:
            self.centroids.append(X[i])

        num_iter = 0 # number of iterations for convergence

        # initialize the cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False

        # iteratively update the centers of clusters till convergence
        while not is_converged:    
            clusters = []
            for i in range(len(init_idx)):
                clusters.append([])

            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                
                distances = [] 
            
                # use euclidean distance to measure the distance between sample and cluster centers
                for j in range(self.num_cluster):
                    
                    dist = np.linalg.norm(X[i] - self.centroids[j])
                    distances.append(dist)
                    cluster_assignment[i] = np.argmin(distances)
                

                # determine the cluster assignment by selecting the cluster whose center is closest to the sample

            # update the centers based on cluster assignment (M step)
                for k in range(len(self.centroids)):
                    self.centroids[k] = X[cluster_assignment==k].mean(axis=0)

            # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1


        # compute the class label of each cluster based on majority voting (remember to update the corresponding class attribute)
        for i in range(self.num_cluster) :
  
            cluster_labels_i = y[np.where(cluster_assignment == i)].astype(int)
            self.cluster_label[i] = np.bincount(cluster_labels_i).argmax()


        return num_iter, self.error_history

    def predict(self,X):
        # predicting the labels of test samples based on their clustering results
        prediction = np.ones([len(X),]) # placeholder

        # iterate through the test samples
        for i in range(len(X)):
            # find the cluster of each sample
            distances = np.array(list(map(np.linalg.norm, X[i] - self.centroids)))
            # use the class label of the selected cluster as the predicted class
            prediction[i] = self.cluster_label[np.argmin(distances)]

        return prediction





    def compute_error(self,X,cluster_assignment):
        # compute the reconstruction error for given cluster assignment and centers
        error = 0 # placeholder
        
        for i in range(len(cluster_assignment)):
            
             
            if (cluster_assignment[i] == 0):
                diff = X[i] - self.centroids[0]
            if (cluster_assignment[i] == 1):
                diff = X[i] - self.centroids[1]
            if (cluster_assignment[i] == 2):
                diff = X[i] - self.centroids[2]
            if (cluster_assignment[i] == 3):
                diff = X[i] - self.centroids[3]
            if (cluster_assignment[i] == 4):
                diff = X[i] - self.centroids[4]
            if (cluster_assignment[i] == 5):
                diff = X[i] - self.centroids[5]
            if (cluster_assignment[i] == 6):
                diff = X[i] - self.centroids[6]
            if (cluster_assignment[i] == 7):
                diff = X[i] - self.centroids[7]
            sum_sq = np.sum(np.square(diff))
            error = error + sum_sq

        return error

    def params(self):
        return self.center, self.cluster_label
