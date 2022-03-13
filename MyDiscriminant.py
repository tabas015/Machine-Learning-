import numpy as np

class GaussianDiscriminant:
    def __init__(self, k=2, d=8, priors=None, shared_cov=False):
        self.mean = np.zeros((k,d)) # mean
        self.shared_cov = shared_cov # using class-independent covariance or not
        if self.shared_cov:
            self.S = np.zeros((d,d)) # class-independent covariance (S1=S2)
        else:
            self.S = np.zeros((k,d,d)) # class-dependent covariance (S1!=S2)
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain): # have to fit gaussian distribution 
        # compute the mean for each class
        x = []
        x = Xtrain[ytrain == 1,] #seperate into two classes
        
        for i in np.arange(Xtrain.shape[0]): #calculating mean for each class 
            self.mean.append(np.mean(x, axis = 0))

        if self.shared_cov: #class independent 
            # compute the class-independent covariance
            b = np.cov(Xtrain.T, ddof = 0)
            self.S  = b 
        else:
            # compute the class-dependent covariance divide the classes 
            x_t = []
            x_t = Xtrain[ytrain == 0,]
            for i in np.arange(Xtrain.shape[0]):
                b = np.cov(Xtrain.T, ddof = 0)
                Self.S = b 
                

    def predict(self, Xtest):
        # predict function to get predictions on test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder
        pred =[]
        for i in np.arange(Xtest.shape[0]): # for each test set example
            g = np.linalg.det(self.S) 
            
            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                if self.shared_cov:
                    pred[0] = -1/2 np.log(g) -1/2 (Xtest.T)*g.T* Xtrain -2* X.T.*s.mean + s.mean.T*s*s.mean +0.7
                else:
                    pred[1] = -1/2 (Xtest-S.mean).T*S*(Xtest-S.mean)+0.3

            # determine the predicted class based on the values of discriminant function
            if pred[0] > pred[1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2


        return predicted_class

    def params(self):
        if self.shared_cov:
            return self.mean[0], self.mean[1], self.S
        else:
            return self.mean[0],self.mean[1],self.S[0,:,:],self.S[1,:,:]


class GaussianDiscriminant_Diagonal:
    def __init__(self,k=2,d=8,priors=None):
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((d,)) # variance
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
        self.k = k
        self.d = d
    
    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        x = []
        x = Xtrain[ytrain == 1,] #seperate into two classes
        
        for i in np.arange(Xtrain.shape[0]): #calculating mean for each class 
            self.mean.append(np.mean(x, axis = 0))
        
        # compute the variance of different features
        b = np.cov (x, ddof=None)
        self.S = b


    def predict(self, Xtest):
        # predict function to get prediction for test set
        predicted_class = np.ones(Xtest.shape[0]) # placeholder
        pred= []
        for i in np.arange(Xtest.shape[0]): # for each test set example
            # calculate the value of discriminant function for each class
            g = np.linalg.det(self.S) 
            for c in np.arange(self.k):
                pred[]= -1/2 ((Xtest.T-S.mean)/g)**2 + 0.3

            # determine the predicted class based on the values of discriminant function
            if pred[0] > pred[1]:
                predicted_class[i] = 1
            else:
                predicted_class[i] = 2

        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S
