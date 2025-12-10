import numpy as np
from collections import Counter

class KNN:                                                      # class for our KNN
    def __init__(self,k,dist_type):
        self.k=k
        self.dist_type=dist_type

    def fit(self,X,y):
        self.m,self.n=X.shape
        self.X_train=X
        self.y_train=y
        if(self.dist_type=="mahalanobis"):
            cov=np.cov(X.T)
            self.cov_inv=np.linalg.inv(cov)
    
    def distance(self,a,b):
        if(self.dist_type=="euclidean"):                               # d=√∑(xi​−yi​)²
            return np.sqrt(np.sum((b-a)**2))
    
        elif(self.dist_type=="manhattan"):                              # d=∑|xi-yi|
            return np.sum(np.abs(b-a))

        elif(self.dist_type=="mahalanobis"):                            # d=√(x-y)^T S^-1(x-y)  here T is transpose and S is the covariance matrix
            diff=(a-b).reshape(-1)
            return np.sqrt(diff.T @ self.cov_inv @ diff)
        
        elif(self.dist_type=="rbf"):                                    # sim=e^(−γ∥x−y∥²)  d=1-sim
            gamma=0.0001
            sq=np.sum((a-b)**2)
            return 1-np.exp(-gamma*sq)
        
        elif(self.dist_type=="rbf_normalized"):                         # d=||x-y||^2/n_features
            sq=np.sum((a-b)**2)
            return sq/self.n
        
        else:
            raise ValueError("Unknow Distance metric.")
        
    def predict_class(self,new_point):                                  # Search for the nearest neighbours and identifies the class of the data point
        distances=[self.distance(point,new_point) for point in self.X_train]

        k_nearest_neighbour=np.argsort(distances)[:self.k]
        k_nearest_labels=[self.y_train[i] for i in k_nearest_neighbour]
        most_common=Counter(k_nearest_labels).most_common(1)                # it returns something like this [('B',4)] representing the class label and its count
        return most_common[0][0]                                            # for the above commented example it will return 'B'
    
    def predict(self,new_points):                                       # Fucntion which should be called to predict the class of a data point
        predictions=[self.predict_class(new_point) for new_point in new_points]
        return np.array(predictions)  
    