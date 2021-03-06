import numpy as np

#------------------------------------------------------------------
def linear_kernel(x, y, b):
    return np.dot(x, y) - b


def polynomial_kernel(x, y, p, b):
    return ((1 + np.dot(x, y)) ** p) - b


def RBF_kernel(x, y, b, sigma):
    numerator = np.array([np.linalg.norm(x[i] - y) for i in range(len(x))])
    denominator = round((sigma ** 2),2)
    return np.exp((-1*numerator) / denominator)

def tanh_kernel(x, y, b, landa):
    return np.tanh(landa * np.dot(x, y) - b)

#------------------------------------------------------------------
class SVM:

    def __init__(self):
        self.lr = 0.001
        self.n_iters = 1000
        self.w = None
        self.b = None


    def fit(self, X, y):
        
        y_ = np.where(y <= 0, -1, 1)        
        self.w = np.zeros(X.shape[1])
        self.b = 0
        
        for _ in range(self.n_iters):
            for index, x_i in enumerate(X):
                temp = y_[index] * (np.dot(x_i, self.w) - self.b) 
                if temp >= 1:
                    self.w -= self.lr * (2 * self.w)
                else:
                    self.w -= self.lr * (2 * self.w - np.dot(x_i, y_[index]))
                    self.b -= self.lr * y_[index]


    def predict(self, X,kernel):
        
        if kernel == linear_kernel:
            Kernel = kernel(X,self.w,self.b)
            
        elif kernel == polynomial_kernel:
            Kernel = kernel(X,self.w,3,self.b)
        
        elif kernel == RBF_kernel:
            Kernel = kernel(X, self.w, self.b,0.1)
            
        elif kernel == tanh_kernel:
            Kernel = kernel(X, self.w, self.b,3)
            
        return np.sign(Kernel)
#------------------------------------------------------------------
        
