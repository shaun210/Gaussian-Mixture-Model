"""
Student1: Dian Basit
McGill ID: 260771254

Student2: Shaun Soobagrah
McGill ID: 260919063
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix

from PIL import Image
from numpy import asarray
from scipy import linalg
from scipy.special import logsumexp
import statistics

import skimage.io
import matplotlib.pyplot as plt
import skimage.filters

class GMM:
  def __init__(self, k, verbose=False, graphing=False, epoch = 1e2, reg = 1e-4, epsilon=1e-3, description = ""):
    """
      # Parameters
        k = hyperparameter which gives number of GMM mixtures
        verbose = gives epoch and mean update after every iteration [default = False]
        graphing = gives log-likelihood graph after the iteration [default = False]
        epoch = number of iterations to run [default = 1e-2]
        reg = regularization constant added to covariance for numerical stability [default = 1e-4]
        epsilon = minimum norm threshold to terminate the iterations [default = 1e-3]

      # Returns
        * None
        * Initializes the parameters to self

    IMPORTANT NOTES: THE GMM METHOD HAS BEEN ADAPTED FROM DIAN BASIT'S (260771254) A2 CODE.
    """
    self.k = k
    # self.D = dimension
    self.verbose = verbose
    self.epoch = epoch
    self.graphing = graphing
    self.reg = reg
    self.epsilon = epsilon
    self.likelihood_history=[] # Record sum of likelihood after each iterations (important)
    self.description = description

  def init_params(self, X):
    """
      # Parameters
        X = input data of size (N, D)

      # Returns
        * None
        * Initializes the required mean, covariance, posterior and likelihood
    """
    self.prior = np.ones(self.k, dtype=float)/self.k # (K,) ; equal priors during initialization, weight

    # self.means = [np.random.choice(X[:, i], size=self.k) for i in range(X.shape[1])] # (K x D)
    x_split = np.array_split(X, self.k, axis=0) 
    self.means = [np.mean(data, axis=0) for data in x_split] # (K x D)
    self.means = np.array(self.means) # match the columns with each dimension of X

    self.cov = [make_spd_matrix(X.shape[1]) for i in range(self.k)] # (K x D x D)
    self.cov = np.array(self.cov)

    self.posterior = np.zeros((X.shape[0], self.k), dtype=float)
    self.likelihood = np.zeros((X.shape[0], self.k), dtype=float)

  def fit(self, X):
    """
      # Parameters
        X = input data of size (N, D)
      
      # Returns
        * None
        * Updates the mean, covariance and likelihood variables
    """
    # initialization of hyperparameters
    self.init_params(X)
    # self.likelihood_history=[]
    oldcov = np.copy(self.cov)
    norm=100 # initialize this norm
    t=0
    
    # break the loop if either epoch is reached or norm is less than epsilon
    while(t<self.epoch and norm>self.epsilon): 
      # Expectation Step
      # calculate the normalization constant
      norm = self.normalization(X, self.likelihood, self.prior, self.means, self.cov)
      # use the bayesian theorem formula to get posterior --> POSTERIOR = (LIKELIHOOD * PRIOR)/NORMALIZATION
      self.posterior = self.expectation(X, self.prior, self.means, self.cov, norm)
   
      # this value will be subtracted from updated norm to get epsilon
      norm_temp = np.copy(norm)

      # Maximization Step
      # update mean, covariance and prior in this method
      self.maximization(X)
      
      # if this norm is < than epsilon then break the loop
      norm = np.linalg.norm(oldcov - self.cov)
      
      # old covariance will be the current covariance
      oldcov = np.copy(self.cov)

      t+=1
      # Verbose if enabled
      if(self.verbose):
        print("%d Epoch Completed" % t)
        print("Mean: \n", self.means)
      # print("%d Epoch Completed" % t)
        
      # self.likelihood_history.append(np.log10(np.sum(self.likelihood, axis=0))) #changed this
      self.likelihood_history.append(np.log10(np.sum(norm_temp)))
    
    print("Cluster %d converged in %d/%d epoch with norm of %.8f" %(self.k, t, self.epoch, norm))
    # print(norm)
    # print(self.epsilon)
    # print(norm>self.epsilon)

    # Graphing if enabled then displays likelihood graph
    if(self.graphing):
      likelihood = np.array(self.likelihood_history)
      # print(np.array(model.likelihood_history).shape)
      fig1 = plt.figure(figsize=(10,5))
      # for i in range(K):
        # plt.plot(np.arange(t), likelihood[:, i], label=str(i+1))
      plt.plot(np.arange(t), likelihood)
      plt.xlabel("Epoch")
      plt.ylabel("Likelihood")
      plt.title("Log-Likelihood vs Iterations for %s" %self.description)
      # plt.legend(loc="best")
    #   plt.show()
    
  def normalization(self, X, likelihood, prior, mean, cov):
    """
      # Parameters
        X --> N, D
        prior --> K,
        mean --> K, D
        cov --> K, D, D
      
      # Returns
        normalization --> N,
    """
    for k in range(self.k):
      self.likelihood[:, k] = multivariate_normal(mean[k, :], cov[k, :]).pdf(X)
    normalization = self.prior*self.likelihood
    normalization = np.sum(normalization, axis=-1) # add all the cluster's likelihood

    # print(normalization.shape)
    assert self.likelihood.shape == (X.shape[0], self.k) # for verification
    assert normalization.shape == (X.shape[0],) # for verification

    return normalization

  def expectation(self, X, prior, mean, cov, norm):
    """
    # Parameters
      X = input data with size N, D
      prior = prior mean with size K,
      mean = size K, D
      cov = size K, D, D
      norm = normalization constant got from normalization() method with size N,

    # Returns 
      posterior = probability with size N, K
    """
    for k in range(self.k):
      self.posterior[:, k] = multivariate_normal(mean[k,:], cov[k,:]).pdf(X)
      self.posterior[:, k] = self.posterior[:, k] / norm
    self.posterior *= self.prior

    assert self.posterior.shape == (X.shape[0], self.k) # for verification

    return self.posterior

  def maximization(self, X):
    """
      # Parameters
        X = input data with size (N, D)

      # Returns
        * None
        * Updates mean, covariance, and prior
    """
    N = X.shape[0]

    w = np.sum(self.posterior, axis=0)
    self.prior = w / N
    assert self.prior.shape == (self.k, )

    for k in range(self.k):
      # print(np.sum(self.posterior[:, k].reshape(-1,1) * X, axis=0).shape)
      self.means[k, :] = np.sum(self.posterior[:, k].reshape(-1,1) * X, axis=0) / w[k]
    assert self.means.shape == (self.k, X.shape[1])

    cov_reg = np.eye(X.shape[1]) * self.reg
    for k in range(self.k):
      self.cov[k] = np.dot((self.posterior[:, k].reshape(-1,1) * (X - self.means[k,:])).T, (X - self.means[k,:]))
      # self.cov[k] = (self.posteior[:,k].reshape(-1,1) * (X-self.mean[k,:]))
      self.cov[k] = self.cov[k] / w[k]
      self.cov[k] += cov_reg
    assert self.cov.shape == (self.k, X.shape[1], X.shape[1])
 
    return None

  def predict_img(self, X):
    """
      # Parameters
        X = input data with size N, D

      # Returns
        labeled_img = returns the pixels correspondence to which cluster with size N,
    """
    labeled_img = np.zeros(X.shape)
    labeled_pixel = np.zeros(X.shape[0], dtype=int)
    
    labeled_pixel = np.argmax(self.likelihood, axis=-1) #look which pixel has maximum likelihood to belong to a cluster

    labeled_img = self.means[labeled_pixel] #which means correspond to which pixel based on likelihood

    return labeled_img

 # Calculating the Bayesian Information Criterion(BIC)
 # implemented from https://github.com/scikit-learn/scikit-learn/blob/baf828ca126bcb2c0ad813226963621cafe38adb/sklearn/mixture/_base.py#L356
  def cost_Fucntion(self,X):
    """
      # Parameters
        X = input data with size N, D

      # Returns
        The first equationn in the report, calculating BIC
    """
    number_of_free_param = (X.shape[1] * self.k)*2 + self.k # number of means, covariance and weight per cluster

    #based on the first equation in the report
    return (-2 * self.score(X) * X.shape[0]) + (number_of_free_param * np.log(
            X.shape[0]))
    
  def score(self, X):
    """
      # Parameters
        X = input data with size N, D

      # Returns
        the maximum log-likelihood
    """
    return statistics.mean(logsumexp(self._estimate_weighted_log_prob(X), axis=1))

  def _estimate_weighted_log_prob(self, X):
    """
      # Parameters
        X = input data with size N, D

      # Returns
        self._estimate_log_gaussian_prob(X) + self._estimate_log_weights()
    """
    return self._estimate_log_gaussian_prob(X) + self._estimate_log_weights()

  
  def _estimate_log_weights(self):
      """
      # Parameters
        self

      # Returns
        The log of the weight matrix
      """
      return np.log(self.prior)
    
  #calculate the the maximum log-likelihood of a Gaussian model
  def _estimate_log_gaussian_prob(self, X):
    
    cholesky = self.cholesky_precision(self.cov) #get the cholesky decomposition of the covariance matrix
    det_cholesky = self._compute_log_det_cholesky(cholesky, X.shape[1]) #get the determinant of the cholesky decomposition

    n_samples, n_features = X.shape
    n_components, _ = self.means.shape
    
    # L^T (x-μ_k) where L is the cholesky decomposition of covariance matrix
    log_probability = np.empty((n_samples, n_components))
    for k, (mu, prec_chol) in enumerate(zip(self.means, cholesky)):
        x = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        log_probability[:,k] = np.sum(np.square(x), axis=1)

    #the third equation in the report
    return - 0.5 * (n_features * np.log(2 * np.pi) + log_probability) + det_cholesky 


  # from https://fossies.org/dox/scikit-learn-1.0.2/__gaussian__mixture_8py_source.html
  def cholesky_precision(self,covariance):
    k, d, _ = covariance.shape
    precision_cholesky = np.empty((k, d, d)) 

    #calculate the cholesky decomposition of the covariance matrix
    for n, covariances in enumerate(covariance):
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError("Cholesky decomposition failed.")
        precision_cholesky[n] = linalg.solve_triangular(cov_chol, 
            np.eye(d), lower=True).T
    return precision_cholesky

  #compute the determinant of the cholesky decomposition
  #of the covariance matrix
  def _compute_log_det_cholesky(self, matrix, d):

    k, _, _ =  matrix.shape
    log_det_chol = (np.sum(np.log(matrix.reshape(k, 
        -1)[:,::d+ 1]),axis=1))
    return log_det_chol


#import Gaussian bkur from skimage
def gaussian_blur(image, sigma=3):

  # apply Gaussian blur, creating a new image
  blurred = skimage.filters.gaussian(
      image, sigma=(sigma, sigma), truncate=3.5, multichannel=True)
  return blurred

if __name__ == '__main__':

    # importing bear image and normalize it to [0,1]
    bear_orig = image.imread("bear.jpg") # I changed this
    bear_copy = bear_orig.copy()
    # bear_copy = bear_copy.reshape(-1, 3)
    bear_copy = bear_copy/255.

    # importing flower image and normalize it to [0, 1]
    flower_orig = image.imread("flower-bouquet-2.jpeg") # changed flower
    flower_copy = flower_orig.copy()
    # flower_copy = flower_copy.reshape(-1, 3)
    flower_copy = flower_copy/255.

    # fig2 = plt.figure(figsize=(10, 5))


    #simple1
    w, h = 512, 512
    data = np.zeros((h, w, 3), dtype=np.uint8)

    data[0:171, 0:512] = [255,0,0]
    data[171:342, 0:512] = [0, 255,0]
    data[342:512, 0:512] = [0,0,255]

    simple1  = Image.fromarray(data, 'RGB')
    simple_img1_orig= asarray(simple1)

    # fig2.add_subplot(1, 2, 1)
    # plt.title("Simple Image 1")
    # plt.imshow(simple1)

    simple_img1_copy = simple_img1_orig.reshape(-1, 3)
    simple_img1_copy = simple_img1_copy/255

    #simple2
    w, h = 512, 512
    data = np.zeros((h, w, 3), dtype=np.uint8)

    data[:,:] = [0,0,255]


    #red 
    data[412:512, 0:101] = [255,0 ,0]
    data[0:101,412:512] = [255,0 ,0]

    #brown
    data[0:101, 0:101] = [0,255,0]
    data[412:512, 412:512] = [0,255,0]

    #blue 
    data[0:200, 262:363] = [0,0,255]
    data[312:512, 150:250] = [0,0,255]

    #green
    data[0:200, 150:250] = [0,255,0]
    data[312:512, 262:363] = [0,255,0]



    simple2  = Image.fromarray(data, 'RGB')
    simple_img2_orig= asarray(simple2)

    simple_img2_copy = simple_img2_orig.reshape(-1, 3)
    simple_img2_copy = simple_img2_copy/255

    #plot all the original images

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    cmap = plt.get_cmap('gray')
    plt.setp(axs, xticks=[], yticks=[])

    axs[0,0].set_title('Simple Image 1', size=8)
    axs[0,0].imshow(simple1, cmap, vmin=0, vmax=1)

    axs[0,1].set_title('Simple Image 2', size=8)
    axs[0,1].imshow(simple2, cmap, vmin=0, vmax=1)

    axs[1,0].set_title('original bear image - from file', size=8)
    axs[1,0].imshow(bear_copy, cmap, vmin=0, vmax=1)
    
    axs[1,1].set_title('original flower image - from file', size=8)
    axs[1,1].imshow(flower_copy, cmap, vmin=0, vmax=1)
    fig.suptitle('Original images', fontsize=16)

    #BLUR

    bear_copy_blur = gaussian_blur(bear_copy, sigma=1)
    flower_copy_blur = gaussian_blur(flower_copy, sigma=1)

    bear_copy_blur = bear_copy_blur.reshape(-1, 3)
    flower_copy_blur = flower_copy_blur.reshape(-1, 3)

    ##test1 for the 2 simple image. Segmentation + log-likelihood ##

    #segmenting simple image1
    K=3
    epoch=100
    epsilon=1e-4
    verbose = False
    Graphing = True

    model = GMM(3, verbose=verbose, graphing = Graphing, epoch=epoch, epsilon=epsilon, description="simple image1")
    model.fit(simple_img1_copy)

    labeled_img_simple_image1 = model.predict_img(simple_img1_copy)
    labeled_img_simple_image1 = np.array(labeled_img_simple_image1 * 255, dtype=int)
    labeled_img_simple_image1 = labeled_img_simple_image1.reshape(simple_img1_orig.shape)

    #plot the segmentation for simple imag1
    fig, axs = plt.subplots(1, 3, constrained_layout=True)

    plt.setp(axs, xticks=[], yticks=[])

    axs[0].set_title('original', size=8)
    axs[0].imshow(simple_img1_orig)

    axs[1].set_title('Labeled', size=8)
    axs[1].imshow(np.mean(labeled_img_simple_image1 , axis=-1))

    axs[2].set_title('Segmented', size=8)
    axs[2].imshow(labeled_img_simple_image1)

    fig.suptitle('Segmentation for simple image1', fontsize=16)
    
    #segmenting simple image2

    K=3
    epoch=100
    epsilon=1e-8
    verbose = False

    model = GMM(3, verbose=verbose, graphing = Graphing, epoch=epoch, epsilon=epsilon, description= "simple image2")
    model.fit(simple_img2_copy)

    labeled_img_simple_image2 = model.predict_img(simple_img2_copy)
    labeled_img_simple_image2 = np.array(labeled_img_simple_image2 * 255, dtype=int)
    labeled_img_simple_image2 = labeled_img_simple_image2.reshape(simple_img2_orig.shape)

    #plot the segmentation for simple imag1
    fig, axs = plt.subplots(1, 3, constrained_layout=True)

    plt.setp(axs, xticks=[], yticks=[])

    axs[0].set_title('original', size=8)
    axs[0].imshow(simple_img2_orig)

    axs[1].set_title('Labeled', size=8)
    axs[1].imshow(np.mean(labeled_img_simple_image2 , axis=-1))

    axs[2].set_title('Segmented', size=8)
    axs[2].imshow(labeled_img_simple_image2)

    fig.suptitle('Segmentation for simple image2', fontsize=16)

    plt.show()

   ###test2 for bear image BIC 

    epoch=100
    epsilon=1e-8
    Verbose = False
    Graphing = False

    #for bear
    BIC_list =[]
    n_components = np.arange(2,11)
    array_length = len(n_components)
    t=0
    for n in n_components:
        temp = GMM(n, verbose=Verbose, graphing = Graphing, epoch=epoch, epsilon=epsilon)
        temp.fit(bear_copy_blur)
        labeled_img = temp.predict_img(bear_copy_blur)
        labeled_img  = np.array(labeled_img * 255, dtype=int)

        labeled_img = labeled_img.reshape(bear_orig.shape) #
        # print(labeled_img.shape)

        fig = plt.figure(figsize=(10,5))

        # t+=1
        fig.add_subplot(1, 3, 1)
        plt.imshow(bear_orig)
        plt.title("Original")
        plt.axis('off')

        # t+=1
        fig.add_subplot(1, 3, 2)
        plt.imshow(np.mean(labeled_img, axis=-1))
        plt.title("Labeled")
        plt.axis('off')

        # t+=1
        fig.add_subplot(1, 3, 3)
       
        plt.title("Segmented with K = %d"%n)
        plt.axis('off')
        plt.imshow(labeled_img)

        BIC_list.append(temp.cost_Fucntion(bear_copy_blur))

    fig = plt.figure(figsize=(10,5))
    plt.plot(n_components, BIC_list)
    plt.title("BIC graph for color segmentation of bear image")
    plt.show()


    ##test3 on flower image

    epoch=200
    epsilon=1e-8
    Verbose = False
    Graphing = False

    #for bear
    BIC_list =[]
    n_components = np.arange(6,13) # we use a limited amount of clusters otherwise it takes too long
    for n in n_components:
        temp = GMM(n, verbose=Verbose, graphing = Graphing, epoch=epoch, epsilon=epsilon, description= "flower image")
        temp.fit(flower_copy_blur)
        labeled_img = temp.predict_img(flower_copy_blur)
        labeled_img  = np.array(labeled_img * 255, dtype=int)

        labeled_img = labeled_img.reshape(flower_orig.shape) #
        # print(labeled_img.shape)

        fig = plt.figure(figsize=(10, 5))
        
        fig.add_subplot(1, 3, 1)
        plt.imshow(flower_orig) #
        plt.title("Original")
        plt.axis('off')

        fig.add_subplot(1, 3, 2)
        plt.imshow(np.mean(labeled_img, axis=-1))
        plt.title("Labeled")
        plt.axis('off')
        
        fig.add_subplot(1, 3, 3)
        plt.imshow(labeled_img)
        plt.title("Segmented with K = %d"% n)
        plt.axis('off')


        BIC_list.append(temp.cost_Fucntion(bear_copy_blur))

    fig = plt.figure(figsize=(10,5))
    plt.plot(n_components, BIC_list)
    plt.title("BIC graph for color segmentation of flower image")
    plt.show()

    ###test4 for bear and flower image with ideal K and log-likelihood graph

    K_bear = 7
    
    epoch=100
    epsilon=1e-8
    Verbose = False
    Graphing = True


    model = GMM(K_bear, verbose=Verbose, graphing=Graphing, epoch=epoch, epsilon=epsilon,description="ideal K value for bear image")
    model.fit(bear_copy_blur)

    labeled_img_simple_image1 = model.predict_img(bear_copy_blur)
    labeled_img_simple_image1 = np.array(labeled_img_simple_image1 * 255, dtype=int)
    labeled_img_simple_image1 = labeled_img_simple_image1.reshape(bear_orig.shape)

    fig = plt.figure(figsize=(10, 5))

    fig.add_subplot(1, 3, 1)
    plt.imshow(bear_orig) #
    plt.title("Original")
    plt.axis('off')

    fig.add_subplot(1, 3, 2)
    plt.imshow(np.mean(labeled_img_simple_image1 , axis=-1))
    plt.title("Labeled")
    plt.axis('off')

    fig.add_subplot(1, 3, 3)
    plt.imshow(labeled_img_simple_image1 )
    plt.title("Segmented")
    plt.axis('off')

    fig.suptitle('Segmentation for bear image with k=7', fontsize=16)
    plt.show()

    ##flower with k = 10

    K_flower = 10
    epoch=200
    epsilon=1e-8
    Verbose = False
    Graphing = True
    
    model = GMM(K_flower, verbose=Verbose, graphing=Graphing, epoch=epoch, epsilon=epsilon,description="ideal K value for flower image")
    model.fit(flower_copy_blur)

    labeled_img_simple_image1 = model.predict_img(flower_copy_blur)
    labeled_img_simple_image1 = np.array(labeled_img_simple_image1 * 255, dtype=int)
    labeled_img_simple_image1 = labeled_img_simple_image1.reshape(flower_orig.shape)

    fig = plt.figure(figsize=(10, 5))

    fig.add_subplot(1, 3, 1)
    plt.imshow(flower_orig) #
    plt.title("Original")
    plt.axis('off')

    fig.add_subplot(1, 3, 2)
    plt.imshow(np.mean(labeled_img_simple_image1 , axis=-1))
    plt.title("Labeled")
    plt.axis('off')

    fig.add_subplot(1, 3, 3)
    plt.imshow(labeled_img_simple_image1 )
    plt.title("Segmented")
    plt.axis('off')
    fig.suptitle('Segmentation for flower image with k=10', fontsize=16)
    plt.show()