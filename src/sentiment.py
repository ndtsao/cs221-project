#!/usr/bin/python

import random
import collections
import math
import sys
import util_sentiment
import util_load

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    features = {}
    words = x.split()
    for word in words:
        if word in features:
            features[word] += 1
        else:
            features[word] = 1
    return features
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    stepSize = 0.1
    numIters = 10

    def gradient(weights, feature, y):
        if len(feature) == 0:
            return 0
        loss = (dotProduct(feature, weights) - y) ** 2
        if loss == 0:
            return -1
        else:
            update = {}
            for k, v in feature.items():
                update[k] = 2 * (feature - y)

            return update

    def predictor(x):
        if dotProduct(x, weights) > 0:
            return 1
        return -1
    
    for i in range(numIters):
        for x, y in trainExamples:
            feature = featureExtractor(x)
            update = gradient(weights, feature, y)
            if update != -1:
                increment(weights, stepSize, update)

     
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        phi = {}
       
        numKeys = random.randint(1, len(weights))
        for i in range(numKeys):
            key = random.choice(weights.keys())
            phi[key] = random.randint(1, 10)
        score = dotProduct(weights, phi)

        if score > 0:
            y = 1
        else:
            y = -1

        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]


def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        x = x.replace("\t", "").replace(" ", "")
        features = {}

        for i in range(0, len(x) + 1 - n):
            ngram = x[i:i + n]
            if ngram in features:
                features[ngram] += 1
            else:
                features[ngram] = 1
        
        return features
        # END_YOUR_CODE
    return extract


reviews = load(IMPORT_PATH, IMPORT_COLS)

