import numpy as np
import sklearn.linear_model
import time
from sarpu.PUmodels import *



def pu_learn_sar_e(x, s, e, classification_model=None, classification_attributes=None):
    start = time.time()
    if classification_model is None:
        classification_model = LogisticRegressionPU()
    if classification_attributes is None:
        classification_attributes = np.ones(x.shape[1]).astype(bool)

    classification_model = LimitedFeaturesModel(classification_model, classification_attributes)

    classification_model.fit(x,s,e=e)

    info = {'time':time.time()-start}

    return classification_model, info


def pu_learn_scar_c(x, s, c, classification_model=None, classification_attributes=None):
    start = time.time()
    if classification_model is None:
        classification_model = LogisticRegressionPU()
    if classification_attributes is None:
        classification_attributes = np.ones(x.shape[1]).astype(bool)

    classification_model = LimitedFeaturesModel(classification_model, classification_attributes)

    e = np.ones_like(s)*c
    classification_model.fit(x,s,e=e)

    info = {'time':time.time()-start}

    return classification_model, info



def pu_learn_neg(x, s, classification_model=None, classification_attributes=None):
    start = time.time()
    if classification_model is None:
        classification_model = LogisticRegressionPU()
    if classification_attributes is None:
        classification_attributes = np.ones(x.shape[1]).astype(bool)

    classification_model = LimitedFeaturesModel(classification_model, classification_attributes)

    e = np.ones_like(s)
    classification_model.fit(x,s)

    info = {'time':time.time()-start}

    return classification_model, info



def pu_learn_sar_em(x,
                 s,
                 propensity_attributes,
                 classification_attributes=None,
                 classification_model=None,
                 propensity_model=None,
                 max_its=500,
                 slope_eps=0.0001,
                 ll_eps=0.0001,
                 convergence_window=10,
                 refit_classifier=True
                 ):

    start = time.time()
    if classification_model is None:
        classification_model = LogisticRegressionPU()
    if propensity_model is None:
        propensity_model = LogisticRegressionPU()
    if classification_attributes is None:
        classification_attributes = np.ones(x.shape[1]).astype(bool)

    if len(propensity_attributes)==0:
        propensity_model = NoFeaturesModel()

    classification_model = LimitedFeaturesModel(classification_model, classification_attributes)
    propensity_model = LimitedFeaturesModel(propensity_model, propensity_attributes)

    info = {}

    initialize_simple(x, s, classification_model, propensity_model)

    expected_prior_y1 = classification_model.predict_proba(x)
    expected_propensity = propensity_model.predict_proba(x)
    expected_posterior_y1 = expectation_y(expected_prior_y1, expected_propensity, s)

    # loglikelihood
    ll = loglikelihood_probs(expected_prior_y1, expected_propensity, s)
    loglikelihoods = [ll]

    # propensity slope
    past_propensities = np.zeros([int(len(s)-sum(s)),convergence_window])
    propensity_slope = []
    max_ll_improvements = []

    i=0
    for i in range(max_its):
        #maximization
        propensity_model.fit(x, s, sample_weight=expected_posterior_y1)

        classification_s = np.concatenate([np.ones_like(expected_posterior_y1), np.zeros_like(expected_posterior_y1)])
        classification_weights = np.concatenate([expected_posterior_y1, 1-expected_posterior_y1])
        classification_model.fit(np.concatenate([x, x], axis=0), classification_s, sample_weight=classification_weights)


        # expectation
        expected_prior_y1 = classification_model.predict_proba(x)
        expected_propensity = propensity_model.predict_proba(x)
        expected_posterior_y1 = expectation_y(expected_prior_y1, expected_propensity, s)


        # loglikelihood
        ll = loglikelihood_probs(expected_prior_y1, expected_propensity, s)
        loglikelihoods.append(ll)

        # convergence
        push(past_propensities, expected_propensity[s==0])
        if i>convergence_window:
            max_ll_improvement = max(loglikelihoods[-convergence_window:]) - loglikelihoods[-convergence_window]
            max_ll_improvements.append(max_ll_improvement)
            average_abs_slope = np.average(np.abs(slope(past_propensities, axis=1)))
            propensity_slope.append(average_abs_slope)
            if average_abs_slope<slope_eps and max_ll_improvement < ll_eps:
                break #converged

    if refit_classifier:
        classification_model.fit(x,s,e=expected_propensity)

    info['nb_iterations']=i
    info['time']=time.time()-start
    info['loglikelihoods']=loglikelihoods
    info['propensity_slopes']=propensity_slope
    info['max_ll_improvements']=max_ll_improvements

    return classification_model, propensity_model, info



def initialize_simple(instances, labels, classification_model, propensity_model):
    """Initialization with unlabeled=negative, but reweighting the examples so that the expected class prior is 0.5"""
    proportion_labeled = labels.sum()/labels.size
    classification_weights =labels*(1-proportion_labeled)+(1-labels)*proportion_labeled
    classification_model.fit(instances,labels, sample_weight=classification_weights)
    classification_expectation = classification_model.predict_proba(instances)
    propensity_model.fit(instances,labels,sample_weight=(labels + (1-labels)*classification_expectation))

def expectation_y(expectation_f,expectation_e, s):
    result= s + (1-s) * (expectation_f*(1-expectation_e))/(1-expectation_f*expectation_e)
    return result



#  Expected loglikelihood of the model probababilities
def loglikelihood_probs(class_probabilities, propensity_scores, labels):
    prob_labeled = class_probabilities*propensity_scores
    prob_unlabeled_pos = class_probabilities*(1-propensity_scores)
    prob_unlabeled_neg = 1-class_probabilities
    prob_pos_given_unl = prob_unlabeled_pos/(prob_unlabeled_pos+prob_unlabeled_neg)
    prob_neg_given_unl = 1-prob_pos_given_unl
    return (
        labels*np.log(prob_labeled)+
        (1-labels)*(
            prob_pos_given_unl*np.log(prob_unlabeled_pos)+
            prob_neg_given_unl*np.log(prob_unlabeled_neg))
    ).mean()


def slope(array, axis=0):
    """Calculate the slope of the values in ar over dimension "axis". The values are assumed to be equidistant."""
    if axis==1:
        array = array.transpose()

    n = array.shape[0]
    norm_x = np.asarray(range(n))-(n-1)/2
    auto_cor_x = np.square(norm_x).mean(0)
    avg_y = array.mean(axis=0)
    norm_y = array - avg_y
    cov_x_y = np.matmul(norm_y.transpose(),norm_x)/n
    result = cov_x_y/auto_cor_x
    if axis==1:
        result = result.transpose()
    return result


def push(array_queue, new_array):
    array_queue[:,:-1]=array_queue[:,1:]
    array_queue[:,-1]= new_array


class NoFeaturesModel:

    def __init__(self, prior=0.5):
        self.prior = prior

    def fit(self, x,y,sample_weight=None):
        if sample_weight is None:
            self.prior=y.mean()
        else:
            self.prior = (y*sample_weight).mean()

    def predict_proba(self, x):
        return np.ones(x.shape[0])*self.prior

class LimitedFeaturesModel:
    def __init__(self, model, features):
        self.model = model
        self.features = features

    def predict_proba(self, x):
        pr = self.model.predict_proba(x[:,self.features])
        if np.ndim(pr)>1 and np.shape(pr)[1]>1:
            pr = pr[:,1]
        return pr

    def fit(self,x,y,e=None,sample_weight=None):
        if issubclass(type(self.model), BasePU):
            self.model.fit(x[:,self.features], y,e, sample_weight)
        else:
            self.model.fit(x[:,self.features], y, sample_weight)
        return self

