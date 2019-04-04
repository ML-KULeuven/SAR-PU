import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class BasePU:
    @staticmethod
    def _make_propensity_weighted_data(x,s,e,sample_weight=None):
        weights_pos = s/e
        weights_neg = (1-s) + s*(1-1/e)
        if sample_weight is not None:
            weights_pos = sample_weight*weights_pos
            weights_neg = sample_weight*weights_neg
            
        Xp = np.concatenate([x,x])
        Yp = np.concatenate([np.ones_like(s), np.zeros_like(s)])
        Wp = np.concatenate([weights_pos, weights_neg])
        return Xp, Yp, Wp

class LogisticRegressionPU(LogisticRegression, BasePU):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        LogisticRegression.__init__(self,penalty=penalty, dual=dual, tol=tol, C=C, 
                         fit_intercept=fit_intercept,intercept_scaling=intercept_scaling,
                        class_weight=class_weight, random_state=random_state,
                        solver=solver, max_iter=max_iter, multi_class=multi_class,
                        verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
               
        
    def fit(self, x, s, e=None, sample_weight=None):
        if e is None:
            super().fit(x,s,sample_weight)
        else:
            Xp,Yp,Wp = self._make_propensity_weighted_data(x,s,e,sample_weight)
            super().fit(Xp,Yp,Wp)
            
        