import numpy as np


def accuracy(true, pred):
    return np.average(1 - abs(pred - true))


def tp(true, pred):
    return (pred * true).mean()
    #return sum(pred[true==1])


def fp(true, pred):
    return (pred * (1 - true)).mean()
    #return sum(pred[true==0])


def tn(true, pred):
    return ((1 - pred) * (1 - true)).mean()
    #return sum((1-pred)[true==0])


def fn(true, pred):
    return ((1 - pred) * true).mean()
    #return sum((1-pred)[true==1])


def tpfptnfn(true, pred):
    return tp(true, pred), fp(true, pred), tn(true, pred), fn(true, pred)


def accuracy_tpfntnfn(tp, fp, tn, fn):
    return (tp + tn) / (tp + fp + tn + fn)


def precision_tpfptnfn(tp, fp, tn, fn):
    if tp + fp == 0.0:
        return 0.0
    return tp / (tp + fp)


def recall_tpfptnfn(tp, fp, tn, fn):
    if tp + fn == 0.0:
        return 0.0
    return tp / (tp + fn)


def f1_score_tpfptnfn(tp, fp, tn, fn):
    prec = precision_tpfptnfn(tp, fp, tn, fn)
    rec = recall_tpfptnfn(tp, fp, tn, fn)
    if (prec + rec) == 0.0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def prp_tpfptnfn(tp, fp, tn, fn):
    return (tp + fp) / (tp + fp + tn + fn)


def rec2_tpfptnfn(tp, fp, tn, fn):
    rec = recall_tpfptnfn(tp, fp, tn, fn)
    prp = prp_tpfptnfn(tp, fp, tn, fn)
    if prp == 0.0:
        return float('inf')
    return rec * rec / prp


def expected_loglikelihood(class_probabilities, propensity_scores, labels):
    prob_labeled = class_probabilities * propensity_scores
    prob_unlabeled_pos = class_probabilities * (1 - propensity_scores)
    prob_unlabeled_neg = 1 - class_probabilities
    prob_pos_given_unl = prob_unlabeled_pos / (
        prob_unlabeled_pos + prob_unlabeled_neg)
    prob_neg_given_unl = 1 - prob_pos_given_unl
    prob_unlabeled_pos[prob_unlabeled_pos ==
                       0] = 0.00000001  #prevent problems of taking log
    prob_unlabeled_neg[prob_unlabeled_neg ==
                       0] = 0.00000001  #prevent problems of taking log
    return (labels * np.log(prob_labeled) + (1 - labels) *
            (prob_pos_given_unl * np.log(prob_unlabeled_pos) +
             prob_neg_given_unl * np.log(prob_unlabeled_neg))).mean()


def label_frequency(class_probabilities, propensity_scores):
    total_pos = class_probabilities.sum()
    total_labeled = (class_probabilities * propensity_scores).sum()
    return total_labeled / total_pos
