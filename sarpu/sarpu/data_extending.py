"""This module extends a dataset with artificially generated attributes.

A dataset can be extended with extra attributes that are independent of all
other attributes, but possibly correlated with the target class. This allows
controlled experiments.
"""
import os
import shutil

import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.cluster import KMeans

from sarpu.data_extending import *
from sarpu.input_output import *
from sarpu.paths_and_names import *


def generate_extended_data(data_folder, data_name, n_extra_attributes, nb_partitions):
    """
    Generate an extra attribute for each neg_pr in extra_attributes_neg_probs.

    Parameters
    ----------
    data_folder : str
        Path to the directory where all datasets are stored.
    data_name : str
        Name of the dataset to extend.
    n_extra_attributes : int

    nb_partitions : int
        Number of splits in the original dataset.
    """
    # load the original dataset
    x_path = data_path(data_folder, data_name)
    y_path = classlabels_path(data_folder, data_name)
    x, y = read_data((x_path, y_path))

    np.random.seed(123)

    # generate the artificial attributes
    extra_atts = []
    for _ in range(n_extra_attributes):
        att_vals = generate_attributes(x, y)
        extra_atts.append(att_vals)

    # append the generated attributes to the original dataset
    x_new = np.concatenate([x, np.stack(extra_atts, axis=1)], axis=1)

    # create a directory to store the extended dataset
    new_data_name = data_name + "_extclustering"
    os.makedirs(partitions_data_path(data_folder, new_data_name), exist_ok=True)

    # copy the true labels and partitions from the original to the extended dataset
    shutil.copy(
        classlabels_path(data_folder, data_name),
        classlabels_path(data_folder, new_data_name))
    for i in range(nb_partitions):
        shutil.copy(
            partition_path(data_folder, data_name, i),
            partition_path(data_folder, new_data_name, i))

    # save the extended dataset
    np.savetxt(data_path(data_folder, new_data_name), x_new)

def generate_attributes(x, y, seed=None):
    """Generates an artificial attribute with `nb_discretizations` possible values.

    Parameters
    ----------
    y : list()
        The target class of each instance.
    seed : int, optional

    Returns
    -------
    list
        a list with the generated attribute value for each instance.
    """
    kmns = KMeans(n_clusters=5)#, init='k-means++', n_init=10, max_iter=1000, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
    kY = kmns.fit_predict(x)
    df = pd.DataFrame({'cluster': kY, 'class': y})
    probs = df.pivot_table(index=['cluster'], columns='class', aggfunc='size', fill_value=0)
    probs['neg_probs'] = probs[0.0] / probs[[0.0, 1.0]].sum(axis=1) - 0.5
    probs['pos_probs'] = probs[1.0] / probs[[0.0, 1.0]].sum(axis=1) - 0.5
    probs['att+'] = np.random.uniform(size=len(probs))
    probs['att-'] = 1 - probs['att+']
    return _generate_attribute_values(y, kY, probs, seed)


def _generate_attribute_values(y, kY, probs, seed=None):
    """Assigns an attribute value to each instance, given the probability
    distributions P(x|y=+1) and P(x|y=-1).
    """
    if seed is not None:
        np.random.seed(seed)
    sigma = 0.1
    vals = [-1,1]
    atts = np.zeros_like(y, dtype=float)
    #atts[y == 1] = sigma * np.random.randn((y == 1).sum()) + probs.loc[kY[y == 1], 'pos_probs']
    #atts[y == 0] = sigma * np.random.randn((y == 0).sum()) + probs.loc[kY[y == 0], 'pos_probs']
    for i, k in enumerate(kY):
        atts[i] = np.random.choice(vals, 1, p=probs.loc[k, ['att-', 'att+']].values)
    #atts[y == 1] = np.random.choice(vals, (y == 1).sum(), p=pos_probs)
    #atts[y == 0] = np.random.choice(vals, (y == 0).sum(), p=neg_probs)
    return atts
