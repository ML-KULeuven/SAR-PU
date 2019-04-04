"""This module labels the data based on the real classes and the attribute values.
"""
import re

import dill as pickle
import numpy as np

from sarpu.input_output import *
from sarpu.paths_and_names import *


def label_data(data_folder,
               data_name,
               labeling_model_type,
               propensity_attributes,
               propensity_attributes_signs,
               nb_assignments,
               relabel_data=False,
               seed=123):
    """
    Labels the data based on the real classes and the attribute values.

    Parameters
    ----------
    data_folder : str
        Path to the directory where all datasets are stored.
    data_name : str
        Name of the dataset to label.
    labeling_model_type : str
        The unique name of the labelling mechanism. Each labeling mechanism
        has a unique name, so given this name the correct labeling mechanism
        (and its parameters) can be instantiated. This is important for the
        file names and experiments. For now, there is only one labeling
        mechanism: `SimplePropensity`. E.g., "simple_0.2_0.8"
    propensity_attributes : list(int)
        The attributes to use for the labelling.
    propensity_attributes_signs : list(-1|+1)
        Indicates whether the attribute values should be negated or not. This
        is useful to switch between positive and negative correlation.
    nb_assignments : int
        Number of label assignments.
    seed : int, optional
    """
    np.random.seed(seed)

    labeling_model = parse_labeling_model(labeling_model_type,
                                          propensity_attributes,
                                          propensity_attributes_signs)

    data_folder_labelings = labelings_data_path(data_folder, data_name)
    os.makedirs(data_folder_labelings, exist_ok=True)

    x, y = read_data((data_path(data_folder, data_name),
                      classlabels_path(data_folder, data_name)))

    if relabel_data or not os.path.exists(propensity_model_path(data_folder, data_name, labeling_model)):
        _save_propensity_model_and_labels(
            data_folder, data_name,
            x, y,
            labeling_model,
            nb_assignments,
            seed=None)

    return labeling_model


def parse_labeling_model(labeling_model_type, propensity_attributes, propensity_attributes_signs):
    """Extracts the model name and parameters from a string and creates
    a corresponding LabelingMechanism object.
    """
    match = re.match(r"^simple_(.*)_(.*)$", labeling_model_type)
    if match:
        min_prob, max_prob = tuple(map(float, match.group(1, 2)))
        return SimpleLabeling(propensity_attributes, propensity_attributes_signs, min_prob, max_prob)

    raise ValueError("Could not parse the labeling model type %s " % labeling_model_type)


def _save_propensity_model_and_labels(data_folder, data_name, x, y,
        propensity_model, nb_assignments=10, seed=None):
    """Computes propensity scores, generates labellings and saves everything.
    """
    if seed is not None:
        np.random.seed(seed)

    # Compute the propensity scores
    propensity_model.fit(x, y)
    propensity_scores = propensity_model.propensity_scores(x)

    # Save the propensity model
    out_model = propensity_model_path(data_folder, data_name, propensity_model)
    with open(out_model, "wb+") as out_model_file:
        pickle.dump(propensity_model, out_model_file)
    print('Propensity model saved to {}'.format(out_model))

    # Save the propensity scores
    out_scores = propensity_scores_path(data_folder, data_name, propensity_model)
    np.savetxt(out_scores, propensity_scores)
    print('Propensity scores saved to {}'.format(out_scores))

    # Save the propensity labels
    for i in range(nb_assignments):
        out_labeling = propensity_labeling_path(data_folder, data_name, propensity_model, i)
        s = _calc_s(y, propensity_scores)
        np.savetxt(out_labeling, s, fmt='%d', delimiter=",")
        print('Propensity labels of assignment {} saved to {}'.format(i, out_labeling))


def _calc_s(y, propensity_scores):
    """Generates a labelling, given the propensity scores."""
    return (y * (np.random.random(y.shape) <= propensity_scores)).astype(int)



# Propensity models ##########################################################

class BaseLabelingMechanism:
    def __init__(self, propensity_attributes):
        self.propensity_attributes = sorted(propensity_attributes)
        self.nb_propensity_attributes = len(self.propensity_attributes)

    def type_name(self):
        pass

    def name(self):
        return self.type_name() + "._." + ".".join(
            map(str, self.propensity_attributes))

    def propensity_scores(self, x):
        pass

    def fit(self, x, y, w=None):
        pass

    def predict_proba(self, x):
        return self.propensity_scores(x)


class SimpleLabeling(BaseLabelingMechanism):
    def __init__(self,
                 propensity_attributes,
                 propensity_attribute_signs,
                 min_prob=0.2,
                 max_prob=0.8):
        propensity_attributes, propensity_attribute_signs = zip(
            *sorted(zip(propensity_attributes, propensity_attribute_signs)))
        super().__init__(propensity_attributes)
        self.propensity_attribute_signs = propensity_attribute_signs
        self.min_prob = min_prob
        self.max_prob = max_prob

    def type_name(self):
        return "simple_" + str(self.min_prob) + "_" + str(self.max_prob)

    def name(self):
        return self.type_name() + "._." + ".".join(
            map(
                lambda a: ("" if a[1] > 0 else "-") + str(a[0]),
                zip(self.propensity_attributes, self.propensity_attribute_signs)))

    def propensity_scores(self, x):
        """
        .. math::
            e(x_e) = \prod_{i=1}^k \left( sc(x_e^{(i)},p^-,p^+)\right)^\frac{1}{k}
            sc(x_e^{(i)},p^-,p^+)=p^-+\frac{x_e^{(i)}-\min x_e^{(i)}}{\max x_e^{(i)} - \min x_e^{(i)}}(p^+-p^-)
        """
        # Select the propensity attributes and optionally negate them
        x_e = x[:, np.abs(self.propensity_attributes)] * np.sign(self.propensity_attributes)
        # Compute the propensity score of each instance
        scaled = self.min_prob + (x_e - self.minx) / (self.maxx - self.minx) * (self.max_prob - self.min_prob)
        return (scaled**(1 / self.nb_propensity_attributes)).prod(1)

    def fit(self, x, y, w=None):
        x_e = x[:, np.abs(self.propensity_attributes)] * np.sign(self.propensity_attributes)
        self.minx = x_e.min(0)
        self.maxx = x_e.max(0)

