"""This module trains and evaluates a PU model."""
import glob
import os
import shutil

import dill as pickle
import numpy as np
import pandas as pd
import sklearn.metrics
from bitarray import bitarray

from km.Kernel_MPE_grad_threshold import wrapper as ramaswamy
from sarpu.evaluation import *
from sarpu.input_output import *
from sarpu.labeling_mechanisms import parse_labeling_model
from sarpu.paths_and_names import *
from sarpu.pu_learning import *
from tice import tice


def train_and_evaluate(data_folder, results_folder, data_name, labeling_model, labeling, partition, settings, pu_method, rerun_experiments=False):
    """Executes an experiment to train and evaluate a pu method for a certain
    dataset and labeling.

    Parameters
    ----------
    data_folder : str
        Path to the directory where all datasets are stored.
    results_folder : str
        Path to the directory where all results are stored.
    data_name : str
        Name of the dataset used in this experiment.
    labeling_model : LabelingMechanism
        The propensity model used to generate the labellings.
    labeling : int
        The ID of the labelling used in this experiment.
    partition : int
        The ID of the partition used in this experiment.
    settings : str
        Settings for the learner, encoded as a string.
    pu_method : ["supervised","negative","sar-e","scar-c","sar-em","scar-km2","scar-tice"]
        The PU learning method to be used.
    rerun_experiments : bool, optional
        Whether to rerun the experiment (default: False).
    """
    out_dir = experiment_method_result_folder_path(results_folder, data_name, labeling_model, labeling, partition, settings, pu_method)
    os.makedirs(out_dir, exist_ok=True)
    print('OUT', experiment_results_path(results_folder, data_name, labeling_model, labeling, partition, settings, pu_method))
    if (
        rerun_experiments
        or not (
            os.path.exists(experiment_results_path(results_folder, data_name, labeling_model, labeling, partition, settings, pu_method)) # results already exist
            # or copy_if_possible(results_folder, data_name, labeling_model, labeling, partition, settings, pu_method) # results were able to be copied
            )
        ):

        classification_model_type, propensity_model_type, classification_attributes = parse_settings(settings)

        # Load data
        x_path = data_path(data_folder,data_name)
        y_path = classlabels_path(data_folder,data_name)
        s_path = propensity_labeling_path(data_folder, data_name, labeling_model, labeling)
        e_path = propensity_scores_path(data_folder, data_name, labeling_model)
        f_path = partition_path(data_folder, data_name, partition)
        (x_train,y_train,s_train,e_train),(x_test,y_test,s_test,e_test) = read_data((x_path,y_path,s_path,e_path),f_path)

        c = (e_train[y_train==1]).mean()

        classification_attributes = classification_attributes(x_train)

        if pu_method == "sar-e":
            f_model, info = pu_learn_sar_e(x_train, s_train, e_train, classification_model_type(), classification_attributes=classification_attributes)
            e_model = pickle.load(open(propensity_model_path(data_folder, data_name, labeling_model), 'rb'))

        elif pu_method == "scar-c":
            f_model, info = pu_learn_scar_c(x_train, s_train, c, classification_model_type(), classification_attributes=classification_attributes)
            e_model = NoFeaturesModel(c)

        elif pu_method == "sar-em":
            f_model, e_model, info = pu_learn_sar_em(x_train, s_train, labeling_model.propensity_attributes, classification_model=classification_model_type(), classification_attributes=classification_attributes, propensity_model=propensity_model_type())

        elif pu_method == "supervised":
            f_model, info = pu_learn_neg(x_train, y_train, classification_model=classification_model_type(), classification_attributes=classification_attributes)
            e_model = NoFeaturesModel(1.0)

        elif pu_method == "negative":
            f_model, info = pu_learn_neg(x_train, s_train, classification_model=classification_model_type(), classification_attributes=classification_attributes)
            e_model = NoFeaturesModel(1.0)

        elif pu_method == "scar-km2":
            start = time.time()
            if len(s_train) > 3200:
                sample = np.random.choice(x_train.shape[0], 3200)
                x_train_ram = x_train[sample]
                s_train_ram = s_train[sample]
            else:
                x_train_ram = x_train
                s_train_ram = s_train
            (_, c_est) = ramaswamy(x_train_ram, x_train_ram[s_train_ram == 1])
            time_c = time.time() - start

            f_model, info = pu_learn_scar_c(x_train, s_train, c_est, classification_model_type(), classification_attributes=classification_attributes)
            e_model = NoFeaturesModel(c_est)

            info['time_c'] = time_c
            info['time_f'] = info['time']
            info['time'] = info['time_c'] + info['time_f']

        elif pu_method == "scar-tice":

            start = time.time()
            x_train_tice = (x_train + 1) / 2
            s_train_tice = bitarray(list(s_train == 1))
            tice_folds = np.random.randint(5, size=len(s_train))
            (c_est, _) = tice.tice(x_train_tice, s_train_tice, 5, tice_folds)
            time_c = time.time() - start

            f_model, info = pu_learn_scar_c(x_train, s_train, c_est, classification_model_type(), classification_attributes=classification_attributes)
            e_model = NoFeaturesModel(c_est)

            info['time_c'] = time_c
            info['time_f'] = info['time']
            info['time'] = info['time_c'] + info['time_f']

        else:
            print("Did not recognize pu_method", pu_method)

        out_info = experiment_info_path(results_folder, data_name, labeling_model, labeling, partition, settings, pu_method)
        print('OUT', out_info)
        with open(out_info, "w+") as out_info_file:
            out_info_file.write("\n".join([k+"\t"+str(v) for k,v in sorted(info.items())]))

        out_classifier = experiment_classifier_path(results_folder, data_name, labeling_model, labeling, partition, settings, pu_method)
        with open(out_classifier, "wb+") as out_classifier_file:
            pickle.dump(f_model, out_classifier_file)

        out_propensity_model = experiment_propensity_model_path(results_folder, data_name, labeling_model, labeling, partition, settings, pu_method)
        with open(out_propensity_model, "wb+") as out_propensity_model_file:
            pickle.dump(e_model, out_propensity_model_file)

        results_test = evaluate_all(y_test,s_test,e_test, f_model.predict_proba(x_test),e_model.predict_proba(x_test))
        results_train = evaluate_all(y_train,s_train,e_train, f_model.predict_proba(x_train),e_model.predict_proba(x_train))

        results = {
            **{"train_" + k: v for k, v in results_train.items()},
            **{"test_" + k: v for k, v in results_test.items()}
        }

        out_results = experiment_results_path(results_folder, data_name, labeling_model, labeling, partition, settings, pu_method)
        with open(out_results, "w+") as out_results_file:
            out_results_file.write("\n".join([k+"\t"+str(v) for k,v in sorted(results.items())]))


def copy_if_possible(results_folder, data_name, labeling_model, labeling, partition, settings, pu_method):

    # For the supervised method, the labeling is not used. Re-use the results from a previous experiment if possible

    if pu_method!="supervised":
        return False
    else:
        try:
            complete_result_path = next(glob.iglob(experiment_method_result_path_nolabel(results_folder, data_name, partition, settings, pu_method)))
            folder = "/".join(complete_result_path.split("/")[:-1])
            out_dir = experiment_method_result_folder_path(results_folder, data_name, labeling_model, labeling, partition, settings, pu_method)
            shutil.rmtree(out_dir)
            shutil.copytree(folder, out_dir)
            return True
        except StopIteration:
            return False


def parse_settings(settings):
    cl_model, pr_model, cl_atts = settings.split('._.')
    return parse_model(cl_model), parse_model(pr_model), parse_cl_atts(cl_atts)


def parse_cl_atts(string):
    if string == "all":
        return lambda x: np.ones(x.shape[1]).astype(bool)
    else:
        atts = []
        for s in string.split('.'):
            if '-' in s:
                begin, end = s.split('-')
                atts += list(range(int(begin), int(end) + 1))
            else:
                atts += [int(s)]
        return lambda x: atts


def parse_model(string):
    return {
        "lr": LogisticRegressionPU,
    }[string]


def evaluate_classification(real, pred):
    tp, fp, tn, fn = tpfptnfn(real, (pred > 0.5).astype(float))
    results = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    results["f1"] = f1_score_tpfptnfn(tp, fp, tn, fn)
    results["accuarcy"] = accuracy_tpfntnfn(tp, fp, tn, fn)
    results["precision"] = precision_tpfptnfn(tp, fp, tn, fn)
    results["recall"] = recall_tpfptnfn(tp, fp, tn, fn)
    results["prp"] = prp_tpfptnfn(tp, fp, tn, fn)
    results["rec2"] = rec2_tpfptnfn(tp, fp, tn, fn)

    tp, fp, tn, fn = tpfptnfn(real, pred)
    results = {**results, **{"p_tp": tp, "p_fp": fp, "p_tn": tn, "p_fn": fn}}

    results["p_f1"] = f1_score_tpfptnfn(tp, fp, tn, fn)
    results["p_accuarcy"] = accuracy_tpfntnfn(tp, fp, tn, fn)
    results["p_precision"] = precision_tpfptnfn(tp, fp, tn, fn)
    results["p_recall"] = recall_tpfptnfn(tp, fp, tn, fn)
    results["p_prp"] = prp_tpfptnfn(tp, fp, tn, fn)
    results["p_rec2"] = rec2_tpfptnfn(tp, fp, tn, fn)

    results["mse"] = sklearn.metrics.mean_squared_error(real, pred)
    results["mae"] = sklearn.metrics.mean_absolute_error(real, pred)
    results["roc_auc"] = sklearn.metrics.roc_auc_score(real, pred)
    results["average_precision"] = sklearn.metrics.average_precision_score(real, pred)
    results["log_loss"] = sklearn.metrics.log_loss(real, pred)

    results["prior"] = real.mean()
    results["pred_prior"] = pred.mean()
    results["prior_err"] = results["pred_prior"] - results["prior"]
    results["prior_abs_err"] = abs(results["prior_err"])
    results["prior_square_err"] = results["prior_err"]**2

    return results



def evaluate_propensity_scores(real, pred):
    tp, fp, tn, fn = tpfptnfn(real, pred)
    results = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    results["f1"] = f1_score_tpfptnfn(tp, fp, tn, fn)
    results["accuarcy"] = accuracy_tpfntnfn(tp, fp, tn, fn)
    results["precision"] = precision_tpfptnfn(tp, fp, tn, fn)
    results["recall"] = recall_tpfptnfn(tp, fp, tn, fn)
    results["prp"] = prp_tpfptnfn(tp, fp, tn, fn)
    results["rec2"] = rec2_tpfptnfn(tp, fp, tn, fn)

    results["mse"] = sklearn.metrics.mean_squared_error(real, pred)
    results["mae"] = sklearn.metrics.mean_absolute_error(real, pred)

    results["prior"] = real.mean()
    results["pred_prior"] = pred.mean()
    results["prior_err"] = results["pred_prior"] - results["prior"]
    results["prior_abs_err"] = abs(results["prior_err"])
    results["prior_square_err"] = results["prior_err"]**2

    return results


def evaluate_all(y, s, e, y_pred, e_pred):
    e = e[y == 1]
    s = s[y == 1]
    e_pred = e_pred[y == 1]
    f_results = evaluate_classification(y, y_pred)
    s_results = evaluate_classification(s, e_pred)
    e_results = evaluate_propensity_scores(e, e_pred)

    return {
        **{"f_" + k: v for k, v in f_results.items()},
        **{"s_" + k: v for k, v in s_results.items()},
        **{"e_" + k: v for k, v in e_results.items()}
    }

def summarize_results(results_folder, experiments_overview_file):
    """Automatically gathers the results for multiple experiments into
    a pandas frame where each row is an experiment and each column an
    evaluation metric from results.csv and some info (time, nb_iterations)
    from info.csv.

    """
    df_experiments = pd.read_csv(experiments_overview_file, dtype={'propensity_attributes': str})

    frames = []
    failed = []
    for _, experiment in df_experiments.iterrows():
        try:
            labeling_model = parse_labeling_model(
                experiment['labeling_model_type'],
                [abs(int(i)) for i in str(experiment['propensity_attributes']).split('.')],
                [int(i) for i in str(experiment['propensity_attributes']).split('.')])

            info_path = experiment_info_path(
                    results_folder,
                    experiment['data_name'],
                    labeling_model,
                    experiment['labeling'],
                    experiment['partition'],
                    experiment['settings'],
                    experiment['pu_method'])
            df_info = pd.Series.from_csv(info_path, index_col=0, header=None, sep="\t")

            results_path = experiment_results_path(
                    results_folder,
                    experiment['data_name'],
                    labeling_model,
                    experiment['labeling'],
                    experiment['partition'],
                    experiment['settings'],
                    experiment['pu_method'])
            df_results = pd.Series.from_csv(results_path, index_col=0, header=None, sep="\t")

            frames.append(experiment.append(df_info).append(df_results))
        except FileNotFoundError as e:
            failed.append(experiment)
            print(e)
    df = pd.concat(frames, axis=1, sort=False).T
    output_file = "{0}_{2}{1}".format(*os.path.splitext(experiments_overview_file), 'results')
    df.to_csv(output_file, index=False)
    print('Results saved to {}'.format(output_file))

    df = pd.DataFrame(failed)
    output_file = "{0}_{2}{1}".format(*os.path.splitext(experiments_overview_file), 'failed')
    df.to_csv(output_file, index=False)
    print('Failures saved to {}'.format(output_file))
