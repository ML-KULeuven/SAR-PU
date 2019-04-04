import sys

from sarpu.experiments import train_and_evaluate, summarize_results
from sarpu.labeling_mechanisms import label_data, parse_labeling_model
from sarpu.paths_and_names import experiment_method_result_folder_path


def print_help():
    print ("Options: label, train_eval, outdir, summ")


def parse_propensity_attributes(arg):
    propensity_attributes_str = arg.split(".")
    propensity_attributes = list(map(lambda s: abs(int(s)), propensity_attributes_str))
    propensity_attributes_signs = list(map(lambda s: -1 if "-" in s else 1, propensity_attributes_str))
    return propensity_attributes, propensity_attributes_signs

def label(args):

    if len(args)==0:
        print("Usage: python -m sarpu label <data_folder> <data_name> <labeling_model_type> <propensity_attributes> <nb_assignments> [relabel] [seed]")

    else:

        data_folder, data_name, labeling_model_type = args[0:3]
        propensity_attributes, propensity_attributes_signs = parse_propensity_attributes(args[3])
        nb_assignments = int(args[4])
        relabel = False
        seed = 123
        try:
            relabel = bool(args[5])
            seed = int(args[6])
        except:
            pass

        label_data(data_folder,data_name, labeling_model_type, propensity_attributes, propensity_attributes_signs,nb_assignments, relabel, seed)


def train_eval(args):
    if len(args)==0:
        print("Usage: python -m sarpu train_eval <data_folder> <results_folder> <data_name> <labeling_model_type> <propensity_attributes> <labeling> <partition> <settings> <pu_method> [rerun]")

    else:

        data_folder, results_folder, data_name, labeling_model_type = args[0:4]
        propensity_attributes, propensity_attributes_signs = parse_propensity_attributes(args[4])
        print(args[4], propensity_attributes, propensity_attributes_signs)
        labeling = int(args[5])
        partition = int(args[6])
        settings, pu_method = args[7:9]
        rerun = False 
        try:
            rerun = bool(args[9])
        except:
            pass

        labeling_model = parse_labeling_model(labeling_model_type, propensity_attributes,propensity_attributes_signs)
        train_and_evaluate(data_folder,results_folder, data_name, labeling_model, labeling, partition, settings, pu_method, rerun_experiments=rerun)


def summarize(args):
    if len(args)==0:
        print("Usage: python -m sarpu summ <results_folder> <experiments>")
    else:
        print(args)
        results_folder, experiments = args[0:]
        summarize_results(results_folder, experiments)


def outdir(args):
    if len(args)==0:
        print("Usage: python -m sarpu outdir <data_folder>  <data_name> <labeling_model_type> <propensity_attributes> <labeling> <partition> <settings> <pu_method>")

    else:
        results_folder, data_name, labeling_model_type = args[0:3]
        propensity_attributes, propensity_attributes_signs = parse_propensity_attributes(args[3])
        labeling = int(args[4])
        partition = int(args[5])
        settings, pu_method = args[6:8]

        labeling_model = parse_labeling_model(labeling_model_type, propensity_attributes,propensity_attributes_signs)

        print(experiment_method_result_folder_path(results_folder, data_name, labeling_model, labeling, partition, settings, pu_method))


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    if len(args) == 0:
        print_help()

    else:

        if args[0] == "label":
            label(args[1:])

        if args[0] == "train_eval":
            train_eval(args[1:])

        if args[0] == "outdir":
            outdir(args[1:])

        if args[0] == "summ":
            summarize(args[1:])




if __name__ == "__main__":
    main()
