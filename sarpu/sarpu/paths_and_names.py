import os




##########
## DATA ##
##########

def original_data_path(data_folder,data_name):
    return os.path.join(data_folder,data_name,"original")

def processed_data_path(data_folder,data_name):
    return os.path.join(data_folder,data_name,"processed")

def partitions_data_path(data_folder,data_name):
    return os.path.join(data_folder,data_name,"processed","partitions")

def labelings_data_path(data_folder,data_name):
    return os.path.join(data_folder,data_name,"processed","labelings")

    
 

def data_fname(data_name):
    return data_name+"._.data.csv"

def data_path(data_folder, data_name):
    return os.path.join(processed_data_path(data_folder,data_name),data_fname(data_name))

def classlabels_fname(data_name):
    return data_name+"._.class.csv"

def classlabels_path(data_folder, data_name):
    return os.path.join(processed_data_path(data_folder,data_name),classlabels_fname(data_name))


def partition_name(data_name, partition):
    return data_name+"._.train_test._."+str(partition)+".csv"


def partition_path(data_folder,data_name, partition):
    return os.path.join(partitions_data_path(data_folder,data_name), partition_name(data_name, partition))


#propensity models, scores, labelings

def propensity_fname(data_name, propensity_model):
    return data_name+"._.propmodel."+propensity_model.name()

def propensity_base_path(data_folder, data_name, propensity_model):
    return os.path.join(labelings_data_path(data_folder,data_name), propensity_fname(data_name,propensity_model))

def propensity_model_path(data_folder, data_name, propensity_model):
    return propensity_base_path(data_folder, data_name, propensity_model)+".model"


def propensity_scores_path(data_folder, data_name, propensity_model):
    return propensity_base_path(data_folder, data_name, propensity_model)+".e.csv"


def propensity_labeling_path(data_folder, data_name, propensity_model,labeling):
    return propensity_base_path(data_folder, data_name, propensity_model)+"._.lab."+str(labeling)+".csv"


#############
## RESULTS ##
#############

def experiment_result_folder_path(results_folder, data_name, propensity_model, labeling, partition, settings):
    return os.path.join(results_folder, data_name, '.__.'.join(list(map(str,[propensity_model.name(),settings, labeling, partition]))))

def experiment_method_result_folder_path(results_folder, data_name, propensity_model, labeling, partition, settings, pu_method):
    return os.path.join(experiment_result_folder_path(results_folder, data_name, propensity_model, labeling, partition, settings), pu_method)
    
def experiment_method_result_path_nolabel(results_folder, data_name, partition, settings, pu_method):
    return "/".join([results_folder, data_name, '.__.'.join(list(map(str,["*",settings, "*", str(partition)]))),pu_method,"results.csv"])


def experiment_classifier_path(results_folder, data_name, propensity_model, labeling, partition, settings, pu_method):
    return os.path.join(experiment_method_result_folder_path(results_folder, data_name, propensity_model, labeling, partition, settings, pu_method),"f.model")

def experiment_propensity_model_path(results_folder, data_name, propensity_model, labeling, partition, settings, pu_method):
    return os.path.join(experiment_method_result_folder_path(results_folder, data_name, propensity_model, labeling, partition, settings, pu_method),"e.model")


def experiment_results_path(results_folder, data_name, propensity_model, labeling, partition, settings, pu_method):
    return os.path.join(experiment_method_result_folder_path(results_folder, data_name, propensity_model, labeling, partition, settings, pu_method),"results.csv")

def experiment_info_path(results_folder, data_name, propensity_model, labeling, partition, settings, pu_method):
    return os.path.join(experiment_method_result_folder_path(results_folder, data_name, propensity_model, labeling, partition, settings, pu_method),"info.csv")
