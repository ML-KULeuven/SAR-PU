#import dill as pickle
import numpy as np


#Reading data
def read_data(data_files, partitions_file=None, delimiter=None):
    datas = list(map(lambda data_path: np.loadtxt(data_path, delimiter=delimiter), data_files))

    if partitions_file == None:
        return tuple(datas)

    partitions = np.loadtxt(partitions_file, delimiter=delimiter, dtype=int)

    data_per_partition = []
    for partition in sorted(list(set(partitions) - set([0]))):
        data_per_partition.append(tuple(map(lambda data: data[partitions == partition], datas)))

    return data_per_partition
