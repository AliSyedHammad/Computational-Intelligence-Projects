import csv
import numpy as np
from tqdm import tqdm

class DataLoader:
    """
    This a class for loading data from a csv file. It supports only timeseries data with labels.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.labels = []
        self.data = {}

    def load_data(self):
        file = open(self.data_path , 'r')
        reader = csv.reader(file)
        raw_data = []

        for row in list(reader)[1:]:
            if row[1] != '':
                if row[2] not in self.data.keys():
                    self.data[row[2]] = []

                float_array = np.array(row[3:])
                float_array[float_array==''] = '0'
                float_array = list(map(float, float_array.tolist()))
                float_array = np.array(float_array).tolist()
                self.data[row[2]].append(row[1:2]+[float_array[0]/10000000]+[float_array[1]]+[float_array[2]])