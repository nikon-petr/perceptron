import csv

import numpy as np


def dataset(csv_path):
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        input_vectors = []
        output_vectors = []
        for row in reader:
            input_vectors.append(row[:-3])
            output_vectors.append(row[-3:])
        dataset = (np.array(input_vectors), np.array(output_vectors))
        return dataset
