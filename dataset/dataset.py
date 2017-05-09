import csv


def dataset(csv_path):
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        datasets = []
        for row in reader:
            input_vector = row[:-3]
            output_vector = row[-3:]
            datasets.append((input_vector, output_vector))
        return datasets
