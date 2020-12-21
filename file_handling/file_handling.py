"""Class file processing."""
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from settings.settings import INPUT_DATA_DIRECTORY


class FileHandling:
    """Class file processing."""

    def change_file_to_csv_format(self, file_path, sentiment):
        """Change file format to csv

        :param file_path: input file
        :return file_path: file in csv format
        """
        reviews = []
        file_name = os.path.join(INPUT_DATA_DIRECTORY, '{}.csv').format(sentiment)

        encoding = 'iso-8859-15'
        file = open(file_path, 'r', encoding=encoding)
        for line in file:
            reviews.append(line)

        with open(file_name, "a") as file:
            fieldnames = ["review", "sentiment"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            try:
                print(f"--- changing the {sentiment} file to csv format ---")
                for review in tqdm(reviews):
                    writer.writerow({"review": review,
                                     "sentiment": sentiment})
            except Exception as e:
                print(f"Error while write review object:{review}, {e}")

    def combine_input_data(self, file_negative_data, file_positive_data):
        """The initial data contains two files with negative and positive reviews,
         this method combines them into one."""

        # Change files to csv format
        self.change_file_to_csv_format(file_path=file_negative_data, sentiment='neg')
        self.change_file_to_csv_format(file_path=file_positive_data, sentiment='pos')

        output_file = os.path.join(INPUT_DATA_DIRECTORY, 'output_data.csv')

        negative_file = os.path.join(INPUT_DATA_DIRECTORY, 'neg.csv')
        positive_file = os.path.join(INPUT_DATA_DIRECTORY, 'pos.csv')

        negative_file = pd.read_csv(negative_file)
        positive_file = pd.read_csv(positive_file)

        print(f"--- merged files ---")
        merged = pd.concat([negative_file, positive_file])
        merged.to_csv(output_file, index=False)

    def get_dataset_info(self, dataset):
        """Return information about a dataset."""
        dataset = pd.read_csv(dataset)
        dataset_head = dataset.head()
        dataset_info = dataset.info()
        dataset_values = dataset['sentiment'].value_counts()
        print(f"--- Explore the dataset ---")
        print(dataset_head)
        print(f"------")
        print(dataset_info)
        print(f"------")
        print(dataset_values)

    def split_dataset(self, dataset):
        """Split the dataset into training and test samples."""
        dataset = pd.read_csv(dataset)
        dataset_train, dataset_test, train_data_label, test_data_label = train_test_split(dataset['review'],
                                                                                          dataset['sentiment'],
                                                                                          test_size=0.25,
                                                                                          random_state=42)

        # convert sentiments to numeric forms.
        train_data_label = (train_data_label.replace({'pos': 1, 'neg': 0})).values
        test_data_label = (test_data_label.replace({'pos': 1, 'neg': 0})).values

        result = {"dataset_train": dataset_train,
                  "dataset_test": dataset_test,
                  "train_data_label": train_data_label,
                  "test_data_label": test_data_label}
        return result
