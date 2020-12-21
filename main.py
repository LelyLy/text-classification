"""main.py"""
import argparse

from file_handling.file_handling import FileHandling
from text_handling.text_handling import TextHandling

from settings.settings import (INPUT_DATASET,
                               INPUT_NEGATIVE_DATA,
                               INPUT_POSITIVE_DATA)


parser = argparse.ArgumentParser()
parser.add_argument('-run', type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    run = args.run

    if run:
        print('-----: RUN TEXT CLASSIFICATION:')

        try:
            # step1: file processing
            print("---RUN STEP1---")
            file_handling = FileHandling()
            file_handling.combine_input_data(file_negative_data=INPUT_NEGATIVE_DATA,
                                             file_positive_data=INPUT_POSITIVE_DATA)
            file_handling.get_dataset_info(dataset=INPUT_DATASET)

            # step2: split the dataset into training and test samples.
            print("---RUN STEP2---")
            result = file_handling.split_dataset(dataset=INPUT_DATASET)

            # step3: cleaning the text.
            print("---RUN STEP3---")
            text_handling = TextHandling()

            print("---clear dataset train result---")
            clear_dataset_result = text_handling.apply_text_cleaning(dataset=result['dataset_train'])

            print("---clear dataset test result---")
            clear_dataset_test = text_handling.apply_text_cleaning(dataset=result['dataset_test'])
        except Exception as e:
            print(e)
