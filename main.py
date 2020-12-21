"""main.py"""
import argparse
from file_handling.file_handling import FileHandling
from generation_report.generation_report import GenerationReport
from linear_SVC.linear_SVC import LinearSVCClass
from settings.settings import (INPUT_DATASET,
                               INPUT_NEGATIVE_DATA,
                               INPUT_POSITIVE_DATA)
from test_result import test_result
from text_handling.text_handling import TextHandling
from text_vectorizer.text_vectorizer import TextVectorizer


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
            result_split_dataset = file_handling.split_dataset(dataset=INPUT_DATASET)

            # step3: cleaning the text.
            print("---RUN STEP3---")
            text_handling = TextHandling()

            print("---clear dataset train result---")
            clear_dataset_train_result = text_handling.apply_text_cleaning(dataset=result_split_dataset['dataset_train'])

            print("---clear dataset test result---")
            clear_dataset_test_result = text_handling.apply_text_cleaning(dataset=result_split_dataset['dataset_test'])

            # step4: TfidfVectorizer.
            text_vectorizer = TextVectorizer()
            result_vec = text_vectorizer.apply_tfidf_vectorizer(dataset_train=clear_dataset_train_result,
                                                                dataset_test=clear_dataset_test_result)

            # step5: LinearSVC.
            linearSVC = LinearSVCClass()
            predict = linearSVC.apply_linear_svc(tfidf_vectorizer_train=result_vec["tfidf_vectorizer_train"],
                                                 tfidf_vectorizer_test=result_vec["tfidf_vectorizer_test"],
                                                 train_data_label=result_split_dataset["train_data_label"])

            # step6: generating a classification report
            create_report = GenerationReport.create_report(test_data_label=result_split_dataset["test_data_label"],
                                                           predict=predict)

            # step7: test result
            test_result(dataset_test=result_split_dataset['dataset_test'],
                        test_data_label=result_split_dataset['test_data_label'],
                        predict=predict)

        except Exception as e:
            print(e)
