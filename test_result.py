"""Test result."""
import pandas as pd
import os
import csv
from settings.settings import OUTPUT_DATA_DIRECTORY


def test_result(dataset_test, test_data_label, predict):
    """Test result"""
    file_name = os.path.join(OUTPUT_DATA_DIRECTORY, 'output_data.csv')

    dataset_predict = dataset_test.copy()
    dataset_predict = pd.DataFrame(dataset_predict)
    dataset_predict.columns = ['review']
    dataset_predict = dataset_predict.reset_index()
    dataset_predict = dataset_predict.drop(['index'], axis=1)
    dataset_predict.head()

    test_actual_label = test_data_label.copy()
    test_actual_label = pd.DataFrame(test_actual_label)
    test_actual_label.columns = ['sentiment']
    test_actual_label['sentiment'] = test_actual_label['sentiment'].replace({1: 'pos', 0: 'neg'})

    test_predicted_label = predict.copy()
    test_predicted_label = pd.DataFrame(test_predicted_label)
    test_predicted_label.columns = ['predicted_sentiment']
    test_predicted_label['predicted_sentiment'] = test_predicted_label['predicted_sentiment'].replace(
        {1: 'pos', 0: 'neg'})

    test_result = pd.concat([dataset_predict, test_actual_label, test_predicted_label], axis=1)
    test_result.to_csv(file_name, index=False)
