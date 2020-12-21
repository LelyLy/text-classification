"""Settings."""
import os

dir_name = os.path.dirname(__file__)
parent_directory = os.path.split(dir_name)[0]

# file handling directory
FILE_HANDLING_DIRECTORY = os.path.join(parent_directory, 'file_handling/')

#

# input data directory
INPUT_DATA_DIRECTORY = os.path.join(parent_directory, 'input_data/')
# input negative data
INPUT_NEGATIVE_DATA= os.path.join(parent_directory, 'input_data/rt-polarity.neg')
# input positive data
INPUT_POSITIVE_DATA= os.path.join(parent_directory, 'input_data/rt-polarity.pos')
# input dataset
INPUT_DATASET= os.path.join(parent_directory, 'input_data/output_data.csv')

# output data directory
OUTPUT_DATA_DIRECTORY = os.path.join(parent_directory, 'output_data/')
