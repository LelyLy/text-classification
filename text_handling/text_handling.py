"""Class text processing."""
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


class TextHandling:
    """Class text processing."""

    @staticmethod
    def apply_text_cleaning(dataset):
        """Cleaning the text from unnecessary words and symbols.

        :param dataset - csv file (output_data.csv)
        :return result - list
        """

        result = []

        print(f"---apply text cleaning---")

        for review in tqdm(range(dataset.shape[0])):

            review = dataset.iloc[review]

            # remove everything except lower/upper case letters using Regular Expressions.
            review = re.sub('\[[^]]*\]', ' ', review)
            review = re.sub('[^a-zA-Z]', ' ', review)

            #  bring everything into lowercase.
            review = review.lower()

            #  split the text.
            review = review.split()

            # remove stop-words.
            review = [word for word in review if not word in set(stopwords.words('english'))]

            # apply lemmatization.
            lem = WordNetLemmatizer()
            review = [lem.lemmatize(word) for word in review]

            # merge the words to form cleaned up version of the text.
            review = ' '.join(review)

            result.append(review)

        return result
