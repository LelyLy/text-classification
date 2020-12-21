"""Class count vectorizer."""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from text_handling.text_handling import TextHandling
from settings.settings import (INPUT_DATASET,
                               INPUT_DATA_DIRECTORY,
                               INPUT_NEGATIVE_DATA,
                               INPUT_POSITIVE_DATA)


class TextVectorizer:
    """Class count vectorizer."""

    result = TextHandling.apply_text_cleaning(dataset=INPUT_DATASET)

    def apply_count_vectorizer(self):
        """Apply count vectorizer.
        Set the numbers for the data depending on how many times it appears in the text.

        :return array_review - array
        """

        print("---apply count vectorizer ---")

        count_vectorizer = CountVectorizer()
        review_count_vectorizer = count_vectorizer.fit_transform(self.result)

        array_review = review_count_vectorizer.toarray()
        return array_review

    def apply_tfidf_vectorizer(self):
        """Apply tfidf vectorizer.
        Text Frequency which means how many times a word (term) appears in a text.

        :return array_review - array
        """

        print("---apply tfidf vectorizer ---")

        tfidf_vectorizer = TfidfVectorizer()
        review_tfidf_vectorizer = tfidf_vectorizer.fit_transform(self.result)

        array_review = review_tfidf_vectorizer.toarray()
        return array_review
