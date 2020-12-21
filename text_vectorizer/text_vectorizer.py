"""Class count vectorizer."""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    """Class count vectorizer."""

    def apply_tfidf_vectorizer(self, dataset_train, dataset_test):
        """Apply tfidf vectorizer.
        Text Frequency which means how many times a word (term) appears in a text.

        :return array_review - array
        """

        print("---apply tfidf vectorizer ---")

        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        tfidf_vectorizer_train = tfidf_vectorizer.fit_transform(dataset_train)
        tfidf_vectorizer_test = tfidf_vectorizer.transform(dataset_test)

        result = {"tfidf_vectorizer_train": tfidf_vectorizer_train,
                  "tfidf_vectorizer_test": tfidf_vectorizer_test}

        return result
