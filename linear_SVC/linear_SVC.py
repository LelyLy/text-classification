"""Class apply LinearSVC."""
from sklearn.svm import LinearSVC


class LinearSVCClass:
    """Class LinearSVC."""

    def apply_linear_svc(self, tfidf_vectorizer_train, tfidf_vectorizer_test, train_data_label):
        """Apply LinearSVC."""

        print("---apply LinearSVC ---")

        linear_svc = LinearSVC(C=0.5, random_state=42)
        linear_svc.fit(tfidf_vectorizer_train, train_data_label)

        predict = linear_svc.predict(tfidf_vectorizer_test)

        return predict
