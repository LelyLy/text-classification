"""Class generation report."""
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class GenerationReport:
    """Class generation report."""

    @staticmethod
    def create_report(test_data_label, predict):
        """Create report."""

        print("---Generation report---")

        print("Classification Report: \n", classification_report(test_data_label, predict, target_names=['Negative', 'Positive']))
        print("Confusion Matrix: \n", confusion_matrix(test_data_label, predict))
        print("Accuracy: \n", accuracy_score(test_data_label, predict))
