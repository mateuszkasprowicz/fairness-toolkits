from fairlearn.metrics import count, false_positive_rate, selection_rate
from sklearn.metrics import recall_score

RANDOM_STATE = 0

DEFAULT_MODEL_CONFIG = {"iterations": 3000,
                        "depth": 3,
                        "learning_rate": 0.01,
                        "loss_function": "Logloss",
                        "verbose": 250,
                        "random_seed": RANDOM_STATE,
                        }

METRICFRAME_METRICS = {
                        "tpr": recall_score,
                        "fpr": false_positive_rate,
                        "sel": selection_rate,
                        "count": count
                    }