from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from fairlearn.metrics import make_derived_metric, true_positive_rate, equalized_odds_difference, equalized_odds_ratio, \
    demographic_parity_ratio, demographic_parity_difference
from aif360.sklearn import metrics as aif360_metrics
import pandas as pd


def plot_roc(clf, X, y, ax):
    y_proba = clf.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y, y_proba)
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc_score(y, y_proba):.2f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")


def print_confusion_matrix(clf, X, y):
    y_pred = clf.predict(X)
    conf_matrix = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)


def calculate_fairlearn_metrics(y_true, y_pred, z_test):
    equal_opportunity_difference = make_derived_metric(metric=true_positive_rate, transform="difference")
    equal_opportunity_ratio = make_derived_metric(metric=true_positive_rate, transform="ratio")

    names = ["equal_opportunity", "equalized_odds", "demographic_parity"]
    ratio_functions = [equal_opportunity_ratio, equalized_odds_ratio, demographic_parity_ratio]
    diff_functions = [equal_opportunity_difference, equalized_odds_difference, demographic_parity_difference]
    methods = ["between_groups", "to_overall"]

    metrics_data = []
    for name, diff_func, ratio_func in zip(names, diff_functions, ratio_functions):
        for method in methods:
            metrics_data.append(
                (name, "difference", method, diff_func(y_true, y_pred, sensitive_features=z_test, method=method)))
            metrics_data.append(
                (name, "ratio", method, ratio_func(y_true, y_pred, sensitive_features=z_test, method=method)))

    metrics = pd.DataFrame(metrics_data, columns=["metric", "type", "method", "value"]).sort_values(by=["metric", "type"])

    return metrics


def calculate_aif360_metrics(y_true, y_pred, z_test, priv_group):
    metrics_data = [("demographic_parity", "difference", "other",
                     aif360_metrics.statistical_parity_difference(y_true, y_pred, prot_attr=z_test, priv_group=priv_group)),
                    ("demographic_parity", "ratio", "other",
                     aif360_metrics.disparate_impact_ratio(y_true, y_pred, prot_attr=z_test, priv_group=priv_group)),
                    ("equal_opportunity", "difference", "other",
                     aif360_metrics.equal_opportunity_difference(y_true, y_pred, prot_attr=z_test, priv_group=priv_group))]

    metrics = pd.DataFrame(metrics_data, columns=["metric", "type", "method", "value"]).sort_values(by=["metric", "type"])

    return metrics
