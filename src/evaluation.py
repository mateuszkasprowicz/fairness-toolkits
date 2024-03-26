from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


def calculate_metrics(clf, X, y):
    y_pred = clf.predict(X)

    metrics = {"Accuracy": accuracy_score(y, y_pred), "Precision": precision_score(y, y_pred),
               "Recall": recall_score(y, y_pred), "F1 Score": f1_score(y, y_pred)}
    return metrics


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