import csv
import numpy as np


def load_csv(path: str):
    file = open(path, "r", encoding="utf-8")
    return csv.reader(file)


def accuracy(predictions: np.ndarray, ground_truth: np.ndarray):
    assert predictions.shape[0] == ground_truth.shape[0]

    truth = 0
    total = 0
    for i in range(predictions.shape[0]):
        if predictions[i] == ground_truth[i]:
            truth += 1
        total += 1

    return truth * 1.0 / total


def f1_score(predictions: np.ndarray, ground_truth: np.ndarray, id2label: dict):
    print("\n---------- Metrics for each label: ----------")
    for tid in id2label.keys():
        positive_label_id = tid
        preds_num = predictions
        labels_num = ground_truth

        tp, fp, fn = 0, 0, 0
        for i in range(len(preds_num)):
            p, l = preds_num[i], labels_num[i]
            if p == positive_label_id:
                if l == p:
                    tp += 1
                else:
                    fp += 1
            elif l == positive_label_id:
                fn += 1
        precision = tp * 1.0 / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp * 1.0 / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        print(id2label[positive_label_id], precision, recall, f1)
    return None


def perform_baseline_models(train_data: dict, test_data: dict, id2label: dict, model_type: str):
    if model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, max_features=50)
    elif model_type == "mlp":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(hidden_layer_sizes=50)
    else:
        raise NotImplementedError("Current model not supported!")

    train_features, train_labels = train_data['features'], train_data['labels']
    test_features, test_labels = test_data['features'], test_data['labels']

    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)
    ground_truth = np.array(test_labels, dtype=np.int)
    print(f"---------- Results for model: {model_type} ----------")
    f1_score(predictions, ground_truth, id2label)
    print(f"accuracy: {accuracy(predictions, ground_truth)}")


if __name__ == '__main__':
    pass
