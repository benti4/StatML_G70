import pandas as pd
import numpy as np

from data_processing_default import process_data
from logistic_regression import logistic_regression
from eval import evaluate_model

process_data = process_data
model_train_evaluate = logistic_regression


def main(data_file):
    # Process the data
    data = process_data(data_file)

    # Do K-Fold Cross Validation and evaluate the model
    k = 5
    fold_size = len(data) // k
    all_results = []
    for fold in range(k):
        start = fold * fold_size
        end = (fold + 1) * fold_size if fold != k - 1 else len(data)

        test_data = data.iloc[start:end]
        train_data = pd.concat([data.iloc[:start], data.iloc[end:]])

        results = model_train_evaluate(train_data, test_data, evaluate_model)
        all_results.append(results)

    # Aggregate results
    agg_results = {key: [] for key in all_results[0].keys()}
    for result in all_results:
        for key, value in result.items():
            agg_results[key].append(value)

    print(agg_results)
    # Print average results
    for key, values in agg_results.items():
        if key == "classification_report":
            # Average each metric in the classification report
            avg_report = {}
            for label in values[0].keys():
                avg_report[label] = {}
                for metric in values[0][label].keys():
                    metric_values = [v[label][metric] for v in values]
                    avg_report[label][metric] = np.average(metric_values)
            print(f"Average {key}:\n{avg_report}\n")
        else:
            avg_value = np.average(values)
            print(f"Average {key}:\n{avg_value}\n")


if __name__ == "__main__":
    main("training_data_ht2025.csv")
    # data = process_data("training_data_ht2025.csv")
    # print(data.head())
