from DataAnalysis import get_data
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB


# splits into equal parts, with odd elements thrown in last bin
def list_split(data: list, n: int):
    split_data = []

    sections = int(len(data) / n)
    for i in range(n):
        if i == n - 1:
            split_data.append(data[sections * i:])
        else:
            split_data.append(data[sections * i:sections * (i + 1)])

    return split_data


# joins sub-lists
def list_concatenate(data: list):
    new_data = []
    for i in data:
        for k in i:
            new_data.append(k)
    return new_data


if __name__ == "__main__":

    fold_validation = 10

    df, target_BOOS, zip_id, normalized_data = get_data()
    normalized_data = normalized_data.tolist()
    target_BOOS = target_BOOS.tolist()
    zip_id = zip_id.tolist()

    # split data into five for five-fold validation
    split_normalized_data = list_split(normalized_data, fold_validation)
    split_normalized_targets = list_split(target_BOOS, fold_validation)
    split_zip_id = list_split(zip_id, fold_validation)

    for i in range(fold_validation):
        temp_split_normalized_data = split_normalized_data.copy()
        temp_split_normalized_targets = split_normalized_targets.copy()
        temp_split_zip_id = split_zip_id.copy()

        test_data = temp_split_normalized_data.pop(i)
        test_data_targets = temp_split_normalized_targets.pop(i)
        test_zip_id = temp_split_zip_id.pop(i)

        training_data = list_concatenate(temp_split_normalized_data)
        training_data_targets = list_concatenate(temp_split_normalized_targets)
        training_zip_id = list_concatenate(temp_split_zip_id)

        reg = LinearRegression(fit_intercept=True, normalize=False).fit(X=training_data, y=training_data_targets)

        print("SLR - fold", i +1, "r^2: ", round(reg.score(X=test_data, y=test_data_targets), 2))

        plt.plot(test_zip_id, reg.predict(test_data), 'o', color='black', label="predicted")
        plt.plot(test_zip_id, test_data_targets, 'x', color='red', label="actual")
        plt.title(("SLR - Predicted vs Actual - FOLD ", i))
        plt.xlabel("Zip Code")
        plt.ylabel(("Simple Linear Regression - predicted vs actual"))
        plt.legend(loc="upper left")
        plt.show()

        gnb = GaussianNB()
        y_pred = gnb.fit(training_data, training_data_targets).predict(test_data)

        print("NB - fold", i + 1, "Mean Accuracy: ", round(gnb.score(X=test_data, y=test_data_targets), 2))

        plt.plot(test_zip_id, y_pred, 'o', color='black', label="predicted")
        plt.plot(test_zip_id, test_data_targets, 'x', color='red', label="actual")
        title = ("Naive Bayes - Predicted vs Actual - FOLD", str(i))
        plt.title(title)
        plt.xlabel("Zip Code")
        plt.ylabel("predicted vs actual")
        plt.legend(loc="upper left")
        plt.show()
