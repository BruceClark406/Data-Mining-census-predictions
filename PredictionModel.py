from DataAnalysis import get_data
from sklearn.linear_model import LinearRegression
import numpy as np



# splits into equal parts, with odd elements thrown in last bin
def list_split(data: list, n: int):
    split_data = []

    sections = int(len(data) / n)
    for i in range(n):
        if i == n-1:
            split_data.append(data[sections * i:])
        else:
            split_data.append(data[sections*i:sections*(i+1)])

    return split_data



# joins sublists
def list_concatenate(data: list):
    new_data = []
    for i in data:
        for k in i:
            new_data.append(k)
    return new_data


if __name__ == "__main__":

    fold_validation = 5

    df, target_BOOS, zip_id, normalized_data = get_data()
    normalized_data = normalized_data.tolist()
    target_BOOS = target_BOOS.tolist()

    # split data into five for five-fold validation
    split_normalized_data = list_split(normalized_data, fold_validation)
    split_normalized_targets = list_split(target_BOOS, fold_validation)


    for i in range(fold_validation):


        training_data = list_concatenate(split_normalized_data[:4])
        training_data_targets = list_concatenate(split_normalized_targets[:4])
        test_data = split_normalized_data[4]
        test_data_targets = split_normalized_targets[4]


        print(len(training_data), len(training_data_targets))
        reg = LinearRegression(fit_intercept=True, normalize=False).fit(X=training_data, y=training_data_targets)

        print(reg.predict(test_data))
        print(test_data_targets)

        print(reg.score(X=test_data, y=test_data_targets))



