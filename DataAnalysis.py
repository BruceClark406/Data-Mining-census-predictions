import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA

def print_correlation_matrix(df):
    print("Correlation Matrix")
    print(list(df.columns))
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(linewidth=np.inf)
    print(np.corrcoef(df.to_numpy().T))


def visulaize_with_PCA(normalized_data):
    pca = PCA(n_components=2)
    pca_transformed2 = pca.fit_transform(normalized_data)
    plt.scatter(pca_transformed2[:, 0], pca_transformed2[:, 1], s=10, c='b', marker='x')
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.title('Scatter plot of PCA-transformed data')
    plt.show()

def visualize_variance_caught_by_PCA(normalized_data):
    pca = PCA()
    pca.fit_transform(normalized_data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='x')
    plt.xlabel('number of principal components')
    plt.ylabel('cumulative explained variance')
    plt.title('Explained variance by number of PCA variables')
    plt.show()


if __name__ == "__main__":

    df = pd.read_csv("ScrapedData.csv")

    # All quantitative (no one-hot encoding necessary)
    # No missing values for returned data (no forward fill or any filling necessary)

    # shuffle the dataset with random seed
    df.sample(frac=1, random_state=1)

    # grab target (BOOS = Born out of state) field from df
    target_BOOS = df["born out of state"]
    zip_id = df["zipcode"]
    df = df.drop(columns=["born out of state", "zipcode"])

    # after looking at the visualization, I decided to drop: "total in poverty""
    df = df.drop(columns=["total in poverty"])

    # perform standard normalization of all attributes
    normalized_data = preprocessing.normalize(df)

    # find correlation matrix
    print_correlation_matrix(df)

    # visualize the data
    visulaize_with_PCA(normalized_data)

    # variance explained by PCA
    visualize_variance_caught_by_PCA(normalized_data)

 