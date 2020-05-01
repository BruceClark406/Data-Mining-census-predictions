import pandas as pd
from sklearn import preprocessing
import numpy as np

if __name__ == "__main__":
    # perform standard normalization of all attributes
    df = pd.read_csv("ScrappedData.csv")
    normalized_data = preprocessing.normalize(df)





