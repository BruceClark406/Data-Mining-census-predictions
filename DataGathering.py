import pandas as pd
import requests
import json
from API_KEY_SECRET import get_api_key



def load_zips():
    f = open("zip_codes_cities_county.csv", "r")
    zip_code_list = []
    for line in f:
        zip_code_list.append(line.split(",")[0])
    f.close()
    return zip_code_list


def call_census_api(zip_codes: list):
    zip_codes_list = ','.join(zip_codes)
    variables = ["B06001_025E", "B01003_001E", "B06010_011E", "B09001_001E", "B14001_009E", "B14001_008E", "B17001_002E", "B02001_003E", "B21001_004E"]
    variables = ','.join(variables)

    # for testing purposes
    # zip_codes_list = "59718"

    URL= ("https://api.census.gov/data/2017/acs/acs5?key=" + get_api_key() + "&get=" + variables + "&for=zip%20code%20tabulation%20area:" + zip_codes_list)
    response = json.loads(requests.get(URL).text)[1:]

    df = pd.DataFrame(columns=['born out of state', 'population',  "income of > $75k", 'under 18', 'grad', 'undergrad', 'total in poverty', 'total African American', 'number of veterans', 'zipcode'], data=response)


    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    print(df.head())
    print()
    print(df.describe())

    df.to_csv('ScrapedData.csv', index=False)


if __name__ == "__main__":
    zip_code_array = load_zips()
    call_census_api(zip_code_array)
