import pandas as pd
import os
import requests


url = 'http://localhost:9696/predict'

path = '../data'
output_file = '../bins/model.bin'

def main():
    try:
        df_test = pd.read_csv(os.path.join(path, 'test_data.csv'))
        X_test = df_test.drop(['outcome'], axis=1)
        y_test = df_test.outcome

    except Exception as e:
        print(f'Failure to upload dataset: {e}')

    # sent request
    X = X_test.sample(1)
    id = X.index[0]
    X = X.to_dict(orient="records")[0]

    try:
        response = requests.post(url, json=X)
        response_json = response.json()
    except requests.exceptions.JSONDecodeError:
        print("Error: Response from server is not JSON or is empty.")
        return

    print('*******************************************')
    print('Case No:', id)
    print('Diabete probability:', round(response_json.get('diabete_probability'), 3))
    print('Diabetes diagnosis:', response_json.get('diabete'))
    print()
    print('Checking True value:')
    print('True value =', bool(y_test.loc[id]))
    print('Prediction result is', 'correct :)' if y_test.loc[id] == response_json.get('diabete') else 'incorrect :(')
    print('*******************************************')


if __name__ == "__main__":
    main()