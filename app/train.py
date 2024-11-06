import pickle

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix


path = '../data'
name = 'diabetes_dataset.csv'
output_file = '../bins/model.bin'
params = {
    'n_estimators' : 678,
    'max_depth' : 10,
}

def main():
    # data preparation
    try:
        df = pd.read_csv(os.path.join(path, name))
        df.columns = df.columns.str.lower()

        df_train, df_test = train_test_split(df, test_size=.2, shuffle=True, random_state=42)

        X_train = df_train.drop(['outcome'], axis=1)
        X_test = df_test.drop(['outcome'], axis=1)

        y_train = df_train.outcome.values
        y_test = df_test.outcome.values
        print('Dataset uploaded')

    except Exception as e:
        print(f'Failure to upload dataset: {e}')


    # training 
    print('Model training...')

    model = XGBClassifier(
            objective='binary:logistic', 
            n_jobs = -1,
            random_state=42,
            **params)

    model.fit(X_train, y_train)
    print('Model training compleated.')

    # rest results
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    print("\nClassification Report:\n", classification_report(y_test, y_pred_binary))
    print()
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_binary))
    print()
    print('Test accuracy =', accuracy_score(y_test, y_pred_binary))
    print('Test precision =', precision_score(y_test, y_pred_binary))
    print('Test recall =', recall_score(y_test, y_pred_binary))
    print('Test f1 =', f1_score(y_test, y_pred_binary))
    print('Test roc_auc =', roc_auc_score(y_test, y_pred_binary))


    with open(output_file, 'wb') as f_out: 
        pickle.dump(model, f_out)
    
    print('Model saved to:', output_file)

    df_test.to_csv(os.path.join(path, 'test_data.csv'), index=False)
    print('Test data saved to:', path)


if __name__ == "__main__":
    main()