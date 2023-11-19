import os
import dill
import json
import pandas as pd
import datetime as dt


# get path
path = os.environ.get('PROJECT_PATH', '..')


# import latest model
def get_latest_model():
    files = [f'{path}/data/models/{name}' for name in os.listdir(f'{path}/data/models/')]
    model_filename = max(files, key=os.path.getctime)
    with open(model_filename, 'rb') as file:
        model = dill.load(file)
    return model


# import jsons and make predictions
def predict():
    model = get_latest_model()
    predictions = []
    for name in os.listdir(f'{path}/data/test/'):
        with open(f'{path}/data/test/{name}', 'r') as json_file:
            form = json.load(json_file)
        df = pd.DataFrame.from_dict([form])
        predictions.append((name[:-5], model.predict(df)[0]))
    df_preds = pd.DataFrame(predictions, columns=['car_id', 'pred'])
    df_preds.to_csv(
        f'{path}/data/predictions/car_price_predictions{dt.datetime.now().strftime("%Y%m%d%H%M")}.csv',
        index=False)


if __name__ == '__main__':
    predict()
