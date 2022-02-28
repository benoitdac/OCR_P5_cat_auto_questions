import pandas as pd
import streamlit as st
import requests


def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/'

    st.title('Quelle est votre question ?')

    Questions = st.text_area('Questions')

    predict_btn = st.button('tags')
    if predict_btn:
        data = [Questions]
        pred = request_prediction(MLFLOW_URI, data)
        st.write(
            'Le prix médian d\'une habitation est de {:.2f}'.format(pred))


if __name__ == '__main__':
    main()
