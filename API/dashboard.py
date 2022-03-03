import pandas as pd
import streamlit as st
import requests
from function import SupervisedModel


def main():
    st.title('Quelle est votre question ?')

    Questions = st.text_area('Question')

    predict_btn = st.button('Tags')
    if predict_btn:
        data = [Questions]
        tags_predict = SupervisedModel()
        predict_tags = tags_predict.predict_tags(text = Questions )       

        st.write(f'Suggestion de Tags : {predict_tags}')


if __name__ == '__main__':
    main()
