import re
import pandas as pd


def read_data() -> list:
    df = pd.read_csv('../../data/raw/reviews.csv')

    # Convertir a una lista
    data = df.content.values.tolist()
    # pprint(data[:1])

    # Eliminar emails
    data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]

    # Eliminar newlines
    data = [re.sub(r'\s+', ' ', sent) for sent in data]

    # Eliminar comillas
    data = [re.sub(r"\'", "", sent) for sent in data]

    return data

