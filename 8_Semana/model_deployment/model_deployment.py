import pandas as pd
import numpy as np
import joblib
import sys
import os
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def limpieza_texto(texto):
    # removiendo caracteres especiales que pueda contener el texto
    patron = r'[^a-zA-Z0-9\s]'
    texto = re.sub(patron, ' ', texto)
    # eliminar espacios adicionales entre texto
    texto = ' '.join(texto.split()) 
    # Convertir todo el texto en minúscula
    texto = texto.lower()
    # Se remueve backslash
    texto = re.sub("\'", "", texto)

    return texto


def eliminar_stopwords(texto):
    # Obtener las stopwords en inglés
    stopwords_lista = stopwords.words('english')
        
    # Tokenizar el texto en palabras
    palabras = nltk.word_tokenize(texto)
        
    # Filtrar las palabras que no son stopwords
    palabras_filtradas = [palabra for palabra in palabras if palabra.lower() not in stopwords_lista]

    # Unir las palabras filtradas en un nuevo texto
    texto_filtrado = ' '.join(palabras_filtradas)

    return texto_filtrado


def predict(plot):

    clf = joblib.load(os.path.dirname(__file__) + '/model.pkl') 

    vecto = joblib.load(os.path.dirname(__file__) + '/vecto.pkl') 

    plot_ = pd.DataFrame([plot], columns=['plot'])

    # Se limpia el texto de caracteres extraños
    plot_['plot_limpio'] = plot_['plot'] .apply(lambda x: limpieza_texto(x))

    # Se eliminan las Stopwords
    plot_['plot_limpio'] = plot_['plot_limpio'] .apply(lambda x: eliminar_stopwords(x))

    # Se vectoriza
    
    X_dtm = vecto.transform(plot_['plot_limpio'])

    # Make prediction
    p1 = clf.predict_proba(X_dtm)


    categorias = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                  'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
                  'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    
    
    etiquetas_valores = list(zip(p1.flatten(), categorias))
    etiquetas_valores_ordenados = sorted(etiquetas_valores, key=lambda x: x[0], reverse=True)
    valores_filtrados = [elemento for elemento in etiquetas_valores_ordenados if elemento[0] > 0.30]
    etiquetas_filtradas = [elemento[1] for elemento in etiquetas_valores_ordenados if elemento[0] > 0.30]

    
    return etiquetas_filtradas


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an URL')
        
    else:

        plot = sys.argv[1]

        valores_texto = predict(plot)
        
        print(plot)
        print('Movie Classification: ', valores_texto)