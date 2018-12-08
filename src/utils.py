from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import gensim
import gensim.models.doc2vec as d2v
import numpy as np
import os


def is_number(string):
    """Devuelve True si el string representa un número."""

    try:
        float(string)
        return True
    except ValueError:
        return False


def parallel_shuffle(a, b):
    # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison/4602224#4602224
    if len(a) == len(b):
        p = np.random.permutation(len(a))
        return a[p], b[p]


def tfidf_filter(data_frame, attribute):
    """Aplica el filtro TF-IDF a las instancias.

    data_frame: instancias a filtrar. Vienen en forma de dataframe.
    attribute: nombre del atributo texto a transformar."""

    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame[attribute].values.astype('U'))
    return tfidf_matrix.A


def doc2vec(data_frame, attribute, vector_size=50, min_count=2, epochs=40, save_path=None):
    """Convierte las instancias en vectores utilizando la estrategia doc2vec.

    data_frame: instancias a filtrar. Vienen en forma de dataframe.
    attribute: nombre del atributo texto a transformar.
    vector_size: tamaño de los vectores resultantes.
    min_count: número mínimo de veces que tiene que aparecer una palabra para ser tomada en cuenta.
    epochs: número de iteraciones sobre los datos."""

    if save_path is not None and os.path.isfile(save_path):
        model = d2v.Doc2Vec.load(save_path)
    else:
        sentences = []
        for i, text in enumerate(data_frame.get(attribute)):
            sentences.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(text), [i]))
        model = _doc2vec_model(sentences, vector_size=vector_size, min_count=min_count,
                               epochs=epochs, save_path=save_path)

    vectors = []
    for i, text in enumerate(data_frame.get(attribute)):
        vectors.append(model.infer_vector(text.split(' ')))
    vectors = np.array(vectors)
    return vectors


def _doc2vec_model(sentences, vector_size=50, min_count=2, epochs=40, save_path=None):
    """Crea un modelo doc2vec entrenado con las instancias.

    sentences: instancias con las que entrenar. Vienen en forma de vector de instancias. Cada instancia es un vector de palabras.
    vector_size: tamaño de los vectores resultantes.
    min_count: número mínimo de veces que tiene que aparecer una palabra para ser tomada en cuenta.
    epochs: número de iteraciones sobre los datos.
    save_path: ruta en la que guardar el modelo."""

    model = d2v.Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs)
    model.build_vocab(documents=sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    if save_path is not None:
        try:
            model.save(save_path)
        except FileNotFoundError:
            pass
    return model


def pca_filter(instances, attributes):
    """Aplica el filtro PCA a las instancias.

    instances: instancias a filtrar. Vienen en forma de array de dos dimensiones.
    attributes: número de atributos a reducir los datos.
    """

    pca = PCA(n_components=attributes)
    return pca.fit_transform(instances.copy())
