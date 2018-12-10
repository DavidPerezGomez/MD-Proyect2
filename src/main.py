"""usage: main.py [-h] -d DATA_PATH -o OUTPUT_FOLDER -a TEXT_ATTRIBUTE -c
               CLASS_ATTRIBUTE
               [-t | -d2v VECTOR_SIZE MIN_COUNT EPOCHS MODEL_PATH]
               [-nb | -nn N_MODELS [NEURONS ...]] [-k K]

optional arguments:
  -h, --help            show this help message and exit
  -d DATA_PATH, --data_path DATA_PATH
                        ruta del archivo csv con las instancias.
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        ruta del directorio en el que guardar los resultados.
  -a TEXT_ATTRIBUTE, --text_attribute TEXT_ATTRIBUTE
                        atributo texto de las instancias sobre el que hacer el
                        clustering.
  -c CLASS_ATTRIBUTE, --class_attribute CLASS_ATTRIBUTE
                        atributo clase de las instancias.
  -t, --tfidf
  -d2v VECTOR_SIZE MIN_COUNT EPOCHS MODEL_PATH, --doc2vec VECTOR_SIZE MIN_COUNT EPOCHS MODEL_PATH
                        vector_size: dimensión de los vectores. min_count:
                        número mínimo de veces que debe aparecer un término
                        para ser tomado en cuenta epochs: número de
                        repeticiones. model_path: ruta en la que leer/guardar
                        el modelo doc2vec.
  -nb, --naive_bayes
  -nn N_MODELS [NEURONS ...], --neural_network N_MODELS [NEURONS ...]
                        neuronas de la red neuronal. Cada número es el número
                        de neuronas de na capa oculta.
  -k K                  número de folds para el k-fold cross-validation
"""

from argparse import ArgumentParser
import pandas as pd
import numpy as np
import utils
import os
from naive_bayes import NaiveBayes
from combined_neural_network import CombinedNN


def _get_args():
    """Devuelve los argumentos introducidos por la terminal."""
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path',
                        type=str,
                        required=True,
                        help='ruta del archivo csv con las instancias.')
    parser.add_argument('-o', '--output_folder',
                        type=str,
                        required=True,
                        help='ruta del directorio en el que guardar los resultados.')
    parser.add_argument('-a', '--text_attribute',
                        type=str,
                        default='open_response',
                        required=True,
                        help='atributo texto de las instancias sobre el que hacer el clustering.')
    parser.add_argument('-c', '--class_attribute',
                        type=str,
                        default='gs_text34',
                        required=True,
                        help='atributo clase de las instancias.')
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('-t', '--tfidf',
                        action='store_true',
                        help='')
    group1.add_argument('-d2v', '--doc2vec',
                        nargs=4,
                        metavar=('VECTOR_SIZE', 'MIN_COUNT', 'EPOCHS', 'MODEL_PATH'),
                        help='vector_size: dimensión de los vectores.'
                             ' min_count: número mínimo de veces que debe aparecer un término para ser tomado en cuenta'
                             ' epochs: número de repeticiones.'
                             ' model_path: ruta en la que leer/guardar el modelo doc2vec.')
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('-nb', '--naive_bayes',
                        action='store_true',
                        help='')
    group2.add_argument('-nn', '--neural_network',
                        nargs='+',
                        metavar=('N_MODELS', 'NEURONS'),
                        help='neuronas de la red neuronal. Cada número es el número de neuronas de na capa oculta.')
    parser.add_argument('-k',
                        type=int,
                        default=10,
                        help='número de folds para el k-fold cross-validation')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='flag para imprimir información adicional por pantalla')
    return parser.parse_args()


def main():
    args = _get_args()

    dataframe = pd.read_csv(args.data_path)
    if args.tfidf:
        instances = utils.tfidf_filter(dataframe, args.text_attribute, args.class_attribute)
    elif args.doc2vec:
        vector_size = int(args.doc2vec[0])
        min_count = int(args.doc2vec[1])
        epochs = int(args.doc2vec[2])
        model_path = args.doc2vec[3]
        instances = utils.doc2vec(dataframe, args.text_attribute,
                                  vector_size=vector_size, min_count=min_count, epochs=epochs,
                                  save_path=model_path)
    else:
        instances = None
        exit(1)

    classes = np.array(dataframe[args.class_attribute])
    for i in range(len(classes)):
        if type(classes[i]) is str:
            classes[i] = classes[i].lower()
    if args.naive_bayes:
        classifier = NaiveBayes()
    elif args.neural_network:
        n_models = int(args.neural_network[0])
        neurons = []
        for neuron in args.neural_network[1:]:
            try:
                neurons.append(int(neuron))
            except ValueError:
                pass
        neurons = tuple(neurons)
        classifier = CombinedNN(neurons=neurons, n_models=n_models)
    else:
        classifier = None
        exit(1)

    if not os.path.isdir(os.path.abspath(args.output_folder)):
        os.mkdir(os.path.abspath(args.output_folder))

    results_path_txt = os.path.join(args.output_folder,
                                    "results_{}-fcv.txt".format(args.k))
    results_path_csv = os.path.join(args.output_folder,
                                    "results_{}-fcv.csv".format(args.k))
    model_path = os.path.join(args.output_folder,
                              "model.pkl")
    classifier.set_input_format(instances[:50], classes[:50])
    # classifier.train(instances, classes)
    # classifier.predict(instances)
    classifier.k_fcv(k=args.k, instances=instances, classes=classes,
                     save_path_txt=results_path_txt, save_path_csv=results_path_csv,
                     verbose=args.verbose)
    # classifier.save_model(model_path)


if __name__ == '__main__':
    main()
