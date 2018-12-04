from argparse import ArgumentParser
import pandas as pd
import utils


def _get_args():
    """Devuelve los argumentos introducidos por la terminal."""
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path',
                        type=str,
                        help='ruta del archivo csv con las instancias',
                        required=True)
    parser.add_argument('-o', '--output_folder',
                        type=str,
                        help='ruta del directorio en el que guardar los resultados',
                        required=True)
    parser.add_argument('-a', '--text_attribute',
                        type=str,
                        help='atributo texto de las instancias sobre el que'
                             ' hacer el clustering',
                        default='open_response')
    parser.add_argument('-c', '--class_attribute',
                        type=str,
                        help='atributo clase de las instancias',
                        default='gs_text34')
    parser.add_argument('-t', '--tfidf',
                        action='store_true',
                        required=False,
                        help='')
    parser.add_argument('-d2v', '--doc2vec',
                        action='store_true',
                        required=False,
                        help='')
    parser.add_argument('-nb', '--naiveBayes',
                        action='store_true',
                        required=False,
                        help='')
    parser.add_argument('-nn', '--neuralNetwork',
                        action='store_true',
                        required=False,
                        help='')
    parser.add_argument('-n', '--neurons',
                        nargs='+',
                        default=[],
                        required=False,
                        help='')
    return parser.parse_args()


def main():
    args = _get_args()
    neurons = []
    for neuron in args.neurons:
        try:
            neurons.append(int(neuron))
        except ValueError:
            pass

    dataframe = pd.read_csv(args.data_path)
    if args.tfidf:
        instances = utils.tfidf_filter(dataframe, args.text_attribute)
    elif args.doc2vec:
        instances = utils.doc2vec(dataframe, args.text_attribute)
    else:
        print("wtf")

    classes = list(dataframe[args.class_attribute])
    print(instances)
    print()


if __name__ == '__main__':
    main()
