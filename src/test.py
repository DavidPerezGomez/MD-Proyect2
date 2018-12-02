import pandas as pd
import utils


def main():
    df = pd.read_csv("../files/verbal_autopsies_clean.csv")
    instances1 = utils.doc2vec(df, "open_response")
    instances2 = utils.tfidf_filter(df, "open_response")
    print(instances1)
    print(instances2)


if __name__ == '__main__':
    main()
