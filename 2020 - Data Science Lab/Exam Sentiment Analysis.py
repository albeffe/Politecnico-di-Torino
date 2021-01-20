# Import ------------------------------------------------------------------------------------------
import csv
import re
import string
import pandas as pd
import numpy as np
import unidecode
import math
from googletrans import Translator
import spacy
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, ParameterGrid
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt

from wordcloud import WordCloud

nlp = spacy.load("it_core_news_sm", disable=['parser', 'tagger', 'ner'])


# Functions ---------------------------------------------------------------------------------------
def custom_csv_print(in_labels):
    list_to_print = []
    for index in range(0, len(in_labels)):
        row_to_print = []
        row_to_print.append(index)
        if in_labels[index] == 1:
            row_to_print.append('pos')
        else:
            row_to_print.append('neg')

        list_to_print.append(row_to_print)

    with open('output.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Predicted'])
        for index in range(0, len(list_to_print)):
            writer.writerow(list_to_print[index])
    return


def custom_import_stopwords(filename):
    in_stopword_list = []
    in_flag = 0
    in_word_cnt = 0

    with open(f'{filename}' + '.csv', encoding="utf8") as f:
        for row in csv.reader(f):
            if in_flag == 0:
                in_flag = 1
            else:
                in_stopword_list.append(row[0])
                in_word_cnt += 1

    print(f"{in_word_cnt} stopwords imported")
    return in_stopword_list


def custom_at_least_n(list1, list2, n):
    elem_list = []
    elem_cnt = 0
    for elem in list1:
        if elem in list2 and len(elem) > 3:
            elem_list.append(elem)
            elem_cnt += 1
            if elem_cnt == n:
                return True
    return False


# Class -------------------------------------------------------------------------------------------
class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, review):
        lemmas = []

        # Sostituzione punteggiatura con spazio
        translator_1 = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        review = review.translate(translator_1)

        # Traduzione delle sole review straniere ma non inglesi (per limiti utilizzo API google)
        # translator = Translator()
        # review_list = review.split(' ')
        # if custom_at_least_n(stopwords_de, review_list, 3):
            # translated = translator.translate(review, src='de', dest='it')
            # print('Traduzione dal tedesco effettuata')
            # review = translated.text
        # if custom_at_least_n(stopwords_es, review_list, 3):
            # translated = translator.translate(review, src='ca', dest='it')
            # print('Traduzione dallo spagnolo effettuata')
            # review = translated.text
        # if custom_at_least_n(stopwords_fr, review_list, 3):
            # translated = translator.translate(review, src='fr', dest='it')
            # print('Traduzione dal francese effettuata')
            # review = translated.text

        # Sostituzione numeri con spazi
        review = re.sub(r'\d+', ' ', review)

        # Rimozione caratteri speciali
        review = re.sub(r"[^a-zA-Z0-9]+", ' ', review)

        # Lemmatizzazione
        doc = nlp(review)
        review_list = []
        for token in doc:
            review_list.append(token.lemma_)
        review = ' '.join(review_list)

        # Rimozione accenti
        review = unidecode.unidecode(review)

        for token in word_tokenize(review):
            # Rimozione spazio iniziale e finale
            token = token.strip()

            if token not in stopwords and len(token) >= 2:
                # Stemming (+ controllo che snowball supporti l'italiano)
                # print(" ".join(SnowballStemmer.languages))
                stemmer = SnowballStemmer("italian")
                token = stemmer.stem(token)
                if len(token) >= 2:
                    lemmas.append(token)

        return lemmas


# Data Exploration --------------------------------------------------------------------------------
print()
print('Data Exploration Phase')
df_dev = pd.read_csv('./development.csv', skiprows=1, names=['review', 'sentiment'])
df_eval = pd.read_csv('./evaluation.csv', skiprows=1, names=['review'])
# print(df_dev.head())
# print(df_eval.head())
# print()
# print(f'Dimensione development dataset: {len(df_dev)}')
print(f"Numero pos: {len(df_dev[df_dev.sentiment == 'pos'])}")
print(f"Numero neg: {len(df_dev[df_dev.sentiment == 'neg'])}")

# Calcolo media lunghezze recensioni in base a sentimento -----------
pos_len_cnt = 0
neg_len_cnt = 0
df_tmp = df_dev
df_tmp['review_len'] = df_tmp['review'].apply(len)
df_tmp_pos = df_tmp[df_tmp.sentiment == "pos"]
df_tmp_neg = df_tmp[df_tmp.sentiment == "neg"]

valor_medio_pos = df_tmp_pos["review_len"].mean()
valor_medio_neg = df_tmp_neg["review_len"].mean()

print(f'Lunghezza media recensioni positive {valor_medio_pos}')
print(f'Lunghezza media recensioni negative {valor_medio_neg}')

# Calcolo deviazione standard
std_pos = df_tmp_pos["review_len"].std()
std_neg = df_tmp_neg["review_len"].std()

print(f'Standard deviation recensioni positive {std_pos}')
print(f'Standard deviation recensioni negative {std_neg}')

# Controllo presenza di null values ---------------------------------
# print()
# print(f'Dimensione evaluation dataset: {len(df_eval)}')
# print()
# print(f'Development dataframe contains NaN values? {df_dev.isnull().values.any()}')
# print(f'Evaluation dataframe contains NaN values? {df_eval.isnull().values.any()}')
# print()

df_dev.loc[df_dev["sentiment"] == 'pos', "sentiment"] = 1
df_dev.loc[df_dev["sentiment"] == 'neg', "sentiment"] = 0
# print(df_dev.head())

# Dataset reduction -------------------------------------------------------------------------------
dataset_reduction_flag = 0

if dataset_reduction_flag == 1:
    df_dev_sampled = df_dev.sample(frac=0.1)
    df_eval_sampled = df_eval.sample(frac=0.1)

    df_dev_review = df_dev_sampled.drop(columns=['sentiment'])
    df_dev_sentiment = df_dev_sampled["sentiment"]
    df_tot = df_dev_review.append(df_eval_sampled)
    # print(f'Dimensione sampled development + evaluation set: {len(df_tot)}')
    # print()
else:
    df_dev_review = df_dev.drop(columns=['sentiment'])
    df_dev_sentiment = df_dev["sentiment"]
    df_tot = df_dev_review.append(df_eval)
    # print(f'Dimensione development + evaluation set: {len(df_tot)}')
    # print()

# Wordcloud ---------------------------------------------------------
stopwords = custom_import_stopwords('italian_stopwords')
# stopwords_en = custom_import_stopwords('english_stopwords')
stopwords_fr = custom_import_stopwords('french_stopwords')
stopwords_es = custom_import_stopwords('spanish_stopwords')
stopwords_de = custom_import_stopwords('german_stopwords')

wordcloud_flag = False

if wordcloud_flag:
    translator_2 = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    df_tmp_pos['review'] = df_tmp_pos['review'].apply(lambda x: x.translate(translator_2))
    df_tmp_pos['review'] = df_tmp_pos['review'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 3
                                                                          and word not in stopwords]))
    df_tmp_pos['review'] = df_tmp_pos['review'].str.lower()

    wordcloud_pos = WordCloud(width=1000, height=700, background_color='white',
                              min_font_size=10).generate(' '.join(df_tmp_pos['review']))
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud_pos)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    df_tmp_neg['review'] = df_tmp_neg['review'].apply(lambda x: x.translate(translator_2))
    df_tmp_neg['review'] = df_tmp_neg['review'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 3
                                                                          and word not in stopwords]))
    df_tmp_neg['review'] = df_tmp_neg['review'].str.lower()

    wordcloud_neg = WordCloud(width=1000, height=700, background_color='white',
                              min_font_size=10).generate(' '.join(df_tmp_neg['review']))
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud_neg)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

# TF-IDF ------------------------------------------------------------------------------------------
print()
print('TF-IDF Phase')

vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(df_tot['review'])

X_train_valid = X_tfidf[:len(df_dev_review)]
X_eval = X_tfidf[len(df_dev_review):]

# Training & Evaluation ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_train_valid, df_dev_sentiment, test_size=0.25, random_state=1)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Classification ----------------------------------------------------------------------------------
print()
print('Classification Phase')
class_type = 'svc'

# Naive Bayes Classifier -------------------------------------------------
if class_type == 'naive':
    print('Multinomia Naive Bayes Classifier')

    hyp_parameters = {
        "alpha": [0.1, 1, 10]
    }

    for config in ParameterGrid(hyp_parameters):
        clf = MultinomialNB(**config)
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        clf_f1_score = f1_score(y_test, y_test_pred, average='weighted')
        print(f"f1 Score: {clf_f1_score}")
        print(f"Configuration: {config}")

# SVC Classifier -------------------------------------------------
elif class_type == 'svc':
    print('SVC Classifier')

    hyp_parameters = {
        "random_state": [0],
        "C": [1, 10, 100, 1000],
        "kernel": ['linear', 'poly', 'rbf'],
        "gamma": [0.1, 1, 10, 100],
        "degree": [2],
        "class_weight": [None, 'balanced', {0: 1, 1: 2}],
        "max_iter": [5000]
    }

    config_cnt = 0
    tot_config = 4 * 3 * 4 * 3
    max_f1 = 0

    # for config in ParameterGrid(hyp_parameters):
    # config_cnt += 1
    # print(f'Analizing config {config_cnt} of {tot_config} || Config: {config}')

    # clf = SVC(**config)
    # clf.fit(X_train, y_train)
    # y_test_pred = clf.predict(X_test)
    # clf_f1_score = f1_score(y_test, y_test_pred, average='weighted')

    # if clf_f1_score > max_f1:
    # max_f1 = clf_f1_score
    # print(f"-----> Score: {clf_f1_score}")
    # print()
    # Config: {'C': 10, 'class_weight': None, 'degree': 2, 'gamma': 0.1,
    #          'kernel': 'linear', 'max_iter': 5000, 'random_state': 0}
    #
    # -----> Score: 0.967768196885848

# Esplorazione recensioni sbagliate ---------------------------------------------------------------
esploration_flag = False

if esploration_flag:
    listytest = y_test.values.tolist()
    yind = y_test.index.values.tolist()

    index_list1 = []
    index_list2 = []
    for i in range(0, len(y_test_pred)):
        if y_test_pred[i] != listytest[i]:
            index_list1.append(yind[i])
            index_list2.append(i)

    cnt = 0
    for index in index_list1:
        # print(df_dev_review.iloc[index, 0])
        # print(f'predicted {y_test_pred[index_list2[cnt]]} but actually {listytest[index_list2[cnt]]}')
        cnt += 1
        # print(f'Review index: {index}')
        # print(f'Totale errori: {len(index_list1)}')
        # print(f'Totale predetti: {len(y_test_pred)}')
        # print('-------------------------------------------')
        # print()

# Creazione file di output ------------------------------------------------------------------------
clf_final = LinearSVC(C=10)
clf_final.fit(X_train_valid, df_dev_sentiment)
y_pred_final = clf_final.predict(X_eval)
custom_csv_print(y_pred_final)
print('Output file aggiornato')
