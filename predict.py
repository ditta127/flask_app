import pandas as pd
import deap
import numpy
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tashaphyne.normalize as normalize
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import py3langid as langid

nltk.download('stopwords')
english_stopwords = stopwords.words('english')
french_stopwords = stopwords.words('french')
arab_stopwords = set(nltk.corpus.stopwords.words("arabic"))


def check_language(SMS_text):
    def remove_stopwords_EN(data):

        data = data.apply(lambda x: ' '.join([word for word in x.split() if word not in (english_stopwords)]))
        # data= data.apply(lambda x: ' '.join([x for x in x.split() if len(x) > 3]))
        return data

    def preprocess_english_text(data):

        data = str(data).lower()
        data = re.sub('\d+', '', data)  # Remove numbers
        data = re.sub('[^\w\s]', '', data)  # Remove punctuation
        data = re.sub('http[^\s]+', '', data)  # Remove URLs
        data = re.sub(
            r'[ŠšŽžÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûýþÿŒ©®¯³´¸¹¾Ð×ØÞð÷øüñÑ¿¬½¼¡«»¦µ±°•·²€„…†‡ˆ‰‹Œ‘’“”˜™š›œÆ¢£¥ƒ¤¶§]',
            '', data)  # remove rpecial characters from other languages
        data = re.sub(r'\n', '', data)
        data = re.sub(r'[\_-]', '', data)  # Remove underscores and dashes
        data = data.rstrip()
        return data

    def add_structural_features_AR_EN(data, data_FE):
        # data_FE['length']=data.apply(len)

        data_FE['link_presence'] = data.str.contains(
            'WWW\\.(.)*|www\\.(.)*|(.)*http(.)*|(.)*HTTP(.)*|(.)*HTTPS(.)*|(.)*https(.)*|(.)@(.)|(.)*\\.com|(.)*\\.net|(.)*\\.org|(.)*\\.ORG|(.)*\\.COM|(.)*\\.NET')

        data_FE['special_char_presence'] = data.str.contains(
            r'[ŠšŽžÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûýþÿŒ©®¯³´¸¹¾Ð×ØÞð÷øüñÑ¿¬½¼¡«»¦µ±°•·²€„…†‡ˆ‰‹Œ‘’“”˜™š›œÆ¢£¥ƒ¤¶§]')

        data_FE['phone_number_presence'] = data.str.contains('[0-9]{3,}')

        data_FE['XXX_presence'] = data.str.contains('[Xx]{3,}')

        data_FE['link_presence'] = data_FE['link_presence'].astype(int)
        data_FE['special_char_presence'] = data_FE['special_char_presence'].astype(int)
        data_FE['phone_number_presence'] = data_FE['phone_number_presence'].astype(int)
        data_FE['XXX_presence'] = data_FE['XXX_presence'].astype(int)

        return (data_FE)

    def FeatureExtraction_EN(data):

        # feature extraction
        # apply bag of words

        transform_data_bow = CountVectorizer().fit(data)
        bow_data = transform_data_bow.transform(data)
        bow_data.shape  # gives nb of rows and cols
        # data_FE = pd.DataFrame(bow_data.A, columns =transform_data_bow.get_feature_names_out())

        # apply tfidf, from occurences to frequencies
        tfidf_transformer = TfidfTransformer()
        data_tfidf = tfidf_transformer.fit_transform(bow_data)

        # turn vector into data frame
        data_FE = pd.DataFrame(data_tfidf.A, columns=transform_data_bow.get_feature_names_out())

        # return the dataframe of the extracted features
        return (data_FE)

    def classification_EN(SMS_text):

        data = pd.read_csv("datasetSpamEN.csv",
                           encoding='latin-1')
        data = data.drop(["Unnamed: 0"], axis=1)
        # data preprocessing
        data['text'] = data['text'].apply(preprocess_english_text)
        # remove stop words
        data['text'] = remove_stopwords_EN(data['text'])
        data_FE = FeatureExtraction_EN(data['text'])

        individual = pd.read_csv("individualLightGBMEN.csv",
                                 encoding="latin-1")
        individual.drop(['Unnamed: 0'], axis=1, inplace=True)
        individual = list(individual['0'])
        cols = [index for index in range(len(individual)) if individual[index] == 0]
        training_data = data_FE.drop(data_FE.columns[cols], axis=1)

        training_data = add_structural_features_AR_EN(data['text'], data_FE)

        # input the sms

        if SMS_text == '':
            print('0')
        else:
            SMS_dic = {"label": [" "],
                       "text": [SMS_text],
                       }
            df_SMS = pd.DataFrame(SMS_dic)
            text = df_SMS['text'].copy(deep=True)
            # preprocessing

            df_SMS["text"] = df_SMS["text"].apply(preprocess_english_text)
            df_SMS["text"] = remove_stopwords_EN(df_SMS["text"])
            SMS_FE = FeatureExtraction_EN(df_SMS["text"])

            SMS_FE = add_structural_features_AR_EN(text, SMS_FE)

            X_training = training_data
            Y_training = data['label']
            Y_testing = df_SMS['label']
            X_testing = SMS_FE
            # matching attributes
            # delet ats in test set that don't exist in training set

            for i in X_testing.columns:
                if i not in X_training.columns:
                    X_testing = X_testing.drop(i, axis=1)

            for j in X_training.columns:
                if j not in X_testing.columns:
                    X_testing[j] = 0

            X_testing = X_testing.reindex(X_training.columns, axis=1)

            clf = lgb.LGBMClassifier()
            y_pred = clf.fit(X_training, Y_training).predict(X_testing)
            return (y_pred)

    ############################french part

    def remove_stopwords_FR(data):

        data = data.apply(lambda x: ' '.join([word for word in x.split() if word not in (french_stopwords)]))
        return data

    def preprocess_french_text(data):

        data = str(data).lower()
        data = re.sub('\d+', '', data)  # Remove numbers
        data = re.sub('[©®¯³´¸¹¾ÃÅÐÒÓÕ×ØÝÞãðõ÷øüýþóñÑ¿¬½¼¡«»¦ßµ±°•·²€„…†‡ˆ‰Š‹Œ‘’“”˜™š›œåÆ¢£¥ƒ¤¶§]', '',
                      data)  # Remove special characters
        data = re.sub('[^\w\s]', '', data)  # Remove punctuation
        data = re.sub(r'\n', '', data)
        data = re.sub(r'[\_-]', '', data)  # Remove underscores and dashes
        data = re.sub('http[^\s]+', '', data)  # Remove URLs
        data = re.sub('[àâä]', 'a', data)  # Remove accents
        data = re.sub('[éèêë]', 'e', data)  # Remove accents
        data = re.sub('[îï]', 'i', data)  # Remove accents
        data = re.sub('[ôö]', 'o', data)  # Remove accents
        data = re.sub('[ùûü]', 'u', data)  # Remove accents
        data = re.sub('[ÿ]', 'y', data)  # Remove accents
        data = re.sub('[ç]', 'c', data)  # Remove accents
        data = data.rstrip()
        return data

    def add_structural_features_FR(data, data_FE):
        # data_FE['length']=data.apply(len)

        data_FE['link_presence'] = data.str.contains(
            'WWW\\.(.)*|www\\.(.)*|(.)*http(.)*|(.)*HTTP(.)*|(.)*HTTPS(.)*|(.)*https(.)*|(.)@(.)|(.)*\\.com|(.)*\\.net|(.)*\\.org|(.)*\\.ORG|(.)*\\.COM|(.)*\\.NET')

        data_FE['special_char_presence'] = data.str.contains(
            '[©®¯³´¸¹¾ÃÅÐÒÓÕ×ØÝÞãðõ÷øüýþóñÑ¿¬½¼¡«»¦ßµ±°•·²€„…†‡ˆ‰Š‹Œ‘’“”˜™š›œåÆ¢£¥ƒ¤¶§]')

        data_FE['phone_number_presence'] = data.str.contains('[0-9]{3,}')

        data_FE['XXX_presence'] = data.str.contains('[Xx]{3,}')

        data_FE['link_presence'] = data_FE['link_presence'].astype(int)
        data_FE['special_char_presence'] = data_FE['special_char_presence'].astype(int)
        data_FE['phone_number_presence'] = data_FE['phone_number_presence'].astype(int)
        data_FE['XXX_presence'] = data_FE['XXX_presence'].astype(int)

        return (data_FE)

    def FeatureExtraction_FR(data):

        # feature extraction
        # apply bag of words
        stop_wordsFR = stopwords.words('french')
        transform_data_bow = CountVectorizer(stop_words=stop_wordsFR).fit(data)
        bow_data = transform_data_bow.transform(data)
        bow_data.shape  # gives nb of rows and cols
        # data_FE = pd.DataFrame(bow_data.A, columns =transform_data_bow.get_feature_names_out())

        # apply tfidf, from occurences to frequencies
        tfidf_transformer = TfidfTransformer()
        data_tfidf = tfidf_transformer.fit_transform(bow_data)

        # turn vector into data frame
        data_FE = pd.DataFrame(data_tfidf.A, columns=transform_data_bow.get_feature_names_out())

        # return the dataframe of the extracted features
        return (data_FE)

    def classification_FR(SMS_text):

        data = pd.read_csv("datasetSpamFR.csv",
                           encoding='utf-8')

        # data preprocessing
        data['text'] = data['text'].apply(preprocess_french_text)
        # remove stop words
        data['text'] = remove_stopwords_FR(data['text'])

        data_FE = FeatureExtraction_FR(data['text'])

        individual = pd.read_csv("individualLightGBMFR.csv",
                                 encoding="utf-8")
        individual.drop(['Unnamed: 0'], axis=1, inplace=True)
        individual = list(individual['0'])
        cols = [index for index in range(len(individual)) if individual[index] == 0]

        # training set = result of the genetic algorithm
        training_data = data_FE.drop(data_FE.columns[cols], axis=1)
        # reread data
        data = pd.read_csv("datasetSpamFR.csv",
                           encoding='utf-8')
        data = data.drop(["Unnamed: 0"], axis=1)
        training_data = add_structural_features_FR(data['text'], data_FE)

        training_data.head()

        # input the sms

        if SMS_text == '':
            print('0')
        else:
            SMS_dic = {"label": [" "],
                       "text": [SMS_text],
                       }
            df_SMS = pd.DataFrame(SMS_dic)
            text = df_SMS['text'].copy(deep=True)
            # preprocessing

            df_SMS["text"] = df_SMS["text"].apply(preprocess_french_text)
            df_SMS["text"] = remove_stopwords_FR(df_SMS["text"])

            SMS_FE = FeatureExtraction_FR(df_SMS["text"])

            SMS_FE = add_structural_features_FR(text, SMS_FE)

            X_training = training_data
            Y_training = data['label']
            Y_testing = df_SMS['label']
            X_testing = SMS_FE

            # matching attributes
            # delet ats in test set that don't exist in training set

            for i in X_testing.columns:
                if i not in X_training.columns:
                    X_testing = X_testing.drop(i, axis=1)

            for j in X_training.columns:
                if j not in X_testing.columns:
                    X_testing[j] = 0

            X_testing = X_testing.reindex(X_training.columns, axis=1)

            clf = lgb.LGBMClassifier()
            y_pred = clf.fit(X_training, Y_training).predict(X_testing)
            return (y_pred)

    ########################### ARABIC PART

    def normalizeArabic(data):
        return normalize.normalize_searchtext(data)

    def remove_stopwords_AR(data):

        data = data.apply(lambda x: ' '.join([word for word in x.split() if word not in (arab_stopwords)]))
        return data

    def preprocess_arabic_text(data):
        data = re.sub('\d+', '', data)  # Remove numbers
        data = re.sub('[^\w\s]', '', data)  # Remove punctuation
        data = re.sub(
            r'[ŠšŽžÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûýþÿŒ©®¯³´¸¹¾Ð×ØÞð÷øüñÑ¿¬½¼¡«»¦µ±°•·²€„…†‡ˆ‰‹Œ‘’“”˜™š›œÆ¢£¥ƒ¤¶§]',
            '', data)  # remove rpecial characters from other languages
        data = re.sub('http[^\s]+', '', data)  # Remove URLs
        data = re.sub(r'[a-zA-Z]', '', data)  # Remove english words
        data = re.sub(r'\n', '', data)
        data = re.sub(r'[\_-]', '', data)  # Remove underscores and dashes
        data = re.sub(r'(.)\1+', r'\1', data)  # remove duplicate letters
        data = data.rstrip()
        return data

    def FeatureExtraction_AR(data):

        # feature extraction
        # apply bag of words

        transform_data_bow = CountVectorizer().fit(data)
        bow_data = transform_data_bow.transform(data)
        bow_data.shape  # gives nb of rows and cols
        # data_FE = pd.DataFrame(bow_data.A, columns =transform_data_bow.get_feature_names_out())

        # apply tfidf, from occurences to frequencies
        tfidf_transformer = TfidfTransformer()
        data_tfidf = tfidf_transformer.fit_transform(bow_data)

        # turn vector into data frame
        data_FE = pd.DataFrame(data_tfidf.A, columns=transform_data_bow.get_feature_names_out())

        # return the dataframe of the extracted features
        return (data_FE)

    def classification_AR(SMS_text):

        data = pd.read_csv("datasetSpamAR.csv",
                           encoding='utf-8-sig')
        data = data.drop(["Unnamed: 0"], axis=1)
        # add structural features

        data['text'] = data['text'].apply(preprocess_arabic_text)
        # remove stop words

        data['text'] = remove_stopwords_AR(data['text'])

        # normlize arabic
        data['text'] = data['text'].apply(normalizeArabic)

        data_FE = FeatureExtraction_AR(data['text'])

        individual = pd.read_csv("individualLightGBMAR.csv",
                                 encoding="utf-8-sig")
        individual.drop(['Unnamed: 0'], axis=1, inplace=True)
        individual = list(individual['0'])
        cols = [index for index in range(len(individual)) if individual[index] == 0]
        # training set = result of the genetic algorithm
        training_data = data_FE.drop(data_FE.columns[cols], axis=1)
        # reread data
        data = pd.read_csv("datasetSpamAR.csv",
                           encoding='utf-8-sig')
        data = data.drop(["Unnamed: 0"], axis=1)
        training_data = add_structural_features_AR_EN(data['text'], data_FE)

        # delete empty lines
        training_data.replace('', np.nan, inplace=True)
        training_data.dropna(inplace=True)

        # input the sms

        if SMS_text == '':
            print('0')
        else:
            SMS_dic = {"label": [" "],
                       "text": [SMS_text],
                       }
            df_SMS = pd.DataFrame(SMS_dic)
            text = df_SMS['text'].copy(deep=True)
            # preprocessing

            df_SMS["text"] = df_SMS["text"].apply(preprocess_arabic_text)
            df_SMS["text"] = remove_stopwords_AR(df_SMS["text"])
            # normlize arabic
            df_SMS["text"] = df_SMS["text"].apply(normalizeArabic)

            SMS_FE = FeatureExtraction_AR(df_SMS["text"])

            SMS_FE = add_structural_features_AR_EN(text, SMS_FE)

            X_training = training_data
            Y_training = data['label']
            Y_testing = df_SMS['label']
            X_testing = SMS_FE
            # matching attributes
            # delet ats in test set that don't exist in training set

            for i in X_testing.columns:
                if i not in X_training.columns:
                    X_testing = X_testing.drop(i, axis=1, inplace=True)

            for j in X_training.columns:
                if j not in X_testing.columns:
                    X_testing[j] = 0

            X_testing = X_testing.reindex(X_training.columns, axis=1)

            clf = lgb.LGBMClassifier()
            y_pred = clf.fit(X_training, Y_training).predict(X_testing)
            return (y_pred)

    language = langid.classify(SMS_text)

    if language[0] == 'ar':
        sms_prediction = classification_AR(SMS_text)
        print('this Arabic message is predicted to be \t: ' + str(sms_prediction))
    elif language[0] == 'en':
        sms_prediction = classification_EN(SMS_text)
        print('this English message is predicted to be \t: ' + str(sms_prediction))
    elif language[0] == 'fr':
        sms_prediction = classification_FR(SMS_text)
        print('this French message is predicted to be \t: ' + str(sms_prediction))

    return (sms_prediction)


