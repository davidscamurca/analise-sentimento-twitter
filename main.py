# Incompleto
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

#Lendo a base de dados
dataset = pd.read_csv('tweets_mg.csv')

# Separando em classes
tweets = dataset['Text'].values
classes = dataset['Classificacao'].values

#Bag of Words, Naive
vectorizer = CountVectorizer(analyzer="word")
freq_tweets = vectorizer.fit_transform(tweets)

modelo = MultinomialNB()
modelo.fit(freq_tweets,classes)

# testes = ['Esse governo está no início, vamos ver o que vai dar',
#          'Estou muito feliz com o governo de Minas esse ano',
#          'O estado de Minas Gerais decretou calamidade financeira!!!',
#          'A segurança desse país está deixando a desejar',
#          'O governador de Minas é do PT']

# freq_testes = vectorizer.transform(testes)
# print(modelo.predict(freq_testes))

sentimento = ['Positivo','Negativo', 'Neutro']

resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)
print(metrics.accuracy_score(classes, resultados))
print("-----------")

print(metrics.classification_report(classes,resultados,sentimento),'')

print("-----------MATRIZ DE CONFUSAO--------")

print(pd.crosstab(classes,resultados, rownames=['Real'], colnames=['Predito'],margins=True),'')
