import csv
import pandas as pd
from sklearn import preprocessing
import numpy as np

fname = "datasets/wordnet/raw/valid.cfacts"	#dataset file name. Since wordnet is tabseperated file.

facts = [] #place where the data after file parsing would be stored

#file is in tab seperated formated
with open(fname) as tsv:
	reader = csv.reader(tsv, dialect='excel', delimiter='\t')
	for row in reader:
		facts.append(row)


def label_one_hot_encoder(fatcs):
	#preprocesssing for converting the data into a one hot encoded format
	facts = np.array(facts)
	facts = np.delete(facts, [0], axis=1)
	facts = facts.transpose()
	le = preprocessing.LabelEncoder()
	facts[0] = le.fit_transform(facts[0])
	facts[1] = le.fit_transform(facts[1])
	facts[2] = le.fit_transform(facts[2])

	ents = np.concatenate([facts[1],facts[2]])
	entites = [[ent] for ent in ents]
	enc = preprocessing.OneHotEncoder()
	enc.fit(entites)

	return le,enc



# #preprocesssing for converting the data into a one hot encoded format
# facts = np.array(facts)
# facts = np.delete(facts, [0], axis=1)
# facts = facts.transpose()
# le = preprocessing.LabelEncoder()
# facts[0] = le.fit_transform(facts[0])
# facts[1] = le.fit_transform(facts[1])
# facts[2] = le.fit_transform(facts[2])

# ents = np.concatenate([facts[1],facts[2]])
# entites = [[ent] for ent in ents]
# enc = preprocessing.OneHotEncoder()
# enc.fit(entites)

# facts[1] = [enc.transform(fact).toarray()[0] for fact in facts[1]]
# facts[2] = [enc.transform(fact) for fact in facts[2]]

# facts_row_1 = [enc.transform(x) for x in facts[1]]
# facts_row_2 = [enc.transform(x) for x in facts[2]]
# data = [facts[0],facts_row_1,facts_row_2]
# data = np.array(data)
# data = data.transpose() #th
















