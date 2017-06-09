''' >Calls dataset reader to create one hot encoding and label encoder '''
#external libraries
import csv
import theano
import numpy as np
import theano.tensor as T
from sklearn import preprocessing
#internal libraries
import dataset_reader,factor_graph_generator

fname = "datasets/wordnet/raw/valid.cfacts"	#dataset file name. Since wordnet is tabseperated file.

facts = [] #place where the data after file parsing would be stored

#file is in tab seperated formated
with open(fname) as tsv:
	reader = csv.reader(tsv, dialect='excel', delimiter='\t')
	for row in reader:
		facts.append(row)

#label_encoder - encodes any relation or entity to a label
#entity_encoder - gives one-hot reperesentation of the label encoded entity
label_encoder,entity_encoder = dataset_reader.label_one_hot_encoder(facts)

'''
	This block will read rules from the wordnet dataset 
	and call factorgraph generator to return the theano function which can be then used to do the NN Magic.
'''
f = open('datasets/wordnet/raw/train-learned.ppr')
for line in f:
	vars,factors,fictional_factor = factor_graph_generator.rule_parser(line)
	#vars - list of variables ; factors - list of relation ; fictional_factor - head
	graph = belief_propagation.Graph(_variables=[vars[key] for key in vars], _factors=factors, _fictional_factor=fictional_factor, _rule = line)
	output = graph.propagate_thy_beliefs()



