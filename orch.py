''' >Calls dataset reader to create one hot encoding and label encoder '''
#external libraries
import csv

import numpy as np

import theano
from theano import sparse
import theano.tensor as T

import scipy.sparse as sps

from pprint import pprint

from sklearn import preprocessing

# Internal libraries
import dataset_reader,factor_graph_generator, utils, graph.belief_propagation as gbp

fname = "datasets/wordnet/raw/valid.cfacts"	#dataset file name. Since wordnet is tabseperated file.
example_file = 'datasets/wordnet/raw/train.examples'
facts = [] #place where the data after file parsing would be stored
rule_lookup = {} # {rule.label - [[vars,factors,fictional_factor,theano_function,symbols],[rulebody2]]}
relation_lookup = {} #{relation.label:sparseMatrix repeeresentation of relation}

def create_relation_matrix(rel_label,number_of_entites,facts):
	''' 
		Function creates sparse relation matrix given relation label and facts.

		For the given predicate, we filter from the facts object, all those facts which have our target relation.
		Then, we make three arrays out of this filtered k
	'''
	left_entity = []
	right_entity = []
	random_number = []

	mat = np.zeros((number_of_entites,number_of_entites))

	for i in xrange(0,len(facts[0])):

		if int(facts[0][i]) == int(rel_label):
			left_entity.append(int(facts[1][i]))
			right_entity.append(int(facts[2][i]))
			mat[left_entity[-1]][right_entity[-1]] = np.random.randn()			

	# for i in range(len(left_entity:	
	# 	for j in range(len(right_entity)):
	# 		mat[left_entity[i]][right_entity[j]] = np.random.randn()

	# print mat
	print mat.shape
	# print mat.sum()
	# raw_input("See matrix")
	mat = sps.csr_matrix(mat)
	return mat
	# return sparse.CSR((random_number,(left_entity,right_entity)),shape=(number_of_entites,number_of_entites))
	# return sps.csr_matrix((random_number,(left_entity,right_entity)),shape=(number_of_entites,number_of_entites))
	# matrix = sparse.coo_matrix((C,(A,B)),shape=(5,5))


#encodes the relations and entites 
def encode(vars,factors,fictional_factor,number_of_entites,facts):

	for var in vars:
		var.u = sparse.csr_matrix(var.label)
		# var.u = T.dmatrix(var.label)

	for rel in factors:
		if rel.label not in relation_lookup:

			#Label encode the relation label.
			rel_label = label_encoder.transform([rel.label])[0]

			#Get a matrix for this relation.
			relation_lookup[rel.label] = create_relation_matrix(rel_label,number_of_entites,facts)
	
		rel.M = sparse.csr_matrix(rel.label)

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# DEBUG
	# print "DEBUGGING ENCODER"
	# print "Vars"
	# for v in vars:
	# 	print v.label
	# 	print v.u, ' ', v.u.__class__

	# print "Factors"
	# for f in factors:
	# 	print f.label
	# 	print f.i.label, ', ', f.o.label
	# 	print f.i.u, ', ', f.o.u

	# print "Factor Shared Vars"
	# for f in factors:
	# 	print f.M.__class__
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	return vars,factors,fictional_factor


#This parses the facts file.
with open(fname) as tsv:
	reader = csv.reader(tsv, dialect='excel', delimiter='\t')
	for row in reader:
		facts.append(row)
#facts - [[rel,ent1,ent2],[rel2,ent3,ent4]]

#label_encoder - encodes any relation or entity to a label
#entity_encoder - gives one-hot reperesentation of the label encoded entity
label_encoder,entity_encoder,number_of_entites,facts = dataset_reader.label_one_hot_encoder(facts)#returned facts is transposed of the original sparse

'''
	This block will read rules from the wordnet dataset 
	and call factorgraph generator to return the theano function which can be then used to do the NN Magic.
'''
f = open('datasets/wordnet/raw/train-learned.ppr')
for line in f:
	temp = []
	if factor_graph_generator.rule_parser(line) == -1:
		continue

	print "Rule: ", line
	vars,factors,fictional_factor = factor_graph_generator.rule_parser(line)
	
	vars,factors,fictional_factor = encode(vars,factors,fictional_factor,number_of_entites,facts)
	
	g = gbp.Graph(vars,factors,fictional_factor)
	# symbols whihc have been used for creating the BP equation in the same sequ as it should be 
	theano_function,symbols = g.propagate_thy_beliefs()
	# print "graph completed and theano function received!!"
	# BP() :- send this to belief propogation to get the factor graph
	# vars - list of variables ; factors - list of relation ; fictional_factor - head
	temp.extend([vars,factors,fictional_factor,theano_function,symbols])
	try:
		rule_lookup[fictional_factor.label].append(temp)
	except:
		rule_lookup[fictional_factor.label] = []
		rule_lookup[fictional_factor.label].append(temp)

pprint(relation_lookup)
raw_input("See the relation lookup!")



''' 
rule_lookup is a dictionary for which the key is rule head and the values are a list, where each element in the list corresponds to 
[vars,factros,fictional_factor]	
'''	 
	# graph = belief_propagation.Graph(_variables=[vars[key] for key in vars], _factors=factors, _fictional_factor=fictional_factor, _rule = line)
	# output = graph.propagate_thy_beliefs()

#parse the example file and start the Neural network process

data = utils.parse_example(example_file)


for node in data:
	# node = [['hypernym', '12455540', 'Y'], ['hypernym', '12455540', '13134302']]
	
	'''
		Creating true output labels
		pseudocode:	@TODO
	'''
	y = []
	for output in node[1:]:

		label = label_encoder.transform([output[2]])
		y.append(entity_encoder.transform(label))

		#Now, since y is a vector, but we want a 1,n matrix, we go through a little reshaping.

	y = np.sum(y,axis=0)	#stores the true label
	print y
	
	# Since np.sum has no roof, there can be a value '2' somewhere. 
	# However, in this usecase, every entity (output) must be unique. So this here is redundant.
	# for bit in xrange(0,len(y)):
	# 	if y[bit] > 1:
	# 		y[bit] = 1
	

	''' 
		Creating true input labels
	'''	
	x = node[0][1]								# x is a string right now. 
	x = label_encoder.transform([output[2]])	# x is label encoded.
	x = entity_encoder.transform(x)				# x is now one hot encoded, and thus, the desired _true input_

	'''
		List of rules having same fictional factor / head.
		each rule is a list containing :- [vars,factors,fictional_factor, belief_propagation_equation]
		vars - list of variables ; factors - list of relation ; fictional_factor - head ; belief_propagation_equation - theano equation
	'''
	rules = rule_lookup[node[0][0]]

	'''
		Purpose of this block:
			-> the real training happens here. Previously we just ran BP and got theano functions.
			-> it deals with multiple entries of one rule (same predicate + same entity)
				-> by treating them as individual rules. We're not adding their equations or something.

		Each rule in rules has the format: vars,factors,fictional_factor,theano_function,symbols

	'''
	for rule in rules:

		print rule

		# Collecting the theano function
		theano_function = rule[3]

		# Collecting the symbols (things used inside the function to be given as inputs alongside x and y)
		relation_list = []
		for rel in rule[-1]:
			relation_list.append(relation_lookup[rel.label])

		
		print ("Symbols: ")
		pprint(relation_list)
		
		print x.shape, y.shape
		print [i.shape for i in relation_list]

		raw_input("Verify")

		if len(relation_list) == 1:
			output,relation_matrix = theano_function(x,y,relation_list[0])
		elif len(relation_list) == 2:
			output,relation_matrix = theano_function(x,y,relation_list[0], relation_list[1])
		elif len(relation_list) == 3:
			output,relation_matrix = theano_function(x,y,relation_list[0], relation_list[1], relation_list[2])
		elif len(relation_list) == 4:
			output,relation_matrix = theano_function(x,y,relation_list[0], relation_list[1], relation_list[2], relation_list[3])
		elif len(relation_list) == 5:
			output,relation_matrix = theano_function(x,y,relation_list[0], relation_list[1], relation_list[2], relation_list[4])
		else:
			print "Too many symbols you have. Hmm. Fuck off, you must."

		'''

		'''
		for i in xrange(0,len(rule[-1])):
			rel = rule[-1][i]
			relation_lookup[rel.label] = relation_matrix[i]





''' 
original code
'''

		# parameters = [x.M for x in rule[1]] #list of all the relations occuring in the rule body. relations are Theano shared variables. 
		
		# belief_propagation_equation = T.nnet.softmax(rule[3])	#softmax over the output

		# loss = -y * T.log(belief_propagation_equation) - (1 - y)*T.log(1-belief_propagation_equation) # unregularized cross-entropy loss

		# cost = loss.mean() 	#+ 0.01*(w**2).sum()   (unregularized )

		# gradients  = T.grad(cost, parameters)
		
		# train = theano.function(
  #         inputs=[rule[2].i.u,y],
  #         outputs= belief_propagation_equation,
  #         updates=tuple([(parameters[i], parameters[i] - 0.1 * gradients[i]) for i in range(len(parameters))]))