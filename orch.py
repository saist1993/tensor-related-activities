''' >Calls dataset reader to create one hot encoding and label encoder '''
#external libraries
import csv
import theano
import numpy as np
import theano.tensor as T
import scipy.sparse as sps
from theano import sparse
from sklearn import preprocessing
#internal libraries
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
		Then, we make three arrays out of this filtered facts:
			- A: left entity of these facts
			- B: right entity of these facts
			- C: random floats (same length as A and B)
	'''
	left_entity = []
	right_entity = []
	random_number = []
	for i in xrange(0,len(facts[0])):
		if facts[0][i] == rel_label:
			left_entity.append(facts[1][i])
			right_entity.append(facts[2][i])
			random_number.append(np.random.randn)
	mat = np.zeros((number_of_entites,number_of_entites))
	for i in range(len(left_entity)):
		
		mat[left_entity[i]][right_entity[j]] = np.random.randn()

	return sps.csr_matrix(mat)
	# return sparse.CSR((random_number,(left_entity,right_entity)),shape=(number_of_entites,number_of_entites))
	# return sps.csr_matrix((random_number,(left_entity,right_entity)),shape=(number_of_entites,number_of_entites))
	# matrix = sparse.coo_matrix((C,(A,B)),shape=(5,5))


#encodes the relations and entites 
def encode(vars,factors,fictional_factor,number_of_entites,facts):
	# print "encoding variables"
	for var in vars:
		var.u = T.dvector(var.label)
	# print "done encoding variables and off to relation encoding"	
	for rel in factors:
		if rel.label not in relation_lookup:
			relation_lookup[rel.label] = create_relation_matrix(rel.label,number_of_entites,facts)
		# try:
		# 	print relation_lookup[rel.label]
		# except KeyError:
		# 	print "key error"
		# 	# print number_of_entites
		# 	relation_lookup[rel.label] = create_relation_matrix(rel.label,number_of_entites,facts)

		# 	# relation_lookup[rel.label] = theano.shared(create_relation_matrix(rel.label,number_of_entites,facts))
		# 	raw_input("see error")
		# 	print "done creating shared varaibles"
		rel.M = T.dmatrix('T'+rel.label)

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


#file is in tab seperated formated.
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

	print "started rule parsing for rule: ", line
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
		rule_lookup[fictional_factor.label] = rule_lookup[fictional_factor.label].append(temp)
	except:
		rule_lookup[fictional_factor.label] = []
		rule_lookup[fictional_factor.label] = rule_lookup[fictional_factor.label].append(temp)

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
		pseudocode:
			@TODO
	'''
	y = []
	for output in node[1:]:
		y.append(entity_encoder.transform(label_encoder.transform(output[2])))
	y = np.sum(y,axis=0)	#stores the true label
	for bit in xrange(0,len(y)):
		if y[bit] > 1:
			y[bit] = 1
	
	''' 
		Creating true input labels
	'''
	x = node[0][1].u #verify this
	# print x, type(x),node[0][1]
	# raw_input("check for types and shit of x")
	'''
		list of rules having same fictional factor / head.
		each rule is a list containing :- [vars,factors,fictional_factor, belief_propagation_equation]
		vars - list of variables ; factors - list of relation ; fictional_factor - head ; belief_propagation_equation - theano equation
	'''
	rules = rule_lookup(node[0][0])


	for rule in rules:
		#each rule is a list containing :- [vars,factors,fictional_factor, belief_propagation_equation]
		theano_function = rule[3]
		relation_list = []
		for rel in rule[-1]:
			relation_list.append(relation_lookup[rel.label])
		output,relation_matrix = theano_function(x,y,relation_list)
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