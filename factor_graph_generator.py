'''
	If this script finally ends humanity, send some nutella to: Priyansh (pc.priyansh@gmail.com)

	Aimed at interpreting the rules in the dataset and then generating a factor graph out of them.
	
	This script will invoke classes in the graph-related-activites folder.
'''

from graph import node

from pprint import pprint
import re


#Open file
f = open('datasets/wordnet/raw/train-learned.ppr')
for line in f:
	
	vars = []
	factors = []

	if not 'learnedPred' == line[:11]:
		print line
		# raw_input("Skipping this line. Press Enter to continue")
		continue

	head, body = line.split(':-')


	'''
		Head parsing
	'''
	head = head.replace('learnedPred','').replace('(','').replace(')','')
	head = head.split(',')

	head_edge = head[0].replace('i_','')
	head_vars = [x.strip() for x in head[1:]]
	factors.append(head_edge)		#@TODO: Shall this predicate be included?
	vars += head[1:]


	'''
		Body parsing
	'''
	body = body.split()

	#First, take away the 'type' of factor graph this is.
	type = body[-1]
	body_triples = body[:-1]

	body = {}
	for triple in body_triples:
		triple = triple.replace('rel(','').replace('),','').replace(')','').strip().split(',')
		body[triple[0]] = triple[1:]
		factors.append(triple[0])
		vars += triple[1:]


	pprint(head_vars)
	pprint(body)



	'''
		Now, we have parsed both the head and the body of the thing. 
		Now,we need to call nodes and create the factor graph on which 
			I can run the belief propagation algorithm.

			First, collect all the variables, and create nodes for them (cannot have more than 1 variable node for one variable)
	'''
	vars = list(set(vars))
	factors = list(set(factors))

	vars = { x: node.Node(x, _type = 'Variable') for x in vars }
	factors = { x: node.Node(x, _type = 'Factor') for x in factors}

	#Now, I'll go through the collected vars and replace the strings with the nodes in the body dictionary.
	head_vars = [ vars[x].set_as_head() for x in head_vars ]
	for key in body.keys():
		body[key] = [ vars[x] for x in body[key]]

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# 				DEBUG
	# for v in head_vars:
	# 	print v, ", ",
	# print "\n"
	# for key in body.keys():
	# 	print key, ": ",
	# 	for v in body[key]:
	# 		print v, ", ",
	# 	print "\n"
	pprint(head_vars)
	pprint(body)
	raw_input()
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


	'''
		Now that the nodes are made properly for all the variables, we need to:
			- create node for factors
			- create the factor graph

	'''

