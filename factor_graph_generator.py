'''
	If this script finally ends humanity, send some nutella to: Priyansh (pc.priyansh@gmail.com)

	Aimed at interpreting the rules in the dataset and then generating a factor graph out of them.
	
	This script will invoke classes in the graph-related-activites folder.

	@TODO: Deal with unary predicates here. !URGENT!
'''

from graph import node, belief_propagation

from pprint import pprint
import re


#Open file
f = open('datasets/wordnet/raw/train-learned.ppr')
def rule_parser(line):
	
	var = []
	factors = []

	if not 'learnedPred' == line[:11]:
		print line
		# raw_input("Skipping this line. Press Enter to continue")
		return -1

	head, body = line.split(':-')


	'''
		Head parsing
	'''
	head = head.replace('learnedPred','').replace('(','').replace(')','')
	head = head.split(',')

	head_factor = head[0].replace('i_','')
	head_var = [x.strip() for x in head[1:]]
	factors.append(head_factor)		#@TODO: Shall this predicate be included?
	var += head_var


	'''
		Body parsing
	'''
	body = body.split()

	#First, take away the 'type' of factor graph this is.
	type = body[-1]
	body_triples = body[:-1]

	body = {}
	for triple in body_triples:
		triple = triple.replace('rel(','').replace('),','').replace(')','').strip().replace(" ","").split(',')
		body[triple[0]] = triple[1:]
		factors.append(triple[0])
		var += triple[1:]
		
	'''
		Now, we have parsed both the head and the body of the thing. 
		Now,we need to call nodes and create the factor graph on which 
			I can run the belief propagation algorithm.

			First, collect all the variables, and create nodes for them (cannot have more than 1 variable node for one variable)
	'''
	var = list(set(var))
	var = { x: node.Variable(x) for x in var }

	#Now, I'll go through the collected var and replace the strings with the nodes in the body dictionary.
	head_var = [ var[x].set_head() for x in head_var ]
	for key in body.keys():
		body[key] = [ var[x] for x in body[key]]

	
	'''
		Now, for every factor in the body 
			(even when the same factor may appear more than once, 
			they will have a different object, unlike variables.) 
		we create a new factor object.
	'''
	factors = [ node.Factor(key, body[key][0], body[key][1]) for key in body.keys()]
	fictional_factor = node.Factor(head_factor, head_var[0], head_var[1])
	
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# 				DEBUG
	
	# print line
	# # for v in head_var:
	# # 	print v, ", ",
	# # print "\n"
	# # for key in body.keys():
	# # 	print key, ": ",
	# # 	for v in body[key]:
	# # 		print v, ", ",
	# # 	print "\n"
	# # pprint(head_var)
	# # pprint(body)
	# print "Summary:"
	# for factor in factors:
	# 	print factor

	# print "Head:", fictional_factor
	# print "~~~~~~~~~~~~"
	# raw_input()
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# print "type is", type(var)
	var = [var[key] for key in var.keys()]
	return var,factors,fictional_factor
# graph = belief_propagation.Graph(_variables=[var[key] for key in var], _factors=factors, _fictional_factor=fictional_factor, _rule = line)
# output = graph.propagate_thy_beliefs()