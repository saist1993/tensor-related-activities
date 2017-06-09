'''
	Authored: 8th June (first version)	
'''
from pprint import pprint
import theano.tensor as T
import numpy as np

import node

class Graph:
	'''
		This class creates a graph based on given data, in an adjacency matrix and runs belief propagation on it,
		as described in the algorithm in the Tensorlog paper (https://arxiv.org/pdf/1605.06523.pdf)

		Input:
			_variables: (type: list of Variable objects) consisting of all the unique variables occuring in the query.
			_factors: (type: list of Factor objeccts) consisting of all the triples existing in the given rule.
			_fictional_factor: (type: Factor) one factor representing the predicate occuring in the head of the query.
			_rule: (type: string) the string from which everything was parsed (bookkeeping purposes).

	'''

	def __init__(self, _variables, _factors, _fictional_factor, _rule = None):

		'''
			Create a null variable.
			Initialize matrix
			Put values in the matrix.
		'''
		print "Creating graph for: ", _rule
		null_node = node.Node("phi")
		
		#Keep the list of variables in order to map the indices of the matrix to actual graph nodes
		self.variables = _variables + [null_node]
		#Keep the constant appearing in the i position of the head variable to be used while doing the BP 
		self.u_c = None 	

		#Keep the predicate that appeared in the head of the query in order to know where do we start the 
		self.head_predicate = _fictional_factor

		#Initialize a numpy matrix
		self.graph = np.zeros(( len(self.variables), len(self.variables) )).astype('object')
		for factor in _factors:
			self.graph[self.variables.index(factor.i)][self.variables.index(factor.o)] = factor

		#Done making the graph and populating it.

		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		# 				DEBUG
	
		# for v in self.variables:
		# 	print v,
		# 	print self.variables.index(v)
		# print ""
		# print self.graph
		self.print_graph()

		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	def _get_neighbours(self, _node, _exclude = None):
		'''
			Internal Function.
			Returns the neighbouring factors of the given node.
			Can exclude a specific factor from it, if required.
		'''
		candidates = self.graph[self.variables.index(_node)].tolist()
		candidates += self.graph[:,self.variables.index(_node)].tolist()
		neighbors = list(set([ x for x in candidates if not x == 0.0]))


		#Exclude the excluding thing.
		neighbors = [ x for x in neighbors if not x == _exclude]		

		return neighbors

	def print_graph(self):
		'''
			Function used to print the matrix with all its fancy tools and contraptions.
		'''
		graph = self.graph.tolist()
		matrix = [[""] + [ v.label for v in self.variables ]] + [ [self.variables[i].label] + [ "--" if v == 0.0 else v.label for v in graph[i]] for i in range(len(graph)) ]
		
		# data = np.asarray(data)
		# print data

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		''' Code to print a matrix beautifully.
		Copied from: https://stackoverflow.com/questions/13214809/pretty-print-2d-python-list'''
		s = [[str(e) for e in row] for row in matrix]
		lens = [max(map(len, col)) for col in zip(*s)]
		fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
		table = [fmt.format(*row) for row in s]
		print '\n'.join(table)
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



	def propagate_thy_beliefs(self, _u_c):
		'''
			Call this function to receive a string containing the path of the belief propagation algorithm.
			We implement the algorithm listed in the paper mentioned in the comments above
			
			Input: the constant appearing in the first position of the head predicate. 

			Pseudocode:
				-> Create an empty theano vector whose definitions will be iteratively changed.
				-> Call compile_message_node_to_factor from the o node of the head predicate. 
				-> Let the functions recursively call each other
				-> Collect their things somehow. @TODO: how. what format. Shall we use theano variables altogether or what
				-> Return said stuff.
		'''	

		self.u_c = _u_c

		print "graph:bp: Starting belief propagation."
		equation = self._comiple_message_node_(self.head_predicate.o, "Fictional Label")
		print equation
		print "graph:bp: Belief propagation complete."
		
		return equation


	def _comiple_message_node_(self, _node, _factor):
		'''
			Pseudocode: 
				(treat _node as X and _factor as L )

				if X is the input variable (global) then
					return u_c , the input
				else
					generate a new variable name v_x
					collect neighbouring L_i of X excluding L
					for [L_1, L_2 .. L_i ], do
						v_i = compile_message(L_i -> X)
					emit(v_x = v1 dot v2 ... dot vi)
					return v_x
		'''

		if _node == self.head_predicate.i:
			#This is the input variable.
			return self.u_c

		#This is NOT the input variable.
		neighbors = self._get_neighbours(_node, _exclude=_factor) #Will be a list of factors.
		
		#Send the neighbour + current node to compilemessage_factor and collect what they have to say.
		# v_x = ' \circle '.join([ self._compile_message_factor_(_factor = factor, _node = _node) for factor in neighbors ])
		neighboring_values = [ self._compile_message_factor_(_factor = factor, _node = _node) for factor in neighbors ]
		if len(neighboring_values) > 0:
			v_x = neighboring_values[0]
			for remaining_values in neighboring_values[1:]:
				v_x = v_x * remaining_values
		else:
			#In this case, since there are no neighbors, there's literally nothing to return.
			#@TODO: What do we do here
			pass

		return v_x

	def _compile_message_factor_(self, _factor, _node):
		'''
			Pseudocode:
				(treat _node as X and _factor as L)

			if L is a unary factor:
				emit v_L,_X = v_L

			elif X is the output node of L
				v_i = compilemessage_node(X_o, L)

			elif X is the output node of L
				v_i =  compilemessage_node(X_i, L)

			return this
		'''
		
		if _factor.o.null:
			#If the factor is unary.
			return _factor.M 	#@TODO: See if we need another variable like self.v to represent the value of unary predicates.

		elif _factor.o == _node:
			#If the node is the output node for this factor
			v_i = self._comiple_message_node_(_factor.i, _factor)
			# return v_i+ " \dot M_"+_factor.label
			return v_i.dot(_factor.label)

		elif _factor.i == _node:
			 #If the node is the input node for this factor
			v_i = self._comiple_message_node_(_factor.o, _factor)
			# return v_i+ " \dot M_"+_factor.label
			return v_i.dot(_factor.label)