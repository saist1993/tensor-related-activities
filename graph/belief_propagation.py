'''
	Authored: 8th June (first version)	
'''
from termcolor import colored, cprint
from pprint import pprint
import sys

from theano import sparse
import theano.tensor as T
import theano

import numpy as np
import scipy as sp
import scipy.sparse as sps


#internal libraries
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
		# print "Creating graph for: ", _rule
		null_node = node.Node("phi")
		
		#Keep the list of variables in order to map the indices of the matrix to actual graph nodes
		self.variables = _variables + [null_node]
		self.factors = _factors

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
		# self.print_graph()

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


	def inspect_inputs(i, node, fn):
	    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs]

	def inspect_outputs(i, node, fn):
	    print " output(s) value(s):", [output[0] for output in fn.outputs]

	def propagate_thy_beliefs(self):
		'''
			Call this function to receive a string containing the path of the belief propagation algorithm.
			We implement the algorithm listed in the paper mentioned in the comments above
			

			Pseudocode:
				-> Create an empty theano vector whose definitions will be iteratively changed.
				-> Call compile_message_node_to_factor from the o node of the head predicate. 
				-> Let the functions recursively call each other
				-> Collect their things somehow. @TODO: how. what format. Shall we use theano variables altogether or what
				-> Return said stuff.
		'''	

		# print "graph:bp: Starting belief propagation."
		equation = self._comiple_message_node_(self.head_predicate.o, "Fictional Label")
		symbols = self._comiple_message_symbols_node_(self.head_predicate.o, "Fictional Label")
		

		#Define an empty dvector to be used as the 'y' label (which will later contain n hot information about desired entities)
		y = sparse.csr_dmatrix('y')

		# Do a softmax over the final BP Equation
		equation = sparse.structured_exp(equation)
		equation = sparse.row_scale(equation, 1.0/sparse.sp_sum(equation, axis = 1))

		# Collect all the parameters (shared vars), found in the factors of this graph.
		#parameters is a list of matrices (relation)		
		parameters = [x.M for x in symbols]
		
		#Cross entropy loss
		# loss = - y * T.log(equation) + (y - 1)*T.log(1-equation) # unregularized cross-entropy loss in theano
		a = sparse.mul(y, sparse.structured_log(equation)) 
		b = sparse.mul(sparse.structured_add(y,-1.0), sparse.structured_log(sparse.structured_add(equation,-1.0)))
		loss = sparse.sub(b, a)

		# Unregularied Loss
		loss_dense = sparse.dense_from_sparse(loss)
		cost = loss_dense.mean()
		# cost = sparse.sp_sum(loss, axis = 1)/float(ne)

		gradients  = theano.grad(cost, parameters)
		
		updated_matrices = [sparse.sub(parameters[i], 0.1*gradients[i] ) for i in range(len(parameters))]
		# updated_matrices = [sparse.sub(parameters[i], sparse.row_scale(gradients[i], 0.1)) for i in range(len(parameters))]
		# updated_matrices = [parameters[i] - 0.1 * gradients[i] for i in range(len(parameters))]

		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#  DEBUG
		# print "Equation: ", equation
		# print "Type of equation: ",type(equation)
		# print "Symbols: ", symbols
		# print "graph:bp: Belief propagation complete."

		# print "Parameters are"
		# for p in parameters:
		# 	print p," and the type is :",type(p)

		# print gradients
		# print "Updated Matrices are :", type(updated_matrices[0])
		
		# print colored(type(self.head_predicate.i.u),'red')

		# print "Inputs: \n"
		# print type(self.head_predicate.i.u)
		# print type(y)
		# print [ type(x) for x in parameters ]

		# raw_input("Verify Symbols and Gradients ")		
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		function = theano.function(
          inputs = [self.head_predicate.i.u, y]+ parameters,		#Inputs to this is the head predicates' symbolic var, and another dvector
          # inputs = [self.head_predicate.i.u,parameters[0]],		#Inputs to this is the head predicates' symbolic var, and another dvector
          # outputs = updated_matrices			#Output to this thing is the BP algorithm's output expression
          outputs = [ equation ] + updated_matrices 
          # mode=theano.compile.MonitorMode(
          #               pre_func=self.inspect_inputs,
          #               post_func=self.inspect_outputs)			#Output to this thing is the BP algorithm's output expression
          # updates=tuple([(parameters[i], parameters[i] - 0.1 * gradients[i]) for i in range(len(parameters))])		#Updates are the gradients of cost wrt parameters
         )

		return function, symbols

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
			if _node.u is None:
				print _node
				# raw_input("Node has nothing")
			return _node.u

		#This is NOT the input variable.
		neighbors = self._get_neighbours(_node, _exclude=_factor) #Will be a list of factors.
		
		#Send the neighbour + current node to compilemessage_factor and collect what they have to say.
		neighboring_values = [ self._compile_message_factor_(_factor = factor, _node = _node) for factor in neighbors ]
		if len(neighboring_values) > 0:
			v_x = neighboring_values[0]
			for remaining_values in neighboring_values[1:]:
				v_x = sparse.mul(v_x, remaining_values)
				# v_x = v_x * remaining_values
		else:
			#In this case, since there are no neighbors, there's literally nothing to return.
			#@TODO: What do we do here
			# print "belief_propagation:Graph:compile_message: Part where there are no neighbours!"
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

			return sparse.structured_dot(v_i, _factor.M)
			# return v_i.dot(_factor.M)

		elif _factor.i == _node:

			#If the node is the input node for this factor
			v_i = self._comiple_message_node_(_factor.o, _factor)
			return sparse.structured_dot(v_i, _factor.M)
			# return v_i.dot(_factor.M)

	def _comiple_message_symbols_node_(self, _node, _factor):
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
			return []

		#This is NOT the input variable.
		neighbors = self._get_neighbours(_node, _exclude=_factor) #Will be a list of factors.
		
		#Send the neighbour + current node to compilemessage_factor and collect what they have to say.
		# v_x = ' \circle '.join([ self._compile_message_factor_(_factor = factor, _node = _node) for factor in neighbors ])
		neighboring_values = []
		for factor in neighbors:
			neighboring_values += self._compile_message_symbols_factor_(_factor = factor, _node = _node)
		# neighboring_values = [ self._compile_message_symbols_factor_(_factor = factor, _node = _node) for factor in neighbors ]

		# if len(neighboring_values) > 0:
		# 	v_x = neighboring_values[0]
		# 	for remaining_values in neighboring_values[1:]:
		# 		v_x = v_x * remaining_values
		# else:
		# 	#In this case, since there are no neighbors, there's literally nothing to return.
		# 	#@TODO: What do we do here
		# 	print "belief_propagation:Graph:compile_message: Part where there are no neighbours!"
		# 	pass

		return neighboring_values

	def _compile_message_symbols_factor_(self, _factor, _node):
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
		
		# if _factor.o.null:
		# 	#If the factor is unary.
		# 	return _factor.M 	#@TODO: See if we need another variable like self.v to represent the value of unary predicates.

		if _factor.o == _node:
			#If the node is the output node for this factor
			v_i = self._comiple_message_symbols_node_(_factor.i, _factor)
			# return v_i+ " \dot M_"+_factor.label
			return v_i + [ _factor ]

		elif _factor.i == _node:
			 #If the node is the input node for this factor
			v_i = self._comiple_message_symbols_node_(_factor.o, _factor)
			# return v_i+ " \dot M_"+_factor.label
			return v_i +  [ _factor ]

if __name__ == "__main__":

	'''
		Testbench for the above code.
		Create a small KB rule and run BP on it.
	'''	
	ENT = 5		#Total entities in the KB
	x = node.Variable('x')
	x.u = sparse.csr_matrix('x')
	y = node.Variable('y')
	y.u = sparse.csr_matrix('y')
	p = node.Factor('p', x, y)
	p.M = sparse.csr_matrix('p')
	q = node.Factor('q', y, x)
	q.M = sparse.csr_matrix('p')

	# uc = np.eye(ENT)[2]

	g = Graph(_variables = [x,y], _factors = [p], _fictional_factor=q)
	f,s = g.propagate_thy_beliefs()

	x = np.random.rand(1,10)
	y = np.random.rand(1,10)
	m = np.asarray([ np.eye(10,10)[i] for i in np.random.randint(0,10,10)])
	x = sps.csr_matrix(x, dtype = np.float64 )
	y = sps.csr_matrix(y, dtype = np.float64 )
	m = sps.csr_matrix(m, dtype = np.float64 )

	op = f(x,y,m)


	sm = sparse.csr_matrix()
	eval = theano.function([sm],sparse.dense_from_sparse(sm))

	for s in op:
		print eval(s)