'''
	This script houses the class which will be used to make nodes for the factor graph.
	A node can be of either variable or factor type, determined by an argument.

'''

class Node:
	'''
		This class will be a node, and will be referenced in the adjacency matrix of the factor graph.
	'''

	def __init__(self, _label):
		self.label = _label
		
	def __str__(self):
		return self.label

class Variable(Node):
	'''
		This class represents the nodes of the query, which will be mapped to the edges of the factor graph matrix.

	'''

	def __init__(self, _label, _is_head = False, _is_null = False):
		self.head =  _is_head
		self.null = _is_null
		self.label = _label# if not self.head else _label+"*"

	def set_head(self):
		self.head = True
		# self.label += "*"
		return self

	def set_null(self):
		self.null = True
		return self

	def __str__(self):
		return self.label if not self.head else self.label+"*"

class Factor(Node):
	'''
		This class represents the factors in the query.
		They hold the matrix representations of the predicate they represent.
		They also keep track of the node they connect.

		Args:
			_label: (type String) the label of the node
			_i, _o: (type Variable) the variabels of the graph connected by this factor
	'''

	def __init__(self, _label, _i, _o):
		self.label = _label
		self.i = _i
		self.o = _o

	def __str__(self):
		return "%(predicate)s(%(i)s, %(o)s)" % { 'predicate':self.label, 'i':self.i, 'o':self.o }
