'''
	This script houses the class which will be used to make nodes for the factor graph.
	A node can be of either variable or factor type, determined by an argument.

'''

class Node:
	'''
		This class will be a node, and will be referenced in the adjacency matrix of the factor graph.
	'''

	def __init__(self, _label, _type, _is_head = False):

		self.label = _label
		if _type == 'Variable':
			self.type = 1
		elif _type == 'Factor':
			self.type = 0
		self.head = _is_head


	def set_as_head(self):
		self.head = True
		return self

	def __str__(self):
		if self.head:
			return self.label+"*"
		else:
			return self.label

		
