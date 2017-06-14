from theano import shared
import theano.tensor as T
import numpy as np

def parse_example(fname):
	with open(fname) as f:
	    content = f.readlines()
	content = [x.strip() for x in content]
	data = []
	for line in content:
		# print line
		temp = line.split('\t')
		#-interp
		parse = []
		for node in temp:
			if "-interp" in node:
				# print "here"
				continue
			else:
				parse.append(node)		
		s = []
		for node in parse:
			s.append(node[node.find("(")+1:node.find(")")].split(","))
		for i in xrange(len(s)):
			s[i][0] = s[i][0][2:]	
		data.append(s)
	return data

def encode(vars,factors,fictional_factor,number_of_entites):
	for var in vars:
		var.u = T.dvector(label_encoder.transform(var.label))
	for rel in factors:
		rel.M = T.shared(np.random.randn(number_of_entites,number_of_entites))
	return vars,factors,fictional_factor