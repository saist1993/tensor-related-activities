'''
	Author: Priyansh Trivedi (geraltofrivia)

	This file is used to create a train/test data in a semi supervised manner from the smokers dataset in order to train our tensorlog implementation over it.
'''

import pickle
import random

data = []
predicates = ['influences','smokes','cancer','stress']

#Generating person data

#Get entities from files.
person_entities = [ x.split('\t')[1] for x in open('datasets/smokers/raw/person.cfacts').read().split('\n') if (len(x.split('\t')) > 1) ]

#Making rules out of these entity sets
query1_template = "stress(%(entity)s,Yes) :- %(answer)s"
query1_answer_tempalte = "stress(%(entity1)s,%(entity2)s)"

for ent in person_entities:
	ans = "yes"
	ans_template = query1_answer_tempalte % {'entity1':ent, 'entity2':ans}
	data.append(query1_template % { 'entity':ent, 'answer':ans_template })


data += [ x.replace('stress','smokes') for x in data[:] ]


#Read friends cfacts
raw =  [ x.split('\t') for x in open('datasets/smokers/raw/friends.cfacts').read().split('\n') if len(x.split('\t')) > 2]
pairs = {}
for x in raw:
    if x[1].strip() in pairs.keys():
    	pairs[x[1].strip()].append(x[2].strip())
    else:
    	pairs[x[1].strip()] = [x[2].strip()]

query2_template = "influences(%(entity)s,X) :- %(answer)s"
query2_answer_tempalte = "influences(%(entity1)s,%(entity2)s)"
for key in pairs:
	answer = []
	for anss in pairs[key]:
		answer.append(query2_answer_tempalte % {'entity1':key,'entity2':anss})
	answer = ','.join(answer)

	data.append(query2_template % { 'entity': key, 'answer': answer })

f = open('data.txt','wb+')
f.write('\n'.join(data))



