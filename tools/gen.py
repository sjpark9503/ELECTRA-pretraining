from pattern3.en import*
import nltk
import argparse
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
import random

"""
variables for variation
"""
ONLY_SAME_WORD = False
ONLY_SAME_POS = True


# convert to wordnet pos
def tag_map(pos):
	_pos = pos[0]
	if (_pos == "J"): return wn.ADJ
	elif (_pos == "V"): return wn.VERB
	elif (_pos == "R"): return wn.ADV
	else: return wn.NOUN

"""
return tokenList, tagDict, lemmaDict 
tokenList = list of tokens
tagDict = dict with {key: token, val: tag}
lemmaDict = dict with {key: token, val: lemma}
"""
def preprocess(tokens, debug=False):
	tokenList = []
	tagDict = dict()
	lemmaDict = dict()

	_lemma = WordNetLemmatizer()
	for token, tag in pos_tag(tokens):
		if (debug):
			print("token: {}, tag: {}".format(token, tag))
		tokenList.append(token)
		tagDict[token] = tag

		lem = _lemma.lemmatize(token)
		lemmaDict[token] = lem 
		if (debug):
			print(token, "=>", lem)
	return tokenList, tagDict, lemmaDict 

# input synset / return name and pos
def get_name_pos_from_syn(syn):
	name = syn.name().split('.')[0]
	pos = syn.name().split('.')[1]
	return name, pos

"""
synDict = dict with {key: token, value: synset}
"""
def make_synDict(tokenList, tagDict, lemmaDict, debug=False, thres=-1):
	synDict = dict()

	# make synset using lemma
	for token in tokenList:
		lem = lemmaDict[token]
		if(len(wn.synsets(lem)) == 0):
			continue
		else:
			synList = wn.synsets(lem)		# list of synset

			"""
			only choose synset with SAME word with lem
			"""
			if ONLY_SAME_WORD:
				tempList = []
				for elem in synList:
					name, pos = get_name_pos_from_syn(elem)
					if (name == lem):
						tempList.append(elem)
			else:
				tempList = synList.copy()

			"""
			only choose synset with SAME pos_tag 
			"""
			appendList = []
			ori_tag = tag_map(tagDict[token])
			for elem in tempList:
				_name, _pos = get_name_pos_from_syn(elem)
				if (ori_tag != _pos):
					if ONLY_SAME_POS:
						# remove the ones that do not match
						continue
					else:
						appendList.append(elem)
				else:
					appendList.append(elem)

			if (len(appendList) == 0):
				# no matching element with same pos
				continue
			elif (thres > 0 and len(appendList) > thres):
				continue
			else:
				synDict[token] = appendList

	return synDict

def get_tense(syn, ori_pos):
	
	"""
	VB:	Verb, base form
	VBD:	Verb, past tense
	VBG:	Verb, gerund or present participle
	VBN:	Verb, past participle
	VBP:	Verb, non-3rd person singular present
	VBZ:	Verb, 3rd person singular present
	"""

	name, pos = get_name_pos_from_syn(syn)
	
	# default values
	_tense = "present"			# infintive, present, past, future
	_person = 1						# 1, 2, 3, or None
	_number = "singular"			# SG, PL
	_mood = "indicative"			# indicative, imperative, conditional, subjunctive
	_aspect = "imperfective"	# imperfective, perfective, progressive

	if (ori_pos == "VBD"):
		_tense = "past"
	elif (ori_pos == "VBG"):
		_aspect = "progressive"
	elif (ori_pos == "VBN"):
		_tense = "past"
		_aspect = "progressive"
	elif (ori_pos == "VBZ"):
		_person = 3

	return conjugate (name,
							tense = _tense, 
							person = _person,
							number = _number,
							mood = _mood, 
							aspect = _aspect,
							negated = False)

"""
hypernymDict = dict of {key: token, value: list of hypernyms}
return only the tokens that have hypernym
"""
def make_hypernymDict(synDict, tokenList, tagDict, lemmaDict, debug=False, thres=-1):
	hypernymDict = dict()

	for token in tokenList:
		if not (token in synDict.keys()):
			continue
		appendList = []
		ori_tag = tag_map(tagDict[token])
		for syn in synDict[token]:
			hyperList = syn.hypernyms()
			
			"""
			only choose synset with SAME pos_tag 
			"""
			hyper = []
			for elem in hyperList:
				_name, _pos = get_name_pos_from_syn(elem)
				if (ori_tag != _pos):
					if ONLY_SAME_POS:
						# remove the ones that do not match
						continue
					else:
						hyper.append(elem)
				else:
					hyper.append(elem)

			if ((thres > 0) and (len(hyper) > thres)):
				continue
			elif (len(hyper) == 0):
				continue
			else:
				for _elem in hyper:
					appendList.append(_elem)
		hypernymDict[token] = appendList

	return hypernymDict

def replace_all(sen, ori_word, ori_hyper):
	print("[Original Sentence] {}".format(sen))
	for elem in ori_hyper:
		_name, _pos = get_name_pos_from_syn(elem)
		print("[Replace Sentence] {}".format(change_word(sen, ori_word, _name)))

def leave_only_char(word):
	word = word.lower()
	charList = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
	retword = ""
	for _char in word:
		if _char in charList:
			retword += _char
	return retword

def change_word(sen, ori_word, replace_word):
	wordList = sen.split(" ")
	retVal = ""
	for i in range(len(wordList)):
		word = wordList[i]
		if leave_only_char(word) == ori_word:
			wordList[i] = replace_word
			for elem in wordList:
				retVal += (elem+" ")
			return retVal
	return sen.replace(ori_word, replace_word, 1)

"""
check the pos tag of the word 
if the tag is in Noun, ADJ, ADV, Verb - return True
else return False
"""
def is_replaceable(word, tagDict):
	nounList = ["NN", "NNS"] 
	adjList = ["JJ"] # JJR, JJS for comparative, superlative
	advList = ["RB"] # RBR, RBS for comparative, superlative
	verbList = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
	if tagDict[word] in nounList+adjList+advList+verbList:
		print(f"{word} have tag {tagDict[word]}")
		return True
	return False

def replace_one(sen, ori_word, ori_hyper):
	if len(ori_hyper) == 0:
		return sen
	idx = random.randint(0, len(ori_hyper)-1)
	elem = ori_hyper[idx]
	_name, _pos = get_name_pos_from_syn(elem)
	return change_word(sen, ori_word, _name)

def replace_sentence (sen, num, total_tokens, _random):

	tokens = word_tokenize(sen)
	tokenList, tagDict, lemmaDict = preprocess(tokens, debug=False)

	if _random:
		if (len(total_tokens) == 0): 
			return sen
		if (len(total_tokens) <= num):
			num = len(total_tokens) - 1
		_senList = sen.split(" ")
		print(f"[Before replaceable]: {_senList}, num: {num}")

		senList = []
		for word in _senList:
			if is_replaceable(word, tagDict):
				senList.append(word)
		if (len(senList) == 0):
			return sen
		if (len(senList) <= num):
			num = max(len(senList) - 1, 1)
		print(f"[After replacement]: {senList}, num: {num}")

		randList = []
		oriList = []
		while len(oriList) != num:
			randNum = random.randint(0, len(senList)-1)
			if randNum not in oriList:
				oriList.append(randNum)
				randList.append(random.randint(0, len(total_tokens) - 1))

		for i in range(len(oriList)):
			ori_word = senList[oriList[i]]
			ori_random = total_tokens[randList[i]].split("\n")[0]
			sen = change_word(sen, ori_word, ori_random)

	else:
		synDict = make_synDict(tokenList, tagDict, lemmaDict, debug=False)
		hypernymDict = make_hypernymDict(synDict, tokenList, tagDict, lemmaDict, debug=False)
		_key = list(hypernymDict.keys())
		print(f"keys: {_key}")	

		if (len(_key) == 0):
			return sen

		if (len(_key) <= num):
			num = len(_key)-1 
		
		_sampleNum = []
		wordNum = []
		while len(_sampleNum) != num:
			randNum = random.randint(0, len(_key)-1)
			if randNum not in _sampleNum:
				_sampleNum.append(randNum)
				wordNum.append(_key[randNum])	
		print(f"[Before replaceable]: {wordNum}, num: {num}")

		sampleNum = []
		for idx in _sampleNum:
			if is_replaceable(_key[idx], tagDict):
				sampleNum.append(idx)
		if (len(sampleNum) == 0):
			return sen
		if (len(sampleNum) <= num):
			num = max(len(sampleNum)-1, 1)
		print(f"[After replaceable]: {sampleNum}, num: {num}")

#	randNum = random.randint(1, len(_key)) # choose the word to change randomly
		for randNum in sampleNum:
			ori_word = _key[randNum]
			ori_hyper = hypernymDict[ori_word]
#		replace_all(sen, ori_word, ori_hyper)
			sen = replace_one(sen, ori_word, ori_hyper)
	return sen

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--sen", required=True,default="I am planning to do this today")
	parser.add_argument("--random", action="store_true")
	args = parser.parse_args()
	print(f"random: {args.random}")
	print(replace_sentence(args.sen, 3, [], args.random))
