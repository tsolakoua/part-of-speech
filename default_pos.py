import spacy

nlp = spacy.load('en')
user_input = input("Give a sentence: ")
doc = nlp(user_input)
for item in doc:
	print (item, item.pos_)
