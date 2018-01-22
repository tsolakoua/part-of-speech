import spacy
import random
import pathlib

# part of speech tag names for my model
pos_tags = {
	'N': {'pos': 'NOUN'},
	'V': {'pos': 'VERB'},
	'A': {'pos': 'ADJ'}
}

# training data
train_data = [ ("I have a beautiful dog", {'tags': ['N', 'V', 'N', 'A', 'N']}),
("The cute puppy is Oscar", {'tags': ['N', 'A', 'N', 'V', 'N']}),
("Oscar is funny", {'tags': ['N', 'V', 'A']}) ] 

nlp = spacy.blank('en')
pos_tagger = nlp.create_pipe('tagger')
for tag, values in pos_tags.items():
	pos_tagger.add_label(tag, values)
nlp.add_pipe(pos_tagger)

# training the model with our train_data
training = nlp.begin_training()
iterations = 20
for i in range(iterations):
	random.shuffle(train_data)
	losses = {}
	annotations = train_data
	for item, annotations in train_data:
		nlp.update([item], [annotations], sgd = training, losses = losses)

# giving user input to test the model
user_input = input("Give a sentence: ")
doc = nlp(user_input)
for item in doc:
	print (item, item.pos_)
