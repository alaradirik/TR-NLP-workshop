import random
import json
import spacy


DATA_PATH = './data/data_astronomy.json'

def load_training_data(filepath):
    # Load json file
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    
    # Convert data to SpaCy format
    TRAIN_DATA = []
    for k in data['astronomy']:
    	ents = data['astronomy'][k]['entities']
    	for i in range(len(ents)):
    		TRAIN_DATA.append((k, {'entities': [(
                ents[i][0], int(ents[i][1])+1, ents[i][2])]}))

    return TRAIN_DATA


TRAIN_DATA = load_training_data(DATA_PATH)

def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('tr')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Iteration: {}".format(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


prdnlp = train_spacy(TRAIN_DATA[:200], 20)

# Save the trained model
modelfile = input("Enter your Model Name: ")
prdnlp.to_disk(modelfile)

#Test your text
test_text = input("Enter your testing text: ")
doc = prdnlp(test_text)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)