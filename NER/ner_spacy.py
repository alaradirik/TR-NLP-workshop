import random
import json
import spacy


DATA_PATH = './data/data_astronomy.json'

# Veri setini yükle
with open(DATA_PATH, 'r') as fp:
    data = json.load(fp)

def convert_to_spacy_format(data):
    # Veriyi SpaCy NER formatına çevir
    DATA = []
    for k in data['astronomy']:
        ents = data['astronomy'][k]['entities']
        for i in range(len(ents)):
            DATA.append((k, {'entities': [(
                ents[i][0], int(ents[i][1])+1, ents[i][2])]}))
    
    return DATA


def train_spacy(data, iterations):
    # Boş Language objesi yarat
    nlp = spacy.blank('tr')  
    
    # NER pipeline'ı oluştur
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    
    # Varlık ismi etiketlerini ekle
    for _, annotations in data:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    
    # NER haricindeki pipeline'ları dondur
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes): 
        optimizer = nlp.begin_training()

        # Modeli eğit
        for itn in range(iterations):
            print("Iteration: {}".format(itn))
            random.shuffle(data)
            losses = {}

            for text, annotations in data:
                nlp.update(
                    [text],  
                    [annotations],  
                    drop=0.2, 
                    sgd=optimizer,
                    losses=losses)
            print(losses)
    return nlp
    

# Modeli eğit
data = convert_to_spacy_format(data)
model = train_spacy(data, 10)

# Eğitilmiş modeli kaydet
model.to_disk("astro-ner")

# Modeli yükle ve test et
nlp = spacy.load('astro-ner')
test_text = input("Enter your testing text: ")
doc = nlp(test_text)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)