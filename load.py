import pandas
import torch
import openpyxl
from gensim.models import fasttext
import numpy
import io
from gensim.models import KeyedVectors

#retrieve our examples
df = pandas.read_excel("Lexique-query-2023-04-11 13-58-53.xlsx")
def find_class(word):
    return int("euse" in word)
def get_feminine_word(neutral_lemma, examples):
    feminine = ""
    for lemma, word in examples:
        if lemma==neutral_lemma:
            feminine = word
    return feminine
examples = list(zip(list(df["lemme"]), list(df["Word"])))



#retrieve fasttext vectors
# run only once !
# model_fasttext = fasttext.load_facebook_model("C:\\Users\\garri\\Desktop\\Cours\\morphology\\cc.fr.300.bin\\cc.fr.300.bin")
# model_kv = model_fasttext.wv
# new_vectors = model_kv.vectors_for_all({lemme:1 for lemme, word in examples if lemme in model_kv.key_to_index})
# new_vectors.save('eur_vectors.kv')

#run everytime !
#load fasttext vectors
loaded_vectors = KeyedVectors.load('eur_vectors.kv')
fasttext_examples = [(loaded_vectors.get_vector(lemma), find_class(get_feminine_word(lemma, examples))) for lemma in loaded_vectors.key_to_index.keys()]

def vec(data, word_to_idx, word):
    return numpy.array(data[word_to_idx[word]], dtype=numpy.float32)

#load FRCOWS(2.5m) vectors
own_model_thin = torch.load("./model106.pth", map_location=torch.device('cpu'))
idx_to_word_thin = own_model_thin["idx_to_word"]
word_to_idx_thin = own_model_thin["word_to_idx"]
data_thin = own_model_thin["cbow_state_dict"]
data_thin = data_thin["embeddings.weight"].data
own_examples_thin = [(vec(data_thin, word_to_idx_thin, lemme), find_class(word)) for lemme, word in examples if lemme in idx_to_word_thin]

#load FRCOWS(500m) vectors
own_model_thick = torch.load("./model4.pth", map_location=torch.device('cpu'))
idx_to_word_thick = own_model_thick["idx_to_word"]
data_thick = own_model_thick["cbow_state_dict"]
word_to_idx_thick = own_model_thick["word_to_idx"]
data_thick = data_thick["embeddings.weight"].data
own_examples_thick = [(vec(data_thick, word_to_idx_thick, lemme), find_class(word)) for lemme, word in examples if lemme in idx_to_word_thick]



#load FRCOWS(8.8bn) vectors
def load_frcows(path):
    embeddings={}
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for e in lines:
            token = e.split(" ")[0]
            # token = token[0:token.rfind("_")] #comment line after saving the file
            embedding = numpy.array(e.split(" ")[1:], dtype=numpy.float32)
            embeddings[token] = embedding
    return embeddings

#clean embeddings, and only keep the items you're interested in before saving them :)
def save_frcows(path, embeddings):
    with open(path, 'w', encoding="utf-8") as file:
        full_text = ""
        for lemma, embedding in embeddings.items():
            full_text += lemma + " " + " ".join(embedding) + "\n"
        file.write(full_text)



embeds = load_frcows("lemma-A-pos-small.txt")
frcows_examples = [(embeds[lemma], find_class(word)) for lemma, word in examples if lemma in embeds.keys()]

#only on first iteration, selects which embeddings to keep
# to_save = {lemma: embeds[lemma] for lemma, word in examples if lemma in embeds.keys()}
# save_frcows("lemma-A-pos-small.txt", to_save)

print("frcow8_8bn:")
print("\ttype:", type(frcows_examples))
print("\t\ttype elt [0]:", type(frcows_examples[0]))
print("\t\t\ttype elt [0][0]", type(frcows_examples[0][0]))
print("\t\t\ttype elt [0][1]", type(frcows_examples[0][1]))
print("\tlen:",len(frcows_examples))
print("\tfirst sample:", frcows_examples[0])

print("frcow2_5m:")
print("\ttype:", type(own_examples_thin))
print("\t\ttype elt [0]:", type(own_examples_thin[0]))
print("\t\t\ttype elt [0][0]", type(own_examples_thin[0][0]))
print("\t\t\ttype elt [0][1]", type(own_examples_thin[0][1]))
print("\tlen:",len(own_examples_thin))
print("\tfirst sample:", own_examples_thin[0])

print("frcow500m:")
print("\ttype:", type(own_examples_thick))
print("\t\ttype elt [0]:", type(own_examples_thick[0]))
print("\t\t\ttype elt [0][0]", type(own_examples_thick[0][0]))
print("\t\t\ttype elt [0][1]", type(own_examples_thick[0][1]))
print("\tlen:",len(own_examples_thick))
print("\tfirst sample:", own_examples_thick[0])

print("fasttext:")
print("\ttype:", type(fasttext_examples))
print("\t\ttype elt [0]:", type(fasttext_examples[0]))
print("\t\t\ttype elt [0][0]", type(fasttext_examples[0][0]))
print("\t\t\ttype elt [0][1]", type(fasttext_examples[0][1]))
print("\tlen:",len(fasttext_examples))
print("\tfirst sample:", fasttext_examples[0])


