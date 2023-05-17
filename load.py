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

#load FRCOWS(2.5m) vectors
def vec( word):
    return numpy.array(data[word_to_idx[word]], dtype=numpy.float32)
own_model = torch.load("./model499.pth", map_location=torch.device('cpu'))
idx_to_word = own_model["idx_to_word"]
word_to_idx = own_model["word_to_idx"]
data = own_model["cbow_state_dict"]
data = data["embeddings.weight"].data
own_examples = [(vec(lemme), find_class(word)) for lemme, word in examples if lemme in idx_to_word]



#load FRCOWS(8.8bn) vectors
def load_frcows(path):
    embeddings={}
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for e in lines:
            token = e.split(" ")[0]
            token = token[0:token.rfind("_")] #comment line after saving the file
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
print("\ttype:", type(own_examples))
print("\t\ttype elt [0]:", type(own_examples[0]))
print("\t\t\ttype elt [0][0]", type(own_examples[0][0]))
print("\t\t\ttype elt [0][1]", type(own_examples[0][1]))
print("\tlen:",len(own_examples))
print("\tfirst sample:", own_examples[0])

print("fasttext:")
print("\ttype:", type(fasttext_examples))
print("\t\ttype elt [0]:", type(fasttext_examples[0]))
print("\t\t\ttype elt [0][0]", type(fasttext_examples[0][0]))
print("\t\t\ttype elt [0][1]", type(fasttext_examples[0][1]))
print("\tlen:",len(fasttext_examples))
print("\tfirst sample:", fasttext_examples[0])


