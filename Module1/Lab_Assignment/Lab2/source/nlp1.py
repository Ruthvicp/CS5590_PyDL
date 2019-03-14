import nltk
from nltk.stem import WordNetLemmatizer

#Opening a file in read mode
f = open("nlp_input.txt","r")

#opening a file in write mode to store the tokenized words in it
with open("tokenize.txt","w") as t:
#reading each sentence
    for sentence in f:
#Performing word tokenization
        wtokens = nltk.word_tokenize(sentence)
#Writing it to a file
        for w in wtokens:
            t.write(str("\n"))
            t.write(str(w))

# creating a lemmatization object
lemmatizer = WordNetLemmatizer()
#Opening a new file in write mode
with open("lemmatize.txt", "w") as l:
#Opening the file consisting of words tokenized
    w = open("tokenize.txt","r")
    for words in w:
#Performing lematization on each word
        le = lemmatizer.lemmatize(words)
        l.write(str(le))

a = open("nlp_input.txt","r")
with open("trigram.txt","w") as tri:
    for sentence in a:
    #performing trigram on each sentence
            trigram = nltk.trigrams(sentence.split())

        #trigram are written on to a file
            for ti in trigram:
                tri.write(str("\n"))
                tri.write(str(ti))

f1 = open("nlp_input.txt", "r")
fileread = f1.read()

tg = []
word_tokens = nltk.word_tokenize(fileread)
for t in nltk.ngrams(word_tokens, 3):
    tg.append(t)

wordFreq = nltk.FreqDist(tg)
mostCommon = wordFreq.most_common()

#     d. Extract the top 10 of the most repeated trigrams based on their count.
Top_ten_trigrams = wordFreq.most_common(10)
print(Top_ten_trigrams)


#     e. Go through the text in the file
#     f. Find all the sentences with the most repeated tri-grams
#     g. Extract those sentences and concatenate
#     h. Print the concatenated result

sent_tokens = nltk.sent_tokenize(fileread)
concat_result = []
# Iterating the Sentences
for s in sent_tokens:
    # Iterating all the trigrams
    for a, b, c in tg:#iterating the top 10 trigrams from all the trigrams
        for ((d, e, f), length) in wordFreq.most_common(20): # Comparing the each with the top 10 trigrams
            if(a, b, c == d, e, f):
                concat_result.append(s)

print("Concatenated Array : ",concat_result)
print("Maximum of Concatenated Array : ", max(concat_result))