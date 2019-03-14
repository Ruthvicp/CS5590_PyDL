
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import bigrams

def Tokenzier():
    f = open('nlp_input.txt')
    data = f.read()
    print('Text from the input file: \n\n,{data}')

    data_word = word_tokenize(data)

    data_sent = sent_tokenize(data)
    lemmatizer = WordNetLemmatizer()
    data_lemmatized = []
    for word in data_word:
        fr_lema = lemmatizer.lemmatize(word.lower())
        data_lemmatized.append(fr_lema)

    print('**Lemmatized Data:** \n\n, {data_lemmatized}', "\n")
    data = []
    for grams in bigrams(data_lemmatized):
        data.append(grams)
    print('**Bigram Data:** \n\n,{bigram_data}', "\n")
    fdist1 = nltk.FreqDist(data)
    bigram_freq = fdist1.most_common()
    print('**Bigrams with Frequency:** \n,{bigram_freq}',"\n")
    top_ten = fdist1.most_common(10)
    print('**Top five Bigrams with frequency:** \n,{top_ten}',"\n")
    rep_sent1 = []
    for sent in data_sent:
        for word, words in data:
            for ((s, t), l) in top_five:
                if (word, words == s, t):
                    rep_sent1.append(sent)
    print("**Summarized Data** \n")
    print(max(rep_sent1, key=len))


if __name__ == '__main__':
    Tokenzier()