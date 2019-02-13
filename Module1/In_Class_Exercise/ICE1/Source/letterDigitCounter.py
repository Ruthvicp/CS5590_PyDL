'''
author : ruthvic
created : 1/25/2019 9:20PM
'''

# Utility method to count letters & digits
def letterDigitCounter(sen):
    num_count = 0
    char_count = 0
    for i in sen:
        if i.isalpha():
            char_count = char_count + 1
        elif i.isdigit():
            num_count = num_count + 1
    return num_count,char_count

# taking input sentence
sen = input("Enter a sentence - combination of letters and digits : ")
x,y = letterDigitCounter(sen)
print("No. of digits = %d  and  No. of Chars = %d" %(x,y))
