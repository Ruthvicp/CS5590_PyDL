'''
author : ruthvic
created : 1/25/2019 9:00PM
'''
def string_reverse(tmp_str):
    rev_str = ''
    index = len(tmp_str) # get length of string
    while index > 0:
        # we do a rev traversal from last index
        rev_str += tmp_str[ index - 1 ]
        index = index - 1
    return rev_str

# standard input to take first & last name
fName = input('Please enter first name: ')
lName = input('Please enter last name: ')
# print the reversed string
print(string_reverse(fName+" "+lName))