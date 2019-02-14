"""
@author : ruthvicp
date : 2/13/2019
"""
# function to convert tuples to dictionary
def tup_to_dict(tup, dict):
    for a, b in tup:
        dict.setdefault(a, []).append(b)
    return dict

# function to sort dictionary whose values are list of tuples
def sort_dict(dict):
    for idx,list_of_tups in dict.items():
        # key - idx, value = list_of_tups
        dict[idx] = sorted(list_of_tups,key=lambda x: x[1]) # sorts on 1st value of list
    return dict

tup = [( 'John', ('Physics', 80)) , ('Daniel', ('Science', 90)), ('John', ('Science', 95)),
       ('Mark',('Maths', 100)), ('Daniel', ('History', 75)), ('Mark', ('Social', 95))]

# empty dictionary
dict = {}
dict = tup_to_dict(tup,dict)
print("Output before sorting is : ")
print(dict)
print("Output after sorting is : ")
print(sort_dict(dict))