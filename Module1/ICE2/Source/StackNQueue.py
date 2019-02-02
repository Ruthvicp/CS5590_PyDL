"""
@author : ruthvicp
date : 2/1/2019
"""
from _collections import deque

def add_element(x, in_stack, in_queue):
    in_stack.append(x)
    in_queue.append(x)
    print(" Stack is : ", in_stack)
    print(" Queue is : ", in_queue)

def rem_element(in_stack, in_queue):
    in_stack.pop()
    in_queue.popleft()
    print(" Stack is : ", in_stack)
    print(" Queue is : ", in_queue)

# program to read list and add to stack/queue
print("Enter your list")
in_list = [int(x) for x in input().split()]
# stack implementation
in_stack = in_list
in_queue = deque(in_list)
y = 'y'
while y == 'y'or y == 'Y':
    choice = int(input(" 1 for adding element, 2 for removing element : ---> "))
    if choice == 1:
        num = int(input("Enter number : ---> "))
        add_element(num,in_stack,in_queue)
    elif choice == 2:
        rem_element(in_stack, in_queue)
    y = input(" Do you want to continue y/n ?  ---> ")

