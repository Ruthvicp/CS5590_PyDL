'''
author : ruthvic
created : 1/25/2019 9:10PM
'''

# Utility method to perform all operations
def arith_opr(num1, num2):
    print("Basic Arithmetic operations :")
    print('num1 + num2 = ', num1+num2)
    print('num1 * num2 = ', num1*num2)
    print('num1 - num2 = ', num1-num2)
    print('num1 / num2 = ', num1/num2)
    print("Specific Arithmetic operations :")
    print('Quotient of (num1/num2) = ', num1//num2)
    print('Remainder of (num1/num2) = ', num1%num2)
    print('num1.pow(num2) = ', num1**num2)

# Reading inputs from console
num1 = float(input("Enter first number : "))
num2 = float(input("Enter Second number : "))
arith_opr(num1, num2)

