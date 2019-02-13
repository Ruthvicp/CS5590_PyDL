"""
@author : ruthvicp
date : 2/11/2019
"""

net_amount = 0
while 1:
    trans_detail = input("Enter transaction: ")
    # we split on space to separate transaction type & amount
    trans_detail = trans_detail.split(" ")
    trans_type = trans_detail [0] # type
    trans_amount = int (trans_detail [1]) #  amount
    if trans_type=="Deposit" or trans_type=="deposit":
        net_amount += trans_amount
    elif trans_type=="Withdraw" or trans_type=="withdraw":
        net_amount -= trans_amount
    else:
        print("Please enter either Deposit or Withdraw amount only")
    #user choice to continue or not
    choice = input("Enter Y/y to Continue or any other character to exit: ")
    if not (choice =="Y" or choice =="y") :
        break

# print the net amount
print("Net amount: ", net_amount)