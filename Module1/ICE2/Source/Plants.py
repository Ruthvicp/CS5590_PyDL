"""
@author : ruthvicp
date : 2/1/2019
"""
def avg_ht(no_of_plants, ht_of_plants):
    sum = 0
    for x in ht_of_plants:
        sum = sum + x
    avg = sum/no_of_plants
    print("Output : ", avg)

print("Input : ")
no_of_plants = int(input("Enter no. of plants : "))
ht_of_plants = [float(x) for x in input().split()]
print(ht_of_plants)
avg_ht(no_of_plants,ht_of_plants)
