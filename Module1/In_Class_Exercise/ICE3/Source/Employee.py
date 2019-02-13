"""
@author : ruthvicp
date : 2/7/2019
"""

#Base Class
class Employee(object):
    empCount = 0
    totalSalary = 0

    def __init__(self, name, family, salary, department):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
        Employee.empCount += 1
        Employee.totalSalary += salary

    def getEmployeeCount(self):
        return Employee.empCount

    def getAvgSalary(self):
        return Employee.totalSalary/Employee.empCount

    def printEmployeeDetails(self):
        print("Name:", self.name, "Family:", self.family, "Salary:", self.salary, "Department:", self.department)

# Inherited class
class FullTimeEmployee(Employee):
    def _init_(self,name,family,salary,department):
        Employee._init_(self,name,family,salary,department)

choice = 'y'
# Take inputs in a while loop
while choice == 'y' or choice == 'Y':
    name = input("Enter Employee Name : ")
    family = input("Enter Employee family : ")
    salary = float(input("Enter Employee salary : "))
    department = input("Enter Employee department : ")
    e_or_fe = int(input("Enter 1 for employee or Enter 2 for a full time employee : "))
    # determining the onject at run time here using the same reference
    if e_or_fe == 1:
        e = Employee(name, family, salary, department)
    elif e_or_fe == 2:
        e = FullTimeEmployee(name, family, salary, department)

    e.printEmployeeDetails()
    # loop terminating condition
    choice = input("Enter y/y to continue adding Employee, Enter n/N to get results : ")

# Results
print("total employee count = ", Employee.empCount)
print("Avg Salary of Employee = ", e.getAvgSalary())
