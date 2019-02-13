class Flight(object):
    flight_count = 0
    def __init__(self, Flight_Number, From, To, Date):
        self.Flight_Number = Flight_Number
        self.From_Loc = From
        self.To_Loc = To
        self.Date = Date
        Flight.flight_count += 1

    def getFlightDetails(self):
        #print("Details of flight are: ", self.Flight_Number, self.From_Loc, self.To_Loc, self.Date)
        return self.Flight_Number, self.From_Loc, self.To_Loc, self.Date
    def getFlightCount(self):
        print("Total number of flights are: " , self.flight_count)


class Person(object):
    person_count = 0
    def __init__(self, Name, Age, Sex):
        self.Name = Name
        self.Age = Age
        self.Sex = Sex
        Person.person_count += 1


    def printPerseonDetails(self):
        return str((self.Name, self.Sex,self.Age))

    def getPersonCount(self):
        print("Total number of persons are: ", self.person_count)

class Employee(Person):
    employee_count = 0
    def __init__(self, Name, Age, Sex, Emp_ID):
        super().__init__(Name, Age, Sex)
        #super.__init__(self,Name, Age, Sex)
        self.Emp_ID = Emp_ID
        Employee.employee_count += 1

    def printEmployeeDetails(self):
        #Person.Print_Person_Details(self)
        print("Employee Details are",self.printPerseonDetails(),self.Emp_ID)

    def getEmployeeCount(self):
        print("Total Number of employees are: ", self.employee_count)

class Passenger(Person):
    flight_details = None
    passenger_count = 0
    def __init__(self, Name, Age, Sex, ID_No, flight):
        Person.__init__(self,Name, Age,Sex)
        self.ID_No = ID_No
        self.flight_details = flight.getFlightDetails()
        Passenger.passenger_count += 1

    def printPassengerDetails(self):
        #Person.Print_Person_Details(self)
        print("Passenger Details are",self.printPerseonDetails(),self.ID_No , "and Flight details are", self.flight_details )

    def getPassengerCount(self):
        print("Total Number of passengers are: ", self.passenger_count)

class Pilot(Person, Flight):
    pilot_count = 0
    assigned_flight = None
    def __init__(self, Name, Age, Sex, Pilot_ID, flight):
        Person.__init__(self,Name, Age, Sex)
        self.assigned_flight = flight.getFlightDetails()
        self.Pilot_ID = Pilot_ID
        Pilot.pilot_count += 1

    def pilotDetails(self):
        print("Pilot Details are",Person.printPerseonDetails(self),self.Pilot_ID,"and assigned flight detils are",self.assigned_flight)

    def getPilotCount(self):
        print("Totlal number of pilots are: ", self.pilot_count)

if __name__ == '__main__':
    person1 = Person('Charan', 23, 'Male')
    flight1 = Flight(9893, 'Kansas-City', 'Chicago-IL','Feb-14-19')
    flight2 = Flight(1235, 'San-Diego', 'Dallas','Feb-15-19')
    passenger1 = Passenger('Kottapalli', 18, 'Male', 'Q789', flight1)
    passenger2 = Passenger('Sita', 19, 'Female', 'A8789', flight2)
    passenger3 = Passenger('Ram', 40, 'Male', 'U4232', flight1)
    Employee1 = Employee('Shankar', 36, 'Male', 'E898202')
    Employee2 = Employee('Rohit', 73, 'Female', 'E987442')
    pilot1 = Pilot('John', 35, 'Male', 'P842748', flight1)

    Employee1.printEmployeeDetails()
    Employee2.printEmployeeDetails()
    passenger1.printPassengerDetails()
    passenger2.printPassengerDetails()
    passenger3.printPassengerDetails()
    pilot1.pilotDetails()
    pilot1.getPilotCount()
    person1.getPersonCount()
    flight1.getFlightCount()
    passenger1.getPersonCount()
    Employee1.getEmployeeCount()