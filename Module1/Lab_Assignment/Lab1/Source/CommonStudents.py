def findStudents(Python, BigData):
    intersectionList = []
    unionList = []
    for Student in Python:
        if Student in BigData:
            intersectionList.append(Student)
            BigData.remove(Student)
        else:
            unionList.append(Student)

    print("Common list of students in both the subjects are: ",intersectionList)
    print("Un Common list of students in both the subjects are: ", unionList + BigData)

if __name__ == '__main__':
    Python = ['Charan','Ram','Kottapalli','Sri']
    WebApplications = ['Charan', 'Kottapalli','Shankar']
    findStudents(Python, WebApplications)