import csv

# prompt the user to create new credentials or enter existing ones, return userID
def initUser():
    choices = ['y','n']
    isNewUser = input("[ATTENTION] Are you a new user? (y/n): ")
    returnUser = ""

    # validate input
    while (isNewUser not in choices):
        print("Please enter (y/n): \n")
        isNewUser = input("Are you a new user? (y/n): ")

    # if the user is not new, allow them to enter their existing id
    if (str(isNewUser) == 'n'):
        returnUser = input("[ATTENTION] Please enter your user id: ")
    
    return isNewUser, returnUser

# enter new user id
def initializeUser():
   newUserId = input("[ATTENTION] Creating new user... Please enter a user id and press <Enter>:  ")
   recordNewUser(newUserId)
   return newUserId

# record new user ids in a csv file
def recordNewUser(newUserId):
    filepath = "Facial_Recognition\\user_store.csv"
    with open(filepath, 'a', newline='', encoding='utf-8') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow([newUserId])
