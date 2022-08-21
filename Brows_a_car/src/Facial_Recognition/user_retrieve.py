import string


class UserData:
    def __init__(this):
        this.usernames = []
        this.currentUser = ""
        this.dataFile = "Facial_Recognition\\user_store.csv"

    def loadUsernames(this):
        # load usernames from csv file
        fileRead = open(this.dataFile)
        [ this.usernames.append(str(x.replace('\n',''))) if x != '\n' else None for x in fileRead ]

    def checkUserExist(this, name) -> bool:
        # check if user exist
        return name in this.usernames

    def setCurrentUser(this, name):
        # set current user logged in
        this.currentUser = name

    def getUsernamesList(this):
        # return list of usernames
        return this.usernames

    def getCurrentUser(this):
        # return current user
        return this.currentUser