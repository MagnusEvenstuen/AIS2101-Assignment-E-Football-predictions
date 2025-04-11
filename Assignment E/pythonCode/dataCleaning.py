from graphing import getColumnNames
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

class SeasonalGameHandler:
    def __init__(self, filename):
        self.outputMatrix = []
        #Opens the file
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            rows = list(reader)

            for row in rows:
                #Changes the name that could be wrong
                if row[1] == "Man City":
                    row[1] = "Manchester City"
                elif row[2] == "Man City":
                    row[2] = "Manchester City"
                if row[1] == "Man United":
                    row[1] = "Manchester Utd"
                elif row[2] == "Man United":
                    row[2] = "Manchester Utd"
                if row[1] == "Ipswich":
                    row[1] = "Ipswich Town"
                elif row[2] == "Ipswich":
                    row[2] = "Ipswich Town"
                if row[1] == "Leicester":
                    row[1] = "Leicester City"
                elif row[2] == "Leicester":
                    row[2] = "Leicester City"
                if row[1] == "Nott'm Forest":
                    row[1] = "Nott'ham Forest"
                elif row[2] == "Nott'm Forest":
                    row[2] = "Nott'ham Forest"
                if row[1] == "Newcastle":
                    row[1] = "Newcastle Utd"
                elif row[2] == "Newcastle":
                    row[2] = "Newcastle Utd"

                if not row:
                    continue

                #Skips the date and the name of the referee
                processedRow = row[1:9] + row[10:]
                self.outputMatrix.append(processedRow)

        self.lettersToNumber()

    def lettersToNumber(self):
        #Turns A, D and H to numbers which is easier to interprate for the program
        for i in range(len(self.outputMatrix)):
            for j in range(len(self.outputMatrix[i])):
                if self.outputMatrix[i][j] == "A":
                    self.outputMatrix[i][j] = -1
                elif self.outputMatrix[i][j] == "D":
                    self.outputMatrix[i][j] = 0
                elif self.outputMatrix[i][j] == "H":
                    self.outputMatrix[i][j] = 1



class SeasonalStatHandler:
    def __init__(self, filenames):
        self.outputMatrix = []
        for i, filename in enumerate(filenames):
            with open(filename, 'r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    if not row:
                        continue
                    team = row[0]
                    if i == 0:  #Take all the values from the first matrix
                        stats = [val for idx, val in enumerate(row[1:])]
                    else:
                        #Removes the stats that is reapeted by all the stat files
                        stats = row[3:] if i > 0 else row[1:]

                    if i == 0:
                        #Finds the teams
                        self.outputMatrix.append([team] + stats)
                    else:
                        for outputRow in self.outputMatrix:
                            #Matches the teams with the statistics
                            if outputRow[0] == team:
                                outputRow.extend(stats)
                                break

def convertToPercent(seasonStats):
    #Conwerts the matrix to a pandas data frame
    seasonStats = pd.DataFrame(seasonStats)
    #Finds the team names
    teamNames = seasonStats.iloc[:, 0]
    #Finds the stats
    stats = seasonStats.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    columnSums = stats.sum(axis=0).replace(0, 1)
    #Converts all the stats to percent of the total stats
    percentStats = stats.div(columnSums, axis=1) * 100

    #Puts the stats back together with the team names
    seasonStats = pd.concat([teamNames, percentStats], axis=1)
    #Converts the dataframe back to list and converts
    return seasonStats.values.tolist()


def removeCorrelation(seasonStats, validationStats, threshold):
    columnNames = getColumnNames()
    addedColumnNames = ['form', 'goal form', 'recent red cards']
    columnNames = columnNames + addedColumnNames
    seasonStats = pd.DataFrame(seasonStats, columns=columnNames)
    validationStats = pd.DataFrame(validationStats, columns=columnNames)

    #Calculate the correlation matrix
    correlationMatrix = seasonStats.corr().abs()

    #Plotting the correlation matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(correlationMatrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(ticks=range(len(correlationMatrix.columns)), labels=correlationMatrix.columns, rotation=90)
    plt.yticks(ticks=range(len(correlationMatrix.columns)), labels=correlationMatrix.columns)
    plt.title("Correlation Matrix Heatmap")
    plt.show()

    droppedValues = set()

    #Removes all but one of the values with high correlation
    for i in range(len(correlationMatrix)):
        for j in range(i + 1, len(correlationMatrix)):
            if correlationMatrix.iloc[i, j] > threshold:
                if j not in droppedValues:
                    droppedValues.add(j)

    #The code line below is generated by deepseek
    keptValues = [i for i in range(len(correlationMatrix)) if i not in droppedValues]

    #Returns the kept values
    return seasonStats.iloc[:, keptValues].values, validationStats.iloc[:, keptValues].values

def getResults(seasonGames):
    #Returns an array with the result of all the games
    outputArray = []
    for game in seasonGames:
        outputArray.append(game[4])
    return outputArray

def createDifferences(seasonGames, seasonStats):
    difference = []
    stat1 = []
    stat2 = []
    #Finds all the stats for each team in each game
    for game in seasonGames:
        for stat in seasonStats:
            if stat[0] == game[0]:
                stat1 = stat
            elif stat[0] == game[1]:
                stat2 = stat
        differenceStats = []

        #Appends the difference in stats between each of the teams
        for i in range(1, len(stat1)):
            stat1[i] = float(stat1[i])
            stat2[i] = float(stat2[i])
            differenceStats.append(-stat1[i] + stat2[i])

        homeForm = []
        awayForm = []
        homeGd = []
        awayGd = []
        homeRed = []
        awayRed = []
        #Find some form based statistics
        for earlyGame in seasonGames:
            if earlyGame == game:
                break
            if earlyGame[0] == game[0]:
                homeForm.append(earlyGame[4])
                homeGd.append(int(earlyGame[2]) - int(earlyGame[3]))
                homeRed.append(int(earlyGame[18]))
            elif earlyGame[1] == game[0]:
                homeForm.append(-earlyGame[4])
                homeGd.append(int(earlyGame[3]) - int(earlyGame[2]))
                homeRed.append(int(earlyGame[19]))
            elif earlyGame[0] == game[1]:
                awayForm.append(earlyGame[4])
                awayGd.append(int(earlyGame[2]) - int(earlyGame[3]))
                awayRed.append(int(earlyGame[18]))
            elif earlyGame[1] == game[1]:
                awayForm.append(-earlyGame[4])
                awayGd.append(int(earlyGame[3]) - int(earlyGame[2]))
                awayRed.append(int(earlyGame[19]))

        homeFormSum = 0
        awayFormSum = 0
        homeGdSum = 0
        awayGdSum = 0
        homeRedLast2 = 0
        awayRedLast2 = 0
        if len(homeForm) >= 2 and len(awayForm) >= 2:
            homeRedLast2 = sum(homeRed[-2:])
            awayRedLast2 = sum(awayRed[-2:])

        if len(homeForm) >= 5 and len(awayForm) >= 5:
            homeFormSum = sum(homeForm[-5:])
            awayFormSum = sum(awayForm[-5:])

            homeGdSum = sum(homeGd[-5:])
            awayGdSum = sum(awayGd[-5:])
        else:
            maxLength = 0
            if len(homeForm) > len(awayForm):
                maxLength = len(awayForm)
            else:
                maxLength = len(homeForm)
            for i in range(maxLength):
                homeFormSum += homeForm[i]
                awayFormSum += awayForm[i]
                homeGdSum += homeGd[i]
                awayGdSum += awayGd[i]

        if len(differenceStats) > 0:
            #Appends the form based stats to the seasonal stats
            differenceStats.append(homeFormSum-awayFormSum)
            differenceStats.append(homeGdSum-awayGdSum)
            differenceStats.append(homeRedLast2-awayRedLast2)
            difference.append(differenceStats)

    return difference

def chooseImportantData(seasonGames, seasonStats, wantedFeatures):
    #Removes the stats that are the least meaningful to the output
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(estimator=model, n_features_to_select=wantedFeatures, step=5)
    selector.fit(seasonStats, seasonGames)

    return np.where(selector.support_)[0]