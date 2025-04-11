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
                    if i == 0:  #skudd.csv - remove shots (4) and shots on target (5)
                        stats = [val for idx, val in enumerate(row[1:]) if idx not in [4, 5]]
                    elif i == 1:  #defence.csv - remove blocks (8) and tkl+int (12)
                        stats = [val for idx, val in enumerate(row[3:]) if idx not in [8, 12]]
                    else:
                        stats = row[3:] if i > 0 else row[1:]

                    if i == 0:
                        self.outputMatrix.append([team] + stats)
                    else:
                        for output_row in self.outputMatrix:
                            if output_row[0] == team:
                                output_row.extend(stats)
                                break


def scatterPlot(matrix, column1, column2, column3):
    xData = []
    yData = []
    zData = []
    teams = []

    columnNames = [
        # From skudd.csv (columns 1-19)
        "Number of Players", "Games", "Goals", "%Shots on target", "Shots per 90",
        "Shots on target per 90", "Goals/shot", "Goals/shot on target", "Average shot distance", "Shots from free kick",
        "Penelty kicks made", "Penelty kicks attempted", "Expected goals", "Non penalty xG", "non penalty xG/shot",
        "Goals - xG", "Non penalty goals - non penalty xG",
        # From defence.csv (columns 20-34)
        "Tackles", "Tackeles won", "Tackles Def 3rd", "Tackles Mid 3rd", "Tackles Att 3rd", "Number of challanges",
        "%Tackle succsess", "Tackles lost", "Shots blocked", "Passes blocked", "Interceptions", "Clearences",
        "Mistakes leading to shots",
        # From pasning.csv (columns 35-48)
        "Passes completed", "Passes attempted", "%Passes completed", "Passing distance",
        "Progressive passing distance",
        "Assists", "Expected assisted goals", "Expected assists", "Assists-Expected assisted goals",
        "Key passes", "Passes into final third", "Passes to penalty area", "Crosses to penalty area",
        "Progressive passes",
        # From keeper.csv (columns 49-69)
        "Goals against", "Penalty kicks allowed", "Free kick goals against", "Corner goals against", "Own goals",
        "Post shot expected goals",
        "Post shot expected goals/shot on target", "Post shot xG-Goals allowed", "Post shot xG-Goals allowed/90",
        "GK passes completed over 40 yards", "GK passes attempted over 40 yards",
        "GK pass completion rate over 40 yards",
        "GK passes completed without goal kick", "Throws attemped", "%Passes over 40 yards", "Average GK pass length",
        "Crosses faced",
        "Crosses stopped", "%Crosses stopped", "Defencive actions outside pen area",
        "Defencive actions outside pen area/90",
        "Average distance of GK defencive actions",
        # From misc.csv (columns 70-85)
        "Yellow cards", "Red cards", "2nd yellow cards", "Fouls commited", "Fouls drawn", "Offsides", "Crosses",
        "Interceprions",
        "Tackles won", "Penalty kicks won", "Penalty kicks conceeded", "Own goals", "Ball recoveries",
        "Aerial duels won", "Aerial duels lost"
    ]

    for row in matrix:
        xData.append(float(row[column1]))
        yData.append(float(row[column2]))
        zData.append(float(row[column3]))
        teams.append(row[0])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(columnNames[column1 - 1], fontsize=12, labelpad=15)
    ax.set_ylabel(columnNames[column2 - 1], fontsize=12, labelpad=15)
    ax.set_zlabel(columnNames[column3 - 1], fontsize=12, labelpad=15)
    ax.scatter(xData, yData, zData, s=50)

    plt.title('3D Scatter Plot of Team Statistics', pad=20)

    for i, team in enumerate(teams):
        ax.text(xData[i], yData[i], zData[i], team,
                fontsize=7, ha='center', va='bottom', alpha=0.8)

    plt.show()

def convertToPercent(seasonStats):
    seasonStats = pd.DataFrame(seasonStats)
    teamNames = seasonStats.iloc[:, 0]
    stats = seasonStats.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    columnSums = stats.sum(axis=0).replace(0, 1)
    percentStats = stats.div(columnSums, axis=1) * 100

    seasonStats = pd.concat([teamNames, percentStats], axis=1)

    return seasonStats.values.tolist()


def linearyRegressionMachineLearning(season0Games, season1Games, season2Games, season3Games,
                                     season0Stats, season1Stats, season2Stats, season3Stats):
        season0Difference = createDifferences(season0Games, season0Stats)
        season1Difference = createDifferences(season1Games, season1Stats)
        season2Difference = createDifferences(season2Games, season2Stats)
        season3Difference = createDifferences(season3Games, season3Stats)
        season0HUB = getResults(season0Games)
        season1HUB = getResults(season1Games)
        season2HUB = getResults(season2Games)
        season3HUB = getResults(season3Games)

        trainingInputData = np.array(season0Difference + season1Difference + season2Difference)
        testingInputData = np.array(season3Difference)
        trainingOutputData = np.array(season0HUB + season1HUB + season2HUB)
        testingOutputData = np.array(season3HUB)

        trainingInputData, testingInputData = removeCorrelation(trainingInputData, testingInputData, 0.9)
        print(len(testingInputData[0]))

        model = LinearRegression()
        model.fit(trainingInputData, trainingOutputData)

        modelValidation = model.predict(testingInputData)
        modelValidation = np.round(modelValidation).astype(int)
        for i in range(len(modelValidation)):
            if modelValidation[i] > 1:
                modelValidation[i] = 1
            elif modelValidation[i] < -1:
                modelValidation[i] = -1

        modelAccuracy = accuracy_score(testingOutputData, modelValidation)
        print("Modellens nøyaktighet lineær:", modelAccuracy)
        print("Koeffisienter lineær:", model.coef_)
        print("Intercept lineær:", model.intercept_)

        scaler = MinMaxScaler()
        trainingInputData = scaler.fit_transform(trainingInputData)
        testingInputData = scaler.transform(testingInputData)

        parameterGrid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }

        model = RandomForestClassifier(random_state=50)
        importantModel = SelectFromModel(
            RandomForestClassifier(n_estimators=60, random_state=50),
            threshold=0.01
        )

        gridSearch = GridSearchCV(model, parameterGrid, cv=5, scoring="accuracy")
        gridSearch.fit(trainingInputData, trainingOutputData)

        print("Best parameters:", gridSearch.best_params_)
        print("Best cross-validation accuracy:", gridSearch.best_score_)

        model = gridSearch.best_estimator_
        predictions = model.predict(testingInputData)

        accuracy = accuracy_score(testingOutputData, predictions)
        print("Modellens nøyaktighet trær:", accuracy)
        print("Feature Importances trær:", model.feature_importances_)

        importantModel.fit(trainingInputData, trainingOutputData)

        importantModelTraining = importantModel.transform(trainingInputData)
        importantModelTest = importantModel.transform(testingInputData)

        modelImportant = RandomForestClassifier(n_estimators=60, random_state=50)
        modelImportant.fit(importantModelTraining, trainingOutputData)
        importantPrediction = modelImportant.predict(importantModelTest)
        importantAccuracy = accuracy_score(testingOutputData, importantPrediction)

        print("Nøyaktighet med viktige features:", importantAccuracy)
        print("Feature Importances trær:", modelImportant.feature_importances_)
        print(f"Antall features før: {trainingInputData.shape[1]}")
        print(f"Antall features etter: {importantModelTraining.shape[1]}")

        return model

def removeCorrelation(seasonStats, validationStats, threshold):
    seasonStats = pd.DataFrame(seasonStats)
    validationStats = pd.DataFrame(validationStats)
    correlationMatrix = seasonStats.corr().abs()
    droppedValues = set()

    for i in range(len(correlationMatrix)):
        for j in range(i + 1, len(correlationMatrix)):
            if correlationMatrix.iloc[i, j] > threshold:
                if j not in droppedValues:
                    droppedValues.add(j)

    #The code line below is generated by deepseek
    keptValues = [i for i in range(len(correlationMatrix)) if i not in droppedValues]

    return seasonStats.iloc[:, keptValues].values, validationStats.iloc[:, keptValues].values

def getResults(seasonGames):
    outputArray = []
    for game in seasonGames:
        outputArray.append(game[4])
    return outputArray

def createDifferences(seasonGames, seasonStats):
    difference = []
    stat1 = []
    stat2 = []
    for game in seasonGames:
        for stat in seasonStats:
            if stat[0] == game[0]:
                stat1 = stat
            elif stat[0] == game[1]:
                stat2 = stat
        differenceStats = []

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
            differenceStats.append(homeFormSum-awayFormSum)
            differenceStats.append(homeGdSum-awayGdSum)
            differenceStats.append(homeRedLast2-awayRedLast2)
            difference.append(differenceStats)

    return difference

def chooseImportantData(seasonGames, seasonStats, wantedFeatures):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(estimator=model, n_features_to_select=wantedFeatures, step=5)
    selector.fit(seasonStats, seasonGames)

    return np.where(selector.support_)[0]

class NeuralNetwork(nn.Module):
    def __init__(self, numInputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(numInputs, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.layers(x)

class NNTrainer:
    def __init__(self, numInputs, learningRate):
        self.model = NeuralNetwork(numInputs)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learningRate)
        self.scaler = StandardScaler()
        self.criterion = nn.CrossEntropyLoss()

    def networkTrainer(self, trainingInput, trainingOutput, wholeNetworkIteration, batchSize):
        trainingOutput = trainingOutput + 1
        trainingInput = self.scaler.fit_transform(trainingInput)
        trainingInput = torch.tensor(trainingInput, dtype=torch.float32)
        trainingOutput = torch.tensor(trainingOutput, dtype=torch.long)

        dataset = data_utils.TensorDataset(trainingInput, trainingOutput)
        loader = data_utils.DataLoader(dataset, batch_size=batchSize, shuffle=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.model.train()

        bestLoss = np.inf
        maxNoImprovement = 20
        noImprovement = 0

        for iteration in range(wholeNetworkIteration):
            iterationLoss = 0
            for inputs, labels in loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                scheduler.step(loss)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                iterationLoss += loss.item()

            if iterationLoss < bestLoss:
                bestLoss = iterationLoss
                print(iterationLoss)
                noImprovement = 0
            else:
                noImprovement += 1
                if noImprovement >= maxNoImprovement:
                    print("Stopping after", iteration)
                    break

    def evaluate(self, testInput, testOutput):
        testOutput = testOutput + 1
        testInput = self.scaler.transform(testInput)
        testInput = torch.tensor(testInput, dtype=torch.float32)
        testOutput = torch.tensor(testOutput, dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(testInput)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == testOutput).sum().item() / testOutput.size(0)
        print(f"Nøyaktighet: {accuracy:.4f}")

    def predict(self, homeTeam, awayTeam, seasonStats, seasonGames, selected):
        homeStats = []
        awayStats = []
        for team in seasonStats:
            if team[0] == homeTeam:
                homeStats = team[1:]
            elif team[0] == awayTeam:
                awayStats = team[1:]
            if len(homeStats) > 0 and len(awayStats) > 0:
                continue

        differenceStats = []
        for i in range(0, len(homeStats)):
            homeStats[i] = float(homeStats[i])
            awayStats[i] = float(awayStats[i])
            differenceStats.append(-homeStats[i] + awayStats[i])

        homeForm = []
        awayForm = []
        homeGd = []
        awayGd = []
        homeRed = []
        awayRed = []
        for earlyGame in seasonGames:
            if earlyGame[0] == homeTeam:
                homeForm.append(earlyGame[4])
                homeGd.append(int(earlyGame[2]) - int(earlyGame[3]))
                homeRed.append(int(earlyGame[18]))
            elif earlyGame[1] == homeTeam:
                homeForm.append(-earlyGame[4])
                homeGd.append(int(earlyGame[3]) - int(earlyGame[2]))
                homeRed.append(int(earlyGame[19]))
            elif earlyGame[0] == awayTeam:
                awayForm.append(earlyGame[4])
                awayGd.append(int(earlyGame[2]) - int(earlyGame[3]))
                awayRed.append(int(earlyGame[18]))
            elif earlyGame[1] == awayTeam:
                awayForm.append(-earlyGame[4])
                awayGd.append(int(earlyGame[3]) - int(earlyGame[2]))
                awayRed.append(int(earlyGame[19]))

        homeRedLast2 = sum(homeRed[-2:])
        awayRedLast2 = sum(awayRed[-2:])

        homeFormSum = sum(homeForm[-5:])
        awayFormSum = sum(awayForm[-5:])

        homeGdSum = sum(homeGd[-5:])
        awayGdSum = sum(awayGd[-5:])
        differenceStats.append(homeFormSum - awayFormSum)
        differenceStats.append(homeGdSum - awayGdSum)
        differenceStats.append(homeRedLast2 - awayRedLast2)

        differenceStats = np.array(differenceStats).reshape(1, -1)
        differenceStats = differenceStats[:, selected]

        differenceStats = self.scaler.transform(differenceStats)
        tensorInput = torch.tensor(differenceStats, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensorInput)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            percentages = probabilities.numpy()[0] * 100

        print(homeTeam, "-", awayTeam)
        print("Home win", percentages[2])
        print("Draw", percentages[1])
        print("Away win", percentages[0])



if __name__ == "__main__":
    season2122 = [
        "../DataFiles/2122skudd.csv",
        "../DataFiles/defence2122.csv",
        "../DataFiles/pasning2122.csv",
        "../DataFiles/2122keeper.csv",
        "../DataFiles/2122misc.csv"
    ]

    season2223 = [
        "../DataFiles/2223skudd.csv",
        "../DataFiles/defence2223.csv",
        "../DataFiles/pasning2223.csv",
        "../DataFiles/2223keeper.csv",
        "../DataFiles/2223misc.csv"
    ]
    season2324 = [
        "../DataFiles/2324skudd.csv",
        "../DataFiles/defence2324.csv",
        "../DataFiles/pasning2324.csv",
        "../DataFiles/2324keeper.csv",
        "../DataFiles/2324misc.csv"
    ]
    season2425 = [
        "../DataFiles/2425skudd.csv",
        "../DataFiles/defence2425.csv",
        "../DataFiles/pasning2425.csv",
        "../DataFiles/2425keeper.csv",
        "../DataFiles/2425misc.csv"
    ]


    stats2122 = SeasonalStatHandler(season2122)
    stats2223 = SeasonalStatHandler(season2223)
    stats2324 = SeasonalStatHandler(season2324)
    stats2425 = SeasonalStatHandler(season2425)

    game2122 = SeasonalGameHandler("../DataFiles/season-2122.csv")
    game2223 = SeasonalGameHandler("../DataFiles/season-2223.csv")
    game2324 = SeasonalGameHandler("../DataFiles/season-2324.csv")
    game2425 = SeasonalGameHandler("../DataFiles/season-2425.csv")

    percentStats2122 = convertToPercent(stats2122.outputMatrix)
    percentStats2223 = convertToPercent(stats2223.outputMatrix)
    percentStats2324 = convertToPercent(stats2324.outputMatrix)
    percentStats2425 = convertToPercent(stats2425.outputMatrix)

    #scatterPlot(stats2425.outputMatrix, 1, 18, 79)
    #linearyRegressionMachineLearning(game2122.outputMatrix, game2223.outputMatrix, game2324.outputMatrix, game2425.outputMatrix,
     #                                stats2122.outputMatrix, stats2223.outputMatrix, stats2324.outputMatrix, stats2425.outputMatrix)

    season0Difference = createDifferences(game2122.outputMatrix, percentStats2122)
    season1Difference = createDifferences(game2223.outputMatrix, percentStats2223)
    season2Difference = createDifferences(game2324.outputMatrix, percentStats2324)
    season3Difference = createDifferences(game2425.outputMatrix, percentStats2425)
    season0HUB = getResults(game2122.outputMatrix)
    season1HUB = getResults(game2223.outputMatrix)
    season2HUB = getResults(game2324.outputMatrix)
    season3HUB = getResults(game2425.outputMatrix)

    trainingInputData = np.array(season0Difference + season1Difference + season2Difference)
    testingInputData = np.array(season3Difference)
    trainingOutputData = np.array(season0HUB + season1HUB + season2HUB)
    testingOutputData = np.array(season3HUB)

    trainingInputData, testingInputData = removeCorrelation(trainingInputData, testingInputData, 0.9)

    selected = chooseImportantData(trainingOutputData, trainingInputData, 30)

    trainingInputData = trainingInputData[:, selected]
    testingInputData = testingInputData[:, selected]

    networkTrainer = NNTrainer(len(trainingInputData[0]), 0.005)
    networkTrainer.networkTrainer(trainingInputData, trainingOutputData, 100, 32)
    networkTrainer.evaluate(testingInputData, testingOutputData)
    networkTrainer.predict("Everton", "Arsenal", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Ipswich Town", "Wolves", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Crystal Palace", "Brighton", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("West Ham", "Bournemouth", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Aston Villa", "Nott'ham Forest", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Tottenham", "Southampton", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Brentford", "Chelsea", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Fulham", "Liverpool", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Manchester Utd", "Manchester City", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Leicester City", "Newcastle Utd", percentStats2425, game2425.outputMatrix, selected)