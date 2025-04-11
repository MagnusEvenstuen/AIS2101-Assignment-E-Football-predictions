from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn.cluster import KMeans
from dataCleaning import *
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

class NeuralNetwork(nn.Module):
    def __init__(self, numInputs):
        #Initiaises the neural network
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(numInputs, 128),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 8),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.layers(x)

class NNTrainer:
    def __init__(self, numInputs, learningRate):
        #Initelizes for the training of the network
        self.model = NeuralNetwork(numInputs)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learningRate)
        self.scaler = StandardScaler()
        self.criterion = nn.CrossEntropyLoss()

    def networkTrainer(self, trainingInput, trainingOutput, wholeNetworkIteration, batchSize):
        #Adds 1 to the input data to make it go from 0 to 2 to make it easier to handle
        trainingOutput = trainingOutput + 1

        #Scales the training data
        trainingInput = self.scaler.fit_transform(trainingInput)
        trainingInput = torch.tensor(trainingInput, dtype=torch.float32)
        trainingOutput = torch.tensor(trainingOutput, dtype=torch.long)

        #Extracts the dataset
        dataset = data_utils.TensorDataset(trainingInput, trainingOutput)
        loader = data_utils.DataLoader(dataset, batch_size=batchSize, shuffle=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.model.train()

        bestLoss = np.inf
        maxNoImprovement = 20
        noImprovement = 0

        #Trains the network and stops either after a given amount of iteration, or when the improvement stalls
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

    def evaluate(self, testInput, testOutput, trainingInput, trainingOutput):
        #Does the same to the testing data as to the training data
        testOutput = testOutput + 1
        trainingOutput = trainingOutput + 1
        trainingInput = self.scaler.transform(trainingInput)
        trainingInput = torch.tensor(trainingInput, dtype=torch.float32)
        trainingOutput = torch.tensor(trainingOutput, dtype=torch.long)
        testInput = self.scaler.transform(testInput)
        testInput = torch.tensor(testInput, dtype=torch.float32)
        testOutput = torch.tensor(testOutput, dtype=torch.long)

        #Evaluates the model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(testInput)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == testOutput).sum().item() / testOutput.size(0)
            outputs = self.model(trainingInput)
            _, predictedTraining = torch.max(outputs.data, 1)
            accuracyTraining = (predictedTraining == trainingOutput).sum().item() / trainingOutput.size(0)

        #Creates a confusion matrix to make it easy to evaluate the result
        confusionMatrix = confusion_matrix(testOutput, predicted)
        display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=["Away win", "Draw", "Home win"])
        display.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix - Neural Network")
        plt.show()
        #Prints the models accuracy
        print(f"Nøyaktighet:", accuracy)
        print(f"Nøyaktighet training:", accuracyTraining)

    def predict(self, homeTeam, awayTeam, seasonStats, seasonGames, selected):
        #Creates data for the inputted games
        differenceStats = predictionPreProcessing(homeTeam, awayTeam, seasonStats, seasonGames, selected)
        tensorInput = torch.tensor(differenceStats, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tensorInput)
            #Finds the probability for each outcome
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            percentages = probabilities.numpy()[0] * 100

        #Prints the chance for each result
        print(homeTeam, "-", awayTeam)
        print("Home win", percentages[2])
        print("Draw", percentages[1])
        print("Away win", percentages[0])

def linearRegressionMachineLearning(trainingInput, trainingOutput, testingInput, testingOutput):
    #Scales the training and testing data
    scaler = StandardScaler()
    trainingInput = scaler.fit_transform(trainingInput)
    testingInput = scaler.transform(testingInput)
    #Runs the lineary regressing model on the training input
    model = LinearRegression()
    model.fit(trainingInput, trainingOutput)

    #Predicts results based on the testing dataset
    prediction = model.predict(testingInput)
    prediction = np.clip(prediction, -1, 1)

    finalPrediction = []
    for val in prediction:
        if val > 0.33:
            finalPrediction.append(-1)
        elif val < -0.33:
            finalPrediction.append(1)
        else:
            finalPrediction.append(0)

    #Creates a confusion matrix and prints the accuracy
    confusionMatrix = confusion_matrix(testingOutput, finalPrediction)
    display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=["Away win", "Draw", "Home win"])
    display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Lineary Regression")
    plt.show()

    print("Modellens nøyaktighet lineær:", accuracy_score(testingOutput, finalPrediction))
    print("Koeffisienter lineær:", model.coef_)
    print("Intercept lineær:", model.intercept_)

    return model

def linearRegressionPrediction(homeTeam, awayTeam, seasonStats, seasonGames, selected, model):
    #Scales the data
    scaler = StandardScaler()
    differenceStats = predictionPreProcessing(homeTeam, awayTeam, seasonStats, seasonGames, selected)
    differenceStats = scaler.fit_transform(differenceStats)
    #Predict the chance of each result
    prediction = model.predict(differenceStats)[0]
    prediction = np.clip(prediction, -1, 1)

    homeWin = (prediction + 1)/2
    awayWin = (-prediction + 1)/2
    draw = 1 - homeWin + awayWin

    homePercent = 100 * homeWin/(homeWin + awayWin + draw)
    awayPercent = 100 * awayWin/(homeWin + awayWin + draw)
    drawPercent = 100 * draw/(homeWin + awayWin + draw)

    #Prints the chance of each result
    print(homeTeam, "-", awayTeam)
    print("Home win", homePercent)
    print("Draw", drawPercent)
    print("Away win", awayPercent)

def randomForestMachineLearning(trainingInput, trainingOutput, testingInput, testingOutput):
    #Creates the model and scales the data
    model = RandomForestClassifier(random_state=50)
    scaler = MinMaxScaler()
    trainingInputData = scaler.fit_transform(trainingInput)
    testingInputData = scaler.transform(testingInput)

    #Creates a grid of which values to test
    parameterGrid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    #Finds the values from the grid that gives the best results
    gridSearch = GridSearchCV(model, parameterGrid, cv=5, scoring="accuracy")
    gridSearch.fit(trainingInputData, trainingOutput)

    #Prints what the best parameters are, and the best score it got while it tested
    print("Best parameters:", gridSearch.best_params_)
    print("Best cross-validation accuracy:", gridSearch.best_score_)

    #Creates a new model with the best estimator and predicts test on the testing data
    model = gridSearch.best_estimator_
    predictions = model.predict(testingInputData)
    predictionTraining = model.predict(trainingInput)

    #Creates the confusion matrix
    confusionMatrix = confusion_matrix(testingOutput, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=["Away win", "Draw", "Home win"])
    display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

    #Prints the accuracy of the model
    accuracyTraining = accuracy_score(trainingOutput, predictionTraining)
    accuracy = accuracy_score(testingOutput, predictions)
    print("Modellens nøyaktighet trær trening:", accuracyTraining)
    print("Modellens nøyaktighet trær:", accuracy)
    print("Feature Importances trær:", model.feature_importances_)

    return model


def importantRandomForestMachineLearning(trainingInput, trainingOutput, testingInput, testingOutput, importanceThreshold):
    #Creates the model
    importantModel = SelectFromModel(
        RandomForestClassifier(n_estimators=60, random_state=50),
        threshold=importanceThreshold
    )

    importantModel.fit(trainingInput, trainingOutput)

    #Scales the model
    importantModelTraining = importantModel.transform(trainingInput)
    importantModelTest = importantModel.transform(testingInput)

    #Creates a new model with the correct values
    modelImportant = RandomForestClassifier(n_estimators=300, random_state=50, class_weight='balanced', max_depth=5)
    modelImportant.fit(importantModelTraining, trainingOutput)
    importantPrediction = modelImportant.predict(importantModelTest)
    importantPredictionTrainining = modelImportant.predict((importantModelTraining))
    importantAccuracy = accuracy_score(testingOutput, importantPrediction)
    importantAccuracyTraining = accuracy_score(trainingOutput, importantPredictionTrainining)

    #Creates the confusion matrix
    confusionMatrix = confusion_matrix(testingOutput, importantPrediction)
    display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=["Away win", "Draw", "Home win"])
    display.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Random Forest Important Features")
    plt.show()

    #Prints the accuracy and the number of features
    print("Nøyaktighet med viktige features på treningsdata:", importantAccuracyTraining)
    print("Nøyaktighet med viktige features:", importantAccuracy)
    print("Feature Importances trær:", modelImportant.feature_importances_)
    print(f"Antall features før: {trainingInput.shape[1]}")
    print(f"Antall features etter: {importantModelTraining.shape[1]}")

    return modelImportant

def randomForestPrediction(homeTeam, awayTeam, seasonStats, seasonGames, selected, model):
    #Pre processes the data to get it on the correct form
    differenceStats = predictionPreProcessing(homeTeam, awayTeam, seasonStats, seasonGames, selected)
    #Predicts the chances for each result
    percentages = model.predict_proba(differenceStats)[0]

    #Prints the chances for each result
    print(homeTeam, "-", awayTeam)
    print("Home win", percentages[2])
    print("Draw", percentages[1])
    print("Away win", percentages[0])

def predictionPreProcessing(homeTeam, awayTeam, seasonStats, seasonGames, selected):
    homeStats = []
    awayStats = []
    #Fills the stats for the home team and the away team
    for team in seasonStats:
        if team[0] == homeTeam:
            homeStats = team[1:]
        elif team[0] == awayTeam:
            awayStats = team[1:]
        if len(homeStats) > 0 and len(awayStats) > 0:
            continue

    differenceStats = []
    #Turns all the statistics into floats
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
    #Fills in form arrays
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
    #Creates the output which is the difference between the home teams and away teams stats
    differenceStats = np.array(differenceStats).reshape(1, -1)
    differenceStats = differenceStats[:, selected]

    return differenceStats

def kMeansClustering(data, point1, point2, label1, label2, kValue):
    teamNames = [row[0] for row in data]

    #This line is generated by Deepseek
    data = np.array([row[1:] for row in data], dtype=float)

    #Creates the map
    kmeans = KMeans(n_clusters=kValue, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_

    silhouettteValues = silhouette_samples(data, labels)
    fig, ax = plt.subplots(figsize=(10, 6))

    yLower = 0
    for i in range(kValue):
        clusterMask = (labels == i)
        thisClusterSilohuetteValue = silhouettteValues[clusterMask]
        thisClusterSilohuetteValue.sort()

        thisClusterSize = thisClusterSilohuetteValue.shape[0]
        yUpper = yLower + thisClusterSize

        color = cm.nipy_spectral(float(i)/kValue)
        ax.fill_betweenx(np.arange(yLower, yUpper),
                         0, thisClusterSilohuetteValue,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, yLower + 0.5 * thisClusterSize, str(i))
        yLower = yUpper + 0.5

    ax.set_title("Silhouette plot")
    ax.set_xlabel("Silhouette-koeffisient")
    ax.set_ylabel("Klynge")

    plt.tight_layout()
    plt.show()

    print("k-Value", str(kValue), "SiluetteScore", str(silhouette_score(data, labels)))
    #Plots based on some datapoints
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, point1], data[:, point2], c=labels, cmap='viridis', label='Datapoints')
    for i, (x, y) in enumerate(zip(data[:, point1], data[:, point2])):
        plt.text(x, y, teamNames[i], fontsize=8, ha='right', va='bottom')

    plt.title("Clustering for stats " + str(kValue))
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.legend()
    plt.show()

    return kmeans.inertia_


def hierarchicalClustering(data, point1, point2, label1, label2, numberOfClusters, linkage_method='complete'):
    teamNames = [row[0] for row in data]
    dataValues = np.array([row[1:] for row in data], dtype=float)

    #Perform hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=numberOfClusters, linkage=linkage_method)
    labels = hierarchical.fit_predict(dataValues)

    #Silhouette plot
    silhouettteValues = silhouette_samples(dataValues, labels)
    fig, ax = plt.subplots(figsize=(10, 6))

    yLower = 0
    for i in range(numberOfClusters):
        clusterMask = (labels == i)
        thisClusterSilohuetteValue = silhouettteValues[clusterMask]
        thisClusterSilohuetteValue.sort()

        thisClusterSize = thisClusterSilohuetteValue.shape[0]
        yUpper = yLower + thisClusterSize

        color = cm.nipy_spectral(float(i) / numberOfClusters)
        ax.fill_betweenx(np.arange(yLower, yUpper),
                         0, thisClusterSilohuetteValue,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, yLower + 0.5 * thisClusterSize, str(i))
        yLower = yUpper + 0.5

    ax.set_title(f"Silhouette plot")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    plt.show()

    #Dendrogram visualization
    plt.figure(figsize=(12, 6))
    linkage_matrix = linkage(dataValues, method=linkage_method)
    dendrogram(linkage_matrix, labels=teamNames, orientation='top')
    plt.title(f"Dendrogram linkage")
    plt.xlabel("Teams")
    plt.ylabel("Distance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    #Scatter plot visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(dataValues[:, point1], dataValues[:, point2], c=labels, cmap='viridis')
    for i, (x, y) in enumerate(zip(dataValues[:, point1], dataValues[:, point2])):
        plt.text(x, y, teamNames[i], fontsize=8, ha='right', va='bottom')

    plt.title(f"Hierarchical Clustering linkage")
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.show()

    print("Number of clusters:", numberOfClusters, "Silhouette Score:", silhouette_score(dataValues, labels))