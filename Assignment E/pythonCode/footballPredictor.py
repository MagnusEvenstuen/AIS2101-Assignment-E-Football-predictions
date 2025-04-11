from dataCleaning import *
from machineLearningAlgorithms import *
from graphing import *

if __name__ == "__main__":
    #Gets long terminal prints
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    #Getting data from the csv files
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

    #Turns the data from the csv file into nxm matrix
    stats2122 = SeasonalStatHandler(season2122)
    stats2223 = SeasonalStatHandler(season2223)
    stats2324 = SeasonalStatHandler(season2324)
    stats2425 = SeasonalStatHandler(season2425)

    #Gets the csv files for the games and turns them into the correct format
    game2122 = SeasonalGameHandler("../DataFiles/season-2122.csv")
    game2223 = SeasonalGameHandler("../DataFiles/season-2223.csv")
    game2324 = SeasonalGameHandler("../DataFiles/season-2324.csv")
    game2425 = SeasonalGameHandler("../DataFiles/season-2425.csv")

    #Converts the stats to percent of total of the data to get a number between 0 and 1
    percentStats2122 = convertToPercent(stats2122.outputMatrix)
    percentStats2223 = convertToPercent(stats2223.outputMatrix)
    percentStats2324 = convertToPercent(stats2324.outputMatrix)
    percentStats2425 = convertToPercent(stats2425.outputMatrix)

    #Creates a scatter plot
    scatterPlot(stats2425.outputMatrix, 3, 4, 15)
    scatterPlot(stats2425.outputMatrix, 34, 49, 85)
    scatterPlot(stats2425.outputMatrix, 48, 46, 44)

    # Calls the function for unsupervised learning, and selects which stats to graph with the output
    #silhouetteScore = []
    #for i in range(2, 9):
    #    silhouetteScore.append(kMeansClustering(stats2425.outputMatrix, 2, 3, "Goals", "Total shots", i))

    kMeansClustering(stats2425.outputMatrix, 2, 3, "Goals", "Total shots", 3)
    kMeansClustering(stats2425.outputMatrix, 47, 43, "Progressive passes", "Key passes", 3)
    kMeansClustering(stats2425.outputMatrix, 33, 48, "Errors leading to shots", "Goals against", 3)
    #Comment in this code and the for loop above to get the wcss plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot([2, 3, 4, 5, 6, 7, 8], silhouetteScore, marker='o')
    plt.title("WCSS score for different kValues")
    plt.xlabel("kValue")
    plt.ylabel("WCSS score")
    plt.grid(True)
    plt.show()
    """
    hierarchicalClustering(stats2425.outputMatrix, 2, 3, "Goals", "Total shots", 3)
    hierarchicalClustering(stats2425.outputMatrix, 47, 43, "Progressive passes", "Key passes", 3)
    hierarchicalClustering(stats2425.outputMatrix, 33, 48, "Errors leading to shots", "Goals against", 3)

    #Finds the difference between the stats of the teams that have faced each other
    season0Difference = createDifferences(game2122.outputMatrix, percentStats2122)
    season1Difference = createDifferences(game2223.outputMatrix, percentStats2223)
    season2Difference = createDifferences(game2324.outputMatrix, percentStats2324)
    season3Difference = createDifferences(game2425.outputMatrix, percentStats2425)

    #Finds the result (home, draw or away) and turns it into numeric values
    season0HUB = getResults(game2122.outputMatrix)
    season1HUB = getResults(game2223.outputMatrix)
    season2HUB = getResults(game2324.outputMatrix)
    season3HUB = getResults(game2425.outputMatrix)

    #Creates a graphes for the statistics
    resultDistribution(np.array(season0HUB + season1HUB + season2HUB + season3HUB))
    boxplot(np.array(stats2122.outputMatrix + stats2223.outputMatrix + stats2324.outputMatrix + stats2425.outputMatrix), "Goals/shot", 9)
    boxplot(np.array(stats2122.outputMatrix + stats2223.outputMatrix + stats2324.outputMatrix + stats2425.outputMatrix),"Goals", 3)
    seasonalChangesStats(stats2122.outputMatrix, stats2223.outputMatrix, stats2324.outputMatrix, stats2425.outputMatrix, "Goals", 3)
    seasonalChangesStats(stats2122.outputMatrix, stats2223.outputMatrix, stats2324.outputMatrix, stats2425.outputMatrix,"Penalty kicks made", 12)
    seasonalChangesStats(stats2122.outputMatrix, stats2223.outputMatrix, stats2324.outputMatrix, stats2425.outputMatrix,"Number of players used", 1)
    getStatisticalStats(np.array(stats2122.outputMatrix + stats2223.outputMatrix + stats2324.outputMatrix + stats2425.outputMatrix))

    #Convert all the training data to single arrays
    trainingInputData = np.array(season0Difference + season1Difference + season2Difference)
    testingInputData = np.array(season3Difference)
    trainingOutputData = np.array(season0HUB + season1HUB + season2HUB)
    testingOutputData = np.array(season3HUB)

    #Removes the data that is strongly correlated to try to remove noice
    trainingInputData, testingInputData = removeCorrelation(trainingInputData, testingInputData, 0.9)
    #Removes some of the unimportant data to remove data that has little to do with the wanted output
    selected = chooseImportantData(trainingOutputData, trainingInputData, 10)

    #Changes the data to the selected part
    trainingInputData = trainingInputData[:, selected]
    testingInputData = testingInputData[:, selected]

    #Runs some supervised learning algorithms
    modelRandomForest = randomForestMachineLearning(trainingInputData, trainingOutputData, testingInputData, testingOutputData)
    importantmodelRandomForest = importantRandomForestMachineLearning(trainingInputData, trainingOutputData, testingInputData, testingOutputData, 0.03)
    linearyRegression = linearRegressionMachineLearning(trainingInputData, trainingOutputData, testingInputData, testingOutputData)

    #Gives the percentage chance for different outcomes (home, draw and away)
    randomForestPrediction("Everton", "Arsenal", percentStats2425, game2425.outputMatrix, selected, modelRandomForest)
    linearRegressionPrediction("Ipswich Town", "Wolves", percentStats2425, game2425.outputMatrix, selected, linearyRegression)

    #Trains a neural network and gives the percent chance for different outcomes
    networkTrainer = NNTrainer(len(trainingInputData[0]), 0.005)
    networkTrainer.networkTrainer(trainingInputData, trainingOutputData, 100, 200)
    networkTrainer.evaluate(testingInputData, testingOutputData, trainingInputData, trainingOutputData)
    networkTrainer.predict("Manchester City", "Crystal Palace", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Brighton", "Leicester City", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Nott'ham Forest", "Everton", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Southampton", "Aston Villa", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Arsenal", "Brentford", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Chelsea", "Ipswich Town", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Liverpool", "West Ham", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Wolves", "Tottenham", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Newcastle Utd", "Manchester Utd", percentStats2425, game2425.outputMatrix, selected)
    networkTrainer.predict("Bournemouth", "Fulham", percentStats2425, game2425.outputMatrix, selected)
