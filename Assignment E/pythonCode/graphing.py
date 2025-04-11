import matplotlib.pyplot as plt
import pandas as pd

def getColumnNames():
    return [
        # From skudd.csv (columns 1–19)
        "Number of players used",  # #PL
        "Number of games",  # 90s
        "Goals scored",  # Gls
        "Total shots",  # Sh
        "Shots on target",  # SoT
        "Percentage of shots on target",  # SoT%
        "Shots per game",  # Sh/90
        "Shots on target per game",  # SoT/90
        "Goals per shot",  # G/Sh
        "Goals per shot on target",  # G/SoT
        "Average shot distance",  # Dist
        "Shots from free kicks",  # FK (skudd)
        "Penalty kicks made",  # PK
        "Penalty kicks attempted",  # PKatt
        "Expected goals",  # xG
        "Non-penalty expected goals",  # npxG
        "Non-penalty xG per shot",  # npxG/Sh
        "Goal difference (Goals - xG)",  # G - xG
        "Non-penalty goal difference (np:G - npxG)",  # np:G - npxG

        # From defence.csv (columns 20–34)
        "Dribblers tackled",  # Tkl
        "Tackles won",  # TklW
        "Tackles in the defensive third",  # Def 3rd
        "Tackles in the middle third",  # Mid 3rd
        "Tackles in the attacking third",  # Att 3rd
        "Dribblers challenged",  # Att (defence)
        "Percent of dribblers tackled",  # Tkl%
        "Challenges lost",  # Lost
        "Blocks made",  # Blocks
        "Shots blocked",  # Sh (defence)
        "Passes blocked",  # Pass
        "Interceptions",  # Int (defence)
        "Total tackles and interceptions",  # Tkl + Int
        "Clearances",  # Clr
        "Errors leading to opposition shots",  # Err

        # From pasning.csv (columns 35–48)
        "Passes completed",  # Cmp
        "Passes attempted",  # Att (pasning)
        "Pass completion percentage",  # Cmp%
        "Total passing distance",  # Tot Dist
        "Progressive passing distance",  # Prg Dist
        "Assists",  # Ast
        "Expected assisted goals",  # xAG
        "Expected assists",  # xA
        "Assist difference (Assists - xAG)",  # A - xAg
        "Key passes",  # KP
        "Passes into final third",  # 1/3
        "Passes into penalty area",  # PPA
        "Crosses into penalty area",  # CrsPA
        "Progressive passes",  # PrgP

        # From keeper.csv (columns 49–70)
        "Goals against",  # GA
        "Penalty kicks allowed",  # PKA
        "Free kick goals against",  # FK (keeper)
        "Corner kick goals against",  # CK
        "Own goals conceded",  # OG
        "Post-shot expected goals",  # PSxG
        "Post-shot expected goals per shot on target",  # PSxG/SoT
        "Post-shot xG difference",  # PSxG+/-
        "Post-shot xG difference per 90",  # /90
        "GK passes completed over 40 yards",  # Cmp (keeper)
        "GK passes attempted over 40 yards",  # Att (keeper)
        "GK pass completion rate over 40 yards",  # Cmp% (keeper)
        "GK pass attempts",  # Att (GK)
        "Throws attempted",  # Thr
        "Percentage of long passes (over 40 yards)",  # Launch%
        "Average GK pass length",  # AvgLen
        "Crosses faced",  # Opp
        "Crosses stopped",  # Stp
        "Percentage of crosses stopped",  # Stp%
        "Defensive actions outside penalty area",  # #OPA
        "Defensive actions outside penalty area per 90",  # #OPA/90
        "Average distance of defensive actions",  # AvgDist

        # From misc.csv (columns 71–86)
        "Yellow cards",  # CrdY
        "Red cards",  # CrdR
        "Second yellow cards",  # 2CrdY
        "Fouls committed",  # Fls
        "Fouls drawn",  # Fld
        "Offsides",  # Off
        "Crosses",  # Crs (misc)
        "Interceptions",  # Int (misc)
        "Tackles won",  # TklW (misc)
        "Penalty kicks won",  # PKwon
        "Penalty kicks conceded",  # PKcon
        "Own goals",  # OG (misc)
        "Ball recoveries",  # Recov
        "Duels won",  # Won
        "Duels lost",  # Lost
        "Duel win percentage"  # Won%
    ]
def scatterPlot(matrix, column1, column2, column3):
    xData = []
    yData = []
    zData = []
    teams = []

    columnNames = getColumnNames()

    #Finds all the data
    for row in matrix:
        xData.append(float(row[column1]))
        yData.append(float(row[column2]))
        zData.append(float(row[column3]))
        teams.append(row[0])

    #Creates the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(columnNames[column1 - 1], fontsize=12, labelpad=15)
    ax.set_ylabel(columnNames[column2 - 1], fontsize=12, labelpad=15)
    ax.set_zlabel(columnNames[column3 - 1], fontsize=12, labelpad=15)
    ax.scatter(xData, yData, zData, s=50)

    plt.title('3D Scatter Plot of Team Statistics', pad=20)

    #Adds team names to the plot
    for i, team in enumerate(teams):
        ax.text(xData[i], yData[i], zData[i], team,
                fontsize=7, ha='center', va='bottom', alpha=0.8)

    plt.show()

def resultDistribution(games):
    result = [0, 0, 0]

    #Goes through all the dat and finds the amount of each result
    for game in games:
        if game == 1:
            result[0] += 1
        elif game == 0:
            result[1] += 1
        else:
            result[2] += 1

    print("home", result[0])
    print("draw", result[1])
    print("away", result[2])
    #Plots the result
    plt.figure(figsize=(8, 6))
    plt.bar(["Home", "Draw", "Away"], result, color=["green", "gray", "red"])
    plt.title("Result Distrebution")
    plt.xlabel("Result")
    plt.ylabel("Number of games")
    plt.tight_layout()
    plt.show()

def boxplot(stat, label, statPlacement):
    #Creates an array with all the stats
    correctStats = []
    for stats in stat:
        correctStats.append(float(stats[statPlacement]))

    #Plots the boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(correctStats)
    plt.title("Boxplot for " + label)
    plt.ylabel(label)
    plt.show()

def getStatisticalStats(stats):
    stats = pd.DataFrame(stats)
    stats = stats.apply(pd.to_numeric, errors='coerce')
    stats = stats.dropna(axis=1, how='all')

    mean = round(stats.mean(), 2)
    median = round(stats.median(), 2)
    mode = round(stats.mode().iloc[0], 2)
    std = round(stats.std(), 2)
    q1 = round(stats.quantile(0.25), 2)
    q3 = round(stats.quantile(0.75), 2)

    # Create a DataFrame with statistics as rows
    stats_df = pd.DataFrame(
        [mean, median, mode, std, q1, q3],
        index=['Mean', 'Median', 'Mode', 'Std Dev', 'Q1', 'Q3']
    )

    # Print the formatted statistics matrix
    print(stats_df)

def seasonalChangesStats(season0, season1, season2, season3, label, statPlacement):
    #Creates array for the total of the stats for each season
    season = [0, 1, 2, 3]
    statTotal = [0, 0, 0, 0]
    seasons = [season0, season1, season2, season3]

    #Fills in the stat total array
    for i in range(len(seasons)):
        for team in seasons[i]:
            statTotal[i] += float(team[statPlacement])

    #Plots the line diagram
    plt.figure(figsize=(8, 6))
    plt.plot(season, statTotal, marker='o')
    plt.title("Seasonal changes")
    plt.xlabel("Season")
    plt.ylabel(label)
    plt.grid(True)
    plt.show()