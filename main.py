from model import LinearRegression


def main():
    
    #    age    |    cost

    #    2       |  20
    #    3       |  30
    #    3       |  10
    #    2       |  25
    #    3       |  4

    X1 = [[2], [3], [3], [2], [3]]
    y1 = [20, 30, 10, 25, 4]

    #    age    |    brand    |    cost

    #    3       |    20     |  20
    #    2       |    15     |  34
    #    3       |    25     |  90
    #    3       |    18     |  30
    #    2       |    22     |  15

    #X2 is a batch, with data points having multiple features (AKA a feature set)
    X2 = [[3, 20], [2, 15], [3, 25], [3, 18], [2, 22]]
    y2= [20, 34, 90, 30, 15]

    # Initializing two models for each training data set
    lrm1 = LinearRegression(0.0001, 100)
    lrm2 = LinearRegression(0.0001, 100)

    print("*******************")
    lrm1.fit(X1, y1)
    print("*******************")
    lrm2.fit(X2, y2)
    print("*******************")


main()