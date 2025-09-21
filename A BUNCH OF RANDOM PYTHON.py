import numpy as np
from scipy import stats
from matplotlib import pyplot as plots
import pandas
from sklearn import linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score as r2
SS = StandardScaler()
def SlopeInterceptForm(x):
    return x*slope+intercept
randlist = np.random.randint(0, 25000000, 1000000)
normalizedlist = np.random.normal(10, 20, 1000000)
scatterX = np.random.uniform(1,6,10)
scatterY = np.random.uniform(1,6,10)
Mode = stats.mode(randlist).mode
Mean = np.mean(randlist)
Median = np.median(randlist)
StandardDeviation = np.std(randlist)
Variance = np.var(randlist)
percentile = np.percentile(randlist, 10)
print(Mode,Mean,Median,StandardDeviation,Variance,percentile,randlist,normalizedlist)
plots.hist(randlist, 100)
plots.show()
plots.hist(normalizedlist, 100)
plots.show()
plots.scatter(scatterX, scatterY)
slope = stats.linregress(scatterX, scatterY).slope
intercept = stats.linregress(scatterX, scatterY).intercept
regressY = list(map(SlopeInterceptForm, scatterX))
plots.plot(scatterX,regressY)
plots.show()
PolynomialModel = np.poly1d(np.polyfit(scatterX, scatterY, 3))
print(r2(scatterY,PolynomialModel(scatterX)))
print(PolynomialModel(6))
LineSpacing = np.linspace(1,6, 100)
plots.plot(LineSpacing, PolynomialModel(LineSpacing))
plots.scatter(scatterX, scatterY)
plots.show()
X = pandas.read_csv("data.csv")[["Weight", "Volume"]]
Y = pandas.read_csv("data.csv")["CO2"]
lmregress = lm.LinearRegression()
lmregress.fit(X,Y)
x = np.random.randint(0, 100000)
y = np.random.randint(0, 100000)
print(x, y, lmregress.predict([[x, y]]), lmregress.coef_, lmregress.intercept_)
X = SS.fit_transform(X)
lmregress.fit(X,Y)
print(lmregress.predict([[x,y]]), lmregress.coef_, lmregress.intercept_)
np.random.seed(2)
scatterX = np.random.normal(3, 1, 100)
scatterY = np.random.normal(150, 40, 100) / scatterX
plots.scatter(scatterX,scatterY)
trainscatterX = scatterX[:80]
trainscatterY = scatterY[:80]
testscatterX = scatterX[80:]
testscatterY = scatterY[80:]
PolynomialModel = np.poly1d(np.polyfit(trainscatterX,trainscatterY,3))
plots.scatter(testscatterX,testscatterY)
plots.plot(LineSpacing, PolynomialModel(LineSpacing))
print(r2(trainscatterY,PolynomialModel(trainscatterX)))
print(r2(testscatterY,PolynomialModel(testscatterX)))
print(PolynomialModel(3))
plots.show()