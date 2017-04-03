# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:18:25 2017

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 23:19:38 2017

@author: user
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, RBF, DotProduct, RationalQuadratic
from sklearn import preprocessing
from matplotlib import pyplot as plt
import seaborn as sns
'''
from subprocess import check_output
#print(check_output(["ls", "C:/Users/user/Downloads/House Prices"]).decode("utf8"))

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# functions by https://www.kaggle.com/masayukixhirose/house-prices-advanced-regression-techniques/linear-model-roughly-modeling
def catcheck(df):
    iscat = np.zeros(df.shape[1])
    for c in range(df.shape[1]):
        if df.dtypes[c] == 'object':
            iscat[c] = 1
        else:
            iscat[c] = 0
    catdf = pd.DataFrame({'variable': df.columns, 'iscat': iscat})
    havecat = catdf[catdf.iscat == 1]
    catcolumn = havecat.variable.tolist()
    return catcolumn  

def NAcheck(df):
    isna = np.zeros(df.shape[1])
    for c in range(df.shape[1]):
        isna[c] = df.iloc[:, c].isnull().values.any()
            # For each columns, it return "True" if they have NaN.
    nandf = pd.DataFrame({'variable': df.columns, 'isna': isna})
    havenan = nandf[nandf.isna == 1]
    NAcolumn = havenan.variable.tolist()
    return NAcolumn



train = pd.read_csv("C:/Users/user/Downloads/House Prices/train.csv")
test = pd.read_csv("C:/Users/user/Downloads/House Prices/test.csv")
y_log = np.log1p(train["SalePrice"].values)
train1 = train.drop(["Id","SalePrice"], axis=1)
test1 = test.drop("Id", axis=1)
train_shape = train.shape[0]




alls= train1.append(test1)

def miss_values(df):
    for column in df:
        # Test whether column has null value
        if len(df[column].apply(pd.isnull).value_counts()) > 0:
            #print(column+" has missing value")
            #if column is numeric, fill null with mean
            if df[column].dtype in ('int64','float64'):
                df[column] = df[column].fillna(df[column].mean())
            else:
                df[column] = df[column].fillna("No")
    return df

#def factor_encoding(df):
#    for column in df:
#        if df[column].dtype == 'object':
#            df = df.merge(pd.get_dummies(data=df[column],prefix=column),right_index=True,left_index=True)
#            del df[column]
#    return df



                
miss_values(alls)
#for column in alls:
#    # Test whether column has null value
#    if len(alls[column].apply(pd.isnull).value_counts()) > 0:
#        print(column+" has missing value")


#factor_encoding(alls)



alls["SimplOverallQual"] = alls.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                           4 : 2, 5 : 2, 6 : 2, # average
                                                           7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                          })


alls = alls.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},
                           "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                           "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                             "ALQ" : 5, "GLQ" : 6},
                           "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                             "ALQ" : 5, "GLQ" : 6},
                           "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                           "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                           "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                           "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5,
                                           "Min2" : 6, "Min1" : 7, "Typ" : 8},
                           "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                           "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                           "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                           "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                           "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                           "Street" : {"Grvl" : 1, "Pave" : 2},
                           "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                         )




alls["lat"] = alls.Neighborhood.replace({'Blmngtn' : 42.062806,
                                           'Blueste' : 42.009408,
                                            'BrDale' : 42.052500,
                                            'BrkSide': 42.033590,
                                            'ClearCr': 42.025425,
                                            'CollgCr': 42.021051,
                                            'Crawfor': 42.025949,
                                            'Edwards': 42.022800,
                                            'Gilbert': 42.027885,
                                            'GrnHill': 42.000854,
                                            'IDOTRR' : 42.019208,
                                            'Landmrk': 42.044777,
                                            'MeadowV': 41.991866,
                                            'Mitchel': 42.031307,
                                            'NAmes'  : 42.042966,
                                            'NoRidge': 42.050307,
                                            'NPkVill': 42.050207,
                                            'NridgHt': 42.060356,
                                            'NWAmes' : 42.051321,
                                            'OldTown': 42.028863,
                                            'SWISU'  : 42.017578,
                                            'Sawyer' : 42.033611,
                                            'SawyerW': 42.035540,
                                            'Somerst': 42.052191,
                                            'StoneBr': 42.060752,
                                            'Timber' : 41.998132,
                                            'Veenker': 42.040106})

alls["lon"] = alls.Neighborhood.replace({'Blmngtn' : -93.639963,
                                           'Blueste' : -93.645543,
                                            'BrDale' : -93.628821,
                                            'BrkSide': -93.627552,
                                            'ClearCr': -93.675741,
                                            'CollgCr': -93.685643,
                                            'Crawfor': -93.620215,
                                            'Edwards': -93.663040,
                                            'Gilbert': -93.615692,
                                            'GrnHill': -93.643377,
                                            'IDOTRR' : -93.623401,
                                            'Landmrk': -93.646239,
                                            'MeadowV': -93.602441,
                                            'Mitchel': -93.626967,
                                            'NAmes'  : -93.613556,
                                            'NoRidge': -93.656045,
                                            'NPkVill': -93.625827,
                                            'NridgHt': -93.657107,
                                            'NWAmes' : -93.633798,
                                            'OldTown': -93.615497,
                                            'SWISU'  : -93.651283,
                                            'Sawyer' : -93.669348,
                                            'SawyerW': -93.685131,
                                            'Somerst': -93.643479,
                                            'StoneBr': -93.628955,
                                            'Timber' : -93.648335,
                                            'Veenker': -93.657032})
alls["lon"] = preprocessing.scale(alls["lon"])
alls["lat"] = preprocessing.scale(alls["lat"])

train1 = alls[:train_shape]
test1 = alls[train_shape:]
      
#heatmap of correlation
cor_train=alls[:train_shape]
cor_train["SalePrice"]=train["SalePrice"]
plt.figure(figsize=(12, 8))
sns.heatmap(cor_train.corr())      


collist = set(NAcheck(train1) + NAcheck(test1))
train1 = train1.drop(collist, axis = 1)
test1 = test1.drop(collist, axis = 1)

train1 = train1.drop(catcheck(train1), axis = 1)
test1 = test1.drop(catcheck(test1), axis = 1)

print(train1.columns)

X = train1.values
X_test = test1.values



#kernel = 1*RBF(length_scale=1.0)
kernel = 1.0**2 * Matern(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0), nu=0.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=5e-9, optimizer='fmin_l_bfgs_b', 
                                n_restarts_optimizer=0, normalize_y=False, copy_X_train=True,
                                random_state=2016)
clf = Pipeline([('scaler', StandardScaler()), ('gp', gp)])     
y_log_centered = y_log - y_log.mean()

y_pred = cross_val_predict(clf, X, y_log_centered, cv=5, n_jobs=-1)
#score=cross_val_score(clf, X, y_log_centered, cv=5, n_jobs=-1)
'''
y = np.expm1(y_log)
y_pred = np.expm1(y_pred + y_log.mean())
score = rmsle(y,y_pred)
print(score) # 0.13096

    
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)





plt.scatter(y_pred, y)
plt.plot([min(y_pred),max(y_pred)], [min(y_pred),max(y_pred)], ls="--", c=".3")
plt.xlim([0,800000])
plt.ylim([0,800000])
plt.xlabel("Prediction")
plt.ylabel("Ground Truth")
plt.show()



clf.fit(X,y_log_centered)
prediction = clf.predict(X_test)

print(clf.steps[1][1].kernel_)


prediction = np.expm1(prediction + y_log.mean())


def write_to_csv(output,score):
    import datetime
    import csv
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    prediction_file_object = csv.writer(f)
    prediction_file_object.writerow(["Id","SalePrice"])  # don't forget the headers

    for i in range(len(test)):
        prediction_file_object.writerow([test["Id"][test.index[i]], (output[i])])
        
        
write_to_csv(prediction, score)
