import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import AllTogether as t
import seaborn as sns
import DataPrepFunctions as dt_prep
import dataframe_image as dfi  
#matplotlib inline

df  = pd.read_csv("survey_results_public.csv")
print(df.head())

a = 'test_score'
b = 'train_score'
c = 'linear model (lm_model)'
d = 'X_train and y_train'
e = 'X_test'
f = 'y_test'
g = 'train and test data sets'
h = 'overfitting'

q1_piat = '''In order to understand how well our {} fit the dataset, 
            we first needed to split our data into {}.  
            Then we were able to fit our {} on the {}.  
            We could then predict using our {}  by providing 
            the linear model the {} for it to make predictions.  
            These predictions were for {}. 

            By looking at the {}, it looked like we were doing awesome because 
            it was 1!  However, looking at the {} suggested our model was not 
            extending well.  The purpose of this notebook will be to see how 
            well we can get our model to extend to new data.
            
            This problem where our data fits the training data well, but does
            not perform well on test data is commonly known as 
            {}.'''.format(c, g, c, d, c, e, f, b, a, h)

print(q1_piat)
t.q1_piat_answer()

col2exclude = ['Respondent', 'ExpectedSalary', 'Salary']
var2predict = 'Salary'

X, y = dt_prep.clean_data(df,var2predict,col2exclude) 
print(X.sum())

cutoff=4000
print(np.where(X.sum() > cutoff))
reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]

cutoffs = range(5000,100,-100)

r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test = dt_prep.find_optimal_lm_mod(X, y, cutoffs)

coef_df = dt_prep.coef_weights(lm_model=lm_model,X_train= X_train)

chart = coef_df.iloc[1:30,].style.bar(subset=['coefs'], align='mid', color=['#d65f5f', '#5fba7d'])
dfi.export(chart, "styled2.png",max_rows=30)