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
# select the column to exclude from the analysis and variable to be predicted
col2exclude = ['Respondent', 'ExpectedSalary', 'Salary',"Currency"]
var2predict = 'Salary'

X, y = dt_prep.clean_data(df,var2predict,col2exclude) 
print(X.shape)
print(y.shape)

# the optimal model will be selected based on the relevance of each varaible, checking how many non-zero obs there are in
cutoff=4000
print(np.where(X.sum() > cutoff))
reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]

# model convergence to the right one
cutoffs = range(5000,100,-100)

r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test = dt_prep.find_optimal_lm_mod(X, y, cutoffs)

coef_df = dt_prep.coef_weights(lm_model=lm_model,X_train= X_train)

chart = coef_df.iloc[1:30,].style.bar(subset=['coefs'], align='mid', color=['#d65f5f', '#5fba7d'])
dfi.export(chart, "styled2.png",max_rows=30)

# visualisation of the output
y_test_preds = lm_model.predict(X_test)

preds_vs_act = pd.DataFrame(np.hstack([y_test.values.reshape(y_test.size,1), y_test_preds.reshape(y_test.size,1)]))
preds_vs_act.columns = ['actual', 'preds']
preds_vs_act['diff'] = preds_vs_act['actual'] - preds_vs_act['preds']

plt.plot(preds_vs_act['preds'], preds_vs_act['diff'], 'bo');
plt.xlabel('predicted');
plt.ylabel('difference');

