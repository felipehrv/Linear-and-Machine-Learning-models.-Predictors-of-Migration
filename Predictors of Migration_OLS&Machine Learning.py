#!/usr/bin/env python
# coding: utf-8

# In[161]:


import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import urllib
import requests

from io import StringIO

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('fivethirtyeight')


# # Created by: Felipe Rodríguez.
# ### Purpose: run Linear and Machine Learning models on the predictors of migration.

# ### Data 
# 
# We use different datasets from the Worl Bank and the Migration Policy Institute. We use variables that measure economic performance, inequality, life expectancy, popultion, remittances and aid. We analyze only the three countries of the northern triangle; El Salvador, Guatemala and Honduras. These countries have the higher levels of emigration to the U.S. after Mexico in Latin America.
# 
# World bank data: governance, trade, labor conditions, economic performance, education and life expectancy
# 
# Migration Policy Institute: Migration the countries of the northern triangle of Central America
# 
# Time Series from 1980-2015 

# ### Models and robustness checks
# 
# OLS, Lasso, Elastic Net, Regression Tree, Random Forrest, PCA and KNN

# ### Findings
# 
# Our dependent variable has missing values, this may affect our results in the different estimations we are running. However, migration data from these countries is scarce and we are interested to analyze this topic even with the issues described. The results are enlightening, anyway.
# 
# It is fair to say, this is merely an exploratory attempt to compare Linear and Machine Learning models on the predictors of migration in the Northern Triangle of Central America. Further analysis should make efforts to use better secondary data and fix the collinearity issues described below. 
# 
# The best predictors according to the different models conducted are related to indicators of economic performance, population growth, remittances, trade, death rates and inequality.

# # 1. Data Cleaning

# Here we read, revise and clean each of the datasets separately and then we merge them to choose our main variables. We also drop variables with too many missing observations. For the rest of the variables we replace the missing values with the mean. Finally, we convert the variables to more appropriate formats. 

# In[162]:


mig1 = pd.read_csv('/User directory+/MPI-Data-Hub-Region-birth_1960-2015_1.csv')


# In[163]:


mig2 = pd.read_csv('/User directory+/Data_Extract_From_Education_Statistics_-_All_Indicators.csv')


# In[164]:


mig3 = pd.read_csv('/User directory+/Data_Extract_From_Poverty_and_Equity_Database-3.csv')


# In[165]:


mig4 = pd.read_csv('/User directory+/Data_Extract_From_Health_Nutrition_and_Population_Statistics_Population_estimates_and_projections.csv')


# In[166]:


mig5 = pd.read_csv('/User directory+/Data_Extract_From_World_Development_Indicators-9.csv')


# In[167]:


mig1.info()


# In[168]:


mig1


# In[169]:


print mig1.isnull().sum()


# ### Replacing missing values with the mean of each decade

# In[170]:


mig1.ix[mig1.Year==1981, 'Migration'] = 1.34203879
mig1.ix[mig1.Year==1982, 'Migration'] = 1.34203879
mig1.ix[mig1.Year==1983, 'Migration'] = 1.34203879
mig1.ix[mig1.Year==1984, 'Migration'] = 1.34203879
mig1.ix[mig1.Year==1985, 'Migration'] = 1.34203879
mig1.ix[mig1.Year==1986, 'Migration'] = 1.34203879
mig1.ix[mig1.Year==1987, 'Migration'] = 1.34203879
mig1.ix[mig1.Year==1988, 'Migration'] = 1.34203879
mig1.ix[mig1.Year==1989, 'Migration'] = 1.34203879

mig1.ix[mig1.Year==1991, 'Migration'] = 4.516020138
mig1.ix[mig1.Year==1992, 'Migration'] = 4.516020138
mig1.ix[mig1.Year==1993, 'Migration'] = 4.516020138
mig1.ix[mig1.Year==1994, 'Migration'] = 4.516020138
mig1.ix[mig1.Year==1995, 'Migration'] = 4.516020138
mig1.ix[mig1.Year==1996, 'Migration'] = 4.516020138
mig1.ix[mig1.Year==1997, 'Migration'] = 4.516020138
mig1.ix[mig1.Year==1998, 'Migration'] = 4.516020138
mig1.ix[mig1.Year==1999, 'Migration'] = 4.516020138

mig1.ix[mig1.Year==2001, 'Migration'] = 7.56872769
mig1.ix[mig1.Year==2002, 'Migration'] = 7.56872769
mig1.ix[mig1.Year==2003, 'Migration'] = 7.56872769
mig1.ix[mig1.Year==2004, 'Migration'] = 7.56872769
mig1.ix[mig1.Year==2005, 'Migration'] = 7.56872769


# In[171]:


mig1


# In[172]:


mig1.rename(columns={
        'Country of Birth': 'Country'
        }, inplace=True)


# In[173]:


mig1


# In[174]:


mig1.info()


# In[175]:


mig2


# In[176]:


del mig2['Time Code']
del mig2['Country Code']


# In[177]:


mig2


# In[178]:


mig2.info()


# In[179]:


mig2.rename(columns={
        'Time': 'Year',
        'Enrolment in tertiary education per 100,000 inhabitants, both sexes [UIS.TE_100000.56]': 'enrolment_tertiary',
        'GDP per capita (constant 2005 US$) [NY.GDP.PCAP.KD]': 'GDP_percapita_constant',
        'Population, ages 0-14 (% of total) [SP.POP.0014.TO.ZS]': 'pop_ages_0-14%',
        'Population, ages 15-64 (% of total) [SP.POP.1564.TO.ZS]': 'pop_ages_14-64%',
        'Primary completion rate, both sexes (%) [SE.PRM.CMPT.ZS]': 'primary_completion'
        }, inplace=True)


# In[180]:


mig2


# In[181]:


mig2['primary_completion'] = pd.to_numeric(mig2['primary_completion'], errors='coerce')


# In[182]:


mig2['enrolment_tertiary'] = pd.to_numeric(mig2['enrolment_tertiary'], errors='coerce')


# In[183]:


mig2


# In[184]:


mig2.info()


# In[185]:


mig2['enrolment_tertiary'].fillna(np.mean(mig2['enrolment_tertiary']), inplace=True)


# In[186]:


mig2['primary_completion'].fillna(np.mean(mig2['primary_completion']), inplace=True)


# In[187]:


mig2


# In[188]:


mig3


# In[189]:


del mig3['Year Code']
del mig3['Country Code']
del mig3['Income share held by fourth 20% [SI.DST.04TH.20]']
del mig3['Income share held by highest 20% [SI.DST.05TH.20]']
del mig3['Income share held by lowest 20% [SI.DST.FRST.20]']
del mig3['Income share held by second 20% [SI.DST.02ND.20]']
del mig3['Income share held by third 20% [SI.DST.03RD.20]']


# In[190]:


mig3.info()


# In[191]:


mig3.rename(columns={
        'GINI index (World Bank estimate) [SI.POV.GINI]': 'gini',
        'Income share held by highest 10% [SI.DST.10TH.10]': 'income_highest%',
        'Income share held by lowest 10% [SI.DST.FRST.10]': 'income_lowest%',
        'Number of poor at $1.90 a day (2011 PPP) (millions) [SI.POV.NOP1]': 'poor_1.90',
        'Number of poor at $3.10 a day (2011 PPP) (millions) [SI.POV.NOP2]': 'poor_3.10',
        'Poverty gap at $1.90 a day (2011 PPP) (%) [SI.POV.GAPS]': 'poverty_gap_1.90',
        'Poverty gap at $3.10 a day (2011 PPP) (%) [SI.POV.GAP2]': 'poverty_gap_3.10',
        'Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population) [SI.POV.DDAY]': 'poverty_headcount_1.90',
        'Poverty headcount ratio at $3.10 a day (2011 PPP) (% of population) [SI.POV.2DAY]': 'poverty_headcount_3.10'
        }, inplace=True)


# In[192]:


mig3.info()


# In[193]:


mig3['gini'] = pd.to_numeric(mig3['gini'], errors='coerce')
mig3['income_highest%'] = pd.to_numeric(mig3['income_highest%'], errors='coerce')
mig3['income_lowest%'] = pd.to_numeric(mig3['income_lowest%'], errors='coerce')
mig3['poor_1.90'] = pd.to_numeric(mig3['poor_1.90'], errors='coerce')
mig3['poor_3.10'] = pd.to_numeric(mig3['poor_3.10'], errors='coerce')
mig3['poverty_gap_1.90'] = pd.to_numeric(mig3['poverty_gap_1.90'], errors='coerce')
mig3['poverty_gap_3.10'] = pd.to_numeric(mig3['poverty_gap_3.10'], errors='coerce')
mig3['poverty_headcount_1.90'] = pd.to_numeric(mig3['poverty_headcount_1.90'], errors='coerce')
mig3['poverty_headcount_3.10'] = pd.to_numeric(mig3['poverty_headcount_3.10'], errors='coerce')


# In[194]:


mig3


# In[195]:


mig3['gini'].fillna(np.mean(mig3['gini']), inplace=True)
mig3['income_highest%'].fillna(np.mean(mig3['income_highest%']), inplace=True)
mig3['income_lowest%'].fillna(np.mean(mig3['income_lowest%']), inplace=True)
mig3['poor_1.90'].fillna(np.mean(mig3['poor_1.90']), inplace=True)
mig3['poor_3.10'].fillna(np.mean(mig3['poor_3.10']), inplace=True)
mig3['poverty_gap_1.90'].fillna(np.mean(mig3['poverty_gap_1.90']), inplace=True)
mig3['poverty_gap_3.10'].fillna(np.mean(mig3['poverty_gap_3.10']), inplace=True)
mig3['poverty_headcount_1.90'].fillna(np.mean(mig3['poverty_headcount_1.90']), inplace=True)
mig3['poverty_headcount_3.10'].fillna(np.mean(mig3['poverty_headcount_3.10']), inplace=True)


# In[196]:


mig3


# In[197]:


mig4


# In[198]:


mig4.info()


# In[199]:


del mig4['Time Code']
del mig4['Country Code']


# In[200]:


mig4.info()


# In[201]:


mig4.rename(columns={
        'Time': 'Year',
        'Country Name': 'Country',
        'Age dependency ratio (% of working-age population) [SP.POP.DPND]': 'age_dependency',
        'Birth rate, crude (per 1,000 people) [SP.DYN.CBRT.IN]': 'birth_rate',
        'Death rate, crude (per 1,000 people) [SP.DYN.CDRT.IN]': 'death_rate',
        'Fertility rate, total (births per woman) [SP.DYN.TFRT.IN]': 'fertility_rate',
        'Life expectancy at birth, total (years) [SP.DYN.LE00.IN]': 'life_expectancy'
        }, inplace=True)


# In[202]:


mig4


# In[203]:


mig5.info()


# In[204]:


del mig5['Time Code']
del mig5['Country Code']
del mig5['Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population) [SI.POV.DDAY]']
del mig5['Poverty headcount ratio at $3.10 a day (2011 PPP) (% of population) [SI.POV.2DAY]']
del mig5['Final consumption expenditure, etc. (% of GDP) [NE.CON.TETC.ZS]']


# In[205]:


mig5.rename(columns={
        'Time': 'Year',
        'Country Name': 'Country',
        'Unemployment, total (% of total labor force) (national estimate) [SL.UEM.TOTL.NE.ZS]': 'unemployment',
        'Trade (% of GDP) [NE.TRD.GNFS.ZS]': 'trade',
        'Short-term debt (% of total external debt) [DT.DOD.DSTC.ZS]': 'short_term_debt',
        'Population growth (annual %) [SP.POP.GROW]': 'pop_growth',
        'Personal remittances, received (% of GDP) [BX.TRF.PWKR.DT.GD.ZS]': 'remittances',
        'Net bilateral aid flows from DAC donors, United States (current US$) [DC.DAC.USAL.CD]': 'net_bilateral_aid',
        'Imports of goods and services (% of GDP) [NE.IMP.GNFS.ZS]': 'imports_%GDP',
        'General government final consumption expenditure (% of GDP) [NE.CON.GOVT.ZS]': 'gov_consumption',
        'Foreign direct investment, net outflows (% of GDP) [BM.KLT.DINV.WD.GD.ZS]': 'FDI',
        'Exports of goods and services (% of GDP) [NE.EXP.GNFS.ZS]': 'exports_%GDP',
        'Employment to population ratio, 15+, total (%) (national estimate) [SL.EMP.TOTL.SP.NE.ZS]': 'employment_15+'
        }, inplace=True)


# In[206]:


mig5


# In[207]:


mig6 = mig5.drop([108,109,110,111,112,113,114,115])


# In[208]:


mig6['unemployment'] = pd.to_numeric(mig6['unemployment'], errors='coerce')
mig6['trade'] = pd.to_numeric(mig6['trade'], errors='coerce')
mig6['short_term_debt'] = pd.to_numeric(mig6['short_term_debt'], errors='coerce')
mig6['pop_growth'] = pd.to_numeric(mig6['pop_growth'], errors='coerce')
mig6['remittances'] = pd.to_numeric(mig6['remittances'], errors='coerce')
mig6['net_bilateral_aid'] = pd.to_numeric(mig6['net_bilateral_aid'], errors='coerce')
mig6['imports_%GDP'] = pd.to_numeric(mig6['imports_%GDP'], errors='coerce')
mig6['gov_consumption'] = pd.to_numeric(mig6['gov_consumption'], errors='coerce')
mig6['FDI'] = pd.to_numeric(mig6['FDI'], errors='coerce')
mig6['exports_%GDP'] = pd.to_numeric(mig6['exports_%GDP'], errors='coerce')
mig6['employment_15+'] = pd.to_numeric(mig6['employment_15+'], errors='coerce')


# In[209]:


mig6['unemployment'].fillna(np.mean(mig6['unemployment']), inplace=True)
mig6['FDI'].fillna(np.mean(mig6['FDI']), inplace=True)
mig6['employment_15+'].fillna(np.mean(mig6['employment_15+']), inplace=True)


# In[210]:


mig6


# In[211]:


mig6.info()


# In[212]:


mig6.Year = [int(float(x)) for x in mig6.Year]


# In[213]:


mig6.info()


# In[214]:


migm1 = pd.merge(mig1, mig2, on=['Year', 'Country'], how='right')


# In[215]:


migm1


# In[216]:


migm2 = pd.merge(migm1, mig3, on=['Year', 'Country'], how='right')


# In[217]:


migm2


# In[218]:


migm3 = pd.merge(migm2, mig4, on=['Year', 'Country'], how='right')


# In[219]:


migm3


# In[220]:


migration_flows = pd.merge(migm3, mig6, on=['Year', 'Country'], how='right')


# In[221]:


migration_flows


# ### We have finally merged and cleaned our datasets. Here we can start with the analysis.

# In[222]:


migration_flows.info()


# # 2. Checking for correlations accross our variables

# In[223]:


fig = plt.figure(figsize=(18,10))
ax = fig.gca()
sns.heatmap(migration_flows.corr(), annot=True, linewidths=.15, cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
plt.show()


# In[224]:


migration_flows.info()


# # Model 

# ### We drop variables with high correlation

# In[228]:


del migration_flows['enrolment_tertiary']
del migration_flows['pop_ages_0-14%']
del migration_flows['pop_ages_14-64%']
del migration_flows['primary_completion']
del migration_flows['gini']
del migration_flows['poor_1.90']
del migration_flows['poor_3.10']
del migration_flows['poverty_gap_1.90']
del migration_flows['poverty_gap_3.10']
del migration_flows['poverty_headcount_3.10']
del migration_flows['age_dependency']
del migration_flows['birth_rate']
del migration_flows['fertility_rate']
del migration_flows['life_expectancy']
del migration_flows['short_term_debt']
del migration_flows['gov_consumption']
del migration_flows['employment_15+']
del migration_flows['imports_%GDP']
del migration_flows['exports_%GDP']


# In[229]:


fig = plt.figure(figsize=(18,10))
ax = fig.gca()
sns.heatmap(migration_flows.corr(), annot=True, linewidths=.15, cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
plt.show()


# In[230]:


migration_flows.info()


# ### We define a new variable for time variation

# In[231]:


migration_flows['Year'] = pd.to_datetime(migration_flows['Year'])


# In[232]:


migration_flows.info()


# ### We have reduced the correlation in our dataset, now we want to see which are our best predictors for migration. We revise our variables one last time.

# In[233]:


fig = plt.figure(figsize=(18,10))
ax = fig.gca()
sns.heatmap(migration_flows.corr(), annot=True, linewidths=.15, cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
plt.show()


# In[234]:


sns.pairplot(migration_flows, hue="Country", plot_kws={"s": 25}, size = 3)


# # 3. Predictions

# In[235]:


y = migration_flows.Migration.values
x = migration_flows[['GDP_percapita_constant', 'income_highest%', 'income_lowest%', 'poverty_headcount_1.90', 'death_rate', 'unemployment', 'trade', 'pop_growth', 'remittances', 'net_bilateral_aid', 'FDI']]


# In[236]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Xn = ss.fit_transform(x)


# In[237]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Xn, y, test_size=0.3, random_state=10)
print x_train.shape, x_test.shape
print "\n======\n"
print y_train.shape, y_test.shape


# # 4. Linear and Maching Learning Models

# ### 4.1 OLS

# In[238]:


from sklearn.linear_model import LinearRegression

## define a linear regression model
lr = LinearRegression()

## fit your model
lr.fit(x_train, y_train)


# In[239]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# In[240]:


''' Function that calls the MSE and R^2 at once, using the name of the method and calling the best model'''

def rsquare_meansquare_error(train_y, test_y, train_x, test_x, test, best_model):
    """ first we need to predict on the test and train data"""
    y_train_pred = best_model.predict(train_x)
    y_test_pred = best_model.predict(test_x)
    
    """ We call the MSE in the following lines"""
    print ('MSE ' + test + ' train data: %.2f, test data: %.2f' % (
        mean_squared_error(train_y, y_train_pred),
        mean_squared_error(test_y, y_test_pred)))
    
    """ We call the R^2 in the following lines"""
    print('R^2 ' + test + ' train data: %.2f, test data: %.2f' % (
        r2_score(train_y, y_train_pred),
        r2_score(test_y, y_test_pred)))


# In[241]:


rsquare_meansquare_error(y_train, y_test, x_train, x_test, "OLS", lr)


# ### 4.2 Regularization

# In[242]:


from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV


# ### 4.2.1 Ridge

# In[243]:


## Find the optimal alpha
ridge_alphas = np.logspace(0, 5, 100)
optimal_ridge = RidgeCV(alphas=ridge_alphas, cv=10)
optimal_ridge.fit(x_train, y_train)
print (optimal_ridge.alpha_)


# In[244]:


## Implement the Ridge Regression
ridge = Ridge(alpha=optimal_ridge.alpha_)

## Fit the Ridge regression
ridge.fit(x_train, y_train)


# In[245]:


## Evaluate the Ridge Regression
rsquare_meansquare_error(y_train, y_test, x_train, x_test, "Ridge", ridge)


# ### 4.2.2 Lasso

# In[246]:


## Find the optimal alpha
optimal_lasso = LassoCV(n_alphas=300, cv=10, verbose=1)
optimal_lasso.fit(x_train, y_train)
print optimal_lasso.alpha_


# In[247]:


## Implement the Lasso Regression
lasso = Lasso(alpha=optimal_lasso.alpha_)

## fit your regression
lasso.fit(x_train, y_train)


# In[248]:


## Evaluate the Lasso Regression
rsquare_meansquare_error(y_train, y_test, x_train, x_test, "Lasso", lasso)


# ### 4.2.3 Elastic Net

# In[249]:


## Find the optimal alphas
l1_ratios = np.linspace(0.01, 1.0, 50)
optimal_enet = ElasticNetCV(l1_ratio=l1_ratios, n_alphas=300, cv=5, verbose=1)
optimal_enet.fit(x_train, y_train)
print optimal_enet.alpha_
print optimal_enet.l1_ratio_


# In[250]:


##  Create a model Enet
enet = ElasticNet(alpha=optimal_enet.alpha_, l1_ratio=optimal_enet.l1_ratio_)

## Fit your model
enet.fit(x_train, y_train)


# In[251]:


## Evaluate the Elastic Net Regression
rsquare_meansquare_error(y_train, y_test, x_train, x_test, "Elastic Net", enet)


# ### Best Predictors

# In[252]:


''' Here I am defining a function to print the coefficients, their absolute values and the non-absolute values'''
def best_reg_method(x, best_regulari):
    method_coefs = pd.DataFrame({'variable':x.columns, 
                                 'coef':best_regulari.coef_, 
                                 'abs_coef':np.abs(best_regulari.coef_)})
    method_coefs.sort_values('abs_coef', inplace=True, ascending=False)
    '''you can change the number inside head to display more or less variables'''
    return method_coefs.head(10)


# In[253]:


best_reg_method(x, ridge)


# ### With ridge our best predictors are related to economic performance, inequality , population growth, death rates and trade

# In[254]:


best_reg_method(x, lasso)


# ### We find similar results with lasso

# In[255]:


best_reg_method(x, enet)


# ### We find similar results with elastic net

# ### 4.3 Regression Tree

# In[256]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()

## Here is the gridsearch
params = {"max_depth": [3,5,10,20],
          "max_features": [None, "auto"],
          "min_samples_leaf": [1, 3, 5, 7, 10],
          "min_samples_split": [2, 5, 7],
           "criterion" : ['mse']
         }

# ## Here crossvalidate 
from sklearn.grid_search import GridSearchCV
dtr_gs = GridSearchCV(dtr, params, n_jobs=-1, cv=5, verbose=1)


# In[257]:


## Fit the regresion tree
dtr_gs.fit(x_train, y_train)


# In[258]:


## Print Best Estimator, parameters and score
''' dtr_best = is the regression tree regressor with best parameters/estimators'''
dtr_best = dtr_gs.best_estimator_ 

print "best estimator", dtr_best
print "\n==========\n"
print "best parameters",  dtr_gs.best_params_
print "\n==========\n"
print "best score", dtr_gs.best_score_


# In[259]:


##features that best explain your Y
''' Here I am defining a function to print feature importance using best models'''
def feature_importance(x, best_model):
    feature_importance = pd.DataFrame({'feature':x.columns, 'importance':best_model.feature_importances_})
    feature_importance.sort_values('importance', ascending=False, inplace=True)
    return feature_importance  


# In[260]:


feature_importance(x, dtr_best)


# ### With regression tree, the findings suggest that remmitances and billateral aid are also relevant predictors.

# In[261]:


## Predict 
y_pred_dtr= dtr_best.predict(x_test)
y_pred_dtr


# In[262]:


## Evaluate the Regression Tree performance on your train and test data
rsquare_meansquare_error(y_train, y_test, x_train, x_test, "Regression tree", dtr_best)


# ### Our R2 is too high we may still have some isssues with correlation among our predictors

# ### 4.4 Random Forrest

# In[263]:


from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor( )

params = {'max_depth':[3,4,5],  
          'max_leaf_nodes':[5,6,7], 
          'min_samples_split':[3,4],
          'n_estimators': [100]
         }

estimator_rfr = GridSearchCV(forest, params, n_jobs=-1,  cv=5,verbose=1)


# In[264]:


## Fit your random forest tree
estimator_rfr.fit(x_train, y_train)


# In[265]:


## Print the best estimator, parameters and score
''' rfr_best = is the random forest regression tree regressor with best parameters/estimators'''
rfr_best = estimator_rfr.best_estimator_
print "best estimator", rfr_best
print "\n==========\n"
print "best parameters", estimator_rfr.best_params_
print "\n==========\n"
print "best score", estimator_rfr.best_score_


# In[266]:


## Print the feauure importance
feature_importance(x, rfr_best)


# ### Remittances might have some colinearity with the levels of migration and that might be overestimating our R2. Economic performance, population, death rates and inequality are important according to our results.

# In[267]:


## Predict
y_pred_rfdtr= rfr_best.predict(x_test)
y_pred_rfdtr


# In[268]:


## Evaluate your model
rsquare_meansquare_error(y_train, y_test, x_train, x_test, "Random Forest Regression tree", rfr_best)


# ### Our findings with random forrest seems to match our expectations about the best predictors from an intuitively point of view. Even though our R2 is still too high, it suggests that remittances might be causing collinearity in our model, based on its high relevance as a predictor.

# # 5. PCA

# ### Now we are going to use PCA and KNN in our estimations.

# In[269]:


print migration_flows['Migration'].mean()


# In[270]:


migration_flows['threshold'] = migration_flows['Migration'] >= 5.86008409129


# In[271]:


migration_flows['threshold']= migration_flows['threshold'].apply(lambda x: 1 if x== True else 0)


# In[272]:


migration_flows


# In[273]:


migration_flows.threshold.value_counts()


# ### Subset of the data without the threshold

# In[274]:


migration_cont = migration_flows.iloc[:, 3:-1]


# In[275]:


migration_cont


# ### Standardize the variables

# In[276]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
migration_cont_n = ss.fit_transform(migration_cont)
migration_cont_n


# ### Fit a PCA on the standardized data

# In[277]:


## Fit the PCA and print the components
from sklearn.decomposition import PCA
migration_pca = PCA().fit(migration_cont_n)
print "Number of PCA components is: \n", migration_pca.n_components_
print "\n======\n"
print "List of PCA components is:\n", migration_pca.components_


# In[278]:


## transform  => Apply dimensionality reduction to X.
migration_pcs = migration_pca.transform(migration_cont_n)
migration_pcs


# In[279]:


## Now create the dataframe
migration_pcs = pd.DataFrame(migration_pcs, columns=['PC'+str(i) for i in range(1, migration_pcs.shape[1]+1)])
migration_pcs['threshold'] = migration_flows.threshold


# In[280]:


migration_pcs


# ### Plot the variance explained by the ratio of the components

# In[281]:


fig, ax = plt.subplots(figsize=(8,6))
ax.plot(range(1, migration_cont.shape[1]+1), migration_pca.explained_variance_ratio_, lw=2)
ax.scatter(range(1, migration_cont.shape[1]+1), migration_pca.explained_variance_ratio_, s=100)
ax.set_title('migration data: explained variance of components')
ax.set_xlabel('principal component')
ax.set_ylabel('explained variance')
plt.show()


# ### Print out the component weights with their corresponding variables for PC1, PC2, and PC3

# In[282]:


for col, comp in zip(migration_cont.columns, migration_pca.components_[0]):
    print col, comp


# In[283]:


for col, comp in zip(migration_cont.columns, migration_pca.components_[1]):
    print col, comp


# In[284]:


for col, comp in zip(migration_cont.columns, migration_pca.components_[3]):
    print col, comp


# ### Plot a seaborn pairplot of PC1, PC2, and PC3 with `hue='threshold'`

# In[285]:


sns.pairplot(data=migration_pcs, vars=['PC1','PC2','PC3'], hue='threshold', size=3)
plt.show()


# ### Horn's parallel analysis

# In[286]:


def horn_parallel_analysis(shape, iters=1000, percentile=95):
    pca = PCA(n_components=shape[1])
    eigenvals = []
    for i in range(iters):
        rdata = np.random.normal(0,1,size=shape)
        pca.fit(rdata)
        eigenvals.append(pca.explained_variance_)
    eigenvals = np.array(eigenvals)
    return np.percentile(eigenvals, percentile, axis=0)


# ### Run parallel analysis for the migration data

# In[287]:


migration_pa = horn_parallel_analysis(migration_cont.shape, percentile=95)
migration_pa


# ### Plot the wine eigenvalues (`.variance_explained_`) against the parallel analysis random eigenvalue cutoffs

# In[288]:


fig, ax = plt.subplots(figsize=(8,6))

ax.plot(range(1, migration_cont.shape[1]+1), migration_pca.explained_variance_, lw=2)
ax.scatter(range(1, migration_cont.shape[1]+1), migration_pca.explained_variance_, s=50)

ax.plot(range(1, len(migration_pa)+1), migration_pa, lw=2, color='darkred')
ax.scatter(range(1, len(migration_pa)+1), migration_pa, s=40, color='darkred')


ax.set_title('Horns parallel analysis on migration data components')
ax.set_xlabel('principal component')
ax.set_ylabel('eigenvalue')
plt.show()


# 
# ### Predict "threshold" from original data and from PCA

# In[289]:


## Explore the noise on the original data
## should you Standarized the data? 
## http://stats.stackexchange.com/questions/48360/is-standardization-needed-before-fitting-logistic-regression
sns.pairplot(data=migration_flows, hue='threshold')
plt.show()


# In[290]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split


# In[291]:


## Define your x and y
columns_ = migration_flows.columns.tolist()
exclude_cols = ['threshold', 'Country', 'Year']
y = migration_flows.threshold.values
X = migration_flows[[i for i in columns_ if i not in exclude_cols]]
X = X.values


# In[292]:


knn = KNeighborsClassifier()

params = {
    'n_neighbors':range(1,20),
    'weights':['uniform','distance']
}

knn_gs = GridSearchCV(knn, params, cv=5, verbose=1)
knn_gs.fit(X, y)

print knn_gs.best_params_
best_knn = knn_gs.best_estimator_


# In[293]:


cv_indices = StratifiedKFold(y, n_folds=5)
## StratifiedKFold = Provides train/test indices to split data in train/test sets.
## http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

logreg = LogisticRegression()

lr_scores_test = []
lr_scores_train = []

knn_scores_test = []
knn_scores_train = []

for train_inds, test_inds in cv_indices:
    
    Xtr, ytr = X[train_inds, :], y[train_inds]
    Xte, yte = X[test_inds, :], y[test_inds]
    print 'Xtrain and ytrain shapes:\n', Xtr.shape, ytr.shape
    print 'Xtest and ytest shapes:\n', Xte.shape, yte.shape

    
    best_knn.fit(Xtr, ytr)
    knn_scores_test.append(best_knn.score(Xte, yte))
    knn_scores_train.append(best_knn.score(Xtr, ytr))
    '''best_knn.score = Returns the mean accuracy on the given test data and labels'''
    y_knn_predict  = best_knn.predict(Xte)
    
    
    logreg.fit(Xtr, ytr)
    lr_scores_test.append(logreg.score(Xte, yte))
    lr_scores_train.append(logreg.score(Xtr, ytr))
    '''logreg.score = Returns the mean accuracy on the given test data and labels'''
    y_log_predict = logreg.predict(Xte)
    
    
print "\n======\n"
print 'KNN accuracy scores on test:\n', knn_scores_test
print 'KNN mean of accuracy scores on test:\n', np.mean(knn_scores_test)
print 'KNN mean of accuracy scores on train :\n', np.mean(knn_scores_train)
print "\n======\n"
print 'Logistic Regression accuracy scores on test:\n', lr_scores_test
print 'Logistic Regression mean of accuracy scores on test:\n', np.mean(lr_scores_test)
print 'Logistic Regression mean of accuracy scores on train:\n', np.mean(lr_scores_train)

print "\n======\n"
print 'Baseline accuracy:\n ', np.mean(y)


# ### We found very similar results with the KNN and logistic estimation

# In[294]:


## Define your x and y
## For your X = only use the number of PCA's that have the greatest explanatory power

columns_ = migration_pcs.columns.tolist()
exclude_cols = ['Year', 'Country', 'PC5','PC6','PC7','PC8','PC9','PC10','PC11', 'threshold']

ypc = migration_pcs.threshold.values

Xpc = migration_pcs[[i for i in columns_ if i not in exclude_cols]]
Xpc = Xpc.values


# ### Perform stratified cross-validation on a KNN classifier and logisitic regression.

# In[295]:


knn = KNeighborsClassifier()

params = {
    'n_neighbors':range(1,20),
    'weights':['uniform','distance']
}

knn_gs_pc = GridSearchCV(knn, params, cv=5, verbose=1)
knn_gs_pc.fit(Xpc, ypc)

print knn_gs_pc.best_params_
best_knn_pc = knn_gs_pc.best_estimator_


# In[296]:


cv_indices_pc = StratifiedKFold(ypc, n_folds=5)

logreg_pc = LogisticRegression()

lr_scores_test_pc = []
lr_scores_train_pc = []

knn_scores_test_pc = []
knn_scores_train_pc = []

for train_inds, test_inds in cv_indices_pc:
    
    Xtr_pc, ytr_pc = Xpc[train_inds, :], ypc[train_inds]
    Xte_pc, yte_pc = Xpc[test_inds, :], ypc[test_inds]
    print 'Xtrain and ytrain shapes:\n', Xtr_pc.shape, ytr_pc.shape
    print 'Xtest and ytest shapes:\n', Xte_pc.shape, yte_pc.shape

    
    best_knn_pc.fit(Xtr_pc, ytr_pc)
    knn_scores_test_pc.append(best_knn_pc.score(Xte_pc, yte_pc))
    knn_scores_train_pc.append(best_knn_pc.score(Xtr_pc, ytr_pc))
    '''best_knn.score = Returns the mean accuracy on the given test data and labels'''
    y_knn_predict_pc  = best_knn_pc.predict(Xte_pc)

    
    
    logreg_pc.fit(Xtr_pc, ytr_pc)
    lr_scores_test_pc.append(logreg_pc.score(Xte_pc, yte_pc))
    lr_scores_train_pc.append(logreg_pc.score(Xtr_pc, ytr_pc))
    '''logreg.score = Returns the mean accuracy on the given test data and labels'''
    y_log_predict_pc = logreg_pc.predict(Xte_pc)
    
print "\n======\n"
print 'KNN accuracy scores on test:\n', knn_scores_test_pc
print 'KNN mean of accuracy scores on test:\n', np.mean(knn_scores_test_pc)
print 'KNN mean of accuracy scores on train :\n', np.mean(knn_scores_train_pc)
print "\n======\n"
print 'Logistic Regression accuracy scores on test:\n', lr_scores_test_pc
print 'Logistic Regression mean of accuracy scores on test:\n', np.mean(lr_scores_test_pc)
print 'Logistic Regression mean of accuracy scores on train:\n', np.mean(lr_scores_train_pc)

print "\n======\n"
print 'Baseline accuracy:\n ', np.mean(ypc)


# ### We found more accurate results using a stratified cross validation in both KNN and the logistic estimation

# In[297]:


'''the mean of the accuracy score on the test data has a significant increase from '''
print 'KNN mean of accuracy scores on test:\n', np.mean(knn_scores_test)
print 'KNN mean of accuracy scores on test PC:\n', np.mean(knn_scores_test_pc)
print "Increase of accuracy of:", (np.mean(knn_scores_test_pc) - np.mean(knn_scores_test))


# ### We have a significant increase in accuracy doing the parallel estimation of KNN and PC

# ### Confusion Matrix for each of your classification methods.

# In[298]:


# Load Confusion Matrix 
from sklearn.metrics import confusion_matrix


# In[299]:


def confus_mat(ytrue, ypred_method, what_predict):
    what_predict = str(what_predict)
    confmat = confusion_matrix(y_true=ytrue, y_pred=ypred_method)
    confusion = pd.DataFrame(confmat, index=['is_not_' + what_predict, 'is_' + what_predict],
                         columns=['predicted_is_not_'+ what_predict, 'predicted_is_'+what_predict])
    return confusion


# In[300]:


# Load Classification Report
from sklearn.metrics import classification_report


# In[301]:


def class_report(ytrue, ypred):
    cls_rep = classification_report(yte, y_knn_predict)
    print cls_rep


# In[302]:


## Confuion Matrix for knn
confus_mat(yte, y_knn_predict, 'threshold')


# In[303]:


## Classification report for knn
class_report(yte, y_knn_predict)


# In[304]:


## Confusion Matrix for logistic
confus_mat(yte, y_log_predict, 'threshold')


# In[305]:


## Classification report for logistic
class_report(yte, y_log_predict)


# In[306]:


## Confuion Matrix for knn with PC
confus_mat(yte, y_knn_predict_pc, 'threshold')


# In[307]:


## Classification report for knn with PC
class_report(yte, y_knn_predict_pc)


# In[308]:


## Confuion Matrix for log with PC
confus_mat(yte, y_log_predict_pc, 'threshold')


# In[309]:


## Classification report for knn with PC
class_report(yte, y_log_predict_pc)


# ### Our results from our confusion matrices suggest that it is better to use the parallel estimation of KNN and PC in order to have more accurate estimations. 
