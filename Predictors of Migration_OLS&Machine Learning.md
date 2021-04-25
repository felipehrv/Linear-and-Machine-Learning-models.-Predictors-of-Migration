```python
import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import urllib
import requests

from io import StringIO

%config InlineBackend.figure_format = 'retina'
%matplotlib inline

plt.style.use('fivethirtyeight')
```

# Created by: Felipe Rodríguez.
### Purpose: run Linear and Machine Learning models on the predictors of migration.

### Data 

We use different datasets from the Worl Bank and the Migration Policy Institute. We use variables that measure economic performance, inequality, life expectancy, popultion, remittances and aid. We analyze only the three countries of the northern triangle; El Salvador, Guatemala and Honduras. These countries have the higher levels of emigration to the U.S. after Mexico in Latin America.

World bank data: governance, trade, labor conditions, economic performance, education and life expectancy

Migration Policy Institute: Migration the countries of the northern triangle of Central America

Time Series from 1980-2015 

### Models and robustness checks

OLS, Lasso, Elastic Net, Regression Tree, Random Forrest, PCA and KNN

### Findings

Our dependent variable has missing values, this may affect our results in the different estimations we are running. However, migration data from these countries is scarce and we are interested to analyze this topic even with the issues described. The results are enlightening, anyway.

It is fair to say, this is merely an exploratory attempt to compare Linear and Machine Learning models on the predictors of migration in the Northern Triangle of Central America. Further analysis should make efforts to use better secondary data and fix the collinearity issues described below. 

The best predictors according to the different models conducted are related to indicators of economic performance, population growth, remittances, trade, death rates and inequality.

# 1. Data Cleaning

Here we read, revise and clean each of the datasets separately and then we merge them to choose our main variables. We also drop variables with too many missing observations. For the rest of the variables we replace the missing values with the mean. Finally, we convert the variables to more appropriate formats. 


```python
mig1 = pd.read_csv('/User directory+/MPI-Data-Hub-Region-birth_1960-2015_1.csv')
```


```python
mig2 = pd.read_csv('/User directory+/Data_Extract_From_Education_Statistics_-_All_Indicators.csv')
```


```python
mig3 = pd.read_csv('/User directory+/Data_Extract_From_Poverty_and_Equity_Database-3.csv')
```


```python
mig4 = pd.read_csv('/User directory+/Data_Extract_From_Health_Nutrition_and_Population_Statistics_Population_estimates_and_projections.csv')
```


```python
mig5 = pd.read_csv('/User directory+/Data_Extract_From_World_Development_Indicators-9.csv')
```


```python
mig1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 108 entries, 0 to 107
    Data columns (total 3 columns):
    Year                108 non-null int64
    Country of Birth    108 non-null object
    Migration           39 non-null float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 2.6+ KB



```python
mig1
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country of Birth</th>
      <th>Migration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>2.063205</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>0.886027</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>1.076884</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>17.546949</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>5.343950</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>5.783592</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>18.448273</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>5.077445</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>6.034761</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>18.237120</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>5.240451</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>6.339264</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>19.096906</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>5.539466</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>6.338030</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>20.105788</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>5.639487</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>6.964149</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>20.886863</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>5.653971</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>6.437598</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>20.945491</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>5.586203</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>6.743448</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>20.560594</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>5.750461</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>6.798242</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>21.537939</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>5.716933</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>7.389157</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>22.073593</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>5.675817</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>7.418273</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 3 columns</p>
</div>




```python
print mig1.isnull().sum()
```

    Year                 0
    Country of Birth     0
    Migration           69
    dtype: int64


### Replacing missing values with the mean of each decade


```python
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
```


```python
mig1
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country of Birth</th>
      <th>Migration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>2.063205</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>0.886027</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>1.076884</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>17.546949</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>5.343950</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>5.783592</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>18.448273</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>5.077445</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>6.034761</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>18.237120</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>5.240451</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>6.339264</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>19.096906</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>5.539466</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>6.338030</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>20.105788</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>5.639487</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>6.964149</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>20.886863</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>5.653971</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>6.437598</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>20.945491</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>5.586203</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>6.743448</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>20.560594</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>5.750461</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>6.798242</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>21.537939</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>5.716933</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>7.389157</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>22.073593</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>5.675817</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>7.418273</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 3 columns</p>
</div>




```python
mig1.rename(columns={
        'Country of Birth': 'Country'
        }, inplace=True)
```


```python
mig1
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>Migration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>2.063205</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>0.886027</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>1.076884</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>1.342039</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>17.546949</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>5.343950</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>5.783592</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>18.448273</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>5.077445</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>6.034761</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>18.237120</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>5.240451</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>6.339264</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>19.096906</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>5.539466</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>6.338030</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>20.105788</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>5.639487</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>6.964149</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>20.886863</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>5.653971</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>6.437598</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>20.945491</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>5.586203</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>6.743448</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>20.560594</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>5.750461</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>6.798242</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>21.537939</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>5.716933</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>7.389157</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>22.073593</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>5.675817</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>7.418273</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 3 columns</p>
</div>




```python
mig1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 108 entries, 0 to 107
    Data columns (total 3 columns):
    Year         108 non-null int64
    Country      108 non-null object
    Migration    108 non-null float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 2.6+ KB



```python
mig2
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>Time Code</th>
      <th>Country</th>
      <th>Country Code</th>
      <th>Enrolment in tertiary education per 100,000 inhabitants, both sexes [UIS.TE_100000.56]</th>
      <th>GDP per capita (constant 2005 US$) [NY.GDP.PCAP.KD]</th>
      <th>Population, ages 0-14 (% of total) [SP.POP.0014.TO.ZS]</th>
      <th>Population, ages 15-64 (% of total) [SP.POP.1564.TO.ZS]</th>
      <th>Primary completion rate, both sexes (%) [SE.PRM.CMPT.ZS]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>YR1980</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>2572.813235</td>
      <td>43.742478</td>
      <td>52.756733</td>
      <td>..</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>YR1980</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>2560.782037</td>
      <td>45.444923</td>
      <td>51.602977</td>
      <td>33.9041481</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>YR1980</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>713.5259399</td>
      <td>1655.946421</td>
      <td>46.957200</td>
      <td>49.818337</td>
      <td>44.57500076</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>YR1981</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>2267.095959</td>
      <td>43.481122</td>
      <td>52.948845</td>
      <td>46.45079041</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>YR1981</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>493.2778625</td>
      <td>2509.736778</td>
      <td>45.617358</td>
      <td>51.409643</td>
      <td>33.95742035</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>YR1981</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>821.0927124</td>
      <td>1645.846419</td>
      <td>46.892259</td>
      <td>49.886066</td>
      <td>..</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>YR1982</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>999.5952759</td>
      <td>2092.554425</td>
      <td>43.204606</td>
      <td>53.154795</td>
      <td>49.3827095</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>YR1982</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>2357.368296</td>
      <td>45.771834</td>
      <td>51.239138</td>
      <td>33.8807106</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>YR1982</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>864.4645996</td>
      <td>1573.671559</td>
      <td>46.745647</td>
      <td>50.043171</td>
      <td>49.94493866</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>YR1983</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>1203.906616</td>
      <td>2094.864582</td>
      <td>42.920848</td>
      <td>53.365841</td>
      <td>50.6135788</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>YR1983</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>571.4475708</td>
      <td>2236.567544</td>
      <td>45.891347</td>
      <td>51.104597</td>
      <td>35.13737869</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>YR1983</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>868.4456787</td>
      <td>1512.185833</td>
      <td>46.554395</td>
      <td>50.246028</td>
      <td>..</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>YR1984</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>1310.496826</td>
      <td>2094.098791</td>
      <td>42.636120</td>
      <td>53.575056</td>
      <td>48.6973114</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>YR1984</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>578.3195801</td>
      <td>2189.829730</td>
      <td>45.951383</td>
      <td>51.025182</td>
      <td>36.57794952</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>YR1984</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>874.7142334</td>
      <td>1530.695403</td>
      <td>46.363681</td>
      <td>50.440498</td>
      <td>54.98490143</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>YR1985</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>1439.984375</td>
      <td>2078.900486</td>
      <td>42.346470</td>
      <td>53.785685</td>
      <td>..</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>YR1985</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>597.5586548</td>
      <td>2121.873660</td>
      <td>45.939359</td>
      <td>51.010453</td>
      <td>38.06546021</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>YR1985</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>868.4234009</td>
      <td>1547.357836</td>
      <td>46.190833</td>
      <td>50.604823</td>
      <td>..</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>YR1986</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>1492.553833</td>
      <td>2055.438830</td>
      <td>41.819358</td>
      <td>54.216902</td>
      <td>..</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>YR1986</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>626.2062378</td>
      <td>2073.066614</td>
      <td>45.963895</td>
      <td>50.941242</td>
      <td>41.00183868</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>YR1986</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>1512.507552</td>
      <td>46.064093</td>
      <td>50.698642</td>
      <td>..</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>YR1987</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>2079.844180</td>
      <td>41.318260</td>
      <td>54.619889</td>
      <td>61.98265839</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>YR1987</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>2095.342199</td>
      <td>45.884100</td>
      <td>50.970232</td>
      <td>..</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>YR1987</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>835.494751</td>
      <td>1556.855276</td>
      <td>45.935714</td>
      <td>50.785360</td>
      <td>..</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>YR1988</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>1515.808716</td>
      <td>2091.693100</td>
      <td>40.835477</td>
      <td>55.003335</td>
      <td>65.72280121</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>YR1988</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>2125.624163</td>
      <td>45.729366</td>
      <td>51.069993</td>
      <td>..</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>YR1988</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>930.5723877</td>
      <td>1581.639092</td>
      <td>45.800132</td>
      <td>50.875323</td>
      <td>..</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>YR1989</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>1568.25354</td>
      <td>2084.671422</td>
      <td>40.362708</td>
      <td>55.377693</td>
      <td>63.61275101</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>YR1989</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>2157.313890</td>
      <td>45.551915</td>
      <td>51.191567</td>
      <td>..</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>YR1989</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>925.7636719</td>
      <td>1603.219717</td>
      <td>45.642320</td>
      <td>50.989531</td>
      <td>..</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>YR2006</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>2093.922607</td>
      <td>3475.866745</td>
      <td>33.230254</td>
      <td>60.147890</td>
      <td>92.3642807</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>YR2006</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>2698.985240</td>
      <td>41.024086</td>
      <td>54.692058</td>
      <td>75.35977936</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>YR2006</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>2017.943010</td>
      <td>38.938629</td>
      <td>56.863273</td>
      <td>88.26262665</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>YR2007</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>2209.102051</td>
      <td>3597.961991</td>
      <td>32.566759</td>
      <td>60.626412</td>
      <td>97.29842377</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>YR2007</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>1695.110107</td>
      <td>2805.169791</td>
      <td>40.575993</td>
      <td>55.103794</td>
      <td>75.83374786</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>YR2007</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>2104.759589</td>
      <td>38.204492</td>
      <td>57.552528</td>
      <td>..</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>YR2008</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>2308.634277</td>
      <td>3633.014903</td>
      <td>31.905088</td>
      <td>61.110033</td>
      <td>99.28109741</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>YR2008</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>2833.735795</td>
      <td>40.091781</td>
      <td>55.548046</td>
      <td>78.54676056</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>YR2008</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>2035.134766</td>
      <td>2155.827865</td>
      <td>37.448905</td>
      <td>58.259473</td>
      <td>88.76953888</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>YR2009</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>2388.975342</td>
      <td>3509.156436</td>
      <td>31.228684</td>
      <td>61.613308</td>
      <td>102.6757813</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>YR2009</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>2787.128287</td>
      <td>39.593279</td>
      <td>55.997451</td>
      <td>82.20065308</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>YR2009</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>2068.185180</td>
      <td>36.665394</td>
      <td>58.987298</td>
      <td>91.86322784</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>YR2010</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>2484.339111</td>
      <td>3547.070983</td>
      <td>30.534690</td>
      <td>62.138073</td>
      <td>105.430397</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>YR2010</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>2805.951416</td>
      <td>39.095628</td>
      <td>56.434019</td>
      <td>84.21375275</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>YR2010</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>2263.870361</td>
      <td>2110.822021</td>
      <td>35.854009</td>
      <td>59.733111</td>
      <td>97.18988037</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>YR2011</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>2648.530029</td>
      <td>3615.583230</td>
      <td>29.801800</td>
      <td>62.706748</td>
      <td>109.1046219</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>YR2011</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>2861.167894</td>
      <td>38.577533</td>
      <td>56.887778</td>
      <td>86.68910217</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>YR2011</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>2157.984444</td>
      <td>35.042579</td>
      <td>60.480535</td>
      <td>100.7206421</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>YR2012</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>2797.323486</td>
      <td>3673.262887</td>
      <td>29.042627</td>
      <td>63.302907</td>
      <td>108.7987289</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>YR2012</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>2884.897429</td>
      <td>38.086602</td>
      <td>57.307763</td>
      <td>86.08334351</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>YR2012</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>2261.272461</td>
      <td>2213.759527</td>
      <td>34.200196</td>
      <td>61.250126</td>
      <td>100.6761017</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>YR2013</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>2891.187012</td>
      <td>3730.422292</td>
      <td>28.295414</td>
      <td>63.885015</td>
      <td>106.8399887</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>YR2013</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>1871.932129</td>
      <td>2930.170750</td>
      <td>37.607424</td>
      <td>57.710289</td>
      <td>86.50177002</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>YR2013</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>2340.688232</td>
      <td>2242.818455</td>
      <td>33.349845</td>
      <td>62.015294</td>
      <td>94.47953033</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>YR2014</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>2886.402832</td>
      <td>3772.401570</td>
      <td>27.615213</td>
      <td>64.393636</td>
      <td>104.6170197</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>YR2014</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>2990.594485</td>
      <td>37.120959</td>
      <td>58.115978</td>
      <td>86.6244278</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>YR2014</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>2334.632813</td>
      <td>2279.309902</td>
      <td>32.529328</td>
      <td>62.734875</td>
      <td>90.7219696</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>YR2015</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>3853.107631</td>
      <td>27.028606</td>
      <td>64.799595</td>
      <td>..</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>YR2015</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>3052.270569</td>
      <td>36.622822</td>
      <td>58.530645</td>
      <td>..</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>YR2015</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>2329.002149</td>
      <td>31.762798</td>
      <td>63.383938</td>
      <td>..</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 9 columns</p>
</div>




```python
del mig2['Time Code']
del mig2['Country Code']
```


```python
mig2
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>Country</th>
      <th>Enrolment in tertiary education per 100,000 inhabitants, both sexes [UIS.TE_100000.56]</th>
      <th>GDP per capita (constant 2005 US$) [NY.GDP.PCAP.KD]</th>
      <th>Population, ages 0-14 (% of total) [SP.POP.0014.TO.ZS]</th>
      <th>Population, ages 15-64 (% of total) [SP.POP.1564.TO.ZS]</th>
      <th>Primary completion rate, both sexes (%) [SE.PRM.CMPT.ZS]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>2572.813235</td>
      <td>43.742478</td>
      <td>52.756733</td>
      <td>..</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2560.782037</td>
      <td>45.444923</td>
      <td>51.602977</td>
      <td>33.9041481</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>713.5259399</td>
      <td>1655.946421</td>
      <td>46.957200</td>
      <td>49.818337</td>
      <td>44.57500076</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>2267.095959</td>
      <td>43.481122</td>
      <td>52.948845</td>
      <td>46.45079041</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>493.2778625</td>
      <td>2509.736778</td>
      <td>45.617358</td>
      <td>51.409643</td>
      <td>33.95742035</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>821.0927124</td>
      <td>1645.846419</td>
      <td>46.892259</td>
      <td>49.886066</td>
      <td>..</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>999.5952759</td>
      <td>2092.554425</td>
      <td>43.204606</td>
      <td>53.154795</td>
      <td>49.3827095</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2357.368296</td>
      <td>45.771834</td>
      <td>51.239138</td>
      <td>33.8807106</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>864.4645996</td>
      <td>1573.671559</td>
      <td>46.745647</td>
      <td>50.043171</td>
      <td>49.94493866</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>1203.906616</td>
      <td>2094.864582</td>
      <td>42.920848</td>
      <td>53.365841</td>
      <td>50.6135788</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>571.4475708</td>
      <td>2236.567544</td>
      <td>45.891347</td>
      <td>51.104597</td>
      <td>35.13737869</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>868.4456787</td>
      <td>1512.185833</td>
      <td>46.554395</td>
      <td>50.246028</td>
      <td>..</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>1310.496826</td>
      <td>2094.098791</td>
      <td>42.636120</td>
      <td>53.575056</td>
      <td>48.6973114</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>578.3195801</td>
      <td>2189.829730</td>
      <td>45.951383</td>
      <td>51.025182</td>
      <td>36.57794952</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>874.7142334</td>
      <td>1530.695403</td>
      <td>46.363681</td>
      <td>50.440498</td>
      <td>54.98490143</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>1439.984375</td>
      <td>2078.900486</td>
      <td>42.346470</td>
      <td>53.785685</td>
      <td>..</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>597.5586548</td>
      <td>2121.873660</td>
      <td>45.939359</td>
      <td>51.010453</td>
      <td>38.06546021</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>868.4234009</td>
      <td>1547.357836</td>
      <td>46.190833</td>
      <td>50.604823</td>
      <td>..</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>1492.553833</td>
      <td>2055.438830</td>
      <td>41.819358</td>
      <td>54.216902</td>
      <td>..</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>626.2062378</td>
      <td>2073.066614</td>
      <td>45.963895</td>
      <td>50.941242</td>
      <td>41.00183868</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>..</td>
      <td>1512.507552</td>
      <td>46.064093</td>
      <td>50.698642</td>
      <td>..</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>2079.844180</td>
      <td>41.318260</td>
      <td>54.619889</td>
      <td>61.98265839</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2095.342199</td>
      <td>45.884100</td>
      <td>50.970232</td>
      <td>..</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>835.494751</td>
      <td>1556.855276</td>
      <td>45.935714</td>
      <td>50.785360</td>
      <td>..</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>1515.808716</td>
      <td>2091.693100</td>
      <td>40.835477</td>
      <td>55.003335</td>
      <td>65.72280121</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2125.624163</td>
      <td>45.729366</td>
      <td>51.069993</td>
      <td>..</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>930.5723877</td>
      <td>1581.639092</td>
      <td>45.800132</td>
      <td>50.875323</td>
      <td>..</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>1568.25354</td>
      <td>2084.671422</td>
      <td>40.362708</td>
      <td>55.377693</td>
      <td>63.61275101</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2157.313890</td>
      <td>45.551915</td>
      <td>51.191567</td>
      <td>..</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>925.7636719</td>
      <td>1603.219717</td>
      <td>45.642320</td>
      <td>50.989531</td>
      <td>..</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>2093.922607</td>
      <td>3475.866745</td>
      <td>33.230254</td>
      <td>60.147890</td>
      <td>92.3642807</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2698.985240</td>
      <td>41.024086</td>
      <td>54.692058</td>
      <td>75.35977936</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>..</td>
      <td>2017.943010</td>
      <td>38.938629</td>
      <td>56.863273</td>
      <td>88.26262665</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>2209.102051</td>
      <td>3597.961991</td>
      <td>32.566759</td>
      <td>60.626412</td>
      <td>97.29842377</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>1695.110107</td>
      <td>2805.169791</td>
      <td>40.575993</td>
      <td>55.103794</td>
      <td>75.83374786</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>..</td>
      <td>2104.759589</td>
      <td>38.204492</td>
      <td>57.552528</td>
      <td>..</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>2308.634277</td>
      <td>3633.014903</td>
      <td>31.905088</td>
      <td>61.110033</td>
      <td>99.28109741</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2833.735795</td>
      <td>40.091781</td>
      <td>55.548046</td>
      <td>78.54676056</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>2035.134766</td>
      <td>2155.827865</td>
      <td>37.448905</td>
      <td>58.259473</td>
      <td>88.76953888</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>2388.975342</td>
      <td>3509.156436</td>
      <td>31.228684</td>
      <td>61.613308</td>
      <td>102.6757813</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2787.128287</td>
      <td>39.593279</td>
      <td>55.997451</td>
      <td>82.20065308</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>..</td>
      <td>2068.185180</td>
      <td>36.665394</td>
      <td>58.987298</td>
      <td>91.86322784</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>2484.339111</td>
      <td>3547.070983</td>
      <td>30.534690</td>
      <td>62.138073</td>
      <td>105.430397</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2805.951416</td>
      <td>39.095628</td>
      <td>56.434019</td>
      <td>84.21375275</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>2263.870361</td>
      <td>2110.822021</td>
      <td>35.854009</td>
      <td>59.733111</td>
      <td>97.18988037</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>2648.530029</td>
      <td>3615.583230</td>
      <td>29.801800</td>
      <td>62.706748</td>
      <td>109.1046219</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2861.167894</td>
      <td>38.577533</td>
      <td>56.887778</td>
      <td>86.68910217</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>..</td>
      <td>2157.984444</td>
      <td>35.042579</td>
      <td>60.480535</td>
      <td>100.7206421</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>2797.323486</td>
      <td>3673.262887</td>
      <td>29.042627</td>
      <td>63.302907</td>
      <td>108.7987289</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2884.897429</td>
      <td>38.086602</td>
      <td>57.307763</td>
      <td>86.08334351</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>2261.272461</td>
      <td>2213.759527</td>
      <td>34.200196</td>
      <td>61.250126</td>
      <td>100.6761017</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>2891.187012</td>
      <td>3730.422292</td>
      <td>28.295414</td>
      <td>63.885015</td>
      <td>106.8399887</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>1871.932129</td>
      <td>2930.170750</td>
      <td>37.607424</td>
      <td>57.710289</td>
      <td>86.50177002</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>2340.688232</td>
      <td>2242.818455</td>
      <td>33.349845</td>
      <td>62.015294</td>
      <td>94.47953033</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>2886.402832</td>
      <td>3772.401570</td>
      <td>27.615213</td>
      <td>64.393636</td>
      <td>104.6170197</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2990.594485</td>
      <td>37.120959</td>
      <td>58.115978</td>
      <td>86.6244278</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>2334.632813</td>
      <td>2279.309902</td>
      <td>32.529328</td>
      <td>62.734875</td>
      <td>90.7219696</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>3853.107631</td>
      <td>27.028606</td>
      <td>64.799595</td>
      <td>..</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>3052.270569</td>
      <td>36.622822</td>
      <td>58.530645</td>
      <td>..</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>..</td>
      <td>2329.002149</td>
      <td>31.762798</td>
      <td>63.383938</td>
      <td>..</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 7 columns</p>
</div>




```python
mig2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 108 entries, 0 to 107
    Data columns (total 7 columns):
    Time                                                                                      108 non-null int64
    Country                                                                                   108 non-null object
    Enrolment in tertiary education per 100,000 inhabitants, both sexes [UIS.TE_100000.56]    108 non-null object
    GDP per capita (constant 2005 US$) [NY.GDP.PCAP.KD]                                       108 non-null float64
    Population, ages 0-14 (% of total) [SP.POP.0014.TO.ZS]                                    108 non-null float64
    Population, ages 15-64 (% of total) [SP.POP.1564.TO.ZS]                                   108 non-null float64
    Primary completion rate, both sexes (%) [SE.PRM.CMPT.ZS]                                  108 non-null object
    dtypes: float64(3), int64(1), object(3)
    memory usage: 6.0+ KB



```python
mig2.rename(columns={
        'Time': 'Year',
        'Enrolment in tertiary education per 100,000 inhabitants, both sexes [UIS.TE_100000.56]': 'enrolment_tertiary',
        'GDP per capita (constant 2005 US$) [NY.GDP.PCAP.KD]': 'GDP_percapita_constant',
        'Population, ages 0-14 (% of total) [SP.POP.0014.TO.ZS]': 'pop_ages_0-14%',
        'Population, ages 15-64 (% of total) [SP.POP.1564.TO.ZS]': 'pop_ages_14-64%',
        'Primary completion rate, both sexes (%) [SE.PRM.CMPT.ZS]': 'primary_completion'
        }, inplace=True)
```


```python
mig2
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>enrolment_tertiary</th>
      <th>GDP_percapita_constant</th>
      <th>pop_ages_0-14%</th>
      <th>pop_ages_14-64%</th>
      <th>primary_completion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>2572.813235</td>
      <td>43.742478</td>
      <td>52.756733</td>
      <td>..</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2560.782037</td>
      <td>45.444923</td>
      <td>51.602977</td>
      <td>33.9041481</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>713.5259399</td>
      <td>1655.946421</td>
      <td>46.957200</td>
      <td>49.818337</td>
      <td>44.57500076</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>2267.095959</td>
      <td>43.481122</td>
      <td>52.948845</td>
      <td>46.45079041</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>493.2778625</td>
      <td>2509.736778</td>
      <td>45.617358</td>
      <td>51.409643</td>
      <td>33.95742035</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>821.0927124</td>
      <td>1645.846419</td>
      <td>46.892259</td>
      <td>49.886066</td>
      <td>..</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>999.5952759</td>
      <td>2092.554425</td>
      <td>43.204606</td>
      <td>53.154795</td>
      <td>49.3827095</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2357.368296</td>
      <td>45.771834</td>
      <td>51.239138</td>
      <td>33.8807106</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>864.4645996</td>
      <td>1573.671559</td>
      <td>46.745647</td>
      <td>50.043171</td>
      <td>49.94493866</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>1203.906616</td>
      <td>2094.864582</td>
      <td>42.920848</td>
      <td>53.365841</td>
      <td>50.6135788</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>571.4475708</td>
      <td>2236.567544</td>
      <td>45.891347</td>
      <td>51.104597</td>
      <td>35.13737869</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>868.4456787</td>
      <td>1512.185833</td>
      <td>46.554395</td>
      <td>50.246028</td>
      <td>..</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>1310.496826</td>
      <td>2094.098791</td>
      <td>42.636120</td>
      <td>53.575056</td>
      <td>48.6973114</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>578.3195801</td>
      <td>2189.829730</td>
      <td>45.951383</td>
      <td>51.025182</td>
      <td>36.57794952</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>874.7142334</td>
      <td>1530.695403</td>
      <td>46.363681</td>
      <td>50.440498</td>
      <td>54.98490143</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>1439.984375</td>
      <td>2078.900486</td>
      <td>42.346470</td>
      <td>53.785685</td>
      <td>..</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>597.5586548</td>
      <td>2121.873660</td>
      <td>45.939359</td>
      <td>51.010453</td>
      <td>38.06546021</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>868.4234009</td>
      <td>1547.357836</td>
      <td>46.190833</td>
      <td>50.604823</td>
      <td>..</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>1492.553833</td>
      <td>2055.438830</td>
      <td>41.819358</td>
      <td>54.216902</td>
      <td>..</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>626.2062378</td>
      <td>2073.066614</td>
      <td>45.963895</td>
      <td>50.941242</td>
      <td>41.00183868</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>..</td>
      <td>1512.507552</td>
      <td>46.064093</td>
      <td>50.698642</td>
      <td>..</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>2079.844180</td>
      <td>41.318260</td>
      <td>54.619889</td>
      <td>61.98265839</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2095.342199</td>
      <td>45.884100</td>
      <td>50.970232</td>
      <td>..</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>835.494751</td>
      <td>1556.855276</td>
      <td>45.935714</td>
      <td>50.785360</td>
      <td>..</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>1515.808716</td>
      <td>2091.693100</td>
      <td>40.835477</td>
      <td>55.003335</td>
      <td>65.72280121</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2125.624163</td>
      <td>45.729366</td>
      <td>51.069993</td>
      <td>..</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>930.5723877</td>
      <td>1581.639092</td>
      <td>45.800132</td>
      <td>50.875323</td>
      <td>..</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>1568.25354</td>
      <td>2084.671422</td>
      <td>40.362708</td>
      <td>55.377693</td>
      <td>63.61275101</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2157.313890</td>
      <td>45.551915</td>
      <td>51.191567</td>
      <td>..</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>925.7636719</td>
      <td>1603.219717</td>
      <td>45.642320</td>
      <td>50.989531</td>
      <td>..</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>2093.922607</td>
      <td>3475.866745</td>
      <td>33.230254</td>
      <td>60.147890</td>
      <td>92.3642807</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2698.985240</td>
      <td>41.024086</td>
      <td>54.692058</td>
      <td>75.35977936</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>..</td>
      <td>2017.943010</td>
      <td>38.938629</td>
      <td>56.863273</td>
      <td>88.26262665</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>2209.102051</td>
      <td>3597.961991</td>
      <td>32.566759</td>
      <td>60.626412</td>
      <td>97.29842377</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>1695.110107</td>
      <td>2805.169791</td>
      <td>40.575993</td>
      <td>55.103794</td>
      <td>75.83374786</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>..</td>
      <td>2104.759589</td>
      <td>38.204492</td>
      <td>57.552528</td>
      <td>..</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>2308.634277</td>
      <td>3633.014903</td>
      <td>31.905088</td>
      <td>61.110033</td>
      <td>99.28109741</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2833.735795</td>
      <td>40.091781</td>
      <td>55.548046</td>
      <td>78.54676056</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>2035.134766</td>
      <td>2155.827865</td>
      <td>37.448905</td>
      <td>58.259473</td>
      <td>88.76953888</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>2388.975342</td>
      <td>3509.156436</td>
      <td>31.228684</td>
      <td>61.613308</td>
      <td>102.6757813</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2787.128287</td>
      <td>39.593279</td>
      <td>55.997451</td>
      <td>82.20065308</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>..</td>
      <td>2068.185180</td>
      <td>36.665394</td>
      <td>58.987298</td>
      <td>91.86322784</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>2484.339111</td>
      <td>3547.070983</td>
      <td>30.534690</td>
      <td>62.138073</td>
      <td>105.430397</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2805.951416</td>
      <td>39.095628</td>
      <td>56.434019</td>
      <td>84.21375275</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>2263.870361</td>
      <td>2110.822021</td>
      <td>35.854009</td>
      <td>59.733111</td>
      <td>97.18988037</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>2648.530029</td>
      <td>3615.583230</td>
      <td>29.801800</td>
      <td>62.706748</td>
      <td>109.1046219</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2861.167894</td>
      <td>38.577533</td>
      <td>56.887778</td>
      <td>86.68910217</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>..</td>
      <td>2157.984444</td>
      <td>35.042579</td>
      <td>60.480535</td>
      <td>100.7206421</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>2797.323486</td>
      <td>3673.262887</td>
      <td>29.042627</td>
      <td>63.302907</td>
      <td>108.7987289</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2884.897429</td>
      <td>38.086602</td>
      <td>57.307763</td>
      <td>86.08334351</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>2261.272461</td>
      <td>2213.759527</td>
      <td>34.200196</td>
      <td>61.250126</td>
      <td>100.6761017</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>2891.187012</td>
      <td>3730.422292</td>
      <td>28.295414</td>
      <td>63.885015</td>
      <td>106.8399887</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>1871.932129</td>
      <td>2930.170750</td>
      <td>37.607424</td>
      <td>57.710289</td>
      <td>86.50177002</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>2340.688232</td>
      <td>2242.818455</td>
      <td>33.349845</td>
      <td>62.015294</td>
      <td>94.47953033</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>2886.402832</td>
      <td>3772.401570</td>
      <td>27.615213</td>
      <td>64.393636</td>
      <td>104.6170197</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>2990.594485</td>
      <td>37.120959</td>
      <td>58.115978</td>
      <td>86.6244278</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>2334.632813</td>
      <td>2279.309902</td>
      <td>32.529328</td>
      <td>62.734875</td>
      <td>90.7219696</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>3853.107631</td>
      <td>27.028606</td>
      <td>64.799595</td>
      <td>..</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>3052.270569</td>
      <td>36.622822</td>
      <td>58.530645</td>
      <td>..</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>..</td>
      <td>2329.002149</td>
      <td>31.762798</td>
      <td>63.383938</td>
      <td>..</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 7 columns</p>
</div>




```python
mig2['primary_completion'] = pd.to_numeric(mig2['primary_completion'], errors='coerce')
```


```python
mig2['enrolment_tertiary'] = pd.to_numeric(mig2['enrolment_tertiary'], errors='coerce')
```


```python
mig2
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>enrolment_tertiary</th>
      <th>GDP_percapita_constant</th>
      <th>pop_ages_0-14%</th>
      <th>pop_ages_14-64%</th>
      <th>primary_completion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>2572.813235</td>
      <td>43.742478</td>
      <td>52.756733</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>2560.782037</td>
      <td>45.444923</td>
      <td>51.602977</td>
      <td>33.904148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>713.525940</td>
      <td>1655.946421</td>
      <td>46.957200</td>
      <td>49.818337</td>
      <td>44.575001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>2267.095959</td>
      <td>43.481122</td>
      <td>52.948845</td>
      <td>46.450790</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>493.277863</td>
      <td>2509.736778</td>
      <td>45.617358</td>
      <td>51.409643</td>
      <td>33.957420</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>821.092712</td>
      <td>1645.846419</td>
      <td>46.892259</td>
      <td>49.886066</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>999.595276</td>
      <td>2092.554425</td>
      <td>43.204606</td>
      <td>53.154795</td>
      <td>49.382709</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>2357.368296</td>
      <td>45.771834</td>
      <td>51.239138</td>
      <td>33.880711</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>864.464600</td>
      <td>1573.671559</td>
      <td>46.745647</td>
      <td>50.043171</td>
      <td>49.944939</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>1203.906616</td>
      <td>2094.864582</td>
      <td>42.920848</td>
      <td>53.365841</td>
      <td>50.613579</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>571.447571</td>
      <td>2236.567544</td>
      <td>45.891347</td>
      <td>51.104597</td>
      <td>35.137379</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>868.445679</td>
      <td>1512.185833</td>
      <td>46.554395</td>
      <td>50.246028</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>1310.496826</td>
      <td>2094.098791</td>
      <td>42.636120</td>
      <td>53.575056</td>
      <td>48.697311</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>578.319580</td>
      <td>2189.829730</td>
      <td>45.951383</td>
      <td>51.025182</td>
      <td>36.577950</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>874.714233</td>
      <td>1530.695403</td>
      <td>46.363681</td>
      <td>50.440498</td>
      <td>54.984901</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>1439.984375</td>
      <td>2078.900486</td>
      <td>42.346470</td>
      <td>53.785685</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>597.558655</td>
      <td>2121.873660</td>
      <td>45.939359</td>
      <td>51.010453</td>
      <td>38.065460</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>868.423401</td>
      <td>1547.357836</td>
      <td>46.190833</td>
      <td>50.604823</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>1492.553833</td>
      <td>2055.438830</td>
      <td>41.819358</td>
      <td>54.216902</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>626.206238</td>
      <td>2073.066614</td>
      <td>45.963895</td>
      <td>50.941242</td>
      <td>41.001839</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>1512.507552</td>
      <td>46.064093</td>
      <td>50.698642</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>2079.844180</td>
      <td>41.318260</td>
      <td>54.619889</td>
      <td>61.982658</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>2095.342199</td>
      <td>45.884100</td>
      <td>50.970232</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>835.494751</td>
      <td>1556.855276</td>
      <td>45.935714</td>
      <td>50.785360</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>1515.808716</td>
      <td>2091.693100</td>
      <td>40.835477</td>
      <td>55.003335</td>
      <td>65.722801</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>2125.624163</td>
      <td>45.729366</td>
      <td>51.069993</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>930.572388</td>
      <td>1581.639092</td>
      <td>45.800132</td>
      <td>50.875323</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>1568.253540</td>
      <td>2084.671422</td>
      <td>40.362708</td>
      <td>55.377693</td>
      <td>63.612751</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>2157.313890</td>
      <td>45.551915</td>
      <td>51.191567</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>925.763672</td>
      <td>1603.219717</td>
      <td>45.642320</td>
      <td>50.989531</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>2093.922607</td>
      <td>3475.866745</td>
      <td>33.230254</td>
      <td>60.147890</td>
      <td>92.364281</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>2698.985240</td>
      <td>41.024086</td>
      <td>54.692058</td>
      <td>75.359779</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>2017.943010</td>
      <td>38.938629</td>
      <td>56.863273</td>
      <td>88.262627</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>2209.102051</td>
      <td>3597.961991</td>
      <td>32.566759</td>
      <td>60.626412</td>
      <td>97.298424</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>1695.110107</td>
      <td>2805.169791</td>
      <td>40.575993</td>
      <td>55.103794</td>
      <td>75.833748</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>2104.759589</td>
      <td>38.204492</td>
      <td>57.552528</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>2308.634277</td>
      <td>3633.014903</td>
      <td>31.905088</td>
      <td>61.110033</td>
      <td>99.281097</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>2833.735795</td>
      <td>40.091781</td>
      <td>55.548046</td>
      <td>78.546761</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>2035.134766</td>
      <td>2155.827865</td>
      <td>37.448905</td>
      <td>58.259473</td>
      <td>88.769539</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>2388.975342</td>
      <td>3509.156436</td>
      <td>31.228684</td>
      <td>61.613308</td>
      <td>102.675781</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>2787.128287</td>
      <td>39.593279</td>
      <td>55.997451</td>
      <td>82.200653</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>2068.185180</td>
      <td>36.665394</td>
      <td>58.987298</td>
      <td>91.863228</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>2484.339111</td>
      <td>3547.070983</td>
      <td>30.534690</td>
      <td>62.138073</td>
      <td>105.430397</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>2805.951416</td>
      <td>39.095628</td>
      <td>56.434019</td>
      <td>84.213753</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>2263.870361</td>
      <td>2110.822021</td>
      <td>35.854009</td>
      <td>59.733111</td>
      <td>97.189880</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>2648.530029</td>
      <td>3615.583230</td>
      <td>29.801800</td>
      <td>62.706748</td>
      <td>109.104622</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>2861.167894</td>
      <td>38.577533</td>
      <td>56.887778</td>
      <td>86.689102</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>2157.984444</td>
      <td>35.042579</td>
      <td>60.480535</td>
      <td>100.720642</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>2797.323486</td>
      <td>3673.262887</td>
      <td>29.042627</td>
      <td>63.302907</td>
      <td>108.798729</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>2884.897429</td>
      <td>38.086602</td>
      <td>57.307763</td>
      <td>86.083344</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>2261.272461</td>
      <td>2213.759527</td>
      <td>34.200196</td>
      <td>61.250126</td>
      <td>100.676102</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>2891.187012</td>
      <td>3730.422292</td>
      <td>28.295414</td>
      <td>63.885015</td>
      <td>106.839989</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>1871.932129</td>
      <td>2930.170750</td>
      <td>37.607424</td>
      <td>57.710289</td>
      <td>86.501770</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>2340.688232</td>
      <td>2242.818455</td>
      <td>33.349845</td>
      <td>62.015294</td>
      <td>94.479530</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>2886.402832</td>
      <td>3772.401570</td>
      <td>27.615213</td>
      <td>64.393636</td>
      <td>104.617020</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>2990.594485</td>
      <td>37.120959</td>
      <td>58.115978</td>
      <td>86.624428</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>2334.632813</td>
      <td>2279.309902</td>
      <td>32.529328</td>
      <td>62.734875</td>
      <td>90.721970</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>3853.107631</td>
      <td>27.028606</td>
      <td>64.799595</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>3052.270569</td>
      <td>36.622822</td>
      <td>58.530645</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>2329.002149</td>
      <td>31.762798</td>
      <td>63.383938</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 7 columns</p>
</div>




```python
mig2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 108 entries, 0 to 107
    Data columns (total 7 columns):
    Year                      108 non-null int64
    Country                   108 non-null object
    enrolment_tertiary        68 non-null float64
    GDP_percapita_constant    108 non-null float64
    pop_ages_0-14%            108 non-null float64
    pop_ages_14-64%           108 non-null float64
    primary_completion        71 non-null float64
    dtypes: float64(5), int64(1), object(1)
    memory usage: 6.0+ KB



```python
mig2['enrolment_tertiary'].fillna(np.mean(mig2['enrolment_tertiary']), inplace=True)
```


```python
mig2['primary_completion'].fillna(np.mean(mig2['primary_completion']), inplace=True)
```


```python
mig2
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>enrolment_tertiary</th>
      <th>GDP_percapita_constant</th>
      <th>pop_ages_0-14%</th>
      <th>pop_ages_14-64%</th>
      <th>primary_completion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>1516.400016</td>
      <td>2572.813235</td>
      <td>43.742478</td>
      <td>52.756733</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>1516.400016</td>
      <td>2560.782037</td>
      <td>45.444923</td>
      <td>51.602977</td>
      <td>33.904148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>713.525940</td>
      <td>1655.946421</td>
      <td>46.957200</td>
      <td>49.818337</td>
      <td>44.575001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>1516.400016</td>
      <td>2267.095959</td>
      <td>43.481122</td>
      <td>52.948845</td>
      <td>46.450790</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>493.277863</td>
      <td>2509.736778</td>
      <td>45.617358</td>
      <td>51.409643</td>
      <td>33.957420</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>821.092712</td>
      <td>1645.846419</td>
      <td>46.892259</td>
      <td>49.886066</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>999.595276</td>
      <td>2092.554425</td>
      <td>43.204606</td>
      <td>53.154795</td>
      <td>49.382709</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>1516.400016</td>
      <td>2357.368296</td>
      <td>45.771834</td>
      <td>51.239138</td>
      <td>33.880711</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>864.464600</td>
      <td>1573.671559</td>
      <td>46.745647</td>
      <td>50.043171</td>
      <td>49.944939</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>1203.906616</td>
      <td>2094.864582</td>
      <td>42.920848</td>
      <td>53.365841</td>
      <td>50.613579</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>571.447571</td>
      <td>2236.567544</td>
      <td>45.891347</td>
      <td>51.104597</td>
      <td>35.137379</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>868.445679</td>
      <td>1512.185833</td>
      <td>46.554395</td>
      <td>50.246028</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>1310.496826</td>
      <td>2094.098791</td>
      <td>42.636120</td>
      <td>53.575056</td>
      <td>48.697311</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>578.319580</td>
      <td>2189.829730</td>
      <td>45.951383</td>
      <td>51.025182</td>
      <td>36.577950</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>874.714233</td>
      <td>1530.695403</td>
      <td>46.363681</td>
      <td>50.440498</td>
      <td>54.984901</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>1439.984375</td>
      <td>2078.900486</td>
      <td>42.346470</td>
      <td>53.785685</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>597.558655</td>
      <td>2121.873660</td>
      <td>45.939359</td>
      <td>51.010453</td>
      <td>38.065460</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>868.423401</td>
      <td>1547.357836</td>
      <td>46.190833</td>
      <td>50.604823</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>1492.553833</td>
      <td>2055.438830</td>
      <td>41.819358</td>
      <td>54.216902</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>626.206238</td>
      <td>2073.066614</td>
      <td>45.963895</td>
      <td>50.941242</td>
      <td>41.001839</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>1516.400016</td>
      <td>1512.507552</td>
      <td>46.064093</td>
      <td>50.698642</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>1516.400016</td>
      <td>2079.844180</td>
      <td>41.318260</td>
      <td>54.619889</td>
      <td>61.982658</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>1516.400016</td>
      <td>2095.342199</td>
      <td>45.884100</td>
      <td>50.970232</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>835.494751</td>
      <td>1556.855276</td>
      <td>45.935714</td>
      <td>50.785360</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>1515.808716</td>
      <td>2091.693100</td>
      <td>40.835477</td>
      <td>55.003335</td>
      <td>65.722801</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>1516.400016</td>
      <td>2125.624163</td>
      <td>45.729366</td>
      <td>51.069993</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>930.572388</td>
      <td>1581.639092</td>
      <td>45.800132</td>
      <td>50.875323</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>1568.253540</td>
      <td>2084.671422</td>
      <td>40.362708</td>
      <td>55.377693</td>
      <td>63.612751</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>1516.400016</td>
      <td>2157.313890</td>
      <td>45.551915</td>
      <td>51.191567</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>925.763672</td>
      <td>1603.219717</td>
      <td>45.642320</td>
      <td>50.989531</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>2093.922607</td>
      <td>3475.866745</td>
      <td>33.230254</td>
      <td>60.147890</td>
      <td>92.364281</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>1516.400016</td>
      <td>2698.985240</td>
      <td>41.024086</td>
      <td>54.692058</td>
      <td>75.359779</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>1516.400016</td>
      <td>2017.943010</td>
      <td>38.938629</td>
      <td>56.863273</td>
      <td>88.262627</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>2209.102051</td>
      <td>3597.961991</td>
      <td>32.566759</td>
      <td>60.626412</td>
      <td>97.298424</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>1695.110107</td>
      <td>2805.169791</td>
      <td>40.575993</td>
      <td>55.103794</td>
      <td>75.833748</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>1516.400016</td>
      <td>2104.759589</td>
      <td>38.204492</td>
      <td>57.552528</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>2308.634277</td>
      <td>3633.014903</td>
      <td>31.905088</td>
      <td>61.110033</td>
      <td>99.281097</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>1516.400016</td>
      <td>2833.735795</td>
      <td>40.091781</td>
      <td>55.548046</td>
      <td>78.546761</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>2035.134766</td>
      <td>2155.827865</td>
      <td>37.448905</td>
      <td>58.259473</td>
      <td>88.769539</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>2388.975342</td>
      <td>3509.156436</td>
      <td>31.228684</td>
      <td>61.613308</td>
      <td>102.675781</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>1516.400016</td>
      <td>2787.128287</td>
      <td>39.593279</td>
      <td>55.997451</td>
      <td>82.200653</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>1516.400016</td>
      <td>2068.185180</td>
      <td>36.665394</td>
      <td>58.987298</td>
      <td>91.863228</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>2484.339111</td>
      <td>3547.070983</td>
      <td>30.534690</td>
      <td>62.138073</td>
      <td>105.430397</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>1516.400016</td>
      <td>2805.951416</td>
      <td>39.095628</td>
      <td>56.434019</td>
      <td>84.213753</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>2263.870361</td>
      <td>2110.822021</td>
      <td>35.854009</td>
      <td>59.733111</td>
      <td>97.189880</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>2648.530029</td>
      <td>3615.583230</td>
      <td>29.801800</td>
      <td>62.706748</td>
      <td>109.104622</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>1516.400016</td>
      <td>2861.167894</td>
      <td>38.577533</td>
      <td>56.887778</td>
      <td>86.689102</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>1516.400016</td>
      <td>2157.984444</td>
      <td>35.042579</td>
      <td>60.480535</td>
      <td>100.720642</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>2797.323486</td>
      <td>3673.262887</td>
      <td>29.042627</td>
      <td>63.302907</td>
      <td>108.798729</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>1516.400016</td>
      <td>2884.897429</td>
      <td>38.086602</td>
      <td>57.307763</td>
      <td>86.083344</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>2261.272461</td>
      <td>2213.759527</td>
      <td>34.200196</td>
      <td>61.250126</td>
      <td>100.676102</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>2891.187012</td>
      <td>3730.422292</td>
      <td>28.295414</td>
      <td>63.885015</td>
      <td>106.839989</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>1871.932129</td>
      <td>2930.170750</td>
      <td>37.607424</td>
      <td>57.710289</td>
      <td>86.501770</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>2340.688232</td>
      <td>2242.818455</td>
      <td>33.349845</td>
      <td>62.015294</td>
      <td>94.479530</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>2886.402832</td>
      <td>3772.401570</td>
      <td>27.615213</td>
      <td>64.393636</td>
      <td>104.617020</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>1516.400016</td>
      <td>2990.594485</td>
      <td>37.120959</td>
      <td>58.115978</td>
      <td>86.624428</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>2334.632813</td>
      <td>2279.309902</td>
      <td>32.529328</td>
      <td>62.734875</td>
      <td>90.721970</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>1516.400016</td>
      <td>3853.107631</td>
      <td>27.028606</td>
      <td>64.799595</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>1516.400016</td>
      <td>3052.270569</td>
      <td>36.622822</td>
      <td>58.530645</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>1516.400016</td>
      <td>2329.002149</td>
      <td>31.762798</td>
      <td>63.383938</td>
      <td>73.720787</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 7 columns</p>
</div>




```python
mig3
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Year Code</th>
      <th>Country</th>
      <th>Country Code</th>
      <th>GINI index (World Bank estimate) [SI.POV.GINI]</th>
      <th>Income share held by fourth 20% [SI.DST.04TH.20]</th>
      <th>Income share held by highest 10% [SI.DST.10TH.10]</th>
      <th>Income share held by highest 20% [SI.DST.05TH.20]</th>
      <th>Income share held by lowest 10% [SI.DST.FRST.10]</th>
      <th>Income share held by lowest 20% [SI.DST.FRST.20]</th>
      <th>Income share held by second 20% [SI.DST.02ND.20]</th>
      <th>Income share held by third 20% [SI.DST.03RD.20]</th>
      <th>Number of poor at $1.90 a day (2011 PPP) (millions) [SI.POV.NOP1]</th>
      <th>Number of poor at $3.10 a day (2011 PPP) (millions) [SI.POV.NOP2]</th>
      <th>Poverty gap at $1.90 a day (2011 PPP) (%) [SI.POV.GAPS]</th>
      <th>Poverty gap at $3.10 a day (2011 PPP) (%) [SI.POV.GAP2]</th>
      <th>Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population) [SI.POV.DDAY]</th>
      <th>Poverty headcount ratio at $3.10 a day (2011 PPP) (% of population) [SI.POV.2DAY]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>YR1980</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>YR1980</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>YR1980</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>YR1981</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>YR1981</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>YR1981</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>YR1982</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>YR1982</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>YR1982</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>YR1983</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>YR1983</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>YR1983</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>YR1984</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>YR1984</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>YR1984</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>YR1985</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>YR1985</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>YR1985</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>YR1986</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>YR1986</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>58.26</td>
      <td>18.35</td>
      <td>46.73</td>
      <td>61.96</td>
      <td>1</td>
      <td>2.77</td>
      <td>6.18</td>
      <td>10.75</td>
      <td>4.238208</td>
      <td>5.79488</td>
      <td>24.99</td>
      <td>39.01</td>
      <td>50.94</td>
      <td>69.65</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>YR1986</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>55.09</td>
      <td>19.35</td>
      <td>43.26</td>
      <td>59.41</td>
      <td>1.23</td>
      <td>3.23</td>
      <td>6.67</td>
      <td>11.36</td>
      <td>0.422176</td>
      <td>0.713758</td>
      <td>9.15</td>
      <td>18.94</td>
      <td>25.28</td>
      <td>42.74</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>YR1987</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>YR1987</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>YR1987</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>YR1988</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>YR1988</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>YR1988</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>YR1989</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>0.983164</td>
      <td>1.632736</td>
      <td>11.24</td>
      <td>16.66</td>
      <td>18.98</td>
      <td>31.52</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>YR1989</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>59.6</td>
      <td>18.78</td>
      <td>46.78</td>
      <td>62.87</td>
      <td>0.68</td>
      <td>2.15</td>
      <td>5.68</td>
      <td>10.53</td>
      <td>3.398988</td>
      <td>4.919682</td>
      <td>18.71</td>
      <td>29.68</td>
      <td>38.02</td>
      <td>55.03</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>YR1989</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>59.49</td>
      <td>17.82</td>
      <td>48.18</td>
      <td>63.31</td>
      <td>1.04</td>
      <td>2.76</td>
      <td>5.87</td>
      <td>10.24</td>
      <td>1.84122</td>
      <td>2.722716</td>
      <td>16.9</td>
      <td>29.14</td>
      <td>38.6</td>
      <td>57.08</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>YR2006</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>45.44</td>
      <td>20.76</td>
      <td>35.48</td>
      <td>51.15</td>
      <td>1.79</td>
      <td>4.89</td>
      <td>9.37</td>
      <td>13.85</td>
      <td>0.379692</td>
      <td>1.035198</td>
      <td>1.73</td>
      <td>5.57</td>
      <td>6.36</td>
      <td>17.34</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>YR2006</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>54.89</td>
      <td>19.09</td>
      <td>43.56</td>
      <td>59.17</td>
      <td>1.07</td>
      <td>3.13</td>
      <td>6.99</td>
      <td>11.63</td>
      <td>1.552699</td>
      <td>3.195781</td>
      <td>3.93</td>
      <td>9.27</td>
      <td>11.51</td>
      <td>23.69</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>YR2006</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>57.42</td>
      <td>19.7</td>
      <td>44.05</td>
      <td>60.63</td>
      <td>0.58</td>
      <td>2.13</td>
      <td>6.16</td>
      <td>11.37</td>
      <td>1.667679</td>
      <td>2.61473</td>
      <td>11.4</td>
      <td>18.85</td>
      <td>23.79</td>
      <td>37.3</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>YR2007</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>45.24</td>
      <td>20.51</td>
      <td>35.72</td>
      <td>51.24</td>
      <td>1.93</td>
      <td>5.18</td>
      <td>9.41</td>
      <td>13.66</td>
      <td>0.268951</td>
      <td>0.835006</td>
      <td>1.08</td>
      <td>4.1</td>
      <td>4.49</td>
      <td>13.94</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>YR2007</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>YR2007</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>56.16</td>
      <td>19.18</td>
      <td>43.81</td>
      <td>60.3</td>
      <td>0.9</td>
      <td>2.81</td>
      <td>6.59</td>
      <td>11.12</td>
      <td>1.242759</td>
      <td>2.279461</td>
      <td>6.91</td>
      <td>13.88</td>
      <td>17.43</td>
      <td>31.97</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>YR2008</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>46.65</td>
      <td>20.75</td>
      <td>36.04</td>
      <td>52.2</td>
      <td>1.7</td>
      <td>4.67</td>
      <td>8.96</td>
      <td>13.41</td>
      <td>0.4152</td>
      <td>1.1148</td>
      <td>1.99</td>
      <td>6.09</td>
      <td>6.92</td>
      <td>18.58</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>YR2008</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>YR2008</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>55.74</td>
      <td>19.25</td>
      <td>43.87</td>
      <td>59.58</td>
      <td>0.91</td>
      <td>2.83</td>
      <td>6.76</td>
      <td>11.57</td>
      <td>1.171764</td>
      <td>2.13081</td>
      <td>6.3</td>
      <td>12.68</td>
      <td>16.14</td>
      <td>29.35</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>YR2009</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>45.93</td>
      <td>20.44</td>
      <td>36.07</td>
      <td>51.74</td>
      <td>1.78</td>
      <td>4.85</td>
      <td>9.21</td>
      <td>13.75</td>
      <td>0.384678</td>
      <td>1.054102</td>
      <td>1.67</td>
      <td>5.52</td>
      <td>6.39</td>
      <td>17.51</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>YR2009</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>YR2009</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>51.56</td>
      <td>20.65</td>
      <td>39.14</td>
      <td>55.81</td>
      <td>1.15</td>
      <td>3.35</td>
      <td>7.54</td>
      <td>12.65</td>
      <td>1.036152</td>
      <td>1.979316</td>
      <td>4.82</td>
      <td>10.88</td>
      <td>14.04</td>
      <td>26.82</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>YR2010</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>44.53</td>
      <td>21.6</td>
      <td>33.7</td>
      <td>50</td>
      <td>1.67</td>
      <td>4.74</td>
      <td>9.44</td>
      <td>14.22</td>
      <td>0.437296</td>
      <td>1.12042</td>
      <td>2.33</td>
      <td>6.3</td>
      <td>7.24</td>
      <td>18.55</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>YR2010</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>YR2010</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>53.39</td>
      <td>20.18</td>
      <td>41.02</td>
      <td>57.59</td>
      <td>1.09</td>
      <td>3.19</td>
      <td>7.13</td>
      <td>11.91</td>
      <td>1.16025</td>
      <td>2.18325</td>
      <td>5.4</td>
      <td>11.9</td>
      <td>15.47</td>
      <td>29.11</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>YR2011</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>42.43</td>
      <td>21.26</td>
      <td>32.86</td>
      <td>48.74</td>
      <td>2.11</td>
      <td>5.57</td>
      <td>10</td>
      <td>14.44</td>
      <td>0.274518</td>
      <td>0.911424</td>
      <td>1.06</td>
      <td>4.39</td>
      <td>4.53</td>
      <td>15.04</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>YR2011</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>52.35</td>
      <td>19.16</td>
      <td>41.83</td>
      <td>57.23</td>
      <td>1.34</td>
      <td>3.87</td>
      <td>7.77</td>
      <td>11.97</td>
      <td>1.735265</td>
      <td>3.983735</td>
      <td>4</td>
      <td>9.84</td>
      <td>11.53</td>
      <td>26.47</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>YR2011</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>57.4</td>
      <td>18.56</td>
      <td>45.67</td>
      <td>61.23</td>
      <td>0.75</td>
      <td>2.61</td>
      <td>6.54</td>
      <td>11.08</td>
      <td>1.42875</td>
      <td>2.489454</td>
      <td>7.88</td>
      <td>14.66</td>
      <td>18.75</td>
      <td>32.67</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>YR2012</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>41.8</td>
      <td>21.3</td>
      <td>32.47</td>
      <td>48.15</td>
      <td>2.15</td>
      <td>5.7</td>
      <td>10.17</td>
      <td>14.67</td>
      <td>0.252512</td>
      <td>0.826127</td>
      <td>0.98</td>
      <td>3.84</td>
      <td>4.16</td>
      <td>13.61</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>YR2012</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>YR2012</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>57.4</td>
      <td>18.56</td>
      <td>45.68</td>
      <td>61.13</td>
      <td>0.79</td>
      <td>2.63</td>
      <td>6.52</td>
      <td>11.16</td>
      <td>1.653264</td>
      <td>2.883924</td>
      <td>9.25</td>
      <td>17.1</td>
      <td>21.36</td>
      <td>37.26</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>YR2013</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>43.51</td>
      <td>20.68</td>
      <td>34.35</td>
      <td>49.79</td>
      <td>2.11</td>
      <td>5.52</td>
      <td>9.82</td>
      <td>14.2</td>
      <td>0.197925</td>
      <td>0.702177</td>
      <td>0.74</td>
      <td>3.16</td>
      <td>3.25</td>
      <td>11.53</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>YR2013</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>YR2013</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>53.67</td>
      <td>19.94</td>
      <td>41.48</td>
      <td>57.66</td>
      <td>0.98</td>
      <td>3.1</td>
      <td>7.2</td>
      <td>12.1</td>
      <td>1.486005</td>
      <td>2.712175</td>
      <td>7.66</td>
      <td>15.24</td>
      <td>18.93</td>
      <td>34.55</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>YR2014</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>41.84</td>
      <td>21.35</td>
      <td>32.31</td>
      <td>48.26</td>
      <td>2.19</td>
      <td>5.72</td>
      <td>10.09</td>
      <td>14.58</td>
      <td>0.181467</td>
      <td>0.689819</td>
      <td>0.64</td>
      <td>3</td>
      <td>2.97</td>
      <td>11.29</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>YR2014</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>48.66</td>
      <td>20.06</td>
      <td>38.36</td>
      <td>53.91</td>
      <td>1.64</td>
      <td>4.44</td>
      <td>8.5</td>
      <td>13.08</td>
      <td>1.493064</td>
      <td>3.85281</td>
      <td>2.72</td>
      <td>8.11</td>
      <td>9.32</td>
      <td>24.05</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>YR2014</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>50.64</td>
      <td>20.68</td>
      <td>38.36</td>
      <td>55.13</td>
      <td>1.15</td>
      <td>3.5</td>
      <td>7.73</td>
      <td>12.95</td>
      <td>1.270416</td>
      <td>2.484316</td>
      <td>6.01</td>
      <td>12.97</td>
      <td>15.96</td>
      <td>31.21</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>YR2015</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>YR2015</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>YR2015</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 18 columns</p>
</div>




```python
del mig3['Year Code']
del mig3['Country Code']
del mig3['Income share held by fourth 20% [SI.DST.04TH.20]']
del mig3['Income share held by highest 20% [SI.DST.05TH.20]']
del mig3['Income share held by lowest 20% [SI.DST.FRST.20]']
del mig3['Income share held by second 20% [SI.DST.02ND.20]']
del mig3['Income share held by third 20% [SI.DST.03RD.20]']
```


```python
mig3.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 108 entries, 0 to 107
    Data columns (total 11 columns):
    Year                                                                                 108 non-null int64
    Country                                                                              108 non-null object
    GINI index (World Bank estimate) [SI.POV.GINI]                                       108 non-null object
    Income share held by highest 10% [SI.DST.10TH.10]                                    108 non-null object
    Income share held by lowest 10% [SI.DST.FRST.10]                                     108 non-null object
    Number of poor at $1.90 a day (2011 PPP) (millions) [SI.POV.NOP1]                    108 non-null object
    Number of poor at $3.10 a day (2011 PPP) (millions) [SI.POV.NOP2]                    108 non-null object
    Poverty gap at $1.90 a day (2011 PPP) (%) [SI.POV.GAPS]                              108 non-null object
    Poverty gap at $3.10 a day (2011 PPP) (%) [SI.POV.GAP2]                              108 non-null object
    Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population) [SI.POV.DDAY]    108 non-null object
    Poverty headcount ratio at $3.10 a day (2011 PPP) (% of population) [SI.POV.2DAY]    108 non-null object
    dtypes: int64(1), object(10)
    memory usage: 9.4+ KB



```python
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
```


```python
mig3.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 108 entries, 0 to 107
    Data columns (total 11 columns):
    Year                      108 non-null int64
    Country                   108 non-null object
    gini                      108 non-null object
    income_highest%           108 non-null object
    income_lowest%            108 non-null object
    poor_1.90                 108 non-null object
    poor_3.10                 108 non-null object
    poverty_gap_1.90          108 non-null object
    poverty_gap_3.10          108 non-null object
    poverty_headcount_1.90    108 non-null object
    poverty_headcount_3.10    108 non-null object
    dtypes: int64(1), object(10)
    memory usage: 9.4+ KB



```python
mig3['gini'] = pd.to_numeric(mig3['gini'], errors='coerce')
mig3['income_highest%'] = pd.to_numeric(mig3['income_highest%'], errors='coerce')
mig3['income_lowest%'] = pd.to_numeric(mig3['income_lowest%'], errors='coerce')
mig3['poor_1.90'] = pd.to_numeric(mig3['poor_1.90'], errors='coerce')
mig3['poor_3.10'] = pd.to_numeric(mig3['poor_3.10'], errors='coerce')
mig3['poverty_gap_1.90'] = pd.to_numeric(mig3['poverty_gap_1.90'], errors='coerce')
mig3['poverty_gap_3.10'] = pd.to_numeric(mig3['poverty_gap_3.10'], errors='coerce')
mig3['poverty_headcount_1.90'] = pd.to_numeric(mig3['poverty_headcount_1.90'], errors='coerce')
mig3['poverty_headcount_3.10'] = pd.to_numeric(mig3['poverty_headcount_3.10'], errors='coerce')
```


```python
mig3
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>gini</th>
      <th>income_highest%</th>
      <th>income_lowest%</th>
      <th>poor_1.90</th>
      <th>poor_3.10</th>
      <th>poverty_gap_1.90</th>
      <th>poverty_gap_3.10</th>
      <th>poverty_headcount_1.90</th>
      <th>poverty_headcount_3.10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>58.26</td>
      <td>46.73</td>
      <td>1.00</td>
      <td>4.238208</td>
      <td>5.794880</td>
      <td>24.99</td>
      <td>39.01</td>
      <td>50.94</td>
      <td>69.65</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>55.09</td>
      <td>43.26</td>
      <td>1.23</td>
      <td>0.422176</td>
      <td>0.713758</td>
      <td>9.15</td>
      <td>18.94</td>
      <td>25.28</td>
      <td>42.74</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.983164</td>
      <td>1.632736</td>
      <td>11.24</td>
      <td>16.66</td>
      <td>18.98</td>
      <td>31.52</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>59.60</td>
      <td>46.78</td>
      <td>0.68</td>
      <td>3.398988</td>
      <td>4.919682</td>
      <td>18.71</td>
      <td>29.68</td>
      <td>38.02</td>
      <td>55.03</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>59.49</td>
      <td>48.18</td>
      <td>1.04</td>
      <td>1.841220</td>
      <td>2.722716</td>
      <td>16.90</td>
      <td>29.14</td>
      <td>38.60</td>
      <td>57.08</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>45.44</td>
      <td>35.48</td>
      <td>1.79</td>
      <td>0.379692</td>
      <td>1.035198</td>
      <td>1.73</td>
      <td>5.57</td>
      <td>6.36</td>
      <td>17.34</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>54.89</td>
      <td>43.56</td>
      <td>1.07</td>
      <td>1.552699</td>
      <td>3.195781</td>
      <td>3.93</td>
      <td>9.27</td>
      <td>11.51</td>
      <td>23.69</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>57.42</td>
      <td>44.05</td>
      <td>0.58</td>
      <td>1.667679</td>
      <td>2.614730</td>
      <td>11.40</td>
      <td>18.85</td>
      <td>23.79</td>
      <td>37.30</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>45.24</td>
      <td>35.72</td>
      <td>1.93</td>
      <td>0.268951</td>
      <td>0.835006</td>
      <td>1.08</td>
      <td>4.10</td>
      <td>4.49</td>
      <td>13.94</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>56.16</td>
      <td>43.81</td>
      <td>0.90</td>
      <td>1.242759</td>
      <td>2.279461</td>
      <td>6.91</td>
      <td>13.88</td>
      <td>17.43</td>
      <td>31.97</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>46.65</td>
      <td>36.04</td>
      <td>1.70</td>
      <td>0.415200</td>
      <td>1.114800</td>
      <td>1.99</td>
      <td>6.09</td>
      <td>6.92</td>
      <td>18.58</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>55.74</td>
      <td>43.87</td>
      <td>0.91</td>
      <td>1.171764</td>
      <td>2.130810</td>
      <td>6.30</td>
      <td>12.68</td>
      <td>16.14</td>
      <td>29.35</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>45.93</td>
      <td>36.07</td>
      <td>1.78</td>
      <td>0.384678</td>
      <td>1.054102</td>
      <td>1.67</td>
      <td>5.52</td>
      <td>6.39</td>
      <td>17.51</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>51.56</td>
      <td>39.14</td>
      <td>1.15</td>
      <td>1.036152</td>
      <td>1.979316</td>
      <td>4.82</td>
      <td>10.88</td>
      <td>14.04</td>
      <td>26.82</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>44.53</td>
      <td>33.70</td>
      <td>1.67</td>
      <td>0.437296</td>
      <td>1.120420</td>
      <td>2.33</td>
      <td>6.30</td>
      <td>7.24</td>
      <td>18.55</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>53.39</td>
      <td>41.02</td>
      <td>1.09</td>
      <td>1.160250</td>
      <td>2.183250</td>
      <td>5.40</td>
      <td>11.90</td>
      <td>15.47</td>
      <td>29.11</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>42.43</td>
      <td>32.86</td>
      <td>2.11</td>
      <td>0.274518</td>
      <td>0.911424</td>
      <td>1.06</td>
      <td>4.39</td>
      <td>4.53</td>
      <td>15.04</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>52.35</td>
      <td>41.83</td>
      <td>1.34</td>
      <td>1.735265</td>
      <td>3.983735</td>
      <td>4.00</td>
      <td>9.84</td>
      <td>11.53</td>
      <td>26.47</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>57.40</td>
      <td>45.67</td>
      <td>0.75</td>
      <td>1.428750</td>
      <td>2.489454</td>
      <td>7.88</td>
      <td>14.66</td>
      <td>18.75</td>
      <td>32.67</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>41.80</td>
      <td>32.47</td>
      <td>2.15</td>
      <td>0.252512</td>
      <td>0.826127</td>
      <td>0.98</td>
      <td>3.84</td>
      <td>4.16</td>
      <td>13.61</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>57.40</td>
      <td>45.68</td>
      <td>0.79</td>
      <td>1.653264</td>
      <td>2.883924</td>
      <td>9.25</td>
      <td>17.10</td>
      <td>21.36</td>
      <td>37.26</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>43.51</td>
      <td>34.35</td>
      <td>2.11</td>
      <td>0.197925</td>
      <td>0.702177</td>
      <td>0.74</td>
      <td>3.16</td>
      <td>3.25</td>
      <td>11.53</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>53.67</td>
      <td>41.48</td>
      <td>0.98</td>
      <td>1.486005</td>
      <td>2.712175</td>
      <td>7.66</td>
      <td>15.24</td>
      <td>18.93</td>
      <td>34.55</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>41.84</td>
      <td>32.31</td>
      <td>2.19</td>
      <td>0.181467</td>
      <td>0.689819</td>
      <td>0.64</td>
      <td>3.00</td>
      <td>2.97</td>
      <td>11.29</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>48.66</td>
      <td>38.36</td>
      <td>1.64</td>
      <td>1.493064</td>
      <td>3.852810</td>
      <td>2.72</td>
      <td>8.11</td>
      <td>9.32</td>
      <td>24.05</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>50.64</td>
      <td>38.36</td>
      <td>1.15</td>
      <td>1.270416</td>
      <td>2.484316</td>
      <td>6.01</td>
      <td>12.97</td>
      <td>15.96</td>
      <td>31.21</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 11 columns</p>
</div>




```python
mig3['gini'].fillna(np.mean(mig3['gini']), inplace=True)
mig3['income_highest%'].fillna(np.mean(mig3['income_highest%']), inplace=True)
mig3['income_lowest%'].fillna(np.mean(mig3['income_lowest%']), inplace=True)
mig3['poor_1.90'].fillna(np.mean(mig3['poor_1.90']), inplace=True)
mig3['poor_3.10'].fillna(np.mean(mig3['poor_3.10']), inplace=True)
mig3['poverty_gap_1.90'].fillna(np.mean(mig3['poverty_gap_1.90']), inplace=True)
mig3['poverty_gap_3.10'].fillna(np.mean(mig3['poverty_gap_3.10']), inplace=True)
mig3['poverty_headcount_1.90'].fillna(np.mean(mig3['poverty_headcount_1.90']), inplace=True)
mig3['poverty_headcount_3.10'].fillna(np.mean(mig3['poverty_headcount_3.10']), inplace=True)
```


```python
mig3
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>gini</th>
      <th>income_highest%</th>
      <th>income_lowest%</th>
      <th>poor_1.90</th>
      <th>poor_3.10</th>
      <th>poverty_gap_1.90</th>
      <th>poverty_gap_3.10</th>
      <th>poverty_headcount_1.90</th>
      <th>poverty_headcount_3.10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>58.260000</td>
      <td>46.730000</td>
      <td>1.000000</td>
      <td>4.238208</td>
      <td>5.794880</td>
      <td>24.99</td>
      <td>39.010</td>
      <td>50.940000</td>
      <td>69.650000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>55.090000</td>
      <td>43.260000</td>
      <td>1.230000</td>
      <td>0.422176</td>
      <td>0.713758</td>
      <td>9.15</td>
      <td>18.940</td>
      <td>25.280000</td>
      <td>42.740000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>0.983164</td>
      <td>1.632736</td>
      <td>11.24</td>
      <td>16.660</td>
      <td>18.980000</td>
      <td>31.520000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>59.600000</td>
      <td>46.780000</td>
      <td>0.680000</td>
      <td>3.398988</td>
      <td>4.919682</td>
      <td>18.71</td>
      <td>29.680</td>
      <td>38.020000</td>
      <td>55.030000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>59.490000</td>
      <td>48.180000</td>
      <td>1.040000</td>
      <td>1.841220</td>
      <td>2.722716</td>
      <td>16.90</td>
      <td>29.140</td>
      <td>38.600000</td>
      <td>57.080000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>45.440000</td>
      <td>35.480000</td>
      <td>1.790000</td>
      <td>0.379692</td>
      <td>1.035198</td>
      <td>1.73</td>
      <td>5.570</td>
      <td>6.360000</td>
      <td>17.340000</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>54.890000</td>
      <td>43.560000</td>
      <td>1.070000</td>
      <td>1.552699</td>
      <td>3.195781</td>
      <td>3.93</td>
      <td>9.270</td>
      <td>11.510000</td>
      <td>23.690000</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>57.420000</td>
      <td>44.050000</td>
      <td>0.580000</td>
      <td>1.667679</td>
      <td>2.614730</td>
      <td>11.40</td>
      <td>18.850</td>
      <td>23.790000</td>
      <td>37.300000</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>45.240000</td>
      <td>35.720000</td>
      <td>1.930000</td>
      <td>0.268951</td>
      <td>0.835006</td>
      <td>1.08</td>
      <td>4.100</td>
      <td>4.490000</td>
      <td>13.940000</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>56.160000</td>
      <td>43.810000</td>
      <td>0.900000</td>
      <td>1.242759</td>
      <td>2.279461</td>
      <td>6.91</td>
      <td>13.880</td>
      <td>17.430000</td>
      <td>31.970000</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>46.650000</td>
      <td>36.040000</td>
      <td>1.700000</td>
      <td>0.415200</td>
      <td>1.114800</td>
      <td>1.99</td>
      <td>6.090</td>
      <td>6.920000</td>
      <td>18.580000</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>55.740000</td>
      <td>43.870000</td>
      <td>0.910000</td>
      <td>1.171764</td>
      <td>2.130810</td>
      <td>6.30</td>
      <td>12.680</td>
      <td>16.140000</td>
      <td>29.350000</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>45.930000</td>
      <td>36.070000</td>
      <td>1.780000</td>
      <td>0.384678</td>
      <td>1.054102</td>
      <td>1.67</td>
      <td>5.520</td>
      <td>6.390000</td>
      <td>17.510000</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>51.560000</td>
      <td>39.140000</td>
      <td>1.150000</td>
      <td>1.036152</td>
      <td>1.979316</td>
      <td>4.82</td>
      <td>10.880</td>
      <td>14.040000</td>
      <td>26.820000</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>44.530000</td>
      <td>33.700000</td>
      <td>1.670000</td>
      <td>0.437296</td>
      <td>1.120420</td>
      <td>2.33</td>
      <td>6.300</td>
      <td>7.240000</td>
      <td>18.550000</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>53.390000</td>
      <td>41.020000</td>
      <td>1.090000</td>
      <td>1.160250</td>
      <td>2.183250</td>
      <td>5.40</td>
      <td>11.900</td>
      <td>15.470000</td>
      <td>29.110000</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>42.430000</td>
      <td>32.860000</td>
      <td>2.110000</td>
      <td>0.274518</td>
      <td>0.911424</td>
      <td>1.06</td>
      <td>4.390</td>
      <td>4.530000</td>
      <td>15.040000</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>52.350000</td>
      <td>41.830000</td>
      <td>1.340000</td>
      <td>1.735265</td>
      <td>3.983735</td>
      <td>4.00</td>
      <td>9.840</td>
      <td>11.530000</td>
      <td>26.470000</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>57.400000</td>
      <td>45.670000</td>
      <td>0.750000</td>
      <td>1.428750</td>
      <td>2.489454</td>
      <td>7.88</td>
      <td>14.660</td>
      <td>18.750000</td>
      <td>32.670000</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>41.800000</td>
      <td>32.470000</td>
      <td>2.150000</td>
      <td>0.252512</td>
      <td>0.826127</td>
      <td>0.98</td>
      <td>3.840</td>
      <td>4.160000</td>
      <td>13.610000</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>57.400000</td>
      <td>45.680000</td>
      <td>0.790000</td>
      <td>1.653264</td>
      <td>2.883924</td>
      <td>9.25</td>
      <td>17.100</td>
      <td>21.360000</td>
      <td>37.260000</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>43.510000</td>
      <td>34.350000</td>
      <td>2.110000</td>
      <td>0.197925</td>
      <td>0.702177</td>
      <td>0.74</td>
      <td>3.160</td>
      <td>3.250000</td>
      <td>11.530000</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>53.670000</td>
      <td>41.480000</td>
      <td>0.980000</td>
      <td>1.486005</td>
      <td>2.712175</td>
      <td>7.66</td>
      <td>15.240</td>
      <td>18.930000</td>
      <td>34.550000</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>41.840000</td>
      <td>32.310000</td>
      <td>2.190000</td>
      <td>0.181467</td>
      <td>0.689819</td>
      <td>0.64</td>
      <td>3.000</td>
      <td>2.970000</td>
      <td>11.290000</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>48.660000</td>
      <td>38.360000</td>
      <td>1.640000</td>
      <td>1.493064</td>
      <td>3.852810</td>
      <td>2.72</td>
      <td>8.110</td>
      <td>9.320000</td>
      <td>24.050000</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>50.640000</td>
      <td>38.360000</td>
      <td>1.150000</td>
      <td>1.270416</td>
      <td>2.484316</td>
      <td>6.01</td>
      <td>12.970</td>
      <td>15.960000</td>
      <td>31.210000</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 11 columns</p>
</div>




```python
mig4
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>Time Code</th>
      <th>Country Name</th>
      <th>Country Code</th>
      <th>Age dependency ratio (% of working-age population) [SP.POP.DPND]</th>
      <th>Birth rate, crude (per 1,000 people) [SP.DYN.CBRT.IN]</th>
      <th>Death rate, crude (per 1,000 people) [SP.DYN.CDRT.IN]</th>
      <th>Fertility rate, total (births per woman) [SP.DYN.TFRT.IN]</th>
      <th>Life expectancy at birth, total (years) [SP.DYN.LE00.IN]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>YR1980</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>89.549266</td>
      <td>37.353</td>
      <td>11.681</td>
      <td>5.087</td>
      <td>56.529927</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>YR1980</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>93.787268</td>
      <td>43.686</td>
      <td>11.568</td>
      <td>6.195</td>
      <td>57.201488</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>YR1980</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>100.729300</td>
      <td>43.476</td>
      <td>10.233</td>
      <td>6.313</td>
      <td>59.612122</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>YR1981</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>88.861531</td>
      <td>36.593</td>
      <td>11.494</td>
      <td>4.952</td>
      <td>56.798976</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>YR1981</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>94.516037</td>
      <td>43.384</td>
      <td>11.300</td>
      <td>6.161</td>
      <td>57.632756</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>YR1981</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>100.456778</td>
      <td>43.020</td>
      <td>9.793</td>
      <td>6.190</td>
      <td>60.405854</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>YR1982</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>88.129782</td>
      <td>35.833</td>
      <td>11.251</td>
      <td>4.819</td>
      <td>57.197537</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>YR1982</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>95.163313</td>
      <td>42.955</td>
      <td>11.016</td>
      <td>6.105</td>
      <td>58.085951</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>YR1982</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>99.827463</td>
      <td>42.524</td>
      <td>9.359</td>
      <td>6.062</td>
      <td>61.212073</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>YR1983</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>87.385786</td>
      <td>35.093</td>
      <td>10.953</td>
      <td>4.688</td>
      <td>57.731659</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>YR1983</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>95.677113</td>
      <td>42.406</td>
      <td>10.718</td>
      <td>6.027</td>
      <td>58.558634</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>YR1983</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>99.020708</td>
      <td>41.998</td>
      <td>8.933</td>
      <td>5.932</td>
      <td>62.024805</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>YR1984</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>86.654026</td>
      <td>34.383</td>
      <td>10.602</td>
      <td>4.562</td>
      <td>58.399829</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>YR1984</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>95.981662</td>
      <td>41.757</td>
      <td>10.406</td>
      <td>5.930</td>
      <td>59.053829</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>YR1984</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>98.253395</td>
      <td>41.452</td>
      <td>8.520</td>
      <td>5.802</td>
      <td>62.831537</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>YR1985</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>85.923075</td>
      <td>33.716</td>
      <td>10.205</td>
      <td>4.442</td>
      <td>59.193976</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>YR1985</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>96.038252</td>
      <td>41.053</td>
      <td>10.085</td>
      <td>5.820</td>
      <td>59.568073</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>YR1985</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>97.609622</td>
      <td>40.898</td>
      <td>8.129</td>
      <td>5.674</td>
      <td>63.613756</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>YR1986</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>84.444327</td>
      <td>33.096</td>
      <td>9.775</td>
      <td>4.328</td>
      <td>60.099854</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>YR1986</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>96.304598</td>
      <td>40.351</td>
      <td>9.762</td>
      <td>5.704</td>
      <td>60.097317</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>YR1986</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>97.243943</td>
      <td>40.353</td>
      <td>7.769</td>
      <td>5.553</td>
      <td>64.350951</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>YR1987</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>83.083490</td>
      <td>32.516</td>
      <td>9.330</td>
      <td>4.221</td>
      <td>61.074000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>YR1987</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>96.192946</td>
      <td>39.705</td>
      <td>9.444</td>
      <td>5.591</td>
      <td>60.633098</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>YR1987</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>96.907141</td>
      <td>39.826</td>
      <td>7.447</td>
      <td>5.439</td>
      <td>65.030146</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>YR1988</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>81.807159</td>
      <td>31.963</td>
      <td>8.893</td>
      <td>4.118</td>
      <td>62.071463</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>YR1988</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>95.809701</td>
      <td>39.153</td>
      <td>9.137</td>
      <td>5.488</td>
      <td>61.169878</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>YR1988</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>96.558949</td>
      <td>39.319</td>
      <td>7.164</td>
      <td>5.333</td>
      <td>65.645293</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>YR1989</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>80.578125</td>
      <td>31.429</td>
      <td>8.480</td>
      <td>4.018</td>
      <td>63.056415</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>YR1989</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>95.344674</td>
      <td>38.707</td>
      <td>8.845</td>
      <td>5.396</td>
      <td>61.704659</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>YR1989</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>96.118688</td>
      <td>38.830</td>
      <td>6.920</td>
      <td>5.233</td>
      <td>66.194439</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>YR2006</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>66.256872</td>
      <td>19.887</td>
      <td>6.672</td>
      <td>2.322</td>
      <td>70.479171</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>YR2006</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>82.841905</td>
      <td>30.710</td>
      <td>5.663</td>
      <td>3.760</td>
      <td>69.887902</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>YR2006</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>75.860438</td>
      <td>26.318</td>
      <td>5.105</td>
      <td>3.164</td>
      <td>71.672463</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>YR2007</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>64.944613</td>
      <td>19.406</td>
      <td>6.662</td>
      <td>2.253</td>
      <td>70.780463</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>YR2007</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>81.475709</td>
      <td>30.085</td>
      <td>5.616</td>
      <td>3.664</td>
      <td>70.110780</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>YR2007</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>73.754313</td>
      <td>25.514</td>
      <td>5.077</td>
      <td>3.039</td>
      <td>71.858732</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>YR2008</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>63.639250</td>
      <td>18.969</td>
      <td>6.659</td>
      <td>2.189</td>
      <td>71.080780</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>YR2008</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>80.024334</td>
      <td>29.519</td>
      <td>5.576</td>
      <td>3.578</td>
      <td>70.328146</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>YR2008</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>71.645904</td>
      <td>24.728</td>
      <td>5.055</td>
      <td>2.918</td>
      <td>72.039976</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>YR2009</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>62.302599</td>
      <td>18.574</td>
      <td>6.663</td>
      <td>2.130</td>
      <td>71.378146</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>YR2009</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>78.579556</td>
      <td>29.016</td>
      <td>5.539</td>
      <td>3.501</td>
      <td>70.547537</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>YR2009</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>69.528024</td>
      <td>23.971</td>
      <td>5.038</td>
      <td>2.802</td>
      <td>72.217220</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>YR2010</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>60.931929</td>
      <td>18.223</td>
      <td>6.673</td>
      <td>2.078</td>
      <td>71.670610</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>YR2010</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>77.198083</td>
      <td>28.574</td>
      <td>5.503</td>
      <td>3.434</td>
      <td>70.775463</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>YR2010</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>67.411337</td>
      <td>23.261</td>
      <td>5.026</td>
      <td>2.695</td>
      <td>72.393976</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>YR2011</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>59.472471</td>
      <td>17.924</td>
      <td>6.692</td>
      <td>2.031</td>
      <td>71.956171</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>YR2011</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>75.784683</td>
      <td>28.182</td>
      <td>5.467</td>
      <td>3.373</td>
      <td>71.010415</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>YR2011</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>65.342451</td>
      <td>22.622</td>
      <td>5.017</td>
      <td>2.599</td>
      <td>72.572732</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>YR2012</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>57.970629</td>
      <td>17.676</td>
      <td>6.718</td>
      <td>1.991</td>
      <td>72.231854</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>YR2012</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>74.496429</td>
      <td>27.819</td>
      <td>5.433</td>
      <td>3.317</td>
      <td>71.249390</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>YR2012</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>63.264970</td>
      <td>22.065</td>
      <td>5.012</td>
      <td>2.514</td>
      <td>72.755024</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>YR2013</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>56.531230</td>
      <td>17.476</td>
      <td>6.751</td>
      <td>1.958</td>
      <td>72.498146</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>YR2013</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>73.279326</td>
      <td>27.465</td>
      <td>5.401</td>
      <td>3.263</td>
      <td>71.486390</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>YR2013</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>61.250546</td>
      <td>21.593</td>
      <td>5.010</td>
      <td>2.442</td>
      <td>72.942854</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>YR2014</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>55.294848</td>
      <td>17.314</td>
      <td>6.790</td>
      <td>1.931</td>
      <td>72.754561</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>YR2014</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>72.069718</td>
      <td>27.112</td>
      <td>5.370</td>
      <td>3.211</td>
      <td>71.722415</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>YR2014</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>59.400971</td>
      <td>21.203</td>
      <td>5.011</td>
      <td>2.382</td>
      <td>73.135707</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>YR2015</td>
      <td>El Salvador</td>
      <td>SLV</td>
      <td>54.321951</td>
      <td>17.175</td>
      <td>6.833</td>
      <td>1.909</td>
      <td>73.001098</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>YR2015</td>
      <td>Guatemala</td>
      <td>GTM</td>
      <td>70.850672</td>
      <td>26.752</td>
      <td>5.339</td>
      <td>3.159</td>
      <td>71.956488</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>YR2015</td>
      <td>Honduras</td>
      <td>HND</td>
      <td>57.768677</td>
      <td>20.881</td>
      <td>5.015</td>
      <td>2.332</td>
      <td>73.333122</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 9 columns</p>
</div>




```python
mig4.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 108 entries, 0 to 107
    Data columns (total 9 columns):
    Time                                                                108 non-null int64
    Time Code                                                           108 non-null object
    Country Name                                                        108 non-null object
    Country Code                                                        108 non-null object
    Age dependency ratio (% of working-age population) [SP.POP.DPND]    108 non-null float64
    Birth rate, crude (per 1,000 people) [SP.DYN.CBRT.IN]               108 non-null float64
    Death rate, crude (per 1,000 people) [SP.DYN.CDRT.IN]               108 non-null float64
    Fertility rate, total (births per woman) [SP.DYN.TFRT.IN]           108 non-null float64
    Life expectancy at birth, total (years) [SP.DYN.LE00.IN]            108 non-null float64
    dtypes: float64(5), int64(1), object(3)
    memory usage: 7.7+ KB



```python
del mig4['Time Code']
del mig4['Country Code']
```


```python
mig4.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 108 entries, 0 to 107
    Data columns (total 7 columns):
    Time                                                                108 non-null int64
    Country Name                                                        108 non-null object
    Age dependency ratio (% of working-age population) [SP.POP.DPND]    108 non-null float64
    Birth rate, crude (per 1,000 people) [SP.DYN.CBRT.IN]               108 non-null float64
    Death rate, crude (per 1,000 people) [SP.DYN.CDRT.IN]               108 non-null float64
    Fertility rate, total (births per woman) [SP.DYN.TFRT.IN]           108 non-null float64
    Life expectancy at birth, total (years) [SP.DYN.LE00.IN]            108 non-null float64
    dtypes: float64(5), int64(1), object(1)
    memory usage: 6.0+ KB



```python
mig4.rename(columns={
        'Time': 'Year',
        'Country Name': 'Country',
        'Age dependency ratio (% of working-age population) [SP.POP.DPND]': 'age_dependency',
        'Birth rate, crude (per 1,000 people) [SP.DYN.CBRT.IN]': 'birth_rate',
        'Death rate, crude (per 1,000 people) [SP.DYN.CDRT.IN]': 'death_rate',
        'Fertility rate, total (births per woman) [SP.DYN.TFRT.IN]': 'fertility_rate',
        'Life expectancy at birth, total (years) [SP.DYN.LE00.IN]': 'life_expectancy'
        }, inplace=True)
```


```python
mig4
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>age_dependency</th>
      <th>birth_rate</th>
      <th>death_rate</th>
      <th>fertility_rate</th>
      <th>life_expectancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>89.549266</td>
      <td>37.353</td>
      <td>11.681</td>
      <td>5.087</td>
      <td>56.529927</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>93.787268</td>
      <td>43.686</td>
      <td>11.568</td>
      <td>6.195</td>
      <td>57.201488</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>100.729300</td>
      <td>43.476</td>
      <td>10.233</td>
      <td>6.313</td>
      <td>59.612122</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>88.861531</td>
      <td>36.593</td>
      <td>11.494</td>
      <td>4.952</td>
      <td>56.798976</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>94.516037</td>
      <td>43.384</td>
      <td>11.300</td>
      <td>6.161</td>
      <td>57.632756</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>100.456778</td>
      <td>43.020</td>
      <td>9.793</td>
      <td>6.190</td>
      <td>60.405854</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>88.129782</td>
      <td>35.833</td>
      <td>11.251</td>
      <td>4.819</td>
      <td>57.197537</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>95.163313</td>
      <td>42.955</td>
      <td>11.016</td>
      <td>6.105</td>
      <td>58.085951</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>99.827463</td>
      <td>42.524</td>
      <td>9.359</td>
      <td>6.062</td>
      <td>61.212073</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>87.385786</td>
      <td>35.093</td>
      <td>10.953</td>
      <td>4.688</td>
      <td>57.731659</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>95.677113</td>
      <td>42.406</td>
      <td>10.718</td>
      <td>6.027</td>
      <td>58.558634</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>99.020708</td>
      <td>41.998</td>
      <td>8.933</td>
      <td>5.932</td>
      <td>62.024805</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>86.654026</td>
      <td>34.383</td>
      <td>10.602</td>
      <td>4.562</td>
      <td>58.399829</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>95.981662</td>
      <td>41.757</td>
      <td>10.406</td>
      <td>5.930</td>
      <td>59.053829</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>98.253395</td>
      <td>41.452</td>
      <td>8.520</td>
      <td>5.802</td>
      <td>62.831537</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>85.923075</td>
      <td>33.716</td>
      <td>10.205</td>
      <td>4.442</td>
      <td>59.193976</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>96.038252</td>
      <td>41.053</td>
      <td>10.085</td>
      <td>5.820</td>
      <td>59.568073</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>97.609622</td>
      <td>40.898</td>
      <td>8.129</td>
      <td>5.674</td>
      <td>63.613756</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>84.444327</td>
      <td>33.096</td>
      <td>9.775</td>
      <td>4.328</td>
      <td>60.099854</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>96.304598</td>
      <td>40.351</td>
      <td>9.762</td>
      <td>5.704</td>
      <td>60.097317</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>97.243943</td>
      <td>40.353</td>
      <td>7.769</td>
      <td>5.553</td>
      <td>64.350951</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>83.083490</td>
      <td>32.516</td>
      <td>9.330</td>
      <td>4.221</td>
      <td>61.074000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>96.192946</td>
      <td>39.705</td>
      <td>9.444</td>
      <td>5.591</td>
      <td>60.633098</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>96.907141</td>
      <td>39.826</td>
      <td>7.447</td>
      <td>5.439</td>
      <td>65.030146</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>81.807159</td>
      <td>31.963</td>
      <td>8.893</td>
      <td>4.118</td>
      <td>62.071463</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>95.809701</td>
      <td>39.153</td>
      <td>9.137</td>
      <td>5.488</td>
      <td>61.169878</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>96.558949</td>
      <td>39.319</td>
      <td>7.164</td>
      <td>5.333</td>
      <td>65.645293</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>80.578125</td>
      <td>31.429</td>
      <td>8.480</td>
      <td>4.018</td>
      <td>63.056415</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>95.344674</td>
      <td>38.707</td>
      <td>8.845</td>
      <td>5.396</td>
      <td>61.704659</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>96.118688</td>
      <td>38.830</td>
      <td>6.920</td>
      <td>5.233</td>
      <td>66.194439</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>66.256872</td>
      <td>19.887</td>
      <td>6.672</td>
      <td>2.322</td>
      <td>70.479171</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>82.841905</td>
      <td>30.710</td>
      <td>5.663</td>
      <td>3.760</td>
      <td>69.887902</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>75.860438</td>
      <td>26.318</td>
      <td>5.105</td>
      <td>3.164</td>
      <td>71.672463</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>64.944613</td>
      <td>19.406</td>
      <td>6.662</td>
      <td>2.253</td>
      <td>70.780463</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>81.475709</td>
      <td>30.085</td>
      <td>5.616</td>
      <td>3.664</td>
      <td>70.110780</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>73.754313</td>
      <td>25.514</td>
      <td>5.077</td>
      <td>3.039</td>
      <td>71.858732</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>63.639250</td>
      <td>18.969</td>
      <td>6.659</td>
      <td>2.189</td>
      <td>71.080780</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>80.024334</td>
      <td>29.519</td>
      <td>5.576</td>
      <td>3.578</td>
      <td>70.328146</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>71.645904</td>
      <td>24.728</td>
      <td>5.055</td>
      <td>2.918</td>
      <td>72.039976</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>62.302599</td>
      <td>18.574</td>
      <td>6.663</td>
      <td>2.130</td>
      <td>71.378146</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>78.579556</td>
      <td>29.016</td>
      <td>5.539</td>
      <td>3.501</td>
      <td>70.547537</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>69.528024</td>
      <td>23.971</td>
      <td>5.038</td>
      <td>2.802</td>
      <td>72.217220</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>60.931929</td>
      <td>18.223</td>
      <td>6.673</td>
      <td>2.078</td>
      <td>71.670610</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>77.198083</td>
      <td>28.574</td>
      <td>5.503</td>
      <td>3.434</td>
      <td>70.775463</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>67.411337</td>
      <td>23.261</td>
      <td>5.026</td>
      <td>2.695</td>
      <td>72.393976</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>59.472471</td>
      <td>17.924</td>
      <td>6.692</td>
      <td>2.031</td>
      <td>71.956171</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>75.784683</td>
      <td>28.182</td>
      <td>5.467</td>
      <td>3.373</td>
      <td>71.010415</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>65.342451</td>
      <td>22.622</td>
      <td>5.017</td>
      <td>2.599</td>
      <td>72.572732</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>57.970629</td>
      <td>17.676</td>
      <td>6.718</td>
      <td>1.991</td>
      <td>72.231854</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>74.496429</td>
      <td>27.819</td>
      <td>5.433</td>
      <td>3.317</td>
      <td>71.249390</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>63.264970</td>
      <td>22.065</td>
      <td>5.012</td>
      <td>2.514</td>
      <td>72.755024</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>56.531230</td>
      <td>17.476</td>
      <td>6.751</td>
      <td>1.958</td>
      <td>72.498146</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>73.279326</td>
      <td>27.465</td>
      <td>5.401</td>
      <td>3.263</td>
      <td>71.486390</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>61.250546</td>
      <td>21.593</td>
      <td>5.010</td>
      <td>2.442</td>
      <td>72.942854</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>55.294848</td>
      <td>17.314</td>
      <td>6.790</td>
      <td>1.931</td>
      <td>72.754561</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>72.069718</td>
      <td>27.112</td>
      <td>5.370</td>
      <td>3.211</td>
      <td>71.722415</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>59.400971</td>
      <td>21.203</td>
      <td>5.011</td>
      <td>2.382</td>
      <td>73.135707</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>54.321951</td>
      <td>17.175</td>
      <td>6.833</td>
      <td>1.909</td>
      <td>73.001098</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>70.850672</td>
      <td>26.752</td>
      <td>5.339</td>
      <td>3.159</td>
      <td>71.956488</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>57.768677</td>
      <td>20.881</td>
      <td>5.015</td>
      <td>2.332</td>
      <td>73.333122</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 7 columns</p>
</div>




```python
mig5.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 116 entries, 0 to 115
    Data columns (total 18 columns):
    Time                                                                                         113 non-null object
    Time Code                                                                                    111 non-null object
    Country Name                                                                                 111 non-null object
    Country Code                                                                                 111 non-null object
    Unemployment, total (% of total labor force) (national estimate) [SL.UEM.TOTL.NE.ZS]         111 non-null object
    Trade (% of GDP) [NE.TRD.GNFS.ZS]                                                            111 non-null object
    Short-term debt (% of total external debt) [DT.DOD.DSTC.ZS]                                  111 non-null object
    Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population) [SI.POV.DDAY]            111 non-null object
    Poverty headcount ratio at $3.10 a day (2011 PPP) (% of population) [SI.POV.2DAY]            111 non-null object
    Population growth (annual %) [SP.POP.GROW]                                                   111 non-null object
    Personal remittances, received (% of GDP) [BX.TRF.PWKR.DT.GD.ZS]                             111 non-null object
    Net bilateral aid flows from DAC donors, United States (current US$) [DC.DAC.USAL.CD]        111 non-null object
    Imports of goods and services (% of GDP) [NE.IMP.GNFS.ZS]                                    111 non-null object
    General government final consumption expenditure (% of GDP) [NE.CON.GOVT.ZS]                 111 non-null object
    Foreign direct investment, net outflows (% of GDP) [BM.KLT.DINV.WD.GD.ZS]                    111 non-null object
    Final consumption expenditure, etc. (% of GDP) [NE.CON.TETC.ZS]                              111 non-null object
    Exports of goods and services (% of GDP) [NE.EXP.GNFS.ZS]                                    111 non-null object
    Employment to population ratio, 15+, total (%) (national estimate) [SL.EMP.TOTL.SP.NE.ZS]    111 non-null object
    dtypes: object(18)
    memory usage: 16.4+ KB



```python
del mig5['Time Code']
del mig5['Country Code']
del mig5['Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population) [SI.POV.DDAY]']
del mig5['Poverty headcount ratio at $3.10 a day (2011 PPP) (% of population) [SI.POV.2DAY]']
del mig5['Final consumption expenditure, etc. (% of GDP) [NE.CON.TETC.ZS]']
```


```python
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
```


```python
mig5
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>unemployment</th>
      <th>trade</th>
      <th>short_term_debt</th>
      <th>pop_growth</th>
      <th>remittances</th>
      <th>net_bilateral_aid</th>
      <th>imports_%GDP</th>
      <th>gov_consumption</th>
      <th>FDI</th>
      <th>exports_%GDP</th>
      <th>employment_15+</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>13.34000015</td>
      <td>67.40646419</td>
      <td>22.07</td>
      <td>1.739183852</td>
      <td>1.372147486</td>
      <td>43000000</td>
      <td>33.24491693</td>
      <td>13.98896501</td>
      <td>..</td>
      <td>34.16154725</td>
      <td>..</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>47.10548695</td>
      <td>26.383</td>
      <td>2.635142783</td>
      <td>0.33254218</td>
      <td>17000000</td>
      <td>24.91908564</td>
      <td>7.958165687</td>
      <td>0.025384899</td>
      <td>22.18640131</td>
      <td>49.61000061</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>..</td>
      <td>81.29383902</td>
      <td>17.3081</td>
      <td>3.14529964</td>
      <td>0.062353858</td>
      <td>19000000</td>
      <td>44.05689509</td>
      <td>12.66562719</td>
      <td>0.038971161</td>
      <td>37.23694394</td>
      <td>27.37999916</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>60.26649248</td>
      <td>17.3809</td>
      <td>1.611672741</td>
      <td>2.108693097</td>
      <td>97000000</td>
      <td>33.58780207</td>
      <td>15.82916235</td>
      <td>..</td>
      <td>26.67869041</td>
      <td>..</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>2.150000095</td>
      <td>40.69125737</td>
      <td>9.9649</td>
      <td>2.658257123</td>
      <td>0.284635482</td>
      <td>18000000</td>
      <td>23.60150949</td>
      <td>7.900086858</td>
      <td>-0.011617775</td>
      <td>17.08974788</td>
      <td>..</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>..</td>
      <td>69.3385352</td>
      <td>13.3695</td>
      <td>3.113439292</td>
      <td>0.062067743</td>
      <td>35000000</td>
      <td>37.70172016</td>
      <td>12.78595496</td>
      <td>0.070934563</td>
      <td>31.63681504</td>
      <td>26.43000031</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>51.24774023</td>
      <td>13.4222</td>
      <td>1.498033738</td>
      <td>3.300787147</td>
      <td>170000000</td>
      <td>28.47033723</td>
      <td>15.7782337</td>
      <td>..</td>
      <td>22.777403</td>
      <td>..</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>2.269999981</td>
      <td>33.47481818</td>
      <td>7.235</td>
      <td>2.669298084</td>
      <td>0.122748654</td>
      <td>20000000</td>
      <td>18.68762138</td>
      <td>7.743489999</td>
      <td>-0.045887348</td>
      <td>14.78719679</td>
      <td>..</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>7.300000191</td>
      <td>54.72705089</td>
      <td>7.7859</td>
      <td>3.083348528</td>
      <td>0.051661787</td>
      <td>68000000</td>
      <td>28.05234841</td>
      <td>13.05321142</td>
      <td>-0.034441191</td>
      <td>26.67470249</td>
      <td>40.09000015</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>54.39780104</td>
      <td>5.1284</td>
      <td>1.413013599</td>
      <td>3.292314653</td>
      <td>231000000</td>
      <td>29.90928339</td>
      <td>15.82944795</td>
      <td>..</td>
      <td>24.48851765</td>
      <td>..</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>27.546959</td>
      <td>6.1899</td>
      <td>2.654604276</td>
      <td>0.043093922</td>
      <td>36000000</td>
      <td>14.55248444</td>
      <td>7.602209609</td>
      <td>0</td>
      <td>12.99447456</td>
      <td>..</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>..</td>
      <td>55.39486765</td>
      <td>6.3325</td>
      <td>3.056877686</td>
      <td>0.058498537</td>
      <td>64000000</td>
      <td>29.23302127</td>
      <td>13.11342238</td>
      <td>0.064998376</td>
      <td>26.16184638</td>
      <td>35.59999847</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>50.29067505</td>
      <td>5.8131</td>
      <td>1.364659688</td>
      <td>4.345542138</td>
      <td>221000000</td>
      <td>28.53653596</td>
      <td>16.036198</td>
      <td>..</td>
      <td>21.75413909</td>
      <td>..</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>28.1531148</td>
      <td>6.1022</td>
      <td>2.607323499</td>
      <td>0.035902852</td>
      <td>29000000</td>
      <td>15.15100301</td>
      <td>7.665258631</td>
      <td>0.05279831</td>
      <td>13.0021118</td>
      <td>..</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>..</td>
      <td>57.72823139</td>
      <td>8.26</td>
      <td>3.037318621</td>
      <td>0.058752638</td>
      <td>123000000</td>
      <td>32.02771919</td>
      <td>13.19674601</td>
      <td>-0.030129557</td>
      <td>25.7005122</td>
      <td>38.72000122</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>16.95000076</td>
      <td>52.21053821</td>
      <td>4.3227</td>
      <td>1.343117088</td>
      <td>4.135388438</td>
      <td>287000000</td>
      <td>29.88677204</td>
      <td>15.4905211</td>
      <td>..</td>
      <td>22.32376617</td>
      <td>..</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>24.93224559</td>
      <td>9.564</td>
      <td>2.541226177</td>
      <td>0.010286318</td>
      <td>50000000</td>
      <td>12.98401607</td>
      <td>6.953550631</td>
      <td>0</td>
      <td>11.94822952</td>
      <td>51.29999924</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>..</td>
      <td>54.96634366</td>
      <td>10.6047</td>
      <td>3.020230628</td>
      <td>0.057700232</td>
      <td>161000000</td>
      <td>29.86674172</td>
      <td>13.09245794</td>
      <td>0</td>
      <td>25.09960194</td>
      <td>..</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>7.900000095</td>
      <td>53.71412272</td>
      <td>6.5223</td>
      <td>1.324193255</td>
      <td>4.170337688</td>
      <td>272000000</td>
      <td>29.04596041</td>
      <td>14.18109655</td>
      <td>..</td>
      <td>24.66816231</td>
      <td>..</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>30.64401925</td>
      <td>10.711</td>
      <td>2.470000701</td>
      <td>0.009679252</td>
      <td>86000000</td>
      <td>14.59211949</td>
      <td>7.096224226</td>
      <td>0</td>
      <td>16.05189976</td>
      <td>..</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>12.11999989</td>
      <td>54.89037607</td>
      <td>11.6717</td>
      <td>2.998073065</td>
      <td>0.055139816</td>
      <td>175000000</td>
      <td>28.30510663</td>
      <td>14.27071138</td>
      <td>0</td>
      <td>26.58526944</td>
      <td>..</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>45.0946222</td>
      <td>10.3707</td>
      <td>1.302156496</td>
      <td>4.715458305</td>
      <td>356000000</td>
      <td>26.10231797</td>
      <td>13.74570501</td>
      <td>..</td>
      <td>18.99230423</td>
      <td>..</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>3.5</td>
      <td>38.14296343</td>
      <td>9.8328</td>
      <td>2.413838221</td>
      <td>0.001411552</td>
      <td>155000000</td>
      <td>22.29405505</td>
      <td>7.902433694</td>
      <td>0.014115522</td>
      <td>15.84890838</td>
      <td>..</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>11.39999962</td>
      <td>48.7898862</td>
      <td>11.9562</td>
      <td>2.966596303</td>
      <td>0.81155933</td>
      <td>153000000</td>
      <td>25.82781488</td>
      <td>14.22034815</td>
      <td>0.024081879</td>
      <td>22.96207132</td>
      <td>..</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>9.369999886</td>
      <td>38.09570441</td>
      <td>11.637</td>
      <td>1.291964388</td>
      <td>5.029738453</td>
      <td>318000000</td>
      <td>22.28502248</td>
      <td>12.73167251</td>
      <td>..</td>
      <td>15.81068193</td>
      <td>..</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>38.03991399</td>
      <td>11.0787</td>
      <td>2.383683528</td>
      <td>0.582789026</td>
      <td>134000000</td>
      <td>21.93623865</td>
      <td>7.981504735</td>
      <td>-0.012752495</td>
      <td>16.10367533</td>
      <td>..</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>..</td>
      <td>55.21565236</td>
      <td>12.1001</td>
      <td>2.927378347</td>
      <td>1.052794273</td>
      <td>155000000</td>
      <td>28.92660253</td>
      <td>14.139012</td>
      <td>-0.025186466</td>
      <td>26.28904983</td>
      <td>..</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>8.350000381</td>
      <td>36.92829582</td>
      <td>9.7005</td>
      <td>1.293850446</td>
      <td>5.439348042</td>
      <td>310000000</td>
      <td>23.69050765</td>
      <td>12.19413417</td>
      <td>..</td>
      <td>13.23778818</td>
      <td>..</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>2</td>
      <td>39.78154615</td>
      <td>12.1191</td>
      <td>2.387014157</td>
      <td>1.010614501</td>
      <td>146000000</td>
      <td>22.47409728</td>
      <td>7.895425786</td>
      <td>-0.047558329</td>
      <td>17.30744887</td>
      <td>..</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>..</td>
      <td>65.34739631</td>
      <td>11.4579</td>
      <td>2.879679331</td>
      <td>1.363847438</td>
      <td>102000000</td>
      <td>34.34294528</td>
      <td>14.27327352</td>
      <td>-0.028062705</td>
      <td>31.00445103</td>
      <td>..</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>2.99000001</td>
      <td>135.7489552</td>
      <td>13.3425</td>
      <td>1.747160287</td>
      <td>20.45975298</td>
      <td>96330000</td>
      <td>84.42367851</td>
      <td>17.11482759</td>
      <td>1.402888726</td>
      <td>51.32527669</td>
      <td>..</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>7.329999924</td>
      <td>61.87164222</td>
      <td>8.3258</td>
      <td>0.285541824</td>
      <td>16.46745075</td>
      <td>82080000</td>
      <td>38.67479793</td>
      <td>10.62920478</td>
      <td>0.014326509</td>
      <td>23.1968443</td>
      <td>58.15999985</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>57.1059931</td>
      <td>8.2714</td>
      <td>2.183077451</td>
      <td>10.65185852</td>
      <td>83890000</td>
      <td>33.13058888</td>
      <td>10.17902302</td>
      <td>0.325370676</td>
      <td>23.97540421</td>
      <td>..</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>3.279999971</td>
      <td>96.90500602</td>
      <td>6.5494</td>
      <td>1.688651031</td>
      <td>17.10144713</td>
      <td>128760000</td>
      <td>57.3747552</td>
      <td>18.69973827</td>
      <td>-0.074316898</td>
      <td>39.53025082</td>
      <td>59.31999969</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>7.050000191</td>
      <td>68.76876316</td>
      <td>7.5574</td>
      <td>0.2809033</td>
      <td>16.20967453</td>
      <td>148160000</td>
      <td>42.84420332</td>
      <td>10.70486453</td>
      <td>0.524806817</td>
      <td>25.92455984</td>
      <td>58.11000061</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>3.74000001</td>
      <td>62.11493226</td>
      <td>10.5394</td>
      <td>2.156000018</td>
      <td>10.23686713</td>
      <td>100500000</td>
      <td>36.30918223</td>
      <td>10.47586426</td>
      <td>0.153036418</td>
      <td>25.80575003</td>
      <td>42.91999817</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>4.099999905</td>
      <td>109.4418382</td>
      <td>9.411</td>
      <td>1.622622006</td>
      <td>16.64314011</td>
      <td>100840000</td>
      <td>63.68293204</td>
      <td>17.92643157</td>
      <td>-2.309011866</td>
      <td>45.7589062</td>
      <td>59.36000061</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>6.619999886</td>
      <td>74.64324301</td>
      <td>10.1925</td>
      <td>0.279521916</td>
      <td>15.74808439</td>
      <td>162440000</td>
      <td>46.6632093</td>
      <td>11.05579325</td>
      <td>-0.414577558</td>
      <td>27.98003371</td>
      <td>58.54999924</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>4.130000114</td>
      <td>63.98419583</td>
      <td>13.9156</td>
      <td>2.129043456</td>
      <td>9.492784648</td>
      <td>93080000</td>
      <td>37.3587707</td>
      <td>10.18922762</td>
      <td>0.274375676</td>
      <td>26.62542512</td>
      <td>59.22999954</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>4.269999981</td>
      <td>122.2169026</td>
      <td>6.0942</td>
      <td>1.554236301</td>
      <td>15.98008307</td>
      <td>46360000</td>
      <td>70.95921592</td>
      <td>16.06436839</td>
      <td>0.172123867</td>
      <td>51.25768668</td>
      <td>49.65999985</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>6.070000172</td>
      <td>69.69882756</td>
      <td>11.7092</td>
      <td>0.280768403</td>
      <td>16.36100124</td>
      <td>150850000</td>
      <td>44.0764941</td>
      <td>11.22845769</td>
      <td>-0.150728911</td>
      <td>25.62233346</td>
      <td>59.40000153</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>2.869999886</td>
      <td>60.98247455</td>
      <td>4.5907</td>
      <td>2.100666307</td>
      <td>9.983741683</td>
      <td>95490000</td>
      <td>36.11378065</td>
      <td>10.34773147</td>
      <td>0.115397798</td>
      <td>24.8686939</td>
      <td>63.54000092</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>..</td>
      <td>121.1882158</td>
      <td>7.9444</td>
      <td>1.493977504</td>
      <td>15.87111001</td>
      <td>52650000</td>
      <td>70.28541699</td>
      <td>16.20178304</td>
      <td>1.177095279</td>
      <td>50.90279882</td>
      <td>..</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>5.929999828</td>
      <td>71.94888074</td>
      <td>13.6861</td>
      <td>0.286321131</td>
      <td>16.24140415</td>
      <td>51090000</td>
      <td>45.57737086</td>
      <td>11.5642543</td>
      <td>0.271335347</td>
      <td>26.37150988</td>
      <td>59.88000107</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>2.99000001</td>
      <td>58.54834136</td>
      <td>4.136</td>
      <td>2.073729463</td>
      <td>9.988782489</td>
      <td>102670000</td>
      <td>34.82864877</td>
      <td>10.57001277</td>
      <td>0.169948314</td>
      <td>23.71969259</td>
      <td>58.34000015</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>3.910000086</td>
      <td>116.3060492</td>
      <td>7.4932</td>
      <td>1.449196017</td>
      <td>16.86376042</td>
      <td>90910000</td>
      <td>68.36455893</td>
      <td>16.73381325</td>
      <td>0.421417344</td>
      <td>47.94149032</td>
      <td>51.58000183</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>69.57077057</td>
      <td>13.8859</td>
      <td>0.296162908</td>
      <td>16.56788621</td>
      <td>45370000</td>
      <td>43.70005827</td>
      <td>11.52261896</td>
      <td>0.791025609</td>
      <td>25.8707123</td>
      <td>..</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>2.910000086</td>
      <td>56.71791511</td>
      <td>3.9421</td>
      <td>2.048252145</td>
      <td>9.941700892</td>
      <td>126040000</td>
      <td>33.55949655</td>
      <td>10.84853306</td>
      <td>-0.198735021</td>
      <td>23.15841855</td>
      <td>58.40999985</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>..</td>
      <td>112.6092346</td>
      <td>6.8729</td>
      <td>1.424638087</td>
      <td>17.38569494</td>
      <td>80450000</td>
      <td>65.73989464</td>
      <td>15.71007371</td>
      <td>0.898222966</td>
      <td>46.86933991</td>
      <td>..</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>67.9890291</td>
      <td>13.4668</td>
      <td>0.308591942</td>
      <td>16.57714726</td>
      <td>47470000</td>
      <td>42.03023574</td>
      <td>11.89313816</td>
      <td>0.346950619</td>
      <td>25.95879336</td>
      <td>..</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>2.420000076</td>
      <td>51.33340339</td>
      <td>3.6161</td>
      <td>2.023674003</td>
      <td>10.30311519</td>
      <td>123500000</td>
      <td>30.04344079</td>
      <td>10.36561731</td>
      <td>0.049518645</td>
      <td>21.2899626</td>
      <td>58.90000153</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>7.380000114</td>
      <td>107.4349161</td>
      <td>6.4723</td>
      <td>1.414026656</td>
      <td>17.95312289</td>
      <td>110380000</td>
      <td>62.58871456</td>
      <td>14.70432469</td>
      <td>0.998199897</td>
      <td>44.84620151</td>
      <td>60.63000107</td>
    </tr>
    <tr>
      <th>108</th>
      <td>2016</td>
      <td>El Salvador</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>109</th>
      <td>2016</td>
      <td>Guatemala</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>110</th>
      <td>2016</td>
      <td>Honduras</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
      <td>..</td>
    </tr>
    <tr>
      <th>111</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>112</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>113</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Data from database: World Development Indicators</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Last Updated: 04/27/2017</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>116 rows × 13 columns</p>
</div>




```python
mig6 = mig5.drop([108,109,110,111,112,113,114,115])
```


```python
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
```


```python
mig6['unemployment'].fillna(np.mean(mig6['unemployment']), inplace=True)
mig6['FDI'].fillna(np.mean(mig6['FDI']), inplace=True)
mig6['employment_15+'].fillna(np.mean(mig6['employment_15+']), inplace=True)
```


```python
mig6
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>unemployment</th>
      <th>trade</th>
      <th>short_term_debt</th>
      <th>pop_growth</th>
      <th>remittances</th>
      <th>net_bilateral_aid</th>
      <th>imports_%GDP</th>
      <th>gov_consumption</th>
      <th>FDI</th>
      <th>exports_%GDP</th>
      <th>employment_15+</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>13.340000</td>
      <td>67.406464</td>
      <td>22.0700</td>
      <td>1.739184</td>
      <td>1.372147</td>
      <td>43000000</td>
      <td>33.244917</td>
      <td>13.988965</td>
      <td>0.014116</td>
      <td>34.161547</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>5.496944</td>
      <td>47.105487</td>
      <td>26.3830</td>
      <td>2.635143</td>
      <td>0.332542</td>
      <td>17000000</td>
      <td>24.919086</td>
      <td>7.958166</td>
      <td>0.025385</td>
      <td>22.186401</td>
      <td>49.610001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>5.496944</td>
      <td>81.293839</td>
      <td>17.3081</td>
      <td>3.145300</td>
      <td>0.062354</td>
      <td>19000000</td>
      <td>44.056895</td>
      <td>12.665627</td>
      <td>0.038971</td>
      <td>37.236944</td>
      <td>27.379999</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>5.496944</td>
      <td>60.266492</td>
      <td>17.3809</td>
      <td>1.611673</td>
      <td>2.108693</td>
      <td>97000000</td>
      <td>33.587802</td>
      <td>15.829162</td>
      <td>0.014116</td>
      <td>26.678690</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>2.150000</td>
      <td>40.691257</td>
      <td>9.9649</td>
      <td>2.658257</td>
      <td>0.284635</td>
      <td>18000000</td>
      <td>23.601509</td>
      <td>7.900087</td>
      <td>-0.011618</td>
      <td>17.089748</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>5.496944</td>
      <td>69.338535</td>
      <td>13.3695</td>
      <td>3.113439</td>
      <td>0.062068</td>
      <td>35000000</td>
      <td>37.701720</td>
      <td>12.785955</td>
      <td>0.070935</td>
      <td>31.636815</td>
      <td>26.430000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>5.496944</td>
      <td>51.247740</td>
      <td>13.4222</td>
      <td>1.498034</td>
      <td>3.300787</td>
      <td>170000000</td>
      <td>28.470337</td>
      <td>15.778234</td>
      <td>0.014116</td>
      <td>22.777403</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>2.270000</td>
      <td>33.474818</td>
      <td>7.2350</td>
      <td>2.669298</td>
      <td>0.122749</td>
      <td>20000000</td>
      <td>18.687621</td>
      <td>7.743490</td>
      <td>-0.045887</td>
      <td>14.787197</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>7.300000</td>
      <td>54.727051</td>
      <td>7.7859</td>
      <td>3.083349</td>
      <td>0.051662</td>
      <td>68000000</td>
      <td>28.052348</td>
      <td>13.053211</td>
      <td>-0.034441</td>
      <td>26.674702</td>
      <td>40.090000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>5.496944</td>
      <td>54.397801</td>
      <td>5.1284</td>
      <td>1.413014</td>
      <td>3.292315</td>
      <td>231000000</td>
      <td>29.909283</td>
      <td>15.829448</td>
      <td>0.014116</td>
      <td>24.488518</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>5.496944</td>
      <td>27.546959</td>
      <td>6.1899</td>
      <td>2.654604</td>
      <td>0.043094</td>
      <td>36000000</td>
      <td>14.552484</td>
      <td>7.602210</td>
      <td>0.000000</td>
      <td>12.994475</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>5.496944</td>
      <td>55.394868</td>
      <td>6.3325</td>
      <td>3.056878</td>
      <td>0.058499</td>
      <td>64000000</td>
      <td>29.233021</td>
      <td>13.113422</td>
      <td>0.064998</td>
      <td>26.161846</td>
      <td>35.599998</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>5.496944</td>
      <td>50.290675</td>
      <td>5.8131</td>
      <td>1.364660</td>
      <td>4.345542</td>
      <td>221000000</td>
      <td>28.536536</td>
      <td>16.036198</td>
      <td>0.014116</td>
      <td>21.754139</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>5.496944</td>
      <td>28.153115</td>
      <td>6.1022</td>
      <td>2.607323</td>
      <td>0.035903</td>
      <td>29000000</td>
      <td>15.151003</td>
      <td>7.665259</td>
      <td>0.052798</td>
      <td>13.002112</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>5.496944</td>
      <td>57.728231</td>
      <td>8.2600</td>
      <td>3.037319</td>
      <td>0.058753</td>
      <td>123000000</td>
      <td>32.027719</td>
      <td>13.196746</td>
      <td>-0.030130</td>
      <td>25.700512</td>
      <td>38.720001</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>16.950001</td>
      <td>52.210538</td>
      <td>4.3227</td>
      <td>1.343117</td>
      <td>4.135388</td>
      <td>287000000</td>
      <td>29.886772</td>
      <td>15.490521</td>
      <td>0.014116</td>
      <td>22.323766</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>5.496944</td>
      <td>24.932246</td>
      <td>9.5640</td>
      <td>2.541226</td>
      <td>0.010286</td>
      <td>50000000</td>
      <td>12.984016</td>
      <td>6.953551</td>
      <td>0.000000</td>
      <td>11.948230</td>
      <td>51.299999</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>5.496944</td>
      <td>54.966344</td>
      <td>10.6047</td>
      <td>3.020231</td>
      <td>0.057700</td>
      <td>161000000</td>
      <td>29.866742</td>
      <td>13.092458</td>
      <td>0.000000</td>
      <td>25.099602</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>7.900000</td>
      <td>53.714123</td>
      <td>6.5223</td>
      <td>1.324193</td>
      <td>4.170338</td>
      <td>272000000</td>
      <td>29.045960</td>
      <td>14.181097</td>
      <td>0.014116</td>
      <td>24.668162</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>5.496944</td>
      <td>30.644019</td>
      <td>10.7110</td>
      <td>2.470001</td>
      <td>0.009679</td>
      <td>86000000</td>
      <td>14.592119</td>
      <td>7.096224</td>
      <td>0.000000</td>
      <td>16.051900</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>12.120000</td>
      <td>54.890376</td>
      <td>11.6717</td>
      <td>2.998073</td>
      <td>0.055140</td>
      <td>175000000</td>
      <td>28.305107</td>
      <td>14.270711</td>
      <td>0.000000</td>
      <td>26.585269</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>5.496944</td>
      <td>45.094622</td>
      <td>10.3707</td>
      <td>1.302156</td>
      <td>4.715458</td>
      <td>356000000</td>
      <td>26.102318</td>
      <td>13.745705</td>
      <td>0.014116</td>
      <td>18.992304</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>3.500000</td>
      <td>38.142963</td>
      <td>9.8328</td>
      <td>2.413838</td>
      <td>0.001412</td>
      <td>155000000</td>
      <td>22.294055</td>
      <td>7.902434</td>
      <td>0.014116</td>
      <td>15.848908</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>11.400000</td>
      <td>48.789886</td>
      <td>11.9562</td>
      <td>2.966596</td>
      <td>0.811559</td>
      <td>153000000</td>
      <td>25.827815</td>
      <td>14.220348</td>
      <td>0.024082</td>
      <td>22.962071</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>9.370000</td>
      <td>38.095704</td>
      <td>11.6370</td>
      <td>1.291964</td>
      <td>5.029738</td>
      <td>318000000</td>
      <td>22.285022</td>
      <td>12.731673</td>
      <td>0.014116</td>
      <td>15.810682</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>5.496944</td>
      <td>38.039914</td>
      <td>11.0787</td>
      <td>2.383684</td>
      <td>0.582789</td>
      <td>134000000</td>
      <td>21.936239</td>
      <td>7.981505</td>
      <td>-0.012752</td>
      <td>16.103675</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>5.496944</td>
      <td>55.215652</td>
      <td>12.1001</td>
      <td>2.927378</td>
      <td>1.052794</td>
      <td>155000000</td>
      <td>28.926603</td>
      <td>14.139012</td>
      <td>-0.025186</td>
      <td>26.289050</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>8.350000</td>
      <td>36.928296</td>
      <td>9.7005</td>
      <td>1.293850</td>
      <td>5.439348</td>
      <td>310000000</td>
      <td>23.690508</td>
      <td>12.194134</td>
      <td>0.014116</td>
      <td>13.237788</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>2.000000</td>
      <td>39.781546</td>
      <td>12.1191</td>
      <td>2.387014</td>
      <td>1.010615</td>
      <td>146000000</td>
      <td>22.474097</td>
      <td>7.895426</td>
      <td>-0.047558</td>
      <td>17.307449</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>5.496944</td>
      <td>65.347396</td>
      <td>11.4579</td>
      <td>2.879679</td>
      <td>1.363847</td>
      <td>102000000</td>
      <td>34.342945</td>
      <td>14.273274</td>
      <td>-0.028063</td>
      <td>31.004451</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>6.570000</td>
      <td>71.849041</td>
      <td>12.6699</td>
      <td>0.341593</td>
      <td>18.773955</td>
      <td>24540000</td>
      <td>46.166991</td>
      <td>9.826583</td>
      <td>-0.141774</td>
      <td>25.682050</td>
      <td>49.169998</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>1.820000</td>
      <td>66.818187</td>
      <td>13.4658</td>
      <td>2.298528</td>
      <td>12.239366</td>
      <td>67250000</td>
      <td>41.886457</td>
      <td>8.369963</td>
      <td>0.276205</td>
      <td>24.931730</td>
      <td>57.549999</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>3.110000</td>
      <td>133.131835</td>
      <td>7.1408</td>
      <td>1.826883</td>
      <td>21.557383</td>
      <td>84100000</td>
      <td>77.077193</td>
      <td>15.003781</td>
      <td>0.452738</td>
      <td>56.054642</td>
      <td>49.080002</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>6.410000</td>
      <td>74.177439</td>
      <td>13.1110</td>
      <td>0.315511</td>
      <td>18.448488</td>
      <td>39040000</td>
      <td>48.294694</td>
      <td>9.281817</td>
      <td>0.473516</td>
      <td>25.882745</td>
      <td>49.500000</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>5.496944</td>
      <td>67.898497</td>
      <td>15.8088</td>
      <td>2.254506</td>
      <td>12.418103</td>
      <td>45710000</td>
      <td>42.333233</td>
      <td>8.658056</td>
      <td>0.408934</td>
      <td>25.565264</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>2.920000</td>
      <td>135.070635</td>
      <td>10.5623</td>
      <td>1.792143</td>
      <td>21.291564</td>
      <td>71100000</td>
      <td>81.561623</td>
      <td>16.602374</td>
      <td>0.332742</td>
      <td>53.509012</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>5.880000</td>
      <td>76.580188</td>
      <td>14.4812</td>
      <td>0.296649</td>
      <td>17.520181</td>
      <td>42370000</td>
      <td>49.698567</td>
      <td>9.175027</td>
      <td>0.370631</td>
      <td>26.881620</td>
      <td>59.020000</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>5.496944</td>
      <td>64.125228</td>
      <td>15.5632</td>
      <td>2.215217</td>
      <td>11.395396</td>
      <td>70350000</td>
      <td>39.406974</td>
      <td>9.013268</td>
      <td>0.034880</td>
      <td>24.718254</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>2.990000</td>
      <td>135.748955</td>
      <td>13.3425</td>
      <td>1.747160</td>
      <td>20.459753</td>
      <td>96330000</td>
      <td>84.423679</td>
      <td>17.114828</td>
      <td>1.402889</td>
      <td>51.325277</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>7.330000</td>
      <td>61.871642</td>
      <td>8.3258</td>
      <td>0.285542</td>
      <td>16.467451</td>
      <td>82080000</td>
      <td>38.674798</td>
      <td>10.629205</td>
      <td>0.014327</td>
      <td>23.196844</td>
      <td>58.160000</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>5.496944</td>
      <td>57.105993</td>
      <td>8.2714</td>
      <td>2.183077</td>
      <td>10.651859</td>
      <td>83890000</td>
      <td>33.130589</td>
      <td>10.179023</td>
      <td>0.325371</td>
      <td>23.975404</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>3.280000</td>
      <td>96.905006</td>
      <td>6.5494</td>
      <td>1.688651</td>
      <td>17.101447</td>
      <td>128760000</td>
      <td>57.374755</td>
      <td>18.699738</td>
      <td>-0.074317</td>
      <td>39.530251</td>
      <td>59.320000</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>7.050000</td>
      <td>68.768763</td>
      <td>7.5574</td>
      <td>0.280903</td>
      <td>16.209675</td>
      <td>148160000</td>
      <td>42.844203</td>
      <td>10.704865</td>
      <td>0.524807</td>
      <td>25.924560</td>
      <td>58.110001</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>3.740000</td>
      <td>62.114932</td>
      <td>10.5394</td>
      <td>2.156000</td>
      <td>10.236867</td>
      <td>100500000</td>
      <td>36.309182</td>
      <td>10.475864</td>
      <td>0.153036</td>
      <td>25.805750</td>
      <td>42.919998</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>4.100000</td>
      <td>109.441838</td>
      <td>9.4110</td>
      <td>1.622622</td>
      <td>16.643140</td>
      <td>100840000</td>
      <td>63.682932</td>
      <td>17.926432</td>
      <td>-2.309012</td>
      <td>45.758906</td>
      <td>59.360001</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>6.620000</td>
      <td>74.643243</td>
      <td>10.1925</td>
      <td>0.279522</td>
      <td>15.748084</td>
      <td>162440000</td>
      <td>46.663209</td>
      <td>11.055793</td>
      <td>-0.414578</td>
      <td>27.980034</td>
      <td>58.549999</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>4.130000</td>
      <td>63.984196</td>
      <td>13.9156</td>
      <td>2.129043</td>
      <td>9.492785</td>
      <td>93080000</td>
      <td>37.358771</td>
      <td>10.189228</td>
      <td>0.274376</td>
      <td>26.625425</td>
      <td>59.230000</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>4.270000</td>
      <td>122.216903</td>
      <td>6.0942</td>
      <td>1.554236</td>
      <td>15.980083</td>
      <td>46360000</td>
      <td>70.959216</td>
      <td>16.064368</td>
      <td>0.172124</td>
      <td>51.257687</td>
      <td>49.660000</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>6.070000</td>
      <td>69.698828</td>
      <td>11.7092</td>
      <td>0.280768</td>
      <td>16.361001</td>
      <td>150850000</td>
      <td>44.076494</td>
      <td>11.228458</td>
      <td>-0.150729</td>
      <td>25.622333</td>
      <td>59.400002</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>2.870000</td>
      <td>60.982475</td>
      <td>4.5907</td>
      <td>2.100666</td>
      <td>9.983742</td>
      <td>95490000</td>
      <td>36.113781</td>
      <td>10.347731</td>
      <td>0.115398</td>
      <td>24.868694</td>
      <td>63.540001</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>5.496944</td>
      <td>121.188216</td>
      <td>7.9444</td>
      <td>1.493978</td>
      <td>15.871110</td>
      <td>52650000</td>
      <td>70.285417</td>
      <td>16.201783</td>
      <td>1.177095</td>
      <td>50.902799</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>5.930000</td>
      <td>71.948881</td>
      <td>13.6861</td>
      <td>0.286321</td>
      <td>16.241404</td>
      <td>51090000</td>
      <td>45.577371</td>
      <td>11.564254</td>
      <td>0.271335</td>
      <td>26.371510</td>
      <td>59.880001</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>2.990000</td>
      <td>58.548341</td>
      <td>4.1360</td>
      <td>2.073729</td>
      <td>9.988782</td>
      <td>102670000</td>
      <td>34.828649</td>
      <td>10.570013</td>
      <td>0.169948</td>
      <td>23.719693</td>
      <td>58.340000</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>3.910000</td>
      <td>116.306049</td>
      <td>7.4932</td>
      <td>1.449196</td>
      <td>16.863760</td>
      <td>90910000</td>
      <td>68.364559</td>
      <td>16.733813</td>
      <td>0.421417</td>
      <td>47.941490</td>
      <td>51.580002</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>5.496944</td>
      <td>69.570771</td>
      <td>13.8859</td>
      <td>0.296163</td>
      <td>16.567886</td>
      <td>45370000</td>
      <td>43.700058</td>
      <td>11.522619</td>
      <td>0.791026</td>
      <td>25.870712</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>2.910000</td>
      <td>56.717915</td>
      <td>3.9421</td>
      <td>2.048252</td>
      <td>9.941701</td>
      <td>126040000</td>
      <td>33.559497</td>
      <td>10.848533</td>
      <td>-0.198735</td>
      <td>23.158419</td>
      <td>58.410000</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>5.496944</td>
      <td>112.609235</td>
      <td>6.8729</td>
      <td>1.424638</td>
      <td>17.385695</td>
      <td>80450000</td>
      <td>65.739895</td>
      <td>15.710074</td>
      <td>0.898223</td>
      <td>46.869340</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>5.496944</td>
      <td>67.989029</td>
      <td>13.4668</td>
      <td>0.308592</td>
      <td>16.577147</td>
      <td>47470000</td>
      <td>42.030236</td>
      <td>11.893138</td>
      <td>0.346951</td>
      <td>25.958793</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>2.420000</td>
      <td>51.333403</td>
      <td>3.6161</td>
      <td>2.023674</td>
      <td>10.303115</td>
      <td>123500000</td>
      <td>30.043441</td>
      <td>10.365617</td>
      <td>0.049519</td>
      <td>21.289963</td>
      <td>58.900002</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>7.380000</td>
      <td>107.434916</td>
      <td>6.4723</td>
      <td>1.414027</td>
      <td>17.953123</td>
      <td>110380000</td>
      <td>62.588715</td>
      <td>14.704325</td>
      <td>0.998200</td>
      <td>44.846202</td>
      <td>60.630001</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 13 columns</p>
</div>




```python
mig6.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 108 entries, 0 to 107
    Data columns (total 13 columns):
    Year                 108 non-null object
    Country              108 non-null object
    unemployment         108 non-null float64
    trade                108 non-null float64
    short_term_debt      108 non-null float64
    pop_growth           108 non-null float64
    remittances          108 non-null float64
    net_bilateral_aid    108 non-null int64
    imports_%GDP         108 non-null float64
    gov_consumption      108 non-null float64
    FDI                  108 non-null float64
    exports_%GDP         108 non-null float64
    employment_15+       108 non-null float64
    dtypes: float64(10), int64(1), object(2)
    memory usage: 11.8+ KB



```python
mig6.Year = [int(float(x)) for x in mig6.Year]
```


```python
mig6.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 108 entries, 0 to 107
    Data columns (total 13 columns):
    Year                 108 non-null int64
    Country              108 non-null object
    unemployment         108 non-null float64
    trade                108 non-null float64
    short_term_debt      108 non-null float64
    pop_growth           108 non-null float64
    remittances          108 non-null float64
    net_bilateral_aid    108 non-null int64
    imports_%GDP         108 non-null float64
    gov_consumption      108 non-null float64
    FDI                  108 non-null float64
    exports_%GDP         108 non-null float64
    employment_15+       108 non-null float64
    dtypes: float64(10), int64(2), object(1)
    memory usage: 11.8+ KB



```python
migm1 = pd.merge(mig1, mig2, on=['Year', 'Country'], how='right')
```


```python
migm1
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>Migration</th>
      <th>enrolment_tertiary</th>
      <th>GDP_percapita_constant</th>
      <th>pop_ages_0-14%</th>
      <th>pop_ages_14-64%</th>
      <th>primary_completion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>2.063205</td>
      <td>1516.400016</td>
      <td>2572.813235</td>
      <td>43.742478</td>
      <td>52.756733</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>0.886027</td>
      <td>1516.400016</td>
      <td>2560.782037</td>
      <td>45.444923</td>
      <td>51.602977</td>
      <td>33.904148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>1.076884</td>
      <td>713.525940</td>
      <td>1655.946421</td>
      <td>46.957200</td>
      <td>49.818337</td>
      <td>44.575001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2267.095959</td>
      <td>43.481122</td>
      <td>52.948845</td>
      <td>46.450790</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>493.277863</td>
      <td>2509.736778</td>
      <td>45.617358</td>
      <td>51.409643</td>
      <td>33.957420</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>821.092712</td>
      <td>1645.846419</td>
      <td>46.892259</td>
      <td>49.886066</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>999.595276</td>
      <td>2092.554425</td>
      <td>43.204606</td>
      <td>53.154795</td>
      <td>49.382709</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2357.368296</td>
      <td>45.771834</td>
      <td>51.239138</td>
      <td>33.880711</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>864.464600</td>
      <td>1573.671559</td>
      <td>46.745647</td>
      <td>50.043171</td>
      <td>49.944939</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1203.906616</td>
      <td>2094.864582</td>
      <td>42.920848</td>
      <td>53.365841</td>
      <td>50.613579</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>571.447571</td>
      <td>2236.567544</td>
      <td>45.891347</td>
      <td>51.104597</td>
      <td>35.137379</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>868.445679</td>
      <td>1512.185833</td>
      <td>46.554395</td>
      <td>50.246028</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1310.496826</td>
      <td>2094.098791</td>
      <td>42.636120</td>
      <td>53.575056</td>
      <td>48.697311</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>578.319580</td>
      <td>2189.829730</td>
      <td>45.951383</td>
      <td>51.025182</td>
      <td>36.577950</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>874.714233</td>
      <td>1530.695403</td>
      <td>46.363681</td>
      <td>50.440498</td>
      <td>54.984901</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1439.984375</td>
      <td>2078.900486</td>
      <td>42.346470</td>
      <td>53.785685</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>597.558655</td>
      <td>2121.873660</td>
      <td>45.939359</td>
      <td>51.010453</td>
      <td>38.065460</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>868.423401</td>
      <td>1547.357836</td>
      <td>46.190833</td>
      <td>50.604823</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1492.553833</td>
      <td>2055.438830</td>
      <td>41.819358</td>
      <td>54.216902</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>626.206238</td>
      <td>2073.066614</td>
      <td>45.963895</td>
      <td>50.941242</td>
      <td>41.001839</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>1512.507552</td>
      <td>46.064093</td>
      <td>50.698642</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2079.844180</td>
      <td>41.318260</td>
      <td>54.619889</td>
      <td>61.982658</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2095.342199</td>
      <td>45.884100</td>
      <td>50.970232</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>835.494751</td>
      <td>1556.855276</td>
      <td>45.935714</td>
      <td>50.785360</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1515.808716</td>
      <td>2091.693100</td>
      <td>40.835477</td>
      <td>55.003335</td>
      <td>65.722801</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2125.624163</td>
      <td>45.729366</td>
      <td>51.069993</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>930.572388</td>
      <td>1581.639092</td>
      <td>45.800132</td>
      <td>50.875323</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1568.253540</td>
      <td>2084.671422</td>
      <td>40.362708</td>
      <td>55.377693</td>
      <td>63.612751</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2157.313890</td>
      <td>45.551915</td>
      <td>51.191567</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>925.763672</td>
      <td>1603.219717</td>
      <td>45.642320</td>
      <td>50.989531</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>17.546949</td>
      <td>2093.922607</td>
      <td>3475.866745</td>
      <td>33.230254</td>
      <td>60.147890</td>
      <td>92.364281</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>5.343950</td>
      <td>1516.400016</td>
      <td>2698.985240</td>
      <td>41.024086</td>
      <td>54.692058</td>
      <td>75.359779</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>5.783592</td>
      <td>1516.400016</td>
      <td>2017.943010</td>
      <td>38.938629</td>
      <td>56.863273</td>
      <td>88.262627</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>18.448273</td>
      <td>2209.102051</td>
      <td>3597.961991</td>
      <td>32.566759</td>
      <td>60.626412</td>
      <td>97.298424</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>5.077445</td>
      <td>1695.110107</td>
      <td>2805.169791</td>
      <td>40.575993</td>
      <td>55.103794</td>
      <td>75.833748</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>6.034761</td>
      <td>1516.400016</td>
      <td>2104.759589</td>
      <td>38.204492</td>
      <td>57.552528</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>18.237120</td>
      <td>2308.634277</td>
      <td>3633.014903</td>
      <td>31.905088</td>
      <td>61.110033</td>
      <td>99.281097</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>5.240451</td>
      <td>1516.400016</td>
      <td>2833.735795</td>
      <td>40.091781</td>
      <td>55.548046</td>
      <td>78.546761</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>6.339264</td>
      <td>2035.134766</td>
      <td>2155.827865</td>
      <td>37.448905</td>
      <td>58.259473</td>
      <td>88.769539</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>19.096906</td>
      <td>2388.975342</td>
      <td>3509.156436</td>
      <td>31.228684</td>
      <td>61.613308</td>
      <td>102.675781</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>5.539466</td>
      <td>1516.400016</td>
      <td>2787.128287</td>
      <td>39.593279</td>
      <td>55.997451</td>
      <td>82.200653</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>6.338030</td>
      <td>1516.400016</td>
      <td>2068.185180</td>
      <td>36.665394</td>
      <td>58.987298</td>
      <td>91.863228</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>20.105788</td>
      <td>2484.339111</td>
      <td>3547.070983</td>
      <td>30.534690</td>
      <td>62.138073</td>
      <td>105.430397</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>5.639487</td>
      <td>1516.400016</td>
      <td>2805.951416</td>
      <td>39.095628</td>
      <td>56.434019</td>
      <td>84.213753</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>6.964149</td>
      <td>2263.870361</td>
      <td>2110.822021</td>
      <td>35.854009</td>
      <td>59.733111</td>
      <td>97.189880</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>20.886863</td>
      <td>2648.530029</td>
      <td>3615.583230</td>
      <td>29.801800</td>
      <td>62.706748</td>
      <td>109.104622</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>5.653971</td>
      <td>1516.400016</td>
      <td>2861.167894</td>
      <td>38.577533</td>
      <td>56.887778</td>
      <td>86.689102</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>6.437598</td>
      <td>1516.400016</td>
      <td>2157.984444</td>
      <td>35.042579</td>
      <td>60.480535</td>
      <td>100.720642</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>20.945491</td>
      <td>2797.323486</td>
      <td>3673.262887</td>
      <td>29.042627</td>
      <td>63.302907</td>
      <td>108.798729</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>5.586203</td>
      <td>1516.400016</td>
      <td>2884.897429</td>
      <td>38.086602</td>
      <td>57.307763</td>
      <td>86.083344</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>6.743448</td>
      <td>2261.272461</td>
      <td>2213.759527</td>
      <td>34.200196</td>
      <td>61.250126</td>
      <td>100.676102</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>20.560594</td>
      <td>2891.187012</td>
      <td>3730.422292</td>
      <td>28.295414</td>
      <td>63.885015</td>
      <td>106.839989</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>5.750461</td>
      <td>1871.932129</td>
      <td>2930.170750</td>
      <td>37.607424</td>
      <td>57.710289</td>
      <td>86.501770</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>6.798242</td>
      <td>2340.688232</td>
      <td>2242.818455</td>
      <td>33.349845</td>
      <td>62.015294</td>
      <td>94.479530</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>21.537939</td>
      <td>2886.402832</td>
      <td>3772.401570</td>
      <td>27.615213</td>
      <td>64.393636</td>
      <td>104.617020</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>5.716933</td>
      <td>1516.400016</td>
      <td>2990.594485</td>
      <td>37.120959</td>
      <td>58.115978</td>
      <td>86.624428</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>7.389157</td>
      <td>2334.632813</td>
      <td>2279.309902</td>
      <td>32.529328</td>
      <td>62.734875</td>
      <td>90.721970</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>22.073593</td>
      <td>1516.400016</td>
      <td>3853.107631</td>
      <td>27.028606</td>
      <td>64.799595</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>5.675817</td>
      <td>1516.400016</td>
      <td>3052.270569</td>
      <td>36.622822</td>
      <td>58.530645</td>
      <td>73.720787</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>7.418273</td>
      <td>1516.400016</td>
      <td>2329.002149</td>
      <td>31.762798</td>
      <td>63.383938</td>
      <td>73.720787</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 8 columns</p>
</div>




```python
migm2 = pd.merge(migm1, mig3, on=['Year', 'Country'], how='right')
```


```python
migm2
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>Migration</th>
      <th>enrolment_tertiary</th>
      <th>GDP_percapita_constant</th>
      <th>pop_ages_0-14%</th>
      <th>pop_ages_14-64%</th>
      <th>primary_completion</th>
      <th>gini</th>
      <th>income_highest%</th>
      <th>income_lowest%</th>
      <th>poor_1.90</th>
      <th>poor_3.10</th>
      <th>poverty_gap_1.90</th>
      <th>poverty_gap_3.10</th>
      <th>poverty_headcount_1.90</th>
      <th>poverty_headcount_3.10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>2.063205</td>
      <td>1516.400016</td>
      <td>2572.813235</td>
      <td>43.742478</td>
      <td>52.756733</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>0.886027</td>
      <td>1516.400016</td>
      <td>2560.782037</td>
      <td>45.444923</td>
      <td>51.602977</td>
      <td>33.904148</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>1.076884</td>
      <td>713.525940</td>
      <td>1655.946421</td>
      <td>46.957200</td>
      <td>49.818337</td>
      <td>44.575001</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2267.095959</td>
      <td>43.481122</td>
      <td>52.948845</td>
      <td>46.450790</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>493.277863</td>
      <td>2509.736778</td>
      <td>45.617358</td>
      <td>51.409643</td>
      <td>33.957420</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>821.092712</td>
      <td>1645.846419</td>
      <td>46.892259</td>
      <td>49.886066</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>999.595276</td>
      <td>2092.554425</td>
      <td>43.204606</td>
      <td>53.154795</td>
      <td>49.382709</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2357.368296</td>
      <td>45.771834</td>
      <td>51.239138</td>
      <td>33.880711</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>864.464600</td>
      <td>1573.671559</td>
      <td>46.745647</td>
      <td>50.043171</td>
      <td>49.944939</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1203.906616</td>
      <td>2094.864582</td>
      <td>42.920848</td>
      <td>53.365841</td>
      <td>50.613579</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>571.447571</td>
      <td>2236.567544</td>
      <td>45.891347</td>
      <td>51.104597</td>
      <td>35.137379</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>868.445679</td>
      <td>1512.185833</td>
      <td>46.554395</td>
      <td>50.246028</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1310.496826</td>
      <td>2094.098791</td>
      <td>42.636120</td>
      <td>53.575056</td>
      <td>48.697311</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>578.319580</td>
      <td>2189.829730</td>
      <td>45.951383</td>
      <td>51.025182</td>
      <td>36.577950</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>874.714233</td>
      <td>1530.695403</td>
      <td>46.363681</td>
      <td>50.440498</td>
      <td>54.984901</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1439.984375</td>
      <td>2078.900486</td>
      <td>42.346470</td>
      <td>53.785685</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>597.558655</td>
      <td>2121.873660</td>
      <td>45.939359</td>
      <td>51.010453</td>
      <td>38.065460</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>868.423401</td>
      <td>1547.357836</td>
      <td>46.190833</td>
      <td>50.604823</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1492.553833</td>
      <td>2055.438830</td>
      <td>41.819358</td>
      <td>54.216902</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>626.206238</td>
      <td>2073.066614</td>
      <td>45.963895</td>
      <td>50.941242</td>
      <td>41.001839</td>
      <td>58.260000</td>
      <td>46.730000</td>
      <td>1.000000</td>
      <td>4.238208</td>
      <td>5.794880</td>
      <td>24.99</td>
      <td>39.010</td>
      <td>50.940000</td>
      <td>69.650000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>1512.507552</td>
      <td>46.064093</td>
      <td>50.698642</td>
      <td>73.720787</td>
      <td>55.090000</td>
      <td>43.260000</td>
      <td>1.230000</td>
      <td>0.422176</td>
      <td>0.713758</td>
      <td>9.15</td>
      <td>18.940</td>
      <td>25.280000</td>
      <td>42.740000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2079.844180</td>
      <td>41.318260</td>
      <td>54.619889</td>
      <td>61.982658</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2095.342199</td>
      <td>45.884100</td>
      <td>50.970232</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>835.494751</td>
      <td>1556.855276</td>
      <td>45.935714</td>
      <td>50.785360</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1515.808716</td>
      <td>2091.693100</td>
      <td>40.835477</td>
      <td>55.003335</td>
      <td>65.722801</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2125.624163</td>
      <td>45.729366</td>
      <td>51.069993</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>930.572388</td>
      <td>1581.639092</td>
      <td>45.800132</td>
      <td>50.875323</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1568.253540</td>
      <td>2084.671422</td>
      <td>40.362708</td>
      <td>55.377693</td>
      <td>63.612751</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>0.983164</td>
      <td>1.632736</td>
      <td>11.24</td>
      <td>16.660</td>
      <td>18.980000</td>
      <td>31.520000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2157.313890</td>
      <td>45.551915</td>
      <td>51.191567</td>
      <td>73.720787</td>
      <td>59.600000</td>
      <td>46.780000</td>
      <td>0.680000</td>
      <td>3.398988</td>
      <td>4.919682</td>
      <td>18.71</td>
      <td>29.680</td>
      <td>38.020000</td>
      <td>55.030000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>925.763672</td>
      <td>1603.219717</td>
      <td>45.642320</td>
      <td>50.989531</td>
      <td>73.720787</td>
      <td>59.490000</td>
      <td>48.180000</td>
      <td>1.040000</td>
      <td>1.841220</td>
      <td>2.722716</td>
      <td>16.90</td>
      <td>29.140</td>
      <td>38.600000</td>
      <td>57.080000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>17.546949</td>
      <td>2093.922607</td>
      <td>3475.866745</td>
      <td>33.230254</td>
      <td>60.147890</td>
      <td>92.364281</td>
      <td>45.440000</td>
      <td>35.480000</td>
      <td>1.790000</td>
      <td>0.379692</td>
      <td>1.035198</td>
      <td>1.73</td>
      <td>5.570</td>
      <td>6.360000</td>
      <td>17.340000</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>5.343950</td>
      <td>1516.400016</td>
      <td>2698.985240</td>
      <td>41.024086</td>
      <td>54.692058</td>
      <td>75.359779</td>
      <td>54.890000</td>
      <td>43.560000</td>
      <td>1.070000</td>
      <td>1.552699</td>
      <td>3.195781</td>
      <td>3.93</td>
      <td>9.270</td>
      <td>11.510000</td>
      <td>23.690000</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>5.783592</td>
      <td>1516.400016</td>
      <td>2017.943010</td>
      <td>38.938629</td>
      <td>56.863273</td>
      <td>88.262627</td>
      <td>57.420000</td>
      <td>44.050000</td>
      <td>0.580000</td>
      <td>1.667679</td>
      <td>2.614730</td>
      <td>11.40</td>
      <td>18.850</td>
      <td>23.790000</td>
      <td>37.300000</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>18.448273</td>
      <td>2209.102051</td>
      <td>3597.961991</td>
      <td>32.566759</td>
      <td>60.626412</td>
      <td>97.298424</td>
      <td>45.240000</td>
      <td>35.720000</td>
      <td>1.930000</td>
      <td>0.268951</td>
      <td>0.835006</td>
      <td>1.08</td>
      <td>4.100</td>
      <td>4.490000</td>
      <td>13.940000</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>5.077445</td>
      <td>1695.110107</td>
      <td>2805.169791</td>
      <td>40.575993</td>
      <td>55.103794</td>
      <td>75.833748</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>6.034761</td>
      <td>1516.400016</td>
      <td>2104.759589</td>
      <td>38.204492</td>
      <td>57.552528</td>
      <td>73.720787</td>
      <td>56.160000</td>
      <td>43.810000</td>
      <td>0.900000</td>
      <td>1.242759</td>
      <td>2.279461</td>
      <td>6.91</td>
      <td>13.880</td>
      <td>17.430000</td>
      <td>31.970000</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>18.237120</td>
      <td>2308.634277</td>
      <td>3633.014903</td>
      <td>31.905088</td>
      <td>61.110033</td>
      <td>99.281097</td>
      <td>46.650000</td>
      <td>36.040000</td>
      <td>1.700000</td>
      <td>0.415200</td>
      <td>1.114800</td>
      <td>1.99</td>
      <td>6.090</td>
      <td>6.920000</td>
      <td>18.580000</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>5.240451</td>
      <td>1516.400016</td>
      <td>2833.735795</td>
      <td>40.091781</td>
      <td>55.548046</td>
      <td>78.546761</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>6.339264</td>
      <td>2035.134766</td>
      <td>2155.827865</td>
      <td>37.448905</td>
      <td>58.259473</td>
      <td>88.769539</td>
      <td>55.740000</td>
      <td>43.870000</td>
      <td>0.910000</td>
      <td>1.171764</td>
      <td>2.130810</td>
      <td>6.30</td>
      <td>12.680</td>
      <td>16.140000</td>
      <td>29.350000</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>19.096906</td>
      <td>2388.975342</td>
      <td>3509.156436</td>
      <td>31.228684</td>
      <td>61.613308</td>
      <td>102.675781</td>
      <td>45.930000</td>
      <td>36.070000</td>
      <td>1.780000</td>
      <td>0.384678</td>
      <td>1.054102</td>
      <td>1.67</td>
      <td>5.520</td>
      <td>6.390000</td>
      <td>17.510000</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>5.539466</td>
      <td>1516.400016</td>
      <td>2787.128287</td>
      <td>39.593279</td>
      <td>55.997451</td>
      <td>82.200653</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>6.338030</td>
      <td>1516.400016</td>
      <td>2068.185180</td>
      <td>36.665394</td>
      <td>58.987298</td>
      <td>91.863228</td>
      <td>51.560000</td>
      <td>39.140000</td>
      <td>1.150000</td>
      <td>1.036152</td>
      <td>1.979316</td>
      <td>4.82</td>
      <td>10.880</td>
      <td>14.040000</td>
      <td>26.820000</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>20.105788</td>
      <td>2484.339111</td>
      <td>3547.070983</td>
      <td>30.534690</td>
      <td>62.138073</td>
      <td>105.430397</td>
      <td>44.530000</td>
      <td>33.700000</td>
      <td>1.670000</td>
      <td>0.437296</td>
      <td>1.120420</td>
      <td>2.33</td>
      <td>6.300</td>
      <td>7.240000</td>
      <td>18.550000</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>5.639487</td>
      <td>1516.400016</td>
      <td>2805.951416</td>
      <td>39.095628</td>
      <td>56.434019</td>
      <td>84.213753</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>6.964149</td>
      <td>2263.870361</td>
      <td>2110.822021</td>
      <td>35.854009</td>
      <td>59.733111</td>
      <td>97.189880</td>
      <td>53.390000</td>
      <td>41.020000</td>
      <td>1.090000</td>
      <td>1.160250</td>
      <td>2.183250</td>
      <td>5.40</td>
      <td>11.900</td>
      <td>15.470000</td>
      <td>29.110000</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>20.886863</td>
      <td>2648.530029</td>
      <td>3615.583230</td>
      <td>29.801800</td>
      <td>62.706748</td>
      <td>109.104622</td>
      <td>42.430000</td>
      <td>32.860000</td>
      <td>2.110000</td>
      <td>0.274518</td>
      <td>0.911424</td>
      <td>1.06</td>
      <td>4.390</td>
      <td>4.530000</td>
      <td>15.040000</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>5.653971</td>
      <td>1516.400016</td>
      <td>2861.167894</td>
      <td>38.577533</td>
      <td>56.887778</td>
      <td>86.689102</td>
      <td>52.350000</td>
      <td>41.830000</td>
      <td>1.340000</td>
      <td>1.735265</td>
      <td>3.983735</td>
      <td>4.00</td>
      <td>9.840</td>
      <td>11.530000</td>
      <td>26.470000</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>6.437598</td>
      <td>1516.400016</td>
      <td>2157.984444</td>
      <td>35.042579</td>
      <td>60.480535</td>
      <td>100.720642</td>
      <td>57.400000</td>
      <td>45.670000</td>
      <td>0.750000</td>
      <td>1.428750</td>
      <td>2.489454</td>
      <td>7.88</td>
      <td>14.660</td>
      <td>18.750000</td>
      <td>32.670000</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>20.945491</td>
      <td>2797.323486</td>
      <td>3673.262887</td>
      <td>29.042627</td>
      <td>63.302907</td>
      <td>108.798729</td>
      <td>41.800000</td>
      <td>32.470000</td>
      <td>2.150000</td>
      <td>0.252512</td>
      <td>0.826127</td>
      <td>0.98</td>
      <td>3.840</td>
      <td>4.160000</td>
      <td>13.610000</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>5.586203</td>
      <td>1516.400016</td>
      <td>2884.897429</td>
      <td>38.086602</td>
      <td>57.307763</td>
      <td>86.083344</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>6.743448</td>
      <td>2261.272461</td>
      <td>2213.759527</td>
      <td>34.200196</td>
      <td>61.250126</td>
      <td>100.676102</td>
      <td>57.400000</td>
      <td>45.680000</td>
      <td>0.790000</td>
      <td>1.653264</td>
      <td>2.883924</td>
      <td>9.25</td>
      <td>17.100</td>
      <td>21.360000</td>
      <td>37.260000</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>20.560594</td>
      <td>2891.187012</td>
      <td>3730.422292</td>
      <td>28.295414</td>
      <td>63.885015</td>
      <td>106.839989</td>
      <td>43.510000</td>
      <td>34.350000</td>
      <td>2.110000</td>
      <td>0.197925</td>
      <td>0.702177</td>
      <td>0.74</td>
      <td>3.160</td>
      <td>3.250000</td>
      <td>11.530000</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>5.750461</td>
      <td>1871.932129</td>
      <td>2930.170750</td>
      <td>37.607424</td>
      <td>57.710289</td>
      <td>86.501770</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>6.798242</td>
      <td>2340.688232</td>
      <td>2242.818455</td>
      <td>33.349845</td>
      <td>62.015294</td>
      <td>94.479530</td>
      <td>53.670000</td>
      <td>41.480000</td>
      <td>0.980000</td>
      <td>1.486005</td>
      <td>2.712175</td>
      <td>7.66</td>
      <td>15.240</td>
      <td>18.930000</td>
      <td>34.550000</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>21.537939</td>
      <td>2886.402832</td>
      <td>3772.401570</td>
      <td>27.615213</td>
      <td>64.393636</td>
      <td>104.617020</td>
      <td>41.840000</td>
      <td>32.310000</td>
      <td>2.190000</td>
      <td>0.181467</td>
      <td>0.689819</td>
      <td>0.64</td>
      <td>3.000</td>
      <td>2.970000</td>
      <td>11.290000</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>5.716933</td>
      <td>1516.400016</td>
      <td>2990.594485</td>
      <td>37.120959</td>
      <td>58.115978</td>
      <td>86.624428</td>
      <td>48.660000</td>
      <td>38.360000</td>
      <td>1.640000</td>
      <td>1.493064</td>
      <td>3.852810</td>
      <td>2.72</td>
      <td>8.110</td>
      <td>9.320000</td>
      <td>24.050000</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>7.389157</td>
      <td>2334.632813</td>
      <td>2279.309902</td>
      <td>32.529328</td>
      <td>62.734875</td>
      <td>90.721970</td>
      <td>50.640000</td>
      <td>38.360000</td>
      <td>1.150000</td>
      <td>1.270416</td>
      <td>2.484316</td>
      <td>6.01</td>
      <td>12.970</td>
      <td>15.960000</td>
      <td>31.210000</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>22.073593</td>
      <td>1516.400016</td>
      <td>3853.107631</td>
      <td>27.028606</td>
      <td>64.799595</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>5.675817</td>
      <td>1516.400016</td>
      <td>3052.270569</td>
      <td>36.622822</td>
      <td>58.530645</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>7.418273</td>
      <td>1516.400016</td>
      <td>2329.002149</td>
      <td>31.762798</td>
      <td>63.383938</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>1.247831</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 17 columns</p>
</div>




```python
migm3 = pd.merge(migm2, mig4, on=['Year', 'Country'], how='right')
```


```python
migm3
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>Migration</th>
      <th>enrolment_tertiary</th>
      <th>GDP_percapita_constant</th>
      <th>pop_ages_0-14%</th>
      <th>pop_ages_14-64%</th>
      <th>primary_completion</th>
      <th>gini</th>
      <th>income_highest%</th>
      <th>...</th>
      <th>poor_3.10</th>
      <th>poverty_gap_1.90</th>
      <th>poverty_gap_3.10</th>
      <th>poverty_headcount_1.90</th>
      <th>poverty_headcount_3.10</th>
      <th>age_dependency</th>
      <th>birth_rate</th>
      <th>death_rate</th>
      <th>fertility_rate</th>
      <th>life_expectancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>2.063205</td>
      <td>1516.400016</td>
      <td>2572.813235</td>
      <td>43.742478</td>
      <td>52.756733</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>89.549266</td>
      <td>37.353</td>
      <td>11.681</td>
      <td>5.087</td>
      <td>56.529927</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>0.886027</td>
      <td>1516.400016</td>
      <td>2560.782037</td>
      <td>45.444923</td>
      <td>51.602977</td>
      <td>33.904148</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>93.787268</td>
      <td>43.686</td>
      <td>11.568</td>
      <td>6.195</td>
      <td>57.201488</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>1.076884</td>
      <td>713.525940</td>
      <td>1655.946421</td>
      <td>46.957200</td>
      <td>49.818337</td>
      <td>44.575001</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>100.729300</td>
      <td>43.476</td>
      <td>10.233</td>
      <td>6.313</td>
      <td>59.612122</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2267.095959</td>
      <td>43.481122</td>
      <td>52.948845</td>
      <td>46.450790</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>88.861531</td>
      <td>36.593</td>
      <td>11.494</td>
      <td>4.952</td>
      <td>56.798976</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>493.277863</td>
      <td>2509.736778</td>
      <td>45.617358</td>
      <td>51.409643</td>
      <td>33.957420</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>94.516037</td>
      <td>43.384</td>
      <td>11.300</td>
      <td>6.161</td>
      <td>57.632756</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>821.092712</td>
      <td>1645.846419</td>
      <td>46.892259</td>
      <td>49.886066</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>100.456778</td>
      <td>43.020</td>
      <td>9.793</td>
      <td>6.190</td>
      <td>60.405854</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>999.595276</td>
      <td>2092.554425</td>
      <td>43.204606</td>
      <td>53.154795</td>
      <td>49.382709</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>88.129782</td>
      <td>35.833</td>
      <td>11.251</td>
      <td>4.819</td>
      <td>57.197537</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2357.368296</td>
      <td>45.771834</td>
      <td>51.239138</td>
      <td>33.880711</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>95.163313</td>
      <td>42.955</td>
      <td>11.016</td>
      <td>6.105</td>
      <td>58.085951</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>864.464600</td>
      <td>1573.671559</td>
      <td>46.745647</td>
      <td>50.043171</td>
      <td>49.944939</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>99.827463</td>
      <td>42.524</td>
      <td>9.359</td>
      <td>6.062</td>
      <td>61.212073</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1203.906616</td>
      <td>2094.864582</td>
      <td>42.920848</td>
      <td>53.365841</td>
      <td>50.613579</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>87.385786</td>
      <td>35.093</td>
      <td>10.953</td>
      <td>4.688</td>
      <td>57.731659</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>571.447571</td>
      <td>2236.567544</td>
      <td>45.891347</td>
      <td>51.104597</td>
      <td>35.137379</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>95.677113</td>
      <td>42.406</td>
      <td>10.718</td>
      <td>6.027</td>
      <td>58.558634</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>868.445679</td>
      <td>1512.185833</td>
      <td>46.554395</td>
      <td>50.246028</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>99.020708</td>
      <td>41.998</td>
      <td>8.933</td>
      <td>5.932</td>
      <td>62.024805</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1310.496826</td>
      <td>2094.098791</td>
      <td>42.636120</td>
      <td>53.575056</td>
      <td>48.697311</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>86.654026</td>
      <td>34.383</td>
      <td>10.602</td>
      <td>4.562</td>
      <td>58.399829</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>578.319580</td>
      <td>2189.829730</td>
      <td>45.951383</td>
      <td>51.025182</td>
      <td>36.577950</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>95.981662</td>
      <td>41.757</td>
      <td>10.406</td>
      <td>5.930</td>
      <td>59.053829</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>874.714233</td>
      <td>1530.695403</td>
      <td>46.363681</td>
      <td>50.440498</td>
      <td>54.984901</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>98.253395</td>
      <td>41.452</td>
      <td>8.520</td>
      <td>5.802</td>
      <td>62.831537</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1439.984375</td>
      <td>2078.900486</td>
      <td>42.346470</td>
      <td>53.785685</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>85.923075</td>
      <td>33.716</td>
      <td>10.205</td>
      <td>4.442</td>
      <td>59.193976</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>597.558655</td>
      <td>2121.873660</td>
      <td>45.939359</td>
      <td>51.010453</td>
      <td>38.065460</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>96.038252</td>
      <td>41.053</td>
      <td>10.085</td>
      <td>5.820</td>
      <td>59.568073</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>868.423401</td>
      <td>1547.357836</td>
      <td>46.190833</td>
      <td>50.604823</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>97.609622</td>
      <td>40.898</td>
      <td>8.129</td>
      <td>5.674</td>
      <td>63.613756</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1492.553833</td>
      <td>2055.438830</td>
      <td>41.819358</td>
      <td>54.216902</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>84.444327</td>
      <td>33.096</td>
      <td>9.775</td>
      <td>4.328</td>
      <td>60.099854</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>626.206238</td>
      <td>2073.066614</td>
      <td>45.963895</td>
      <td>50.941242</td>
      <td>41.001839</td>
      <td>58.260000</td>
      <td>46.730000</td>
      <td>...</td>
      <td>5.794880</td>
      <td>24.99</td>
      <td>39.010</td>
      <td>50.940000</td>
      <td>69.650000</td>
      <td>96.304598</td>
      <td>40.351</td>
      <td>9.762</td>
      <td>5.704</td>
      <td>60.097317</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>1512.507552</td>
      <td>46.064093</td>
      <td>50.698642</td>
      <td>73.720787</td>
      <td>55.090000</td>
      <td>43.260000</td>
      <td>...</td>
      <td>0.713758</td>
      <td>9.15</td>
      <td>18.940</td>
      <td>25.280000</td>
      <td>42.740000</td>
      <td>97.243943</td>
      <td>40.353</td>
      <td>7.769</td>
      <td>5.553</td>
      <td>64.350951</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2079.844180</td>
      <td>41.318260</td>
      <td>54.619889</td>
      <td>61.982658</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>83.083490</td>
      <td>32.516</td>
      <td>9.330</td>
      <td>4.221</td>
      <td>61.074000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2095.342199</td>
      <td>45.884100</td>
      <td>50.970232</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>96.192946</td>
      <td>39.705</td>
      <td>9.444</td>
      <td>5.591</td>
      <td>60.633098</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>835.494751</td>
      <td>1556.855276</td>
      <td>45.935714</td>
      <td>50.785360</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>96.907141</td>
      <td>39.826</td>
      <td>7.447</td>
      <td>5.439</td>
      <td>65.030146</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1515.808716</td>
      <td>2091.693100</td>
      <td>40.835477</td>
      <td>55.003335</td>
      <td>65.722801</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>81.807159</td>
      <td>31.963</td>
      <td>8.893</td>
      <td>4.118</td>
      <td>62.071463</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2125.624163</td>
      <td>45.729366</td>
      <td>51.069993</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>95.809701</td>
      <td>39.153</td>
      <td>9.137</td>
      <td>5.488</td>
      <td>61.169878</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>930.572388</td>
      <td>1581.639092</td>
      <td>45.800132</td>
      <td>50.875323</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>96.558949</td>
      <td>39.319</td>
      <td>7.164</td>
      <td>5.333</td>
      <td>65.645293</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1568.253540</td>
      <td>2084.671422</td>
      <td>40.362708</td>
      <td>55.377693</td>
      <td>63.612751</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>1.632736</td>
      <td>11.24</td>
      <td>16.660</td>
      <td>18.980000</td>
      <td>31.520000</td>
      <td>80.578125</td>
      <td>31.429</td>
      <td>8.480</td>
      <td>4.018</td>
      <td>63.056415</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2157.313890</td>
      <td>45.551915</td>
      <td>51.191567</td>
      <td>73.720787</td>
      <td>59.600000</td>
      <td>46.780000</td>
      <td>...</td>
      <td>4.919682</td>
      <td>18.71</td>
      <td>29.680</td>
      <td>38.020000</td>
      <td>55.030000</td>
      <td>95.344674</td>
      <td>38.707</td>
      <td>8.845</td>
      <td>5.396</td>
      <td>61.704659</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>925.763672</td>
      <td>1603.219717</td>
      <td>45.642320</td>
      <td>50.989531</td>
      <td>73.720787</td>
      <td>59.490000</td>
      <td>48.180000</td>
      <td>...</td>
      <td>2.722716</td>
      <td>16.90</td>
      <td>29.140</td>
      <td>38.600000</td>
      <td>57.080000</td>
      <td>96.118688</td>
      <td>38.830</td>
      <td>6.920</td>
      <td>5.233</td>
      <td>66.194439</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>17.546949</td>
      <td>2093.922607</td>
      <td>3475.866745</td>
      <td>33.230254</td>
      <td>60.147890</td>
      <td>92.364281</td>
      <td>45.440000</td>
      <td>35.480000</td>
      <td>...</td>
      <td>1.035198</td>
      <td>1.73</td>
      <td>5.570</td>
      <td>6.360000</td>
      <td>17.340000</td>
      <td>66.256872</td>
      <td>19.887</td>
      <td>6.672</td>
      <td>2.322</td>
      <td>70.479171</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>5.343950</td>
      <td>1516.400016</td>
      <td>2698.985240</td>
      <td>41.024086</td>
      <td>54.692058</td>
      <td>75.359779</td>
      <td>54.890000</td>
      <td>43.560000</td>
      <td>...</td>
      <td>3.195781</td>
      <td>3.93</td>
      <td>9.270</td>
      <td>11.510000</td>
      <td>23.690000</td>
      <td>82.841905</td>
      <td>30.710</td>
      <td>5.663</td>
      <td>3.760</td>
      <td>69.887902</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>5.783592</td>
      <td>1516.400016</td>
      <td>2017.943010</td>
      <td>38.938629</td>
      <td>56.863273</td>
      <td>88.262627</td>
      <td>57.420000</td>
      <td>44.050000</td>
      <td>...</td>
      <td>2.614730</td>
      <td>11.40</td>
      <td>18.850</td>
      <td>23.790000</td>
      <td>37.300000</td>
      <td>75.860438</td>
      <td>26.318</td>
      <td>5.105</td>
      <td>3.164</td>
      <td>71.672463</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>18.448273</td>
      <td>2209.102051</td>
      <td>3597.961991</td>
      <td>32.566759</td>
      <td>60.626412</td>
      <td>97.298424</td>
      <td>45.240000</td>
      <td>35.720000</td>
      <td>...</td>
      <td>0.835006</td>
      <td>1.08</td>
      <td>4.100</td>
      <td>4.490000</td>
      <td>13.940000</td>
      <td>64.944613</td>
      <td>19.406</td>
      <td>6.662</td>
      <td>2.253</td>
      <td>70.780463</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>5.077445</td>
      <td>1695.110107</td>
      <td>2805.169791</td>
      <td>40.575993</td>
      <td>55.103794</td>
      <td>75.833748</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>81.475709</td>
      <td>30.085</td>
      <td>5.616</td>
      <td>3.664</td>
      <td>70.110780</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>6.034761</td>
      <td>1516.400016</td>
      <td>2104.759589</td>
      <td>38.204492</td>
      <td>57.552528</td>
      <td>73.720787</td>
      <td>56.160000</td>
      <td>43.810000</td>
      <td>...</td>
      <td>2.279461</td>
      <td>6.91</td>
      <td>13.880</td>
      <td>17.430000</td>
      <td>31.970000</td>
      <td>73.754313</td>
      <td>25.514</td>
      <td>5.077</td>
      <td>3.039</td>
      <td>71.858732</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>18.237120</td>
      <td>2308.634277</td>
      <td>3633.014903</td>
      <td>31.905088</td>
      <td>61.110033</td>
      <td>99.281097</td>
      <td>46.650000</td>
      <td>36.040000</td>
      <td>...</td>
      <td>1.114800</td>
      <td>1.99</td>
      <td>6.090</td>
      <td>6.920000</td>
      <td>18.580000</td>
      <td>63.639250</td>
      <td>18.969</td>
      <td>6.659</td>
      <td>2.189</td>
      <td>71.080780</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>5.240451</td>
      <td>1516.400016</td>
      <td>2833.735795</td>
      <td>40.091781</td>
      <td>55.548046</td>
      <td>78.546761</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>80.024334</td>
      <td>29.519</td>
      <td>5.576</td>
      <td>3.578</td>
      <td>70.328146</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>6.339264</td>
      <td>2035.134766</td>
      <td>2155.827865</td>
      <td>37.448905</td>
      <td>58.259473</td>
      <td>88.769539</td>
      <td>55.740000</td>
      <td>43.870000</td>
      <td>...</td>
      <td>2.130810</td>
      <td>6.30</td>
      <td>12.680</td>
      <td>16.140000</td>
      <td>29.350000</td>
      <td>71.645904</td>
      <td>24.728</td>
      <td>5.055</td>
      <td>2.918</td>
      <td>72.039976</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>19.096906</td>
      <td>2388.975342</td>
      <td>3509.156436</td>
      <td>31.228684</td>
      <td>61.613308</td>
      <td>102.675781</td>
      <td>45.930000</td>
      <td>36.070000</td>
      <td>...</td>
      <td>1.054102</td>
      <td>1.67</td>
      <td>5.520</td>
      <td>6.390000</td>
      <td>17.510000</td>
      <td>62.302599</td>
      <td>18.574</td>
      <td>6.663</td>
      <td>2.130</td>
      <td>71.378146</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>5.539466</td>
      <td>1516.400016</td>
      <td>2787.128287</td>
      <td>39.593279</td>
      <td>55.997451</td>
      <td>82.200653</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>78.579556</td>
      <td>29.016</td>
      <td>5.539</td>
      <td>3.501</td>
      <td>70.547537</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>6.338030</td>
      <td>1516.400016</td>
      <td>2068.185180</td>
      <td>36.665394</td>
      <td>58.987298</td>
      <td>91.863228</td>
      <td>51.560000</td>
      <td>39.140000</td>
      <td>...</td>
      <td>1.979316</td>
      <td>4.82</td>
      <td>10.880</td>
      <td>14.040000</td>
      <td>26.820000</td>
      <td>69.528024</td>
      <td>23.971</td>
      <td>5.038</td>
      <td>2.802</td>
      <td>72.217220</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>20.105788</td>
      <td>2484.339111</td>
      <td>3547.070983</td>
      <td>30.534690</td>
      <td>62.138073</td>
      <td>105.430397</td>
      <td>44.530000</td>
      <td>33.700000</td>
      <td>...</td>
      <td>1.120420</td>
      <td>2.33</td>
      <td>6.300</td>
      <td>7.240000</td>
      <td>18.550000</td>
      <td>60.931929</td>
      <td>18.223</td>
      <td>6.673</td>
      <td>2.078</td>
      <td>71.670610</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>5.639487</td>
      <td>1516.400016</td>
      <td>2805.951416</td>
      <td>39.095628</td>
      <td>56.434019</td>
      <td>84.213753</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>77.198083</td>
      <td>28.574</td>
      <td>5.503</td>
      <td>3.434</td>
      <td>70.775463</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>6.964149</td>
      <td>2263.870361</td>
      <td>2110.822021</td>
      <td>35.854009</td>
      <td>59.733111</td>
      <td>97.189880</td>
      <td>53.390000</td>
      <td>41.020000</td>
      <td>...</td>
      <td>2.183250</td>
      <td>5.40</td>
      <td>11.900</td>
      <td>15.470000</td>
      <td>29.110000</td>
      <td>67.411337</td>
      <td>23.261</td>
      <td>5.026</td>
      <td>2.695</td>
      <td>72.393976</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>20.886863</td>
      <td>2648.530029</td>
      <td>3615.583230</td>
      <td>29.801800</td>
      <td>62.706748</td>
      <td>109.104622</td>
      <td>42.430000</td>
      <td>32.860000</td>
      <td>...</td>
      <td>0.911424</td>
      <td>1.06</td>
      <td>4.390</td>
      <td>4.530000</td>
      <td>15.040000</td>
      <td>59.472471</td>
      <td>17.924</td>
      <td>6.692</td>
      <td>2.031</td>
      <td>71.956171</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>5.653971</td>
      <td>1516.400016</td>
      <td>2861.167894</td>
      <td>38.577533</td>
      <td>56.887778</td>
      <td>86.689102</td>
      <td>52.350000</td>
      <td>41.830000</td>
      <td>...</td>
      <td>3.983735</td>
      <td>4.00</td>
      <td>9.840</td>
      <td>11.530000</td>
      <td>26.470000</td>
      <td>75.784683</td>
      <td>28.182</td>
      <td>5.467</td>
      <td>3.373</td>
      <td>71.010415</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>6.437598</td>
      <td>1516.400016</td>
      <td>2157.984444</td>
      <td>35.042579</td>
      <td>60.480535</td>
      <td>100.720642</td>
      <td>57.400000</td>
      <td>45.670000</td>
      <td>...</td>
      <td>2.489454</td>
      <td>7.88</td>
      <td>14.660</td>
      <td>18.750000</td>
      <td>32.670000</td>
      <td>65.342451</td>
      <td>22.622</td>
      <td>5.017</td>
      <td>2.599</td>
      <td>72.572732</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>20.945491</td>
      <td>2797.323486</td>
      <td>3673.262887</td>
      <td>29.042627</td>
      <td>63.302907</td>
      <td>108.798729</td>
      <td>41.800000</td>
      <td>32.470000</td>
      <td>...</td>
      <td>0.826127</td>
      <td>0.98</td>
      <td>3.840</td>
      <td>4.160000</td>
      <td>13.610000</td>
      <td>57.970629</td>
      <td>17.676</td>
      <td>6.718</td>
      <td>1.991</td>
      <td>72.231854</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>5.586203</td>
      <td>1516.400016</td>
      <td>2884.897429</td>
      <td>38.086602</td>
      <td>57.307763</td>
      <td>86.083344</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>74.496429</td>
      <td>27.819</td>
      <td>5.433</td>
      <td>3.317</td>
      <td>71.249390</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>6.743448</td>
      <td>2261.272461</td>
      <td>2213.759527</td>
      <td>34.200196</td>
      <td>61.250126</td>
      <td>100.676102</td>
      <td>57.400000</td>
      <td>45.680000</td>
      <td>...</td>
      <td>2.883924</td>
      <td>9.25</td>
      <td>17.100</td>
      <td>21.360000</td>
      <td>37.260000</td>
      <td>63.264970</td>
      <td>22.065</td>
      <td>5.012</td>
      <td>2.514</td>
      <td>72.755024</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>20.560594</td>
      <td>2891.187012</td>
      <td>3730.422292</td>
      <td>28.295414</td>
      <td>63.885015</td>
      <td>106.839989</td>
      <td>43.510000</td>
      <td>34.350000</td>
      <td>...</td>
      <td>0.702177</td>
      <td>0.74</td>
      <td>3.160</td>
      <td>3.250000</td>
      <td>11.530000</td>
      <td>56.531230</td>
      <td>17.476</td>
      <td>6.751</td>
      <td>1.958</td>
      <td>72.498146</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>5.750461</td>
      <td>1871.932129</td>
      <td>2930.170750</td>
      <td>37.607424</td>
      <td>57.710289</td>
      <td>86.501770</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>73.279326</td>
      <td>27.465</td>
      <td>5.401</td>
      <td>3.263</td>
      <td>71.486390</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>6.798242</td>
      <td>2340.688232</td>
      <td>2242.818455</td>
      <td>33.349845</td>
      <td>62.015294</td>
      <td>94.479530</td>
      <td>53.670000</td>
      <td>41.480000</td>
      <td>...</td>
      <td>2.712175</td>
      <td>7.66</td>
      <td>15.240</td>
      <td>18.930000</td>
      <td>34.550000</td>
      <td>61.250546</td>
      <td>21.593</td>
      <td>5.010</td>
      <td>2.442</td>
      <td>72.942854</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>21.537939</td>
      <td>2886.402832</td>
      <td>3772.401570</td>
      <td>27.615213</td>
      <td>64.393636</td>
      <td>104.617020</td>
      <td>41.840000</td>
      <td>32.310000</td>
      <td>...</td>
      <td>0.689819</td>
      <td>0.64</td>
      <td>3.000</td>
      <td>2.970000</td>
      <td>11.290000</td>
      <td>55.294848</td>
      <td>17.314</td>
      <td>6.790</td>
      <td>1.931</td>
      <td>72.754561</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>5.716933</td>
      <td>1516.400016</td>
      <td>2990.594485</td>
      <td>37.120959</td>
      <td>58.115978</td>
      <td>86.624428</td>
      <td>48.660000</td>
      <td>38.360000</td>
      <td>...</td>
      <td>3.852810</td>
      <td>2.72</td>
      <td>8.110</td>
      <td>9.320000</td>
      <td>24.050000</td>
      <td>72.069718</td>
      <td>27.112</td>
      <td>5.370</td>
      <td>3.211</td>
      <td>71.722415</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>7.389157</td>
      <td>2334.632813</td>
      <td>2279.309902</td>
      <td>32.529328</td>
      <td>62.734875</td>
      <td>90.721970</td>
      <td>50.640000</td>
      <td>38.360000</td>
      <td>...</td>
      <td>2.484316</td>
      <td>6.01</td>
      <td>12.970</td>
      <td>15.960000</td>
      <td>31.210000</td>
      <td>59.400971</td>
      <td>21.203</td>
      <td>5.011</td>
      <td>2.382</td>
      <td>73.135707</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>22.073593</td>
      <td>1516.400016</td>
      <td>3853.107631</td>
      <td>27.028606</td>
      <td>64.799595</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>54.321951</td>
      <td>17.175</td>
      <td>6.833</td>
      <td>1.909</td>
      <td>73.001098</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>5.675817</td>
      <td>1516.400016</td>
      <td>3052.270569</td>
      <td>36.622822</td>
      <td>58.530645</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>70.850672</td>
      <td>26.752</td>
      <td>5.339</td>
      <td>3.159</td>
      <td>71.956488</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>7.418273</td>
      <td>1516.400016</td>
      <td>2329.002149</td>
      <td>31.762798</td>
      <td>63.383938</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>2.223817</td>
      <td>8.20</td>
      <td>15.195</td>
      <td>18.977037</td>
      <td>33.377593</td>
      <td>57.768677</td>
      <td>20.881</td>
      <td>5.015</td>
      <td>2.332</td>
      <td>73.333122</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 22 columns</p>
</div>




```python
migration_flows = pd.merge(migm3, mig6, on=['Year', 'Country'], how='right')
```


```python
migration_flows
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>Migration</th>
      <th>enrolment_tertiary</th>
      <th>GDP_percapita_constant</th>
      <th>pop_ages_0-14%</th>
      <th>pop_ages_14-64%</th>
      <th>primary_completion</th>
      <th>gini</th>
      <th>income_highest%</th>
      <th>...</th>
      <th>trade</th>
      <th>short_term_debt</th>
      <th>pop_growth</th>
      <th>remittances</th>
      <th>net_bilateral_aid</th>
      <th>imports_%GDP</th>
      <th>gov_consumption</th>
      <th>FDI</th>
      <th>exports_%GDP</th>
      <th>employment_15+</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1980</td>
      <td>El Salvador</td>
      <td>2.063205</td>
      <td>1516.400016</td>
      <td>2572.813235</td>
      <td>43.742478</td>
      <td>52.756733</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>67.406464</td>
      <td>22.0700</td>
      <td>1.739184</td>
      <td>1.372147</td>
      <td>43000000</td>
      <td>33.244917</td>
      <td>13.988965</td>
      <td>0.014116</td>
      <td>34.161547</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980</td>
      <td>Guatemala</td>
      <td>0.886027</td>
      <td>1516.400016</td>
      <td>2560.782037</td>
      <td>45.444923</td>
      <td>51.602977</td>
      <td>33.904148</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>47.105487</td>
      <td>26.3830</td>
      <td>2.635143</td>
      <td>0.332542</td>
      <td>17000000</td>
      <td>24.919086</td>
      <td>7.958166</td>
      <td>0.025385</td>
      <td>22.186401</td>
      <td>49.610001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1980</td>
      <td>Honduras</td>
      <td>1.076884</td>
      <td>713.525940</td>
      <td>1655.946421</td>
      <td>46.957200</td>
      <td>49.818337</td>
      <td>44.575001</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>81.293839</td>
      <td>17.3081</td>
      <td>3.145300</td>
      <td>0.062354</td>
      <td>19000000</td>
      <td>44.056895</td>
      <td>12.665627</td>
      <td>0.038971</td>
      <td>37.236944</td>
      <td>27.379999</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2267.095959</td>
      <td>43.481122</td>
      <td>52.948845</td>
      <td>46.450790</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>60.266492</td>
      <td>17.3809</td>
      <td>1.611673</td>
      <td>2.108693</td>
      <td>97000000</td>
      <td>33.587802</td>
      <td>15.829162</td>
      <td>0.014116</td>
      <td>26.678690</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>493.277863</td>
      <td>2509.736778</td>
      <td>45.617358</td>
      <td>51.409643</td>
      <td>33.957420</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>40.691257</td>
      <td>9.9649</td>
      <td>2.658257</td>
      <td>0.284635</td>
      <td>18000000</td>
      <td>23.601509</td>
      <td>7.900087</td>
      <td>-0.011618</td>
      <td>17.089748</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>821.092712</td>
      <td>1645.846419</td>
      <td>46.892259</td>
      <td>49.886066</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>69.338535</td>
      <td>13.3695</td>
      <td>3.113439</td>
      <td>0.062068</td>
      <td>35000000</td>
      <td>37.701720</td>
      <td>12.785955</td>
      <td>0.070935</td>
      <td>31.636815</td>
      <td>26.430000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1982</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>999.595276</td>
      <td>2092.554425</td>
      <td>43.204606</td>
      <td>53.154795</td>
      <td>49.382709</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>51.247740</td>
      <td>13.4222</td>
      <td>1.498034</td>
      <td>3.300787</td>
      <td>170000000</td>
      <td>28.470337</td>
      <td>15.778234</td>
      <td>0.014116</td>
      <td>22.777403</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1982</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2357.368296</td>
      <td>45.771834</td>
      <td>51.239138</td>
      <td>33.880711</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>33.474818</td>
      <td>7.2350</td>
      <td>2.669298</td>
      <td>0.122749</td>
      <td>20000000</td>
      <td>18.687621</td>
      <td>7.743490</td>
      <td>-0.045887</td>
      <td>14.787197</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1982</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>864.464600</td>
      <td>1573.671559</td>
      <td>46.745647</td>
      <td>50.043171</td>
      <td>49.944939</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>54.727051</td>
      <td>7.7859</td>
      <td>3.083349</td>
      <td>0.051662</td>
      <td>68000000</td>
      <td>28.052348</td>
      <td>13.053211</td>
      <td>-0.034441</td>
      <td>26.674702</td>
      <td>40.090000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1983</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1203.906616</td>
      <td>2094.864582</td>
      <td>42.920848</td>
      <td>53.365841</td>
      <td>50.613579</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>54.397801</td>
      <td>5.1284</td>
      <td>1.413014</td>
      <td>3.292315</td>
      <td>231000000</td>
      <td>29.909283</td>
      <td>15.829448</td>
      <td>0.014116</td>
      <td>24.488518</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1983</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>571.447571</td>
      <td>2236.567544</td>
      <td>45.891347</td>
      <td>51.104597</td>
      <td>35.137379</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>27.546959</td>
      <td>6.1899</td>
      <td>2.654604</td>
      <td>0.043094</td>
      <td>36000000</td>
      <td>14.552484</td>
      <td>7.602210</td>
      <td>0.000000</td>
      <td>12.994475</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1983</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>868.445679</td>
      <td>1512.185833</td>
      <td>46.554395</td>
      <td>50.246028</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>55.394868</td>
      <td>6.3325</td>
      <td>3.056878</td>
      <td>0.058499</td>
      <td>64000000</td>
      <td>29.233021</td>
      <td>13.113422</td>
      <td>0.064998</td>
      <td>26.161846</td>
      <td>35.599998</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1984</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1310.496826</td>
      <td>2094.098791</td>
      <td>42.636120</td>
      <td>53.575056</td>
      <td>48.697311</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>50.290675</td>
      <td>5.8131</td>
      <td>1.364660</td>
      <td>4.345542</td>
      <td>221000000</td>
      <td>28.536536</td>
      <td>16.036198</td>
      <td>0.014116</td>
      <td>21.754139</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1984</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>578.319580</td>
      <td>2189.829730</td>
      <td>45.951383</td>
      <td>51.025182</td>
      <td>36.577950</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>28.153115</td>
      <td>6.1022</td>
      <td>2.607323</td>
      <td>0.035903</td>
      <td>29000000</td>
      <td>15.151003</td>
      <td>7.665259</td>
      <td>0.052798</td>
      <td>13.002112</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1984</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>874.714233</td>
      <td>1530.695403</td>
      <td>46.363681</td>
      <td>50.440498</td>
      <td>54.984901</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>57.728231</td>
      <td>8.2600</td>
      <td>3.037319</td>
      <td>0.058753</td>
      <td>123000000</td>
      <td>32.027719</td>
      <td>13.196746</td>
      <td>-0.030130</td>
      <td>25.700512</td>
      <td>38.720001</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1985</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1439.984375</td>
      <td>2078.900486</td>
      <td>42.346470</td>
      <td>53.785685</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>52.210538</td>
      <td>4.3227</td>
      <td>1.343117</td>
      <td>4.135388</td>
      <td>287000000</td>
      <td>29.886772</td>
      <td>15.490521</td>
      <td>0.014116</td>
      <td>22.323766</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1985</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>597.558655</td>
      <td>2121.873660</td>
      <td>45.939359</td>
      <td>51.010453</td>
      <td>38.065460</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>24.932246</td>
      <td>9.5640</td>
      <td>2.541226</td>
      <td>0.010286</td>
      <td>50000000</td>
      <td>12.984016</td>
      <td>6.953551</td>
      <td>0.000000</td>
      <td>11.948230</td>
      <td>51.299999</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1985</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>868.423401</td>
      <td>1547.357836</td>
      <td>46.190833</td>
      <td>50.604823</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>54.966344</td>
      <td>10.6047</td>
      <td>3.020231</td>
      <td>0.057700</td>
      <td>161000000</td>
      <td>29.866742</td>
      <td>13.092458</td>
      <td>0.000000</td>
      <td>25.099602</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1986</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1492.553833</td>
      <td>2055.438830</td>
      <td>41.819358</td>
      <td>54.216902</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>53.714123</td>
      <td>6.5223</td>
      <td>1.324193</td>
      <td>4.170338</td>
      <td>272000000</td>
      <td>29.045960</td>
      <td>14.181097</td>
      <td>0.014116</td>
      <td>24.668162</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1986</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>626.206238</td>
      <td>2073.066614</td>
      <td>45.963895</td>
      <td>50.941242</td>
      <td>41.001839</td>
      <td>58.260000</td>
      <td>46.730000</td>
      <td>...</td>
      <td>30.644019</td>
      <td>10.7110</td>
      <td>2.470001</td>
      <td>0.009679</td>
      <td>86000000</td>
      <td>14.592119</td>
      <td>7.096224</td>
      <td>0.000000</td>
      <td>16.051900</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1986</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>1512.507552</td>
      <td>46.064093</td>
      <td>50.698642</td>
      <td>73.720787</td>
      <td>55.090000</td>
      <td>43.260000</td>
      <td>...</td>
      <td>54.890376</td>
      <td>11.6717</td>
      <td>2.998073</td>
      <td>0.055140</td>
      <td>175000000</td>
      <td>28.305107</td>
      <td>14.270711</td>
      <td>0.000000</td>
      <td>26.585269</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1987</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2079.844180</td>
      <td>41.318260</td>
      <td>54.619889</td>
      <td>61.982658</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>45.094622</td>
      <td>10.3707</td>
      <td>1.302156</td>
      <td>4.715458</td>
      <td>356000000</td>
      <td>26.102318</td>
      <td>13.745705</td>
      <td>0.014116</td>
      <td>18.992304</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1987</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2095.342199</td>
      <td>45.884100</td>
      <td>50.970232</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>38.142963</td>
      <td>9.8328</td>
      <td>2.413838</td>
      <td>0.001412</td>
      <td>155000000</td>
      <td>22.294055</td>
      <td>7.902434</td>
      <td>0.014116</td>
      <td>15.848908</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1987</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>835.494751</td>
      <td>1556.855276</td>
      <td>45.935714</td>
      <td>50.785360</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>48.789886</td>
      <td>11.9562</td>
      <td>2.966596</td>
      <td>0.811559</td>
      <td>153000000</td>
      <td>25.827815</td>
      <td>14.220348</td>
      <td>0.024082</td>
      <td>22.962071</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1988</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1515.808716</td>
      <td>2091.693100</td>
      <td>40.835477</td>
      <td>55.003335</td>
      <td>65.722801</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>38.095704</td>
      <td>11.6370</td>
      <td>1.291964</td>
      <td>5.029738</td>
      <td>318000000</td>
      <td>22.285022</td>
      <td>12.731673</td>
      <td>0.014116</td>
      <td>15.810682</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1988</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2125.624163</td>
      <td>45.729366</td>
      <td>51.069993</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>38.039914</td>
      <td>11.0787</td>
      <td>2.383684</td>
      <td>0.582789</td>
      <td>134000000</td>
      <td>21.936239</td>
      <td>7.981505</td>
      <td>-0.012752</td>
      <td>16.103675</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1988</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>930.572388</td>
      <td>1581.639092</td>
      <td>45.800132</td>
      <td>50.875323</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>55.215652</td>
      <td>12.1001</td>
      <td>2.927378</td>
      <td>1.052794</td>
      <td>155000000</td>
      <td>28.926603</td>
      <td>14.139012</td>
      <td>-0.025186</td>
      <td>26.289050</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1989</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>1568.253540</td>
      <td>2084.671422</td>
      <td>40.362708</td>
      <td>55.377693</td>
      <td>63.612751</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>36.928296</td>
      <td>9.7005</td>
      <td>1.293850</td>
      <td>5.439348</td>
      <td>310000000</td>
      <td>23.690508</td>
      <td>12.194134</td>
      <td>0.014116</td>
      <td>13.237788</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1989</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>1516.400016</td>
      <td>2157.313890</td>
      <td>45.551915</td>
      <td>51.191567</td>
      <td>73.720787</td>
      <td>59.600000</td>
      <td>46.780000</td>
      <td>...</td>
      <td>39.781546</td>
      <td>12.1191</td>
      <td>2.387014</td>
      <td>1.010615</td>
      <td>146000000</td>
      <td>22.474097</td>
      <td>7.895426</td>
      <td>-0.047558</td>
      <td>17.307449</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1989</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>925.763672</td>
      <td>1603.219717</td>
      <td>45.642320</td>
      <td>50.989531</td>
      <td>73.720787</td>
      <td>59.490000</td>
      <td>48.180000</td>
      <td>...</td>
      <td>65.347396</td>
      <td>11.4579</td>
      <td>2.879679</td>
      <td>1.363847</td>
      <td>102000000</td>
      <td>34.342945</td>
      <td>14.273274</td>
      <td>-0.028063</td>
      <td>31.004451</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2006</td>
      <td>El Salvador</td>
      <td>17.546949</td>
      <td>2093.922607</td>
      <td>3475.866745</td>
      <td>33.230254</td>
      <td>60.147890</td>
      <td>92.364281</td>
      <td>45.440000</td>
      <td>35.480000</td>
      <td>...</td>
      <td>71.849041</td>
      <td>12.6699</td>
      <td>0.341593</td>
      <td>18.773955</td>
      <td>24540000</td>
      <td>46.166991</td>
      <td>9.826583</td>
      <td>-0.141774</td>
      <td>25.682050</td>
      <td>49.169998</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2006</td>
      <td>Guatemala</td>
      <td>5.343950</td>
      <td>1516.400016</td>
      <td>2698.985240</td>
      <td>41.024086</td>
      <td>54.692058</td>
      <td>75.359779</td>
      <td>54.890000</td>
      <td>43.560000</td>
      <td>...</td>
      <td>66.818187</td>
      <td>13.4658</td>
      <td>2.298528</td>
      <td>12.239366</td>
      <td>67250000</td>
      <td>41.886457</td>
      <td>8.369963</td>
      <td>0.276205</td>
      <td>24.931730</td>
      <td>57.549999</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2006</td>
      <td>Honduras</td>
      <td>5.783592</td>
      <td>1516.400016</td>
      <td>2017.943010</td>
      <td>38.938629</td>
      <td>56.863273</td>
      <td>88.262627</td>
      <td>57.420000</td>
      <td>44.050000</td>
      <td>...</td>
      <td>133.131835</td>
      <td>7.1408</td>
      <td>1.826883</td>
      <td>21.557383</td>
      <td>84100000</td>
      <td>77.077193</td>
      <td>15.003781</td>
      <td>0.452738</td>
      <td>56.054642</td>
      <td>49.080002</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2007</td>
      <td>El Salvador</td>
      <td>18.448273</td>
      <td>2209.102051</td>
      <td>3597.961991</td>
      <td>32.566759</td>
      <td>60.626412</td>
      <td>97.298424</td>
      <td>45.240000</td>
      <td>35.720000</td>
      <td>...</td>
      <td>74.177439</td>
      <td>13.1110</td>
      <td>0.315511</td>
      <td>18.448488</td>
      <td>39040000</td>
      <td>48.294694</td>
      <td>9.281817</td>
      <td>0.473516</td>
      <td>25.882745</td>
      <td>49.500000</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2007</td>
      <td>Guatemala</td>
      <td>5.077445</td>
      <td>1695.110107</td>
      <td>2805.169791</td>
      <td>40.575993</td>
      <td>55.103794</td>
      <td>75.833748</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>67.898497</td>
      <td>15.8088</td>
      <td>2.254506</td>
      <td>12.418103</td>
      <td>45710000</td>
      <td>42.333233</td>
      <td>8.658056</td>
      <td>0.408934</td>
      <td>25.565264</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2007</td>
      <td>Honduras</td>
      <td>6.034761</td>
      <td>1516.400016</td>
      <td>2104.759589</td>
      <td>38.204492</td>
      <td>57.552528</td>
      <td>73.720787</td>
      <td>56.160000</td>
      <td>43.810000</td>
      <td>...</td>
      <td>135.070635</td>
      <td>10.5623</td>
      <td>1.792143</td>
      <td>21.291564</td>
      <td>71100000</td>
      <td>81.561623</td>
      <td>16.602374</td>
      <td>0.332742</td>
      <td>53.509012</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2008</td>
      <td>El Salvador</td>
      <td>18.237120</td>
      <td>2308.634277</td>
      <td>3633.014903</td>
      <td>31.905088</td>
      <td>61.110033</td>
      <td>99.281097</td>
      <td>46.650000</td>
      <td>36.040000</td>
      <td>...</td>
      <td>76.580188</td>
      <td>14.4812</td>
      <td>0.296649</td>
      <td>17.520181</td>
      <td>42370000</td>
      <td>49.698567</td>
      <td>9.175027</td>
      <td>0.370631</td>
      <td>26.881620</td>
      <td>59.020000</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2008</td>
      <td>Guatemala</td>
      <td>5.240451</td>
      <td>1516.400016</td>
      <td>2833.735795</td>
      <td>40.091781</td>
      <td>55.548046</td>
      <td>78.546761</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>64.125228</td>
      <td>15.5632</td>
      <td>2.215217</td>
      <td>11.395396</td>
      <td>70350000</td>
      <td>39.406974</td>
      <td>9.013268</td>
      <td>0.034880</td>
      <td>24.718254</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2008</td>
      <td>Honduras</td>
      <td>6.339264</td>
      <td>2035.134766</td>
      <td>2155.827865</td>
      <td>37.448905</td>
      <td>58.259473</td>
      <td>88.769539</td>
      <td>55.740000</td>
      <td>43.870000</td>
      <td>...</td>
      <td>135.748955</td>
      <td>13.3425</td>
      <td>1.747160</td>
      <td>20.459753</td>
      <td>96330000</td>
      <td>84.423679</td>
      <td>17.114828</td>
      <td>1.402889</td>
      <td>51.325277</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2009</td>
      <td>El Salvador</td>
      <td>19.096906</td>
      <td>2388.975342</td>
      <td>3509.156436</td>
      <td>31.228684</td>
      <td>61.613308</td>
      <td>102.675781</td>
      <td>45.930000</td>
      <td>36.070000</td>
      <td>...</td>
      <td>61.871642</td>
      <td>8.3258</td>
      <td>0.285542</td>
      <td>16.467451</td>
      <td>82080000</td>
      <td>38.674798</td>
      <td>10.629205</td>
      <td>0.014327</td>
      <td>23.196844</td>
      <td>58.160000</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2009</td>
      <td>Guatemala</td>
      <td>5.539466</td>
      <td>1516.400016</td>
      <td>2787.128287</td>
      <td>39.593279</td>
      <td>55.997451</td>
      <td>82.200653</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>57.105993</td>
      <td>8.2714</td>
      <td>2.183077</td>
      <td>10.651859</td>
      <td>83890000</td>
      <td>33.130589</td>
      <td>10.179023</td>
      <td>0.325371</td>
      <td>23.975404</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2009</td>
      <td>Honduras</td>
      <td>6.338030</td>
      <td>1516.400016</td>
      <td>2068.185180</td>
      <td>36.665394</td>
      <td>58.987298</td>
      <td>91.863228</td>
      <td>51.560000</td>
      <td>39.140000</td>
      <td>...</td>
      <td>96.905006</td>
      <td>6.5494</td>
      <td>1.688651</td>
      <td>17.101447</td>
      <td>128760000</td>
      <td>57.374755</td>
      <td>18.699738</td>
      <td>-0.074317</td>
      <td>39.530251</td>
      <td>59.320000</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2010</td>
      <td>El Salvador</td>
      <td>20.105788</td>
      <td>2484.339111</td>
      <td>3547.070983</td>
      <td>30.534690</td>
      <td>62.138073</td>
      <td>105.430397</td>
      <td>44.530000</td>
      <td>33.700000</td>
      <td>...</td>
      <td>68.768763</td>
      <td>7.5574</td>
      <td>0.280903</td>
      <td>16.209675</td>
      <td>148160000</td>
      <td>42.844203</td>
      <td>10.704865</td>
      <td>0.524807</td>
      <td>25.924560</td>
      <td>58.110001</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2010</td>
      <td>Guatemala</td>
      <td>5.639487</td>
      <td>1516.400016</td>
      <td>2805.951416</td>
      <td>39.095628</td>
      <td>56.434019</td>
      <td>84.213753</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>62.114932</td>
      <td>10.5394</td>
      <td>2.156000</td>
      <td>10.236867</td>
      <td>100500000</td>
      <td>36.309182</td>
      <td>10.475864</td>
      <td>0.153036</td>
      <td>25.805750</td>
      <td>42.919998</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2010</td>
      <td>Honduras</td>
      <td>6.964149</td>
      <td>2263.870361</td>
      <td>2110.822021</td>
      <td>35.854009</td>
      <td>59.733111</td>
      <td>97.189880</td>
      <td>53.390000</td>
      <td>41.020000</td>
      <td>...</td>
      <td>109.441838</td>
      <td>9.4110</td>
      <td>1.622622</td>
      <td>16.643140</td>
      <td>100840000</td>
      <td>63.682932</td>
      <td>17.926432</td>
      <td>-2.309012</td>
      <td>45.758906</td>
      <td>59.360001</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2011</td>
      <td>El Salvador</td>
      <td>20.886863</td>
      <td>2648.530029</td>
      <td>3615.583230</td>
      <td>29.801800</td>
      <td>62.706748</td>
      <td>109.104622</td>
      <td>42.430000</td>
      <td>32.860000</td>
      <td>...</td>
      <td>74.643243</td>
      <td>10.1925</td>
      <td>0.279522</td>
      <td>15.748084</td>
      <td>162440000</td>
      <td>46.663209</td>
      <td>11.055793</td>
      <td>-0.414578</td>
      <td>27.980034</td>
      <td>58.549999</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2011</td>
      <td>Guatemala</td>
      <td>5.653971</td>
      <td>1516.400016</td>
      <td>2861.167894</td>
      <td>38.577533</td>
      <td>56.887778</td>
      <td>86.689102</td>
      <td>52.350000</td>
      <td>41.830000</td>
      <td>...</td>
      <td>63.984196</td>
      <td>13.9156</td>
      <td>2.129043</td>
      <td>9.492785</td>
      <td>93080000</td>
      <td>37.358771</td>
      <td>10.189228</td>
      <td>0.274376</td>
      <td>26.625425</td>
      <td>59.230000</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2011</td>
      <td>Honduras</td>
      <td>6.437598</td>
      <td>1516.400016</td>
      <td>2157.984444</td>
      <td>35.042579</td>
      <td>60.480535</td>
      <td>100.720642</td>
      <td>57.400000</td>
      <td>45.670000</td>
      <td>...</td>
      <td>122.216903</td>
      <td>6.0942</td>
      <td>1.554236</td>
      <td>15.980083</td>
      <td>46360000</td>
      <td>70.959216</td>
      <td>16.064368</td>
      <td>0.172124</td>
      <td>51.257687</td>
      <td>49.660000</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2012</td>
      <td>El Salvador</td>
      <td>20.945491</td>
      <td>2797.323486</td>
      <td>3673.262887</td>
      <td>29.042627</td>
      <td>63.302907</td>
      <td>108.798729</td>
      <td>41.800000</td>
      <td>32.470000</td>
      <td>...</td>
      <td>69.698828</td>
      <td>11.7092</td>
      <td>0.280768</td>
      <td>16.361001</td>
      <td>150850000</td>
      <td>44.076494</td>
      <td>11.228458</td>
      <td>-0.150729</td>
      <td>25.622333</td>
      <td>59.400002</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2012</td>
      <td>Guatemala</td>
      <td>5.586203</td>
      <td>1516.400016</td>
      <td>2884.897429</td>
      <td>38.086602</td>
      <td>57.307763</td>
      <td>86.083344</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>60.982475</td>
      <td>4.5907</td>
      <td>2.100666</td>
      <td>9.983742</td>
      <td>95490000</td>
      <td>36.113781</td>
      <td>10.347731</td>
      <td>0.115398</td>
      <td>24.868694</td>
      <td>63.540001</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2012</td>
      <td>Honduras</td>
      <td>6.743448</td>
      <td>2261.272461</td>
      <td>2213.759527</td>
      <td>34.200196</td>
      <td>61.250126</td>
      <td>100.676102</td>
      <td>57.400000</td>
      <td>45.680000</td>
      <td>...</td>
      <td>121.188216</td>
      <td>7.9444</td>
      <td>1.493978</td>
      <td>15.871110</td>
      <td>52650000</td>
      <td>70.285417</td>
      <td>16.201783</td>
      <td>1.177095</td>
      <td>50.902799</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2013</td>
      <td>El Salvador</td>
      <td>20.560594</td>
      <td>2891.187012</td>
      <td>3730.422292</td>
      <td>28.295414</td>
      <td>63.885015</td>
      <td>106.839989</td>
      <td>43.510000</td>
      <td>34.350000</td>
      <td>...</td>
      <td>71.948881</td>
      <td>13.6861</td>
      <td>0.286321</td>
      <td>16.241404</td>
      <td>51090000</td>
      <td>45.577371</td>
      <td>11.564254</td>
      <td>0.271335</td>
      <td>26.371510</td>
      <td>59.880001</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2013</td>
      <td>Guatemala</td>
      <td>5.750461</td>
      <td>1871.932129</td>
      <td>2930.170750</td>
      <td>37.607424</td>
      <td>57.710289</td>
      <td>86.501770</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>58.548341</td>
      <td>4.1360</td>
      <td>2.073729</td>
      <td>9.988782</td>
      <td>102670000</td>
      <td>34.828649</td>
      <td>10.570013</td>
      <td>0.169948</td>
      <td>23.719693</td>
      <td>58.340000</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2013</td>
      <td>Honduras</td>
      <td>6.798242</td>
      <td>2340.688232</td>
      <td>2242.818455</td>
      <td>33.349845</td>
      <td>62.015294</td>
      <td>94.479530</td>
      <td>53.670000</td>
      <td>41.480000</td>
      <td>...</td>
      <td>116.306049</td>
      <td>7.4932</td>
      <td>1.449196</td>
      <td>16.863760</td>
      <td>90910000</td>
      <td>68.364559</td>
      <td>16.733813</td>
      <td>0.421417</td>
      <td>47.941490</td>
      <td>51.580002</td>
    </tr>
    <tr>
      <th>102</th>
      <td>2014</td>
      <td>El Salvador</td>
      <td>21.537939</td>
      <td>2886.402832</td>
      <td>3772.401570</td>
      <td>27.615213</td>
      <td>64.393636</td>
      <td>104.617020</td>
      <td>41.840000</td>
      <td>32.310000</td>
      <td>...</td>
      <td>69.570771</td>
      <td>13.8859</td>
      <td>0.296163</td>
      <td>16.567886</td>
      <td>45370000</td>
      <td>43.700058</td>
      <td>11.522619</td>
      <td>0.791026</td>
      <td>25.870712</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2014</td>
      <td>Guatemala</td>
      <td>5.716933</td>
      <td>1516.400016</td>
      <td>2990.594485</td>
      <td>37.120959</td>
      <td>58.115978</td>
      <td>86.624428</td>
      <td>48.660000</td>
      <td>38.360000</td>
      <td>...</td>
      <td>56.717915</td>
      <td>3.9421</td>
      <td>2.048252</td>
      <td>9.941701</td>
      <td>126040000</td>
      <td>33.559497</td>
      <td>10.848533</td>
      <td>-0.198735</td>
      <td>23.158419</td>
      <td>58.410000</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2014</td>
      <td>Honduras</td>
      <td>7.389157</td>
      <td>2334.632813</td>
      <td>2279.309902</td>
      <td>32.529328</td>
      <td>62.734875</td>
      <td>90.721970</td>
      <td>50.640000</td>
      <td>38.360000</td>
      <td>...</td>
      <td>112.609235</td>
      <td>6.8729</td>
      <td>1.424638</td>
      <td>17.385695</td>
      <td>80450000</td>
      <td>65.739895</td>
      <td>15.710074</td>
      <td>0.898223</td>
      <td>46.869340</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2015</td>
      <td>El Salvador</td>
      <td>22.073593</td>
      <td>1516.400016</td>
      <td>3853.107631</td>
      <td>27.028606</td>
      <td>64.799595</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>67.989029</td>
      <td>13.4668</td>
      <td>0.308592</td>
      <td>16.577147</td>
      <td>47470000</td>
      <td>42.030236</td>
      <td>11.893138</td>
      <td>0.346951</td>
      <td>25.958793</td>
      <td>51.230213</td>
    </tr>
    <tr>
      <th>106</th>
      <td>2015</td>
      <td>Guatemala</td>
      <td>5.675817</td>
      <td>1516.400016</td>
      <td>3052.270569</td>
      <td>36.622822</td>
      <td>58.530645</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>51.333403</td>
      <td>3.6161</td>
      <td>2.023674</td>
      <td>10.303115</td>
      <td>123500000</td>
      <td>30.043441</td>
      <td>10.365617</td>
      <td>0.049519</td>
      <td>21.289963</td>
      <td>58.900002</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2015</td>
      <td>Honduras</td>
      <td>7.418273</td>
      <td>1516.400016</td>
      <td>2329.002149</td>
      <td>31.762798</td>
      <td>63.383938</td>
      <td>73.720787</td>
      <td>52.529231</td>
      <td>40.809231</td>
      <td>...</td>
      <td>107.434916</td>
      <td>6.4723</td>
      <td>1.414027</td>
      <td>17.953123</td>
      <td>110380000</td>
      <td>62.588715</td>
      <td>14.704325</td>
      <td>0.998200</td>
      <td>44.846202</td>
      <td>60.630001</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 33 columns</p>
</div>



### We have finally merged and cleaned our datasets. Here we can start with the analysis.


```python
migration_flows.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 108 entries, 0 to 107
    Data columns (total 33 columns):
    Year                      108 non-null int64
    Country                   108 non-null object
    Migration                 108 non-null float64
    enrolment_tertiary        108 non-null float64
    GDP_percapita_constant    108 non-null float64
    pop_ages_0-14%            108 non-null float64
    pop_ages_14-64%           108 non-null float64
    primary_completion        108 non-null float64
    gini                      108 non-null float64
    income_highest%           108 non-null float64
    income_lowest%            108 non-null float64
    poor_1.90                 108 non-null float64
    poor_3.10                 108 non-null float64
    poverty_gap_1.90          108 non-null float64
    poverty_gap_3.10          108 non-null float64
    poverty_headcount_1.90    108 non-null float64
    poverty_headcount_3.10    108 non-null float64
    age_dependency            108 non-null float64
    birth_rate                108 non-null float64
    death_rate                108 non-null float64
    fertility_rate            108 non-null float64
    life_expectancy           108 non-null float64
    unemployment              108 non-null float64
    trade                     108 non-null float64
    short_term_debt           108 non-null float64
    pop_growth                108 non-null float64
    remittances               108 non-null float64
    net_bilateral_aid         108 non-null int64
    imports_%GDP              108 non-null float64
    gov_consumption           108 non-null float64
    FDI                       108 non-null float64
    exports_%GDP              108 non-null float64
    employment_15+            108 non-null float64
    dtypes: float64(30), int64(2), object(1)
    memory usage: 28.7+ KB


# 2. Checking for correlations accross our variables


```python
fig = plt.figure(figsize=(18,10))
ax = fig.gca()
sns.heatmap(migration_flows.corr(), annot=True, linewidths=.15, cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
plt.show()
```


    
![png](output_71_0.png)
    



```python
migration_flows.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 108 entries, 0 to 107
    Data columns (total 33 columns):
    Year                      108 non-null int64
    Country                   108 non-null object
    Migration                 108 non-null float64
    enrolment_tertiary        108 non-null float64
    GDP_percapita_constant    108 non-null float64
    pop_ages_0-14%            108 non-null float64
    pop_ages_14-64%           108 non-null float64
    primary_completion        108 non-null float64
    gini                      108 non-null float64
    income_highest%           108 non-null float64
    income_lowest%            108 non-null float64
    poor_1.90                 108 non-null float64
    poor_3.10                 108 non-null float64
    poverty_gap_1.90          108 non-null float64
    poverty_gap_3.10          108 non-null float64
    poverty_headcount_1.90    108 non-null float64
    poverty_headcount_3.10    108 non-null float64
    age_dependency            108 non-null float64
    birth_rate                108 non-null float64
    death_rate                108 non-null float64
    fertility_rate            108 non-null float64
    life_expectancy           108 non-null float64
    unemployment              108 non-null float64
    trade                     108 non-null float64
    short_term_debt           108 non-null float64
    pop_growth                108 non-null float64
    remittances               108 non-null float64
    net_bilateral_aid         108 non-null int64
    imports_%GDP              108 non-null float64
    gov_consumption           108 non-null float64
    FDI                       108 non-null float64
    exports_%GDP              108 non-null float64
    employment_15+            108 non-null float64
    dtypes: float64(30), int64(2), object(1)
    memory usage: 28.7+ KB


# Model 

### We drop variables with high correlation


```python
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
```


```python
fig = plt.figure(figsize=(18,10))
ax = fig.gca()
sns.heatmap(migration_flows.corr(), annot=True, linewidths=.15, cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
plt.show()
```


    
![png](output_76_0.png)
    



```python
migration_flows.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 108 entries, 0 to 107
    Data columns (total 14 columns):
    Year                      108 non-null datetime64[ns]
    Country                   108 non-null object
    Migration                 108 non-null float64
    GDP_percapita_constant    108 non-null float64
    income_highest%           108 non-null float64
    income_lowest%            108 non-null float64
    poverty_headcount_1.90    108 non-null float64
    death_rate                108 non-null float64
    unemployment              108 non-null float64
    trade                     108 non-null float64
    pop_growth                108 non-null float64
    remittances               108 non-null float64
    net_bilateral_aid         108 non-null int64
    FDI                       108 non-null float64
    dtypes: datetime64[ns](1), float64(11), int64(1), object(1)
    memory usage: 12.7+ KB


### We define a new variable for time variation


```python
migration_flows['Year'] = pd.to_datetime(migration_flows['Year'])
```


```python
migration_flows.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 108 entries, 0 to 107
    Data columns (total 14 columns):
    Year                      108 non-null datetime64[ns]
    Country                   108 non-null object
    Migration                 108 non-null float64
    GDP_percapita_constant    108 non-null float64
    income_highest%           108 non-null float64
    income_lowest%            108 non-null float64
    poverty_headcount_1.90    108 non-null float64
    death_rate                108 non-null float64
    unemployment              108 non-null float64
    trade                     108 non-null float64
    pop_growth                108 non-null float64
    remittances               108 non-null float64
    net_bilateral_aid         108 non-null int64
    FDI                       108 non-null float64
    dtypes: datetime64[ns](1), float64(11), int64(1), object(1)
    memory usage: 12.7+ KB


### We have reduced the correlation in our dataset, now we want to see which are our best predictors for migration. We revise our variables one last time.


```python
fig = plt.figure(figsize=(18,10))
ax = fig.gca()
sns.heatmap(migration_flows.corr(), annot=True, linewidths=.15, cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
plt.show()
```


    
![png](output_82_0.png)
    



```python
sns.pairplot(migration_flows, hue="Country", plot_kws={"s": 25}, size = 3)
```




    <seaborn.axisgrid.PairGrid at 0x13d861850>




    
![png](output_83_1.png)
    


# 3. Predictions


```python
y = migration_flows.Migration.values
x = migration_flows[['GDP_percapita_constant', 'income_highest%', 'income_lowest%', 'poverty_headcount_1.90', 'death_rate', 'unemployment', 'trade', 'pop_growth', 'remittances', 'net_bilateral_aid', 'FDI']]
```


```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Xn = ss.fit_transform(x)
```


```python
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Xn, y, test_size=0.3, random_state=10)
print x_train.shape, x_test.shape
print "\n======\n"
print y_train.shape, y_test.shape
```

    (75, 11) (33, 11)
    
    ======
    
    (75,) (33,)


# 4. Linear and Maching Learning Models

### 4.1 OLS


```python
from sklearn.linear_model import LinearRegression

## define a linear regression model
lr = LinearRegression()

## fit your model
lr.fit(x_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
```


```python
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
```


```python
rsquare_meansquare_error(y_train, y_test, x_train, x_test, "OLS", lr)
```

    MSE OLS train data: 4.49, test data: 5.22
    R^2 OLS train data: 0.85, test data: 0.72


### 4.2 Regularization


```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
```

### 4.2.1 Ridge


```python
## Find the optimal alpha
ridge_alphas = np.logspace(0, 5, 100)
optimal_ridge = RidgeCV(alphas=ridge_alphas, cv=10)
optimal_ridge.fit(x_train, y_train)
print (optimal_ridge.alpha_)
```

    11.497569954



```python
## Implement the Ridge Regression
ridge = Ridge(alpha=optimal_ridge.alpha_)

## Fit the Ridge regression
ridge.fit(x_train, y_train)
```




    Ridge(alpha=11.497569953977356, copy_X=True, fit_intercept=True,
       max_iter=None, normalize=False, random_state=None, solver='auto',
       tol=0.001)




```python
## Evaluate the Ridge Regression
rsquare_meansquare_error(y_train, y_test, x_train, x_test, "Ridge", ridge)
```

    MSE Ridge train data: 4.84, test data: 5.37
    R^2 Ridge train data: 0.84, test data: 0.71


### 4.2.2 Lasso


```python
## Find the optimal alpha
optimal_lasso = LassoCV(n_alphas=300, cv=10, verbose=1)
optimal_lasso.fit(x_train, y_train)
print optimal_lasso.alpha_
```

    ............................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................

    0.0282291816094


    ............................................................................................................[Parallel(n_jobs=1)]: Done  10 out of  10 | elapsed:    0.8s finished



```python
## Implement the Lasso Regression
lasso = Lasso(alpha=optimal_lasso.alpha_)

## fit your regression
lasso.fit(x_train, y_train)
```




    Lasso(alpha=0.028229181609392653, copy_X=True, fit_intercept=True,
       max_iter=1000, normalize=False, positive=False, precompute=False,
       random_state=None, selection='cyclic', tol=0.0001, warm_start=False)




```python
## Evaluate the Lasso Regression
rsquare_meansquare_error(y_train, y_test, x_train, x_test, "Lasso", lasso)
```

    MSE Lasso train data: 4.50, test data: 5.17
    R^2 Lasso train data: 0.85, test data: 0.72


### 4.2.3 Elastic Net


```python
## Find the optimal alphas
l1_ratios = np.linspace(0.01, 1.0, 50)
optimal_enet = ElasticNetCV(l1_ratio=l1_ratios, n_alphas=300, cv=5, verbose=1)
optimal_enet.fit(x_train, y_train)
print optimal_enet.alpha_
print optimal_enet.l1_ratio_
```

    ........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................

    0.064443056188
    0.070612244898


    [Parallel(n_jobs=1)]: Done 250 out of 250 | elapsed:   19.6s finished



```python
##  Create a model Enet
enet = ElasticNet(alpha=optimal_enet.alpha_, l1_ratio=optimal_enet.l1_ratio_)

## Fit your model
enet.fit(x_train, y_train)
```




    ElasticNet(alpha=0.064443056188017989, copy_X=True, fit_intercept=True,
          l1_ratio=0.070612244897959184, max_iter=1000, normalize=False,
          positive=False, precompute=False, random_state=None,
          selection='cyclic', tol=0.0001, warm_start=False)




```python
## Evaluate the Elastic Net Regression
rsquare_meansquare_error(y_train, y_test, x_train, x_test, "Elastic Net", enet)
```

    MSE Elastic Net train data: 4.60, test data: 5.27
    R^2 Elastic Net train data: 0.84, test data: 0.71


### Best Predictors


```python
''' Here I am defining a function to print the coefficients, their absolute values and the non-absolute values'''
def best_reg_method(x, best_regulari):
    method_coefs = pd.DataFrame({'variable':x.columns, 
                                 'coef':best_regulari.coef_, 
                                 'abs_coef':np.abs(best_regulari.coef_)})
    method_coefs.sort_values('abs_coef', inplace=True, ascending=False)
    '''you can change the number inside head to display more or less variables'''
    return method_coefs.head(10)
```


```python
best_reg_method(x, ridge)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>abs_coef</th>
      <th>coef</th>
      <th>variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1.740302</td>
      <td>1.740302</td>
      <td>income_lowest%</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.326838</td>
      <td>1.326838</td>
      <td>GDP_percapita_constant</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.202375</td>
      <td>-1.202375</td>
      <td>pop_growth</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.729386</td>
      <td>-0.729386</td>
      <td>death_rate</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.669221</td>
      <td>0.669221</td>
      <td>trade</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.668478</td>
      <td>0.668478</td>
      <td>remittances</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.584685</td>
      <td>-0.584685</td>
      <td>net_bilateral_aid</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.546543</td>
      <td>-0.546543</td>
      <td>income_highest%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.307262</td>
      <td>0.307262</td>
      <td>poverty_headcount_1.90</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.184696</td>
      <td>-0.184696</td>
      <td>FDI</td>
    </tr>
  </tbody>
</table>
</div>



### With ridge our best predictors are related to economic performance, inequality , population growth, death rates and trade


```python
best_reg_method(x, lasso)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>abs_coef</th>
      <th>coef</th>
      <th>variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2.253235</td>
      <td>2.253235</td>
      <td>income_lowest%</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.783755</td>
      <td>-1.783755</td>
      <td>pop_growth</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.584851</td>
      <td>1.584851</td>
      <td>GDP_percapita_constant</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.989429</td>
      <td>0.989429</td>
      <td>trade</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.912235</td>
      <td>-0.912235</td>
      <td>death_rate</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.696059</td>
      <td>-0.696059</td>
      <td>net_bilateral_aid</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.672069</td>
      <td>0.672069</td>
      <td>poverty_headcount_1.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.244444</td>
      <td>-0.244444</td>
      <td>income_highest%</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.168705</td>
      <td>-0.168705</td>
      <td>FDI</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.045264</td>
      <td>-0.045264</td>
      <td>unemployment</td>
    </tr>
  </tbody>
</table>
</div>



### We find similar results with lasso


```python
best_reg_method(x, enet)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>abs_coef</th>
      <th>coef</th>
      <th>variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2.014602</td>
      <td>2.014602</td>
      <td>income_lowest%</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1.456899</td>
      <td>1.456899</td>
      <td>GDP_percapita_constant</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.445306</td>
      <td>-1.445306</td>
      <td>pop_growth</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.810558</td>
      <td>0.810558</td>
      <td>trade</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.801342</td>
      <td>-0.801342</td>
      <td>death_rate</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.650087</td>
      <td>-0.650087</td>
      <td>net_bilateral_aid</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.584547</td>
      <td>0.584547</td>
      <td>poverty_headcount_1.90</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.469064</td>
      <td>-0.469064</td>
      <td>income_highest%</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.429958</td>
      <td>0.429958</td>
      <td>remittances</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.193813</td>
      <td>-0.193813</td>
      <td>FDI</td>
    </tr>
  </tbody>
</table>
</div>



### We find similar results with elastic net

### 4.3 Regression Tree


```python
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
```


```python
## Fit the regresion tree
dtr_gs.fit(x_train, y_train)
```

    Fitting 5 folds for each of 120 candidates, totalling 600 fits


    [Parallel(n_jobs=-1)]: Done 600 out of 600 | elapsed:    1.3s finished





    GridSearchCV(cv=5, error_score='raise',
           estimator=DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
               max_leaf_nodes=None, min_impurity_split=1e-07,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, presort=False, random_state=None,
               splitter='best'),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid={'max_features': [None, 'auto'], 'min_samples_split': [2, 5, 7], 'criterion': ['mse'], 'max_depth': [3, 5, 10, 20], 'min_samples_leaf': [1, 3, 5, 7, 10]},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=1)




```python
## Print Best Estimator, parameters and score
''' dtr_best = is the regression tree regressor with best parameters/estimators'''
dtr_best = dtr_gs.best_estimator_ 

print "best estimator", dtr_best
print "\n==========\n"
print "best parameters",  dtr_gs.best_params_
print "\n==========\n"
print "best score", dtr_gs.best_score_
```

    best estimator DecisionTreeRegressor(criterion='mse', max_depth=5, max_features='auto',
               max_leaf_nodes=None, min_impurity_split=1e-07,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, presort=False, random_state=None,
               splitter='best')
    
    ==========
    
    best parameters {'max_features': 'auto', 'min_samples_split': 2, 'criterion': 'mse', 'max_depth': 5, 'min_samples_leaf': 1}
    
    ==========
    
    best score 0.885738615734



```python
##features that best explain your Y
''' Here I am defining a function to print feature importance using best models'''
def feature_importance(x, best_model):
    feature_importance = pd.DataFrame({'feature':x.columns, 'importance':best_model.feature_importances_})
    feature_importance.sort_values('importance', ascending=False, inplace=True)
    return feature_importance  
```


```python
feature_importance(x, dtr_best)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GDP_percapita_constant</td>
      <td>8.039375e-01</td>
    </tr>
    <tr>
      <th>8</th>
      <td>remittances</td>
      <td>1.250507e-01</td>
    </tr>
    <tr>
      <th>9</th>
      <td>net_bilateral_aid</td>
      <td>3.330661e-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>death_rate</td>
      <td>2.985446e-02</td>
    </tr>
    <tr>
      <th>6</th>
      <td>trade</td>
      <td>6.407658e-03</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pop_growth</td>
      <td>1.004094e-03</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FDI</td>
      <td>3.731590e-04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>poverty_headcount_1.90</td>
      <td>6.509575e-05</td>
    </tr>
    <tr>
      <th>5</th>
      <td>unemployment</td>
      <td>7.798122e-07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>income_highest%</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>income_lowest%</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>



### With regression tree, the findings suggest that remmitances and billateral aid are also relevant predictors.


```python
## Predict 
y_pred_dtr= dtr_best.predict(x_test)
y_pred_dtr
```




    array([  6.97976052,  19.09690622,   6.97976052,   4.97569521,
             1.34203879,   2.22139377,   4.97569521,   6.97976052,
             1.34203879,   7.56872769,   1.34203879,   2.26399794,
             4.97569521,   1.34203879,   6.97976052,   4.47564148,
             6.97976052,   7.56872769,   4.97569521,   6.97976052,
             1.34203879,   4.47564148,   4.97569521,   1.34203879,
            20.94549073,   7.56872769,   4.47564148,   6.97976052,
             6.97976052,   4.97569521,   4.97569521,   7.56872769,   6.97976052])




```python
## Evaluate the Regression Tree performance on your train and test data
rsquare_meansquare_error(y_train, y_test, x_train, x_test, "Regression tree", dtr_best)
```

    MSE Regression tree train data: 0.19, test data: 1.55
    R^2 Regression tree train data: 0.99, test data: 0.92


### Our R2 is too high we may still have some isssues with correlation among our predictors

### 4.4 Random Forrest


```python
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor( )

params = {'max_depth':[3,4,5],  
          'max_leaf_nodes':[5,6,7], 
          'min_samples_split':[3,4],
          'n_estimators': [100]
         }

estimator_rfr = GridSearchCV(forest, params, n_jobs=-1,  cv=5,verbose=1)
```


```python
## Fit your random forest tree
estimator_rfr.fit(x_train, y_train)
```

    Fitting 5 folds for each of 18 candidates, totalling 90 fits


    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    8.1s
    [Parallel(n_jobs=-1)]: Done  90 out of  90 | elapsed:   16.2s finished





    GridSearchCV(cv=5, error_score='raise',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_split=1e-07, min_samples_leaf=1,
               min_samples_split=2, min_weight_fraction_leaf=0.0,
               n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
               verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid={'min_samples_split': [3, 4], 'max_leaf_nodes': [5, 6, 7], 'n_estimators': [100], 'max_depth': [3, 4, 5]},
           pre_dispatch='2*n_jobs', refit=True, scoring=None, verbose=1)




```python
## Print the best estimator, parameters and score
''' rfr_best = is the random forest regression tree regressor with best parameters/estimators'''
rfr_best = estimator_rfr.best_estimator_
print "best estimator", rfr_best
print "\n==========\n"
print "best parameters", estimator_rfr.best_params_
print "\n==========\n"
print "best score", estimator_rfr.best_score_
```

    best estimator RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=5,
               max_features='auto', max_leaf_nodes=7, min_impurity_split=1e-07,
               min_samples_leaf=1, min_samples_split=3,
               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False)
    
    ==========
    
    best parameters {'min_samples_split': 3, 'max_leaf_nodes': 7, 'n_estimators': 100, 'max_depth': 5}
    
    ==========
    
    best score 0.879425646607



```python
## Print the feauure importance
feature_importance(x, rfr_best)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GDP_percapita_constant</td>
      <td>0.504375</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pop_growth</td>
      <td>0.237784</td>
    </tr>
    <tr>
      <th>8</th>
      <td>remittances</td>
      <td>0.088999</td>
    </tr>
    <tr>
      <th>4</th>
      <td>death_rate</td>
      <td>0.076235</td>
    </tr>
    <tr>
      <th>3</th>
      <td>poverty_headcount_1.90</td>
      <td>0.040469</td>
    </tr>
    <tr>
      <th>9</th>
      <td>net_bilateral_aid</td>
      <td>0.018331</td>
    </tr>
    <tr>
      <th>2</th>
      <td>income_lowest%</td>
      <td>0.017777</td>
    </tr>
    <tr>
      <th>1</th>
      <td>income_highest%</td>
      <td>0.008949</td>
    </tr>
    <tr>
      <th>6</th>
      <td>trade</td>
      <td>0.003667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>unemployment</td>
      <td>0.002310</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FDI</td>
      <td>0.001104</td>
    </tr>
  </tbody>
</table>
</div>



### Remittances might have some colinearity with the levels of migration and that might be overestimating our R2. Economic performance, population, death rates and inequality are important according to our results.


```python
## Predict
y_pred_rfdtr= rfr_best.predict(x_test)
y_pred_rfdtr
```




    array([  6.7452667 ,  18.92962689,   6.35557839,   5.85822173,
             2.26808502,   3.41398234,   5.77375152,   6.31265132,
             1.49663576,   4.95788407,   1.50335546,   1.89007663,
             5.26840944,   1.54260286,   6.85693493,   4.73685239,
            13.14452424,   4.38669293,   6.56894847,   7.79768882,
             1.82853601,   4.31161426,   5.77664529,   1.82853601,
            20.33043929,   5.08040759,   3.545363  ,   6.78950492,
             6.74578188,   5.50926387,   5.77375152,   5.08040759,   7.47567285])




```python
## Evaluate your model
rsquare_meansquare_error(y_train, y_test, x_train, x_test, "Random Forest Regression tree", rfr_best)
```

    MSE Random Forest Regression tree train data: 0.84, test data: 1.45
    R^2 Random Forest Regression tree train data: 0.97, test data: 0.92


### Our findings with random forrest seems to match our expectations about the best predictors from an intuitively point of view. Even though our R2 is still too high, it suggests that remittances might be causing collinearity in our model, based on its high relevance as a predictor.

# 5. PCA

### Now we are going to use PCA and KNN in our estimations.


```python
print migration_flows['Migration'].mean()
```

    5.86008409129



```python
migration_flows['threshold'] = migration_flows['Migration'] >= 5.86008409129
```


```python
migration_flows['threshold']= migration_flows['threshold'].apply(lambda x: 1 if x== True else 0)
```


```python
migration_flows
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Country</th>
      <th>Migration</th>
      <th>GDP_percapita_constant</th>
      <th>income_highest%</th>
      <th>income_lowest%</th>
      <th>poverty_headcount_1.90</th>
      <th>death_rate</th>
      <th>unemployment</th>
      <th>trade</th>
      <th>pop_growth</th>
      <th>remittances</th>
      <th>net_bilateral_aid</th>
      <th>FDI</th>
      <th>threshold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1970-01-01 00:00:00.000001980</td>
      <td>El Salvador</td>
      <td>2.063205</td>
      <td>2572.813235</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>11.681</td>
      <td>13.340000</td>
      <td>67.406464</td>
      <td>1.739184</td>
      <td>1.372147</td>
      <td>43000000</td>
      <td>0.014116</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1970-01-01 00:00:00.000001980</td>
      <td>Guatemala</td>
      <td>0.886027</td>
      <td>2560.782037</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>11.568</td>
      <td>5.496944</td>
      <td>47.105487</td>
      <td>2.635143</td>
      <td>0.332542</td>
      <td>17000000</td>
      <td>0.025385</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1970-01-01 00:00:00.000001980</td>
      <td>Honduras</td>
      <td>1.076884</td>
      <td>1655.946421</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.233</td>
      <td>5.496944</td>
      <td>81.293839</td>
      <td>3.145300</td>
      <td>0.062354</td>
      <td>19000000</td>
      <td>0.038971</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1970-01-01 00:00:00.000001981</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>2267.095959</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>11.494</td>
      <td>5.496944</td>
      <td>60.266492</td>
      <td>1.611673</td>
      <td>2.108693</td>
      <td>97000000</td>
      <td>0.014116</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1970-01-01 00:00:00.000001981</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>2509.736778</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>11.300</td>
      <td>2.150000</td>
      <td>40.691257</td>
      <td>2.658257</td>
      <td>0.284635</td>
      <td>18000000</td>
      <td>-0.011618</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1970-01-01 00:00:00.000001981</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>1645.846419</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>9.793</td>
      <td>5.496944</td>
      <td>69.338535</td>
      <td>3.113439</td>
      <td>0.062068</td>
      <td>35000000</td>
      <td>0.070935</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1970-01-01 00:00:00.000001982</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>2092.554425</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>11.251</td>
      <td>5.496944</td>
      <td>51.247740</td>
      <td>1.498034</td>
      <td>3.300787</td>
      <td>170000000</td>
      <td>0.014116</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1970-01-01 00:00:00.000001982</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>2357.368296</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>11.016</td>
      <td>2.270000</td>
      <td>33.474818</td>
      <td>2.669298</td>
      <td>0.122749</td>
      <td>20000000</td>
      <td>-0.045887</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1970-01-01 00:00:00.000001982</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>1573.671559</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>9.359</td>
      <td>7.300000</td>
      <td>54.727051</td>
      <td>3.083349</td>
      <td>0.051662</td>
      <td>68000000</td>
      <td>-0.034441</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1970-01-01 00:00:00.000001983</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>2094.864582</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.953</td>
      <td>5.496944</td>
      <td>54.397801</td>
      <td>1.413014</td>
      <td>3.292315</td>
      <td>231000000</td>
      <td>0.014116</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1970-01-01 00:00:00.000001983</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>2236.567544</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.718</td>
      <td>5.496944</td>
      <td>27.546959</td>
      <td>2.654604</td>
      <td>0.043094</td>
      <td>36000000</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1970-01-01 00:00:00.000001983</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>1512.185833</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>8.933</td>
      <td>5.496944</td>
      <td>55.394868</td>
      <td>3.056878</td>
      <td>0.058499</td>
      <td>64000000</td>
      <td>0.064998</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1970-01-01 00:00:00.000001984</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>2094.098791</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.602</td>
      <td>5.496944</td>
      <td>50.290675</td>
      <td>1.364660</td>
      <td>4.345542</td>
      <td>221000000</td>
      <td>0.014116</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1970-01-01 00:00:00.000001984</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>2189.829730</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.406</td>
      <td>5.496944</td>
      <td>28.153115</td>
      <td>2.607323</td>
      <td>0.035903</td>
      <td>29000000</td>
      <td>0.052798</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1970-01-01 00:00:00.000001984</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>1530.695403</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>8.520</td>
      <td>5.496944</td>
      <td>57.728231</td>
      <td>3.037319</td>
      <td>0.058753</td>
      <td>123000000</td>
      <td>-0.030130</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1970-01-01 00:00:00.000001985</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>2078.900486</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.205</td>
      <td>16.950001</td>
      <td>52.210538</td>
      <td>1.343117</td>
      <td>4.135388</td>
      <td>287000000</td>
      <td>0.014116</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1970-01-01 00:00:00.000001985</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>2121.873660</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.085</td>
      <td>5.496944</td>
      <td>24.932246</td>
      <td>2.541226</td>
      <td>0.010286</td>
      <td>50000000</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1970-01-01 00:00:00.000001985</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>1547.357836</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>8.129</td>
      <td>5.496944</td>
      <td>54.966344</td>
      <td>3.020231</td>
      <td>0.057700</td>
      <td>161000000</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1970-01-01 00:00:00.000001986</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>2055.438830</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>9.775</td>
      <td>7.900000</td>
      <td>53.714123</td>
      <td>1.324193</td>
      <td>4.170338</td>
      <td>272000000</td>
      <td>0.014116</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1970-01-01 00:00:00.000001986</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>2073.066614</td>
      <td>46.730000</td>
      <td>1.000000</td>
      <td>50.940000</td>
      <td>9.762</td>
      <td>5.496944</td>
      <td>30.644019</td>
      <td>2.470001</td>
      <td>0.009679</td>
      <td>86000000</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1970-01-01 00:00:00.000001986</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>1512.507552</td>
      <td>43.260000</td>
      <td>1.230000</td>
      <td>25.280000</td>
      <td>7.769</td>
      <td>12.120000</td>
      <td>54.890376</td>
      <td>2.998073</td>
      <td>0.055140</td>
      <td>175000000</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1970-01-01 00:00:00.000001987</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>2079.844180</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>9.330</td>
      <td>5.496944</td>
      <td>45.094622</td>
      <td>1.302156</td>
      <td>4.715458</td>
      <td>356000000</td>
      <td>0.014116</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1970-01-01 00:00:00.000001987</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>2095.342199</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>9.444</td>
      <td>3.500000</td>
      <td>38.142963</td>
      <td>2.413838</td>
      <td>0.001412</td>
      <td>155000000</td>
      <td>0.014116</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1970-01-01 00:00:00.000001987</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>1556.855276</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>7.447</td>
      <td>11.400000</td>
      <td>48.789886</td>
      <td>2.966596</td>
      <td>0.811559</td>
      <td>153000000</td>
      <td>0.024082</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1970-01-01 00:00:00.000001988</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>2091.693100</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>8.893</td>
      <td>9.370000</td>
      <td>38.095704</td>
      <td>1.291964</td>
      <td>5.029738</td>
      <td>318000000</td>
      <td>0.014116</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1970-01-01 00:00:00.000001988</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>2125.624163</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>9.137</td>
      <td>5.496944</td>
      <td>38.039914</td>
      <td>2.383684</td>
      <td>0.582789</td>
      <td>134000000</td>
      <td>-0.012752</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1970-01-01 00:00:00.000001988</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>1581.639092</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>7.164</td>
      <td>5.496944</td>
      <td>55.215652</td>
      <td>2.927378</td>
      <td>1.052794</td>
      <td>155000000</td>
      <td>-0.025186</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1970-01-01 00:00:00.000001989</td>
      <td>El Salvador</td>
      <td>1.342039</td>
      <td>2084.671422</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.980000</td>
      <td>8.480</td>
      <td>8.350000</td>
      <td>36.928296</td>
      <td>1.293850</td>
      <td>5.439348</td>
      <td>310000000</td>
      <td>0.014116</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1970-01-01 00:00:00.000001989</td>
      <td>Guatemala</td>
      <td>1.342039</td>
      <td>2157.313890</td>
      <td>46.780000</td>
      <td>0.680000</td>
      <td>38.020000</td>
      <td>8.845</td>
      <td>2.000000</td>
      <td>39.781546</td>
      <td>2.387014</td>
      <td>1.010615</td>
      <td>146000000</td>
      <td>-0.047558</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1970-01-01 00:00:00.000001989</td>
      <td>Honduras</td>
      <td>1.342039</td>
      <td>1603.219717</td>
      <td>48.180000</td>
      <td>1.040000</td>
      <td>38.600000</td>
      <td>6.920</td>
      <td>5.496944</td>
      <td>65.347396</td>
      <td>2.879679</td>
      <td>1.363847</td>
      <td>102000000</td>
      <td>-0.028063</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1970-01-01 00:00:00.000002006</td>
      <td>El Salvador</td>
      <td>17.546949</td>
      <td>3475.866745</td>
      <td>35.480000</td>
      <td>1.790000</td>
      <td>6.360000</td>
      <td>6.672</td>
      <td>6.570000</td>
      <td>71.849041</td>
      <td>0.341593</td>
      <td>18.773955</td>
      <td>24540000</td>
      <td>-0.141774</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1970-01-01 00:00:00.000002006</td>
      <td>Guatemala</td>
      <td>5.343950</td>
      <td>2698.985240</td>
      <td>43.560000</td>
      <td>1.070000</td>
      <td>11.510000</td>
      <td>5.663</td>
      <td>1.820000</td>
      <td>66.818187</td>
      <td>2.298528</td>
      <td>12.239366</td>
      <td>67250000</td>
      <td>0.276205</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>1970-01-01 00:00:00.000002006</td>
      <td>Honduras</td>
      <td>5.783592</td>
      <td>2017.943010</td>
      <td>44.050000</td>
      <td>0.580000</td>
      <td>23.790000</td>
      <td>5.105</td>
      <td>3.110000</td>
      <td>133.131835</td>
      <td>1.826883</td>
      <td>21.557383</td>
      <td>84100000</td>
      <td>0.452738</td>
      <td>0</td>
    </tr>
    <tr>
      <th>81</th>
      <td>1970-01-01 00:00:00.000002007</td>
      <td>El Salvador</td>
      <td>18.448273</td>
      <td>3597.961991</td>
      <td>35.720000</td>
      <td>1.930000</td>
      <td>4.490000</td>
      <td>6.662</td>
      <td>6.410000</td>
      <td>74.177439</td>
      <td>0.315511</td>
      <td>18.448488</td>
      <td>39040000</td>
      <td>0.473516</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82</th>
      <td>1970-01-01 00:00:00.000002007</td>
      <td>Guatemala</td>
      <td>5.077445</td>
      <td>2805.169791</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.616</td>
      <td>5.496944</td>
      <td>67.898497</td>
      <td>2.254506</td>
      <td>12.418103</td>
      <td>45710000</td>
      <td>0.408934</td>
      <td>0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>1970-01-01 00:00:00.000002007</td>
      <td>Honduras</td>
      <td>6.034761</td>
      <td>2104.759589</td>
      <td>43.810000</td>
      <td>0.900000</td>
      <td>17.430000</td>
      <td>5.077</td>
      <td>2.920000</td>
      <td>135.070635</td>
      <td>1.792143</td>
      <td>21.291564</td>
      <td>71100000</td>
      <td>0.332742</td>
      <td>1</td>
    </tr>
    <tr>
      <th>84</th>
      <td>1970-01-01 00:00:00.000002008</td>
      <td>El Salvador</td>
      <td>18.237120</td>
      <td>3633.014903</td>
      <td>36.040000</td>
      <td>1.700000</td>
      <td>6.920000</td>
      <td>6.659</td>
      <td>5.880000</td>
      <td>76.580188</td>
      <td>0.296649</td>
      <td>17.520181</td>
      <td>42370000</td>
      <td>0.370631</td>
      <td>1</td>
    </tr>
    <tr>
      <th>85</th>
      <td>1970-01-01 00:00:00.000002008</td>
      <td>Guatemala</td>
      <td>5.240451</td>
      <td>2833.735795</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.576</td>
      <td>5.496944</td>
      <td>64.125228</td>
      <td>2.215217</td>
      <td>11.395396</td>
      <td>70350000</td>
      <td>0.034880</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>1970-01-01 00:00:00.000002008</td>
      <td>Honduras</td>
      <td>6.339264</td>
      <td>2155.827865</td>
      <td>43.870000</td>
      <td>0.910000</td>
      <td>16.140000</td>
      <td>5.055</td>
      <td>2.990000</td>
      <td>135.748955</td>
      <td>1.747160</td>
      <td>20.459753</td>
      <td>96330000</td>
      <td>1.402889</td>
      <td>1</td>
    </tr>
    <tr>
      <th>87</th>
      <td>1970-01-01 00:00:00.000002009</td>
      <td>El Salvador</td>
      <td>19.096906</td>
      <td>3509.156436</td>
      <td>36.070000</td>
      <td>1.780000</td>
      <td>6.390000</td>
      <td>6.663</td>
      <td>7.330000</td>
      <td>61.871642</td>
      <td>0.285542</td>
      <td>16.467451</td>
      <td>82080000</td>
      <td>0.014327</td>
      <td>1</td>
    </tr>
    <tr>
      <th>88</th>
      <td>1970-01-01 00:00:00.000002009</td>
      <td>Guatemala</td>
      <td>5.539466</td>
      <td>2787.128287</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.539</td>
      <td>5.496944</td>
      <td>57.105993</td>
      <td>2.183077</td>
      <td>10.651859</td>
      <td>83890000</td>
      <td>0.325371</td>
      <td>0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>1970-01-01 00:00:00.000002009</td>
      <td>Honduras</td>
      <td>6.338030</td>
      <td>2068.185180</td>
      <td>39.140000</td>
      <td>1.150000</td>
      <td>14.040000</td>
      <td>5.038</td>
      <td>3.280000</td>
      <td>96.905006</td>
      <td>1.688651</td>
      <td>17.101447</td>
      <td>128760000</td>
      <td>-0.074317</td>
      <td>1</td>
    </tr>
    <tr>
      <th>90</th>
      <td>1970-01-01 00:00:00.000002010</td>
      <td>El Salvador</td>
      <td>20.105788</td>
      <td>3547.070983</td>
      <td>33.700000</td>
      <td>1.670000</td>
      <td>7.240000</td>
      <td>6.673</td>
      <td>7.050000</td>
      <td>68.768763</td>
      <td>0.280903</td>
      <td>16.209675</td>
      <td>148160000</td>
      <td>0.524807</td>
      <td>1</td>
    </tr>
    <tr>
      <th>91</th>
      <td>1970-01-01 00:00:00.000002010</td>
      <td>Guatemala</td>
      <td>5.639487</td>
      <td>2805.951416</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.503</td>
      <td>3.740000</td>
      <td>62.114932</td>
      <td>2.156000</td>
      <td>10.236867</td>
      <td>100500000</td>
      <td>0.153036</td>
      <td>0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>1970-01-01 00:00:00.000002010</td>
      <td>Honduras</td>
      <td>6.964149</td>
      <td>2110.822021</td>
      <td>41.020000</td>
      <td>1.090000</td>
      <td>15.470000</td>
      <td>5.026</td>
      <td>4.100000</td>
      <td>109.441838</td>
      <td>1.622622</td>
      <td>16.643140</td>
      <td>100840000</td>
      <td>-2.309012</td>
      <td>1</td>
    </tr>
    <tr>
      <th>93</th>
      <td>1970-01-01 00:00:00.000002011</td>
      <td>El Salvador</td>
      <td>20.886863</td>
      <td>3615.583230</td>
      <td>32.860000</td>
      <td>2.110000</td>
      <td>4.530000</td>
      <td>6.692</td>
      <td>6.620000</td>
      <td>74.643243</td>
      <td>0.279522</td>
      <td>15.748084</td>
      <td>162440000</td>
      <td>-0.414578</td>
      <td>1</td>
    </tr>
    <tr>
      <th>94</th>
      <td>1970-01-01 00:00:00.000002011</td>
      <td>Guatemala</td>
      <td>5.653971</td>
      <td>2861.167894</td>
      <td>41.830000</td>
      <td>1.340000</td>
      <td>11.530000</td>
      <td>5.467</td>
      <td>4.130000</td>
      <td>63.984196</td>
      <td>2.129043</td>
      <td>9.492785</td>
      <td>93080000</td>
      <td>0.274376</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1970-01-01 00:00:00.000002011</td>
      <td>Honduras</td>
      <td>6.437598</td>
      <td>2157.984444</td>
      <td>45.670000</td>
      <td>0.750000</td>
      <td>18.750000</td>
      <td>5.017</td>
      <td>4.270000</td>
      <td>122.216903</td>
      <td>1.554236</td>
      <td>15.980083</td>
      <td>46360000</td>
      <td>0.172124</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>1970-01-01 00:00:00.000002012</td>
      <td>El Salvador</td>
      <td>20.945491</td>
      <td>3673.262887</td>
      <td>32.470000</td>
      <td>2.150000</td>
      <td>4.160000</td>
      <td>6.718</td>
      <td>6.070000</td>
      <td>69.698828</td>
      <td>0.280768</td>
      <td>16.361001</td>
      <td>150850000</td>
      <td>-0.150729</td>
      <td>1</td>
    </tr>
    <tr>
      <th>97</th>
      <td>1970-01-01 00:00:00.000002012</td>
      <td>Guatemala</td>
      <td>5.586203</td>
      <td>2884.897429</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.433</td>
      <td>2.870000</td>
      <td>60.982475</td>
      <td>2.100666</td>
      <td>9.983742</td>
      <td>95490000</td>
      <td>0.115398</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1970-01-01 00:00:00.000002012</td>
      <td>Honduras</td>
      <td>6.743448</td>
      <td>2213.759527</td>
      <td>45.680000</td>
      <td>0.790000</td>
      <td>21.360000</td>
      <td>5.012</td>
      <td>5.496944</td>
      <td>121.188216</td>
      <td>1.493978</td>
      <td>15.871110</td>
      <td>52650000</td>
      <td>1.177095</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1970-01-01 00:00:00.000002013</td>
      <td>El Salvador</td>
      <td>20.560594</td>
      <td>3730.422292</td>
      <td>34.350000</td>
      <td>2.110000</td>
      <td>3.250000</td>
      <td>6.751</td>
      <td>5.930000</td>
      <td>71.948881</td>
      <td>0.286321</td>
      <td>16.241404</td>
      <td>51090000</td>
      <td>0.271335</td>
      <td>1</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1970-01-01 00:00:00.000002013</td>
      <td>Guatemala</td>
      <td>5.750461</td>
      <td>2930.170750</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.401</td>
      <td>2.990000</td>
      <td>58.548341</td>
      <td>2.073729</td>
      <td>9.988782</td>
      <td>102670000</td>
      <td>0.169948</td>
      <td>0</td>
    </tr>
    <tr>
      <th>101</th>
      <td>1970-01-01 00:00:00.000002013</td>
      <td>Honduras</td>
      <td>6.798242</td>
      <td>2242.818455</td>
      <td>41.480000</td>
      <td>0.980000</td>
      <td>18.930000</td>
      <td>5.010</td>
      <td>3.910000</td>
      <td>116.306049</td>
      <td>1.449196</td>
      <td>16.863760</td>
      <td>90910000</td>
      <td>0.421417</td>
      <td>1</td>
    </tr>
    <tr>
      <th>102</th>
      <td>1970-01-01 00:00:00.000002014</td>
      <td>El Salvador</td>
      <td>21.537939</td>
      <td>3772.401570</td>
      <td>32.310000</td>
      <td>2.190000</td>
      <td>2.970000</td>
      <td>6.790</td>
      <td>5.496944</td>
      <td>69.570771</td>
      <td>0.296163</td>
      <td>16.567886</td>
      <td>45370000</td>
      <td>0.791026</td>
      <td>1</td>
    </tr>
    <tr>
      <th>103</th>
      <td>1970-01-01 00:00:00.000002014</td>
      <td>Guatemala</td>
      <td>5.716933</td>
      <td>2990.594485</td>
      <td>38.360000</td>
      <td>1.640000</td>
      <td>9.320000</td>
      <td>5.370</td>
      <td>2.910000</td>
      <td>56.717915</td>
      <td>2.048252</td>
      <td>9.941701</td>
      <td>126040000</td>
      <td>-0.198735</td>
      <td>0</td>
    </tr>
    <tr>
      <th>104</th>
      <td>1970-01-01 00:00:00.000002014</td>
      <td>Honduras</td>
      <td>7.389157</td>
      <td>2279.309902</td>
      <td>38.360000</td>
      <td>1.150000</td>
      <td>15.960000</td>
      <td>5.011</td>
      <td>5.496944</td>
      <td>112.609235</td>
      <td>1.424638</td>
      <td>17.385695</td>
      <td>80450000</td>
      <td>0.898223</td>
      <td>1</td>
    </tr>
    <tr>
      <th>105</th>
      <td>1970-01-01 00:00:00.000002015</td>
      <td>El Salvador</td>
      <td>22.073593</td>
      <td>3853.107631</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>6.833</td>
      <td>5.496944</td>
      <td>67.989029</td>
      <td>0.308592</td>
      <td>16.577147</td>
      <td>47470000</td>
      <td>0.346951</td>
      <td>1</td>
    </tr>
    <tr>
      <th>106</th>
      <td>1970-01-01 00:00:00.000002015</td>
      <td>Guatemala</td>
      <td>5.675817</td>
      <td>3052.270569</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.339</td>
      <td>2.420000</td>
      <td>51.333403</td>
      <td>2.023674</td>
      <td>10.303115</td>
      <td>123500000</td>
      <td>0.049519</td>
      <td>0</td>
    </tr>
    <tr>
      <th>107</th>
      <td>1970-01-01 00:00:00.000002015</td>
      <td>Honduras</td>
      <td>7.418273</td>
      <td>2329.002149</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.015</td>
      <td>7.380000</td>
      <td>107.434916</td>
      <td>1.414027</td>
      <td>17.953123</td>
      <td>110380000</td>
      <td>0.998200</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 15 columns</p>
</div>




```python
migration_flows.threshold.value_counts()
```




    0    72
    1    36
    Name: threshold, dtype: int64



### Subset of the data without the threshold


```python
migration_cont = migration_flows.iloc[:, 3:-1]
```


```python
migration_cont
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GDP_percapita_constant</th>
      <th>income_highest%</th>
      <th>income_lowest%</th>
      <th>poverty_headcount_1.90</th>
      <th>death_rate</th>
      <th>unemployment</th>
      <th>trade</th>
      <th>pop_growth</th>
      <th>remittances</th>
      <th>net_bilateral_aid</th>
      <th>FDI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2572.813235</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>11.681</td>
      <td>13.340000</td>
      <td>67.406464</td>
      <td>1.739184</td>
      <td>1.372147</td>
      <td>43000000</td>
      <td>0.014116</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2560.782037</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>11.568</td>
      <td>5.496944</td>
      <td>47.105487</td>
      <td>2.635143</td>
      <td>0.332542</td>
      <td>17000000</td>
      <td>0.025385</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1655.946421</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.233</td>
      <td>5.496944</td>
      <td>81.293839</td>
      <td>3.145300</td>
      <td>0.062354</td>
      <td>19000000</td>
      <td>0.038971</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2267.095959</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>11.494</td>
      <td>5.496944</td>
      <td>60.266492</td>
      <td>1.611673</td>
      <td>2.108693</td>
      <td>97000000</td>
      <td>0.014116</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2509.736778</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>11.300</td>
      <td>2.150000</td>
      <td>40.691257</td>
      <td>2.658257</td>
      <td>0.284635</td>
      <td>18000000</td>
      <td>-0.011618</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1645.846419</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>9.793</td>
      <td>5.496944</td>
      <td>69.338535</td>
      <td>3.113439</td>
      <td>0.062068</td>
      <td>35000000</td>
      <td>0.070935</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2092.554425</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>11.251</td>
      <td>5.496944</td>
      <td>51.247740</td>
      <td>1.498034</td>
      <td>3.300787</td>
      <td>170000000</td>
      <td>0.014116</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2357.368296</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>11.016</td>
      <td>2.270000</td>
      <td>33.474818</td>
      <td>2.669298</td>
      <td>0.122749</td>
      <td>20000000</td>
      <td>-0.045887</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1573.671559</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>9.359</td>
      <td>7.300000</td>
      <td>54.727051</td>
      <td>3.083349</td>
      <td>0.051662</td>
      <td>68000000</td>
      <td>-0.034441</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2094.864582</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.953</td>
      <td>5.496944</td>
      <td>54.397801</td>
      <td>1.413014</td>
      <td>3.292315</td>
      <td>231000000</td>
      <td>0.014116</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2236.567544</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.718</td>
      <td>5.496944</td>
      <td>27.546959</td>
      <td>2.654604</td>
      <td>0.043094</td>
      <td>36000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1512.185833</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>8.933</td>
      <td>5.496944</td>
      <td>55.394868</td>
      <td>3.056878</td>
      <td>0.058499</td>
      <td>64000000</td>
      <td>0.064998</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2094.098791</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.602</td>
      <td>5.496944</td>
      <td>50.290675</td>
      <td>1.364660</td>
      <td>4.345542</td>
      <td>221000000</td>
      <td>0.014116</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2189.829730</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.406</td>
      <td>5.496944</td>
      <td>28.153115</td>
      <td>2.607323</td>
      <td>0.035903</td>
      <td>29000000</td>
      <td>0.052798</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1530.695403</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>8.520</td>
      <td>5.496944</td>
      <td>57.728231</td>
      <td>3.037319</td>
      <td>0.058753</td>
      <td>123000000</td>
      <td>-0.030130</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2078.900486</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.205</td>
      <td>16.950001</td>
      <td>52.210538</td>
      <td>1.343117</td>
      <td>4.135388</td>
      <td>287000000</td>
      <td>0.014116</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2121.873660</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>10.085</td>
      <td>5.496944</td>
      <td>24.932246</td>
      <td>2.541226</td>
      <td>0.010286</td>
      <td>50000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1547.357836</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>8.129</td>
      <td>5.496944</td>
      <td>54.966344</td>
      <td>3.020231</td>
      <td>0.057700</td>
      <td>161000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2055.438830</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>9.775</td>
      <td>7.900000</td>
      <td>53.714123</td>
      <td>1.324193</td>
      <td>4.170338</td>
      <td>272000000</td>
      <td>0.014116</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2073.066614</td>
      <td>46.730000</td>
      <td>1.000000</td>
      <td>50.940000</td>
      <td>9.762</td>
      <td>5.496944</td>
      <td>30.644019</td>
      <td>2.470001</td>
      <td>0.009679</td>
      <td>86000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1512.507552</td>
      <td>43.260000</td>
      <td>1.230000</td>
      <td>25.280000</td>
      <td>7.769</td>
      <td>12.120000</td>
      <td>54.890376</td>
      <td>2.998073</td>
      <td>0.055140</td>
      <td>175000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2079.844180</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>9.330</td>
      <td>5.496944</td>
      <td>45.094622</td>
      <td>1.302156</td>
      <td>4.715458</td>
      <td>356000000</td>
      <td>0.014116</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2095.342199</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>9.444</td>
      <td>3.500000</td>
      <td>38.142963</td>
      <td>2.413838</td>
      <td>0.001412</td>
      <td>155000000</td>
      <td>0.014116</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1556.855276</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>7.447</td>
      <td>11.400000</td>
      <td>48.789886</td>
      <td>2.966596</td>
      <td>0.811559</td>
      <td>153000000</td>
      <td>0.024082</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2091.693100</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>8.893</td>
      <td>9.370000</td>
      <td>38.095704</td>
      <td>1.291964</td>
      <td>5.029738</td>
      <td>318000000</td>
      <td>0.014116</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2125.624163</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>9.137</td>
      <td>5.496944</td>
      <td>38.039914</td>
      <td>2.383684</td>
      <td>0.582789</td>
      <td>134000000</td>
      <td>-0.012752</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1581.639092</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>7.164</td>
      <td>5.496944</td>
      <td>55.215652</td>
      <td>2.927378</td>
      <td>1.052794</td>
      <td>155000000</td>
      <td>-0.025186</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2084.671422</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.980000</td>
      <td>8.480</td>
      <td>8.350000</td>
      <td>36.928296</td>
      <td>1.293850</td>
      <td>5.439348</td>
      <td>310000000</td>
      <td>0.014116</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2157.313890</td>
      <td>46.780000</td>
      <td>0.680000</td>
      <td>38.020000</td>
      <td>8.845</td>
      <td>2.000000</td>
      <td>39.781546</td>
      <td>2.387014</td>
      <td>1.010615</td>
      <td>146000000</td>
      <td>-0.047558</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1603.219717</td>
      <td>48.180000</td>
      <td>1.040000</td>
      <td>38.600000</td>
      <td>6.920</td>
      <td>5.496944</td>
      <td>65.347396</td>
      <td>2.879679</td>
      <td>1.363847</td>
      <td>102000000</td>
      <td>-0.028063</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>3475.866745</td>
      <td>35.480000</td>
      <td>1.790000</td>
      <td>6.360000</td>
      <td>6.672</td>
      <td>6.570000</td>
      <td>71.849041</td>
      <td>0.341593</td>
      <td>18.773955</td>
      <td>24540000</td>
      <td>-0.141774</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2698.985240</td>
      <td>43.560000</td>
      <td>1.070000</td>
      <td>11.510000</td>
      <td>5.663</td>
      <td>1.820000</td>
      <td>66.818187</td>
      <td>2.298528</td>
      <td>12.239366</td>
      <td>67250000</td>
      <td>0.276205</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2017.943010</td>
      <td>44.050000</td>
      <td>0.580000</td>
      <td>23.790000</td>
      <td>5.105</td>
      <td>3.110000</td>
      <td>133.131835</td>
      <td>1.826883</td>
      <td>21.557383</td>
      <td>84100000</td>
      <td>0.452738</td>
    </tr>
    <tr>
      <th>81</th>
      <td>3597.961991</td>
      <td>35.720000</td>
      <td>1.930000</td>
      <td>4.490000</td>
      <td>6.662</td>
      <td>6.410000</td>
      <td>74.177439</td>
      <td>0.315511</td>
      <td>18.448488</td>
      <td>39040000</td>
      <td>0.473516</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2805.169791</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.616</td>
      <td>5.496944</td>
      <td>67.898497</td>
      <td>2.254506</td>
      <td>12.418103</td>
      <td>45710000</td>
      <td>0.408934</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2104.759589</td>
      <td>43.810000</td>
      <td>0.900000</td>
      <td>17.430000</td>
      <td>5.077</td>
      <td>2.920000</td>
      <td>135.070635</td>
      <td>1.792143</td>
      <td>21.291564</td>
      <td>71100000</td>
      <td>0.332742</td>
    </tr>
    <tr>
      <th>84</th>
      <td>3633.014903</td>
      <td>36.040000</td>
      <td>1.700000</td>
      <td>6.920000</td>
      <td>6.659</td>
      <td>5.880000</td>
      <td>76.580188</td>
      <td>0.296649</td>
      <td>17.520181</td>
      <td>42370000</td>
      <td>0.370631</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2833.735795</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.576</td>
      <td>5.496944</td>
      <td>64.125228</td>
      <td>2.215217</td>
      <td>11.395396</td>
      <td>70350000</td>
      <td>0.034880</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2155.827865</td>
      <td>43.870000</td>
      <td>0.910000</td>
      <td>16.140000</td>
      <td>5.055</td>
      <td>2.990000</td>
      <td>135.748955</td>
      <td>1.747160</td>
      <td>20.459753</td>
      <td>96330000</td>
      <td>1.402889</td>
    </tr>
    <tr>
      <th>87</th>
      <td>3509.156436</td>
      <td>36.070000</td>
      <td>1.780000</td>
      <td>6.390000</td>
      <td>6.663</td>
      <td>7.330000</td>
      <td>61.871642</td>
      <td>0.285542</td>
      <td>16.467451</td>
      <td>82080000</td>
      <td>0.014327</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2787.128287</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.539</td>
      <td>5.496944</td>
      <td>57.105993</td>
      <td>2.183077</td>
      <td>10.651859</td>
      <td>83890000</td>
      <td>0.325371</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2068.185180</td>
      <td>39.140000</td>
      <td>1.150000</td>
      <td>14.040000</td>
      <td>5.038</td>
      <td>3.280000</td>
      <td>96.905006</td>
      <td>1.688651</td>
      <td>17.101447</td>
      <td>128760000</td>
      <td>-0.074317</td>
    </tr>
    <tr>
      <th>90</th>
      <td>3547.070983</td>
      <td>33.700000</td>
      <td>1.670000</td>
      <td>7.240000</td>
      <td>6.673</td>
      <td>7.050000</td>
      <td>68.768763</td>
      <td>0.280903</td>
      <td>16.209675</td>
      <td>148160000</td>
      <td>0.524807</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2805.951416</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.503</td>
      <td>3.740000</td>
      <td>62.114932</td>
      <td>2.156000</td>
      <td>10.236867</td>
      <td>100500000</td>
      <td>0.153036</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2110.822021</td>
      <td>41.020000</td>
      <td>1.090000</td>
      <td>15.470000</td>
      <td>5.026</td>
      <td>4.100000</td>
      <td>109.441838</td>
      <td>1.622622</td>
      <td>16.643140</td>
      <td>100840000</td>
      <td>-2.309012</td>
    </tr>
    <tr>
      <th>93</th>
      <td>3615.583230</td>
      <td>32.860000</td>
      <td>2.110000</td>
      <td>4.530000</td>
      <td>6.692</td>
      <td>6.620000</td>
      <td>74.643243</td>
      <td>0.279522</td>
      <td>15.748084</td>
      <td>162440000</td>
      <td>-0.414578</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2861.167894</td>
      <td>41.830000</td>
      <td>1.340000</td>
      <td>11.530000</td>
      <td>5.467</td>
      <td>4.130000</td>
      <td>63.984196</td>
      <td>2.129043</td>
      <td>9.492785</td>
      <td>93080000</td>
      <td>0.274376</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2157.984444</td>
      <td>45.670000</td>
      <td>0.750000</td>
      <td>18.750000</td>
      <td>5.017</td>
      <td>4.270000</td>
      <td>122.216903</td>
      <td>1.554236</td>
      <td>15.980083</td>
      <td>46360000</td>
      <td>0.172124</td>
    </tr>
    <tr>
      <th>96</th>
      <td>3673.262887</td>
      <td>32.470000</td>
      <td>2.150000</td>
      <td>4.160000</td>
      <td>6.718</td>
      <td>6.070000</td>
      <td>69.698828</td>
      <td>0.280768</td>
      <td>16.361001</td>
      <td>150850000</td>
      <td>-0.150729</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2884.897429</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.433</td>
      <td>2.870000</td>
      <td>60.982475</td>
      <td>2.100666</td>
      <td>9.983742</td>
      <td>95490000</td>
      <td>0.115398</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2213.759527</td>
      <td>45.680000</td>
      <td>0.790000</td>
      <td>21.360000</td>
      <td>5.012</td>
      <td>5.496944</td>
      <td>121.188216</td>
      <td>1.493978</td>
      <td>15.871110</td>
      <td>52650000</td>
      <td>1.177095</td>
    </tr>
    <tr>
      <th>99</th>
      <td>3730.422292</td>
      <td>34.350000</td>
      <td>2.110000</td>
      <td>3.250000</td>
      <td>6.751</td>
      <td>5.930000</td>
      <td>71.948881</td>
      <td>0.286321</td>
      <td>16.241404</td>
      <td>51090000</td>
      <td>0.271335</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2930.170750</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.401</td>
      <td>2.990000</td>
      <td>58.548341</td>
      <td>2.073729</td>
      <td>9.988782</td>
      <td>102670000</td>
      <td>0.169948</td>
    </tr>
    <tr>
      <th>101</th>
      <td>2242.818455</td>
      <td>41.480000</td>
      <td>0.980000</td>
      <td>18.930000</td>
      <td>5.010</td>
      <td>3.910000</td>
      <td>116.306049</td>
      <td>1.449196</td>
      <td>16.863760</td>
      <td>90910000</td>
      <td>0.421417</td>
    </tr>
    <tr>
      <th>102</th>
      <td>3772.401570</td>
      <td>32.310000</td>
      <td>2.190000</td>
      <td>2.970000</td>
      <td>6.790</td>
      <td>5.496944</td>
      <td>69.570771</td>
      <td>0.296163</td>
      <td>16.567886</td>
      <td>45370000</td>
      <td>0.791026</td>
    </tr>
    <tr>
      <th>103</th>
      <td>2990.594485</td>
      <td>38.360000</td>
      <td>1.640000</td>
      <td>9.320000</td>
      <td>5.370</td>
      <td>2.910000</td>
      <td>56.717915</td>
      <td>2.048252</td>
      <td>9.941701</td>
      <td>126040000</td>
      <td>-0.198735</td>
    </tr>
    <tr>
      <th>104</th>
      <td>2279.309902</td>
      <td>38.360000</td>
      <td>1.150000</td>
      <td>15.960000</td>
      <td>5.011</td>
      <td>5.496944</td>
      <td>112.609235</td>
      <td>1.424638</td>
      <td>17.385695</td>
      <td>80450000</td>
      <td>0.898223</td>
    </tr>
    <tr>
      <th>105</th>
      <td>3853.107631</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>6.833</td>
      <td>5.496944</td>
      <td>67.989029</td>
      <td>0.308592</td>
      <td>16.577147</td>
      <td>47470000</td>
      <td>0.346951</td>
    </tr>
    <tr>
      <th>106</th>
      <td>3052.270569</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.339</td>
      <td>2.420000</td>
      <td>51.333403</td>
      <td>2.023674</td>
      <td>10.303115</td>
      <td>123500000</td>
      <td>0.049519</td>
    </tr>
    <tr>
      <th>107</th>
      <td>2329.002149</td>
      <td>40.809231</td>
      <td>1.104423</td>
      <td>18.977037</td>
      <td>5.015</td>
      <td>7.380000</td>
      <td>107.434916</td>
      <td>1.414027</td>
      <td>17.953123</td>
      <td>110380000</td>
      <td>0.998200</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 11 columns</p>
</div>



### Standardize the variables


```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
migration_cont_n = ss.fit_transform(migration_cont)
migration_cont_n
```




    array([[  3.09990924e-01,   2.49079191e-15,  -6.78250390e-16, ...,
             -1.03044645e+00,  -7.97539512e-01,   1.39632884e-17],
           [  2.90287283e-01,   2.49079191e-15,  -6.78250390e-16, ...,
             -1.19160228e+00,  -1.16667096e+00,   9.07069956e-03],
           [ -1.19157316e+00,   2.49079191e-15,  -6.78250390e-16, ...,
             -1.23348589e+00,  -1.13827623e+00,   2.00066728e-02],
           ..., 
           [  2.40674484e+00,   2.49079191e-15,  -6.78250390e-16, ...,
              1.32657722e+00,  -7.34077298e-01,   2.67908201e-01],
           [  1.09520411e+00,   2.49079191e-15,  -6.78250390e-16, ...,
              3.53999624e-01,   3.45348233e-01,   2.84966462e-02],
           [ -8.93014977e-02,   2.49079191e-15,  -6.78250390e-16, ...,
              1.53987595e+00,   1.59078827e-01,   7.92117484e-01]])



### Fit a PCA on the standardized data


```python
## Fit the PCA and print the components
from sklearn.decomposition import PCA
migration_pca = PCA().fit(migration_cont_n)
print "Number of PCA components is: \n", migration_pca.n_components_
print "\n======\n"
print "List of PCA components is:\n", migration_pca.components_
```

    Number of PCA components is: 
    11
    
    ======
    
    List of PCA components is:
    [[  4.44953785e-01  -4.33783948e-01   3.34104401e-01  -4.24088228e-01
       -1.06912171e-01   9.48163627e-02   1.37408511e-02  -4.19167628e-01
        3.51743673e-01  -3.09566472e-02   3.71274547e-02]
     [ -1.21666677e-02   2.08096986e-01  -2.33634354e-01   7.65655577e-02
       -5.38365104e-01  -3.05100519e-01   5.50558972e-01  -5.63939732e-02
        3.96597653e-01  -2.22691862e-01  -4.28025433e-04]
     [ -1.48957745e-01   1.06678958e-01  -2.21507866e-01   1.50903978e-01
       -6.02258276e-03   5.57226489e-01   1.98828498e-01  -3.10661936e-01
        2.04701578e-01   5.77502700e-01   2.69444055e-01]
     [ -3.77536854e-02   1.75339147e-02   2.85664035e-02  -2.71247993e-02
       -6.39828471e-02  -3.73895302e-02   4.62671591e-02  -7.30312143e-02
        4.20089511e-02   3.94499973e-01  -9.09040404e-01]
     [  3.48318407e-01   1.49689192e-01  -5.76707332e-01   1.23864812e-01
        2.74527548e-01   2.22756631e-01  -1.79594534e-01  -3.23647293e-01
        7.19247322e-02  -4.33903578e-01  -2.30000987e-01]
     [ -3.00602698e-01  -1.04249633e-01   1.84363842e-01  -1.63764722e-01
        1.71659942e-02   6.24551610e-01   3.94588512e-01   1.95176281e-01
       -8.54131778e-02  -4.53507556e-01  -2.02097371e-01]
     [  2.80858239e-01   1.43466519e-01   3.58370256e-01   5.50577572e-01
       -5.02721107e-01   3.09970041e-01  -3.22754836e-01   6.82383505e-02
        1.50828276e-04  -9.21459990e-02  -5.33207368e-02]
     [  1.64146336e-02   8.28126884e-02   4.49229254e-01   4.96082911e-01
        5.24508557e-01  -1.89871602e-01   3.63412439e-01  -2.90460448e-01
        9.75618391e-02  -8.63290902e-02  -2.00023246e-02]
     [  2.08960689e-01   7.59519081e-01   2.40702424e-01  -3.90100905e-01
        1.83567171e-01   9.13798292e-02  -4.82576106e-02   2.20294840e-01
        2.70961964e-01   5.01216013e-02   2.26156889e-02]
     [ -6.17462634e-02   3.36679898e-01   1.24097352e-01  -2.14175080e-01
       -2.36145089e-01  -4.73309909e-02   7.86232942e-02  -5.80332938e-01
       -6.45080067e-01  -5.76723427e-02   3.37026563e-02]
     [  6.63278746e-01  -7.68197951e-02  -1.17269424e-01   3.65260156e-02
        4.05405163e-02   6.01316506e-02   4.71506725e-01   3.19519314e-01
       -4.04701878e-01   2.11156656e-01   3.21322350e-02]]



```python
## transform  => Apply dimensionality reduction to X.
migration_pcs = migration_pca.transform(migration_cont_n)
migration_pcs
```




    array([[-0.09475011, -2.55902429,  1.13077881, ...,  0.46564207,
             0.03408878,  0.69305284],
           [-0.91575677, -1.98239954, -1.4239706 , ...,  0.36246894,
            -0.35480985,  0.45654761],
           [-1.75284795, -0.93167753, -1.1262195 , ..., -0.01598467,
            -0.32290655,  0.26322268],
           ..., 
           [ 2.36805819,  0.86440093,  0.13611338, ...,  0.38771311,
             0.16832743,  0.30748464],
           [ 0.49046578,  0.62675761, -0.78621404, ...,  0.11704958,
            -0.18836398,  0.30903495],
           [ 0.96671037,  1.8038798 ,  1.53175003, ...,  0.09377444,
            -0.31034041, -0.10472607]])




```python
## Now create the dataframe
migration_pcs = pd.DataFrame(migration_pcs, columns=['PC'+str(i) for i in range(1, migration_pcs.shape[1]+1)])
migration_pcs['threshold'] = migration_flows.threshold
```


```python
migration_pcs
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PC1</th>
      <th>PC2</th>
      <th>PC3</th>
      <th>PC4</th>
      <th>PC5</th>
      <th>PC6</th>
      <th>PC7</th>
      <th>PC8</th>
      <th>PC9</th>
      <th>PC10</th>
      <th>PC11</th>
      <th>threshold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.094750</td>
      <td>-2.559024</td>
      <td>1.130779</td>
      <td>-0.640239</td>
      <td>1.828362</td>
      <td>2.409852</td>
      <td>-0.045526</td>
      <td>0.654374</td>
      <td>0.465642</td>
      <td>0.034089</td>
      <td>0.693053</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.915757</td>
      <td>-1.982400</td>
      <td>-1.423971</td>
      <td>-0.787201</td>
      <td>1.006300</td>
      <td>0.456770</td>
      <td>-0.686899</td>
      <td>0.673341</td>
      <td>0.362469</td>
      <td>-0.354810</td>
      <td>0.456548</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.752848</td>
      <td>-0.931678</td>
      <td>-1.126220</td>
      <td>-0.671409</td>
      <td>-0.153935</td>
      <td>1.499758</td>
      <td>-1.110063</td>
      <td>0.545378</td>
      <td>-0.015985</td>
      <td>-0.322907</td>
      <td>0.263223</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.540747</td>
      <td>-1.759718</td>
      <td>-0.162042</td>
      <td>-0.185934</td>
      <td>0.667831</td>
      <td>0.016410</td>
      <td>-1.147953</td>
      <td>1.109195</td>
      <td>0.090771</td>
      <td>0.194729</td>
      <td>0.100909</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.089305</td>
      <td>-1.614602</td>
      <td>-2.248583</td>
      <td>-0.703059</td>
      <td>0.659168</td>
      <td>-0.483891</td>
      <td>-0.992225</td>
      <td>0.765459</td>
      <td>0.205943</td>
      <td>-0.280907</td>
      <td>0.212460</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.730771</td>
      <td>-1.095974</td>
      <td>-1.060670</td>
      <td>-0.607031</td>
      <td>-0.237551</td>
      <td>1.209526</td>
      <td>-0.876000</td>
      <td>0.249212</td>
      <td>-0.038364</td>
      <td>-0.290328</td>
      <td>0.069523</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.568150</td>
      <td>-2.019582</td>
      <td>0.493532</td>
      <td>0.244410</td>
      <td>0.200131</td>
      <td>-0.544758</td>
      <td>-1.158457</td>
      <td>0.881710</td>
      <td>0.094860</td>
      <td>0.117630</td>
      <td>-0.151819</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-1.199071</td>
      <td>-1.708567</td>
      <td>-2.236150</td>
      <td>-0.663807</td>
      <td>0.577186</td>
      <td>-0.488600</td>
      <td>-0.882825</td>
      <td>0.565188</td>
      <td>0.139945</td>
      <td>-0.246635</td>
      <td>-0.063284</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-1.697105</td>
      <td>-1.599834</td>
      <td>-0.470535</td>
      <td>-0.376321</td>
      <td>-0.249953</td>
      <td>1.295664</td>
      <td>-0.422549</td>
      <td>-0.245478</td>
      <td>0.001974</td>
      <td>-0.313593</td>
      <td>-0.143937</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.531903</td>
      <td>-2.055599</td>
      <td>1.048910</td>
      <td>0.609119</td>
      <td>-0.206761</td>
      <td>-0.915087</td>
      <td>-1.200603</td>
      <td>0.794274</td>
      <td>0.080778</td>
      <td>0.175153</td>
      <td>0.050038</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-1.147356</td>
      <td>-2.206898</td>
      <td>-1.352877</td>
      <td>-0.649862</td>
      <td>0.702034</td>
      <td>0.211710</td>
      <td>-0.391360</td>
      <td>0.124095</td>
      <td>0.207912</td>
      <td>-0.271063</td>
      <td>-0.175285</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-1.769911</td>
      <td>-1.215291</td>
      <td>-0.870798</td>
      <td>-0.421142</td>
      <td>-0.504485</td>
      <td>0.863736</td>
      <td>-0.577620</td>
      <td>-0.203546</td>
      <td>-0.139473</td>
      <td>-0.191269</td>
      <td>-0.273811</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.427941</td>
      <td>-1.937353</td>
      <td>0.989404</td>
      <td>0.569404</td>
      <td>-0.139926</td>
      <td>-0.939164</td>
      <td>-1.046679</td>
      <td>0.683846</td>
      <td>0.077178</td>
      <td>0.144913</td>
      <td>-0.145186</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-1.134854</td>
      <td>-2.077429</td>
      <td>-1.364454</td>
      <td>-0.708813</td>
      <td>0.676491</td>
      <td>0.266166</td>
      <td>-0.331920</td>
      <td>0.066289</td>
      <td>0.142851</td>
      <td>-0.183613</td>
      <td>-0.259661</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-1.750131</td>
      <td>-1.232270</td>
      <td>-0.386277</td>
      <td>-0.002139</td>
      <td>-0.909319</td>
      <td>0.516002</td>
      <td>-0.559029</td>
      <td>-0.353133</td>
      <td>-0.143436</td>
      <td>-0.170585</td>
      <td>-0.055022</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.009909</td>
      <td>-3.464194</td>
      <td>4.222059</td>
      <td>0.778410</td>
      <td>0.446583</td>
      <td>1.659190</td>
      <td>0.429358</td>
      <td>-0.390377</td>
      <td>0.499373</td>
      <td>-0.041849</td>
      <td>0.354199</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-1.146214</td>
      <td>-2.111590</td>
      <td>-1.185993</td>
      <td>-0.537097</td>
      <td>0.517247</td>
      <td>0.107555</td>
      <td>-0.267502</td>
      <td>-0.072676</td>
      <td>0.088735</td>
      <td>-0.114849</td>
      <td>-0.359251</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>-1.723924</td>
      <td>-1.293844</td>
      <td>-0.085023</td>
      <td>0.197953</td>
      <td>-1.172703</td>
      <td>0.210061</td>
      <td>-0.463852</td>
      <td>-0.542510</td>
      <td>-0.148757</td>
      <td>-0.148434</td>
      <td>0.014237</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>-0.322516</td>
      <td>-2.100750</td>
      <td>2.014988</td>
      <td>0.856887</td>
      <td>-0.384499</td>
      <td>-0.585060</td>
      <td>-0.638002</td>
      <td>0.251468</td>
      <td>0.085435</td>
      <td>0.221362</td>
      <td>0.063409</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>-3.977446</td>
      <td>-1.171330</td>
      <td>0.134423</td>
      <td>-0.394938</td>
      <td>1.238367</td>
      <td>-1.018212</td>
      <td>2.236757</td>
      <td>2.061839</td>
      <td>-0.147960</td>
      <td>-0.296691</td>
      <td>-0.203067</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-2.066231</td>
      <td>-1.923692</td>
      <td>1.726760</td>
      <td>0.192129</td>
      <td>-0.693138</td>
      <td>1.700350</td>
      <td>1.187756</td>
      <td>-0.517069</td>
      <td>0.473613</td>
      <td>-0.071320</td>
      <td>0.087511</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-0.374719</td>
      <td>-2.071261</td>
      <td>2.100620</td>
      <td>1.369580</td>
      <td>-1.106210</td>
      <td>-1.909042</td>
      <td>-0.825242</td>
      <td>0.112817</td>
      <td>0.049800</td>
      <td>0.190495</td>
      <td>0.077678</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>-1.182717</td>
      <td>-1.722051</td>
      <td>-0.634126</td>
      <td>0.129705</td>
      <td>-0.467755</td>
      <td>-0.921688</td>
      <td>-0.670680</td>
      <td>-0.003336</td>
      <td>-0.043533</td>
      <td>0.052566</td>
      <td>0.045835</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-1.373889</td>
      <td>-1.899490</td>
      <td>1.229482</td>
      <td>0.065273</td>
      <td>-0.603294</td>
      <td>1.676373</td>
      <td>0.571713</td>
      <td>-1.249533</td>
      <td>0.035502</td>
      <td>-0.227589</td>
      <td>-0.041483</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>-0.151607</td>
      <td>-2.441351</td>
      <td>2.652865</td>
      <td>1.101502</td>
      <td>-0.515248</td>
      <td>-0.771171</td>
      <td>-0.065139</td>
      <td>-0.358824</td>
      <td>0.154412</td>
      <td>0.154986</td>
      <td>-0.081862</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>-1.008282</td>
      <td>-1.785772</td>
      <td>-0.323846</td>
      <td>0.015558</td>
      <td>-0.156833</td>
      <td>-0.294206</td>
      <td>-0.286456</td>
      <td>-0.204561</td>
      <td>0.013758</td>
      <td>0.028432</td>
      <td>0.008690</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-1.539755</td>
      <td>-0.920739</td>
      <td>-0.076727</td>
      <td>0.229274</td>
      <td>-1.209815</td>
      <td>0.195572</td>
      <td>-0.186215</td>
      <td>-0.757890</td>
      <td>-0.221267</td>
      <td>-0.057763</td>
      <td>-0.082202</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-0.149081</td>
      <td>-2.163703</td>
      <td>2.356188</td>
      <td>1.087927</td>
      <td>-0.614977</td>
      <td>-1.008840</td>
      <td>-0.063041</td>
      <td>-0.395872</td>
      <td>0.085767</td>
      <td>0.189655</td>
      <td>-0.193659</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-3.522453</td>
      <td>-0.295340</td>
      <td>-0.131563</td>
      <td>0.106544</td>
      <td>0.795555</td>
      <td>-2.155864</td>
      <td>0.570972</td>
      <td>0.872353</td>
      <td>0.159828</td>
      <td>0.074185</td>
      <td>0.073905</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>-3.750821</td>
      <td>0.334077</td>
      <td>0.305371</td>
      <td>-0.066868</td>
      <td>-0.123965</td>
      <td>-0.081913</td>
      <td>1.593498</td>
      <td>0.836644</td>
      <td>0.587575</td>
      <td>0.324458</td>
      <td>-0.161960</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>4.478286</td>
      <td>0.058006</td>
      <td>-0.649471</td>
      <td>-0.088195</td>
      <td>0.118626</td>
      <td>0.659358</td>
      <td>0.266967</td>
      <td>0.567403</td>
      <td>0.102336</td>
      <td>-0.024233</td>
      <td>-0.381972</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.170862</td>
      <td>1.358790</td>
      <td>-1.205971</td>
      <td>-0.252739</td>
      <td>-0.281869</td>
      <td>-0.894103</td>
      <td>-0.282680</td>
      <td>-0.682349</td>
      <td>1.196525</td>
      <td>0.057017</td>
      <td>-0.088277</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>-0.741774</td>
      <td>3.792263</td>
      <td>1.001756</td>
      <td>-0.114708</td>
      <td>0.169414</td>
      <td>-0.190819</td>
      <td>-0.738841</td>
      <td>0.380811</td>
      <td>0.242901</td>
      <td>-0.728481</td>
      <td>-0.133418</td>
      <td>0</td>
    </tr>
    <tr>
      <th>81</th>
      <td>4.783592</td>
      <td>-0.039442</td>
      <td>-0.571420</td>
      <td>-0.437174</td>
      <td>-0.305688</td>
      <td>0.507650</td>
      <td>0.255915</td>
      <td>0.662332</td>
      <td>0.399101</td>
      <td>0.165755</td>
      <td>-0.208843</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.468604</td>
      <td>0.857043</td>
      <td>-0.465914</td>
      <td>-0.567331</td>
      <td>0.154463</td>
      <td>0.089279</td>
      <td>0.716434</td>
      <td>-0.460483</td>
      <td>0.249326</td>
      <td>-0.510813</td>
      <td>0.146788</td>
      <td>0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0.049049</td>
      <td>3.578193</td>
      <td>0.467543</td>
      <td>-0.046734</td>
      <td>-0.385710</td>
      <td>0.173173</td>
      <td>-0.854540</td>
      <td>0.448518</td>
      <td>0.733687</td>
      <td>-0.389061</td>
      <td>-0.188824</td>
      <td>1</td>
    </tr>
    <tr>
      <th>84</th>
      <td>4.321318</td>
      <td>0.223923</td>
      <td>-0.486102</td>
      <td>-0.364267</td>
      <td>0.106546</td>
      <td>0.195278</td>
      <td>0.118753</td>
      <td>0.583612</td>
      <td>0.130796</td>
      <td>0.160910</td>
      <td>0.002010</td>
      <td>1</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.431876</td>
      <td>0.652951</td>
      <td>-0.397503</td>
      <td>-0.165698</td>
      <td>0.111361</td>
      <td>-0.074007</td>
      <td>0.766345</td>
      <td>-0.547999</td>
      <td>0.219227</td>
      <td>-0.420196</td>
      <td>0.224023</td>
      <td>0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>0.163871</td>
      <td>3.444238</td>
      <td>0.875077</td>
      <td>-0.686279</td>
      <td>-0.739195</td>
      <td>-0.128048</td>
      <td>-0.988705</td>
      <td>0.331269</td>
      <td>0.832277</td>
      <td>-0.219633</td>
      <td>0.006447</td>
      <td>1</td>
    </tr>
    <tr>
      <th>87</th>
      <td>4.308394</td>
      <td>-0.510793</td>
      <td>-0.071237</td>
      <td>0.081610</td>
      <td>-0.064405</td>
      <td>0.288794</td>
      <td>0.437382</td>
      <td>0.287628</td>
      <td>0.239987</td>
      <td>0.221250</td>
      <td>-0.214290</td>
      <td>1</td>
    </tr>
    <tr>
      <th>88</th>
      <td>0.374888</td>
      <td>0.434577</td>
      <td>-0.275455</td>
      <td>-0.312340</td>
      <td>-0.006826</td>
      <td>-0.286532</td>
      <td>0.806405</td>
      <td>-0.675913</td>
      <td>0.187347</td>
      <td>-0.337647</td>
      <td>0.131908</td>
      <td>0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>0.965129</td>
      <td>1.776229</td>
      <td>0.155755</td>
      <td>0.493735</td>
      <td>-1.152353</td>
      <td>-0.162479</td>
      <td>-0.635827</td>
      <td>-0.218792</td>
      <td>-0.251284</td>
      <td>-0.424330</td>
      <td>-0.484578</td>
      <td>1</td>
    </tr>
    <tr>
      <th>90</th>
      <td>4.501801</td>
      <td>-0.647759</td>
      <td>0.554353</td>
      <td>0.063195</td>
      <td>-0.532458</td>
      <td>-0.202094</td>
      <td>0.047849</td>
      <td>0.152657</td>
      <td>-0.481531</td>
      <td>-0.115392</td>
      <td>0.274090</td>
      <td>1</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.302127</td>
      <td>0.695477</td>
      <td>-0.556860</td>
      <td>-0.057335</td>
      <td>-0.263189</td>
      <td>-0.761664</td>
      <td>0.520295</td>
      <td>-0.492758</td>
      <td>0.098087</td>
      <td>-0.243340</td>
      <td>0.256112</td>
      <td>0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>0.560663</td>
      <td>2.190915</td>
      <td>-0.134079</td>
      <td>1.982935</td>
      <td>-0.302484</td>
      <td>0.614406</td>
      <td>-0.394815</td>
      <td>0.037583</td>
      <td>0.054757</td>
      <td>-0.194704</td>
      <td>-0.358342</td>
      <td>1</td>
    </tr>
    <tr>
      <th>93</th>
      <td>5.209716</td>
      <td>-0.955935</td>
      <td>-0.004407</td>
      <td>0.882745</td>
      <td>-1.353030</td>
      <td>0.143220</td>
      <td>0.207319</td>
      <td>0.662414</td>
      <td>-0.267244</td>
      <td>0.056912</td>
      <td>0.339676</td>
      <td>1</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0.852504</td>
      <td>0.502216</td>
      <td>-0.786472</td>
      <td>-0.141094</td>
      <td>-0.673724</td>
      <td>-0.368209</td>
      <td>0.341459</td>
      <td>-0.647959</td>
      <td>0.920779</td>
      <td>0.281212</td>
      <td>0.226478</td>
      <td>0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>-0.536637</td>
      <td>3.183446</td>
      <td>0.555293</td>
      <td>-0.132371</td>
      <td>0.446959</td>
      <td>0.328886</td>
      <td>-0.437220</td>
      <td>0.124292</td>
      <td>0.829259</td>
      <td>0.386729</td>
      <td>-0.150190</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>5.392972</td>
      <td>-0.982346</td>
      <td>-0.251387</td>
      <td>0.626731</td>
      <td>-1.402727</td>
      <td>-0.032393</td>
      <td>0.215352</td>
      <td>0.685838</td>
      <td>-0.289379</td>
      <td>-0.021494</td>
      <td>0.230838</td>
      <td>1</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.343826</td>
      <td>0.806332</td>
      <td>-0.823722</td>
      <td>-0.045419</td>
      <td>-0.245563</td>
      <td>-1.016474</td>
      <td>0.480068</td>
      <td>-0.434475</td>
      <td>0.057321</td>
      <td>-0.161355</td>
      <td>0.277174</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>-0.505971</td>
      <td>2.982403</td>
      <td>1.135349</td>
      <td>-0.858154</td>
      <td>0.370471</td>
      <td>0.354728</td>
      <td>-0.056654</td>
      <td>0.238542</td>
      <td>0.793398</td>
      <td>0.370396</td>
      <td>-0.050095</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99</th>
      <td>5.200854</td>
      <td>-0.465391</td>
      <td>-0.934898</td>
      <td>-0.229329</td>
      <td>-0.706470</td>
      <td>0.441633</td>
      <td>0.283417</td>
      <td>0.788753</td>
      <td>0.176302</td>
      <td>0.312095</td>
      <td>0.010279</td>
      <td>1</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.394541</td>
      <td>0.729059</td>
      <td>-0.743733</td>
      <td>-0.050470</td>
      <td>-0.240843</td>
      <td>-1.104907</td>
      <td>0.540461</td>
      <td>-0.484981</td>
      <td>0.077747</td>
      <td>-0.157398</td>
      <td>0.298249</td>
      <td>0</td>
    </tr>
    <tr>
      <th>101</th>
      <td>0.458918</td>
      <td>2.561368</td>
      <td>0.585001</td>
      <td>-0.066151</td>
      <td>-0.391602</td>
      <td>0.021122</td>
      <td>-0.395216</td>
      <td>0.270839</td>
      <td>-0.056034</td>
      <td>-0.088304</td>
      <td>-0.095397</td>
      <td>1</td>
    </tr>
    <tr>
      <th>102</th>
      <td>5.649544</td>
      <td>-0.642510</td>
      <td>-1.127469</td>
      <td>-0.646168</td>
      <td>-1.015027</td>
      <td>0.349322</td>
      <td>0.214588</td>
      <td>0.835208</td>
      <td>-0.265599</td>
      <td>0.081387</td>
      <td>0.008645</td>
      <td>1</td>
    </tr>
    <tr>
      <th>103</th>
      <td>1.897351</td>
      <td>-0.025992</td>
      <td>-1.321641</td>
      <td>0.414625</td>
      <td>-1.506061</td>
      <td>-0.673880</td>
      <td>0.316538</td>
      <td>-0.508213</td>
      <td>0.347161</td>
      <td>0.023346</td>
      <td>0.209247</td>
      <td>0</td>
    </tr>
    <tr>
      <th>104</th>
      <td>1.424259</td>
      <td>1.969073</td>
      <td>0.670027</td>
      <td>-0.495009</td>
      <td>-0.719318</td>
      <td>0.616269</td>
      <td>-0.328903</td>
      <td>0.061407</td>
      <td>-0.508717</td>
      <td>-0.365781</td>
      <td>-0.133354</td>
      <td>1</td>
    </tr>
    <tr>
      <th>105</th>
      <td>2.368058</td>
      <td>0.864401</td>
      <td>0.136113</td>
      <td>-0.420829</td>
      <td>1.738602</td>
      <td>-0.927647</td>
      <td>0.705547</td>
      <td>0.658094</td>
      <td>0.387713</td>
      <td>0.168327</td>
      <td>0.307485</td>
      <td>1</td>
    </tr>
    <tr>
      <th>106</th>
      <td>0.490466</td>
      <td>0.626758</td>
      <td>-0.786214</td>
      <td>0.151879</td>
      <td>-0.268408</td>
      <td>-1.550888</td>
      <td>0.600004</td>
      <td>-0.552887</td>
      <td>0.117050</td>
      <td>-0.188364</td>
      <td>0.309035</td>
      <td>0</td>
    </tr>
    <tr>
      <th>107</th>
      <td>0.966710</td>
      <td>1.803880</td>
      <td>1.531750</td>
      <td>-0.437454</td>
      <td>-0.413422</td>
      <td>0.607254</td>
      <td>0.252542</td>
      <td>0.029014</td>
      <td>0.093774</td>
      <td>-0.310340</td>
      <td>-0.104726</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 12 columns</p>
</div>



### Plot the variance explained by the ratio of the components


```python
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(range(1, migration_cont.shape[1]+1), migration_pca.explained_variance_ratio_, lw=2)
ax.scatter(range(1, migration_cont.shape[1]+1), migration_pca.explained_variance_ratio_, s=100)
ax.set_title('migration data: explained variance of components')
ax.set_xlabel('principal component')
ax.set_ylabel('explained variance')
plt.show()
```


    
![png](output_153_0.png)
    


### Print out the component weights with their corresponding variables for PC1, PC2, and PC3


```python
for col, comp in zip(migration_cont.columns, migration_pca.components_[0]):
    print col, comp
```

    GDP_percapita_constant 0.444953784686
    income_highest% -0.433783947612
    income_lowest% 0.334104401133
    poverty_headcount_1.90 -0.42408822848
    death_rate -0.106912170586
    unemployment 0.0948163627333
    trade 0.0137408511283
    pop_growth -0.419167628048
    remittances 0.351743673355
    net_bilateral_aid -0.0309566471764
    FDI 0.0371274546902



```python
for col, comp in zip(migration_cont.columns, migration_pca.components_[1]):
    print col, comp
```

    GDP_percapita_constant -0.012166667719
    income_highest% 0.208096986157
    income_lowest% -0.233634353696
    poverty_headcount_1.90 0.0765655576751
    death_rate -0.538365104029
    unemployment -0.305100519113
    trade 0.55055897166
    pop_growth -0.0563939732101
    remittances 0.396597653328
    net_bilateral_aid -0.222691861952
    FDI -0.0004280254334



```python
for col, comp in zip(migration_cont.columns, migration_pca.components_[3]):
    print col, comp
```

    GDP_percapita_constant -0.037753685352
    income_highest% 0.0175339147193
    income_lowest% 0.028566403528
    poverty_headcount_1.90 -0.0271247993477
    death_rate -0.0639828470952
    unemployment -0.0373895301668
    trade 0.0462671590709
    pop_growth -0.0730312142768
    remittances 0.0420089510669
    net_bilateral_aid 0.394499973419
    FDI -0.909040403928


### Plot a seaborn pairplot of PC1, PC2, and PC3 with `hue='threshold'`


```python
sns.pairplot(data=migration_pcs, vars=['PC1','PC2','PC3'], hue='threshold', size=3)
plt.show()
```


    
![png](output_159_0.png)
    


### Horn's parallel analysis


```python
def horn_parallel_analysis(shape, iters=1000, percentile=95):
    pca = PCA(n_components=shape[1])
    eigenvals = []
    for i in range(iters):
        rdata = np.random.normal(0,1,size=shape)
        pca.fit(rdata)
        eigenvals.append(pca.explained_variance_)
    eigenvals = np.array(eigenvals)
    return np.percentile(eigenvals, percentile, axis=0)
```

### Run parallel analysis for the migration data


```python
migration_pa = horn_parallel_analysis(migration_cont.shape, percentile=95)
migration_pa
```




    array([ 1.78259703,  1.5597615 ,  1.39057351,  1.27295842,  1.15745361,
            1.05150508,  0.95585737,  0.86531285,  0.78163868,  0.68858056,
            0.6034336 ])



### Plot the wine eigenvalues (`.variance_explained_`) against the parallel analysis random eigenvalue cutoffs


```python
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(range(1, migration_cont.shape[1]+1), migration_pca.explained_variance_, lw=2)
ax.scatter(range(1, migration_cont.shape[1]+1), migration_pca.explained_variance_, s=50)

ax.plot(range(1, len(migration_pa)+1), migration_pa, lw=2, color='darkred')
ax.scatter(range(1, len(migration_pa)+1), migration_pa, s=40, color='darkred')


ax.set_title('Horns parallel analysis on migration data components')
ax.set_xlabel('principal component')
ax.set_ylabel('eigenvalue')
plt.show()
```


    
![png](output_165_0.png)
    



### Predict "threshold" from original data and from PCA


```python
## Explore the noise on the original data
## should you Standarized the data? 
## http://stats.stackexchange.com/questions/48360/is-standardization-needed-before-fitting-logistic-regression
sns.pairplot(data=migration_flows, hue='threshold')
plt.show()
```


    
![png](output_167_0.png)
    



```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
```


```python
## Define your x and y
columns_ = migration_flows.columns.tolist()
exclude_cols = ['threshold', 'Country', 'Year']
y = migration_flows.threshold.values
X = migration_flows[[i for i in columns_ if i not in exclude_cols]]
X = X.values
```


```python
knn = KNeighborsClassifier()

params = {
    'n_neighbors':range(1,20),
    'weights':['uniform','distance']
}

knn_gs = GridSearchCV(knn, params, cv=5, verbose=1)
knn_gs.fit(X, y)

print knn_gs.best_params_
best_knn = knn_gs.best_estimator_
```

    Fitting 5 folds for each of 38 candidates, totalling 190 fits
    {'n_neighbors': 10, 'weights': 'uniform'}


    [Parallel(n_jobs=1)]: Done 190 out of 190 | elapsed:    0.8s finished



```python
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
```

    Xtrain and ytrain shapes:
    (85, 12) (85,)
    Xtest and ytest shapes:
    (23, 12) (23,)
    Xtrain and ytrain shapes:
    (86, 12) (86,)
    Xtest and ytest shapes:
    (22, 12) (22,)
    Xtrain and ytrain shapes:
    (87, 12) (87,)
    Xtest and ytest shapes:
    (21, 12) (21,)
    Xtrain and ytrain shapes:
    (87, 12) (87,)
    Xtest and ytest shapes:
    (21, 12) (21,)
    Xtrain and ytrain shapes:
    (87, 12) (87,)
    Xtest and ytest shapes:
    (21, 12) (21,)
    
    ======
    
    KNN accuracy scores on test:
    [0.60869565217391308, 0.68181818181818177, 0.61904761904761907, 0.52380952380952384, 0.61904761904761907]
    KNN mean of accuracy scores on test:
    0.610483719179
    KNN mean of accuracy scores on train :
    0.662051354624
    
    ======
    
    Logistic Regression accuracy scores on test:
    [0.65217391304347827, 0.68181818181818177, 0.66666666666666663, 0.61904761904761907, 0.66666666666666663]
    Logistic Regression mean of accuracy scores on test:
    0.657274609449
    Logistic Regression mean of accuracy scores on train:
    0.657399562872
    
    ======
    
    Baseline accuracy:
      0.333333333333


### We found very similar results with the KNN and logistic estimation


```python
## Define your x and y
## For your X = only use the number of PCA's that have the greatest explanatory power

columns_ = migration_pcs.columns.tolist()
exclude_cols = ['Year', 'Country', 'PC5','PC6','PC7','PC8','PC9','PC10','PC11', 'threshold']

ypc = migration_pcs.threshold.values

Xpc = migration_pcs[[i for i in columns_ if i not in exclude_cols]]
Xpc = Xpc.values
```

### Perform stratified cross-validation on a KNN classifier and logisitic regression.


```python
knn = KNeighborsClassifier()

params = {
    'n_neighbors':range(1,20),
    'weights':['uniform','distance']
}

knn_gs_pc = GridSearchCV(knn, params, cv=5, verbose=1)
knn_gs_pc.fit(Xpc, ypc)

print knn_gs_pc.best_params_
best_knn_pc = knn_gs_pc.best_estimator_
```

    Fitting 5 folds for each of 38 candidates, totalling 190 fits
    {'n_neighbors': 17, 'weights': 'uniform'}


    [Parallel(n_jobs=1)]: Done 190 out of 190 | elapsed:    0.9s finished



```python
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
```

    Xtrain and ytrain shapes:
    (85, 4) (85,)
    Xtest and ytest shapes:
    (23, 4) (23,)
    Xtrain and ytrain shapes:
    (86, 4) (86,)
    Xtest and ytest shapes:
    (22, 4) (22,)
    Xtrain and ytrain shapes:
    (87, 4) (87,)
    Xtest and ytest shapes:
    (21, 4) (21,)
    Xtrain and ytrain shapes:
    (87, 4) (87,)
    Xtest and ytest shapes:
    (21, 4) (21,)
    Xtrain and ytrain shapes:
    (87, 4) (87,)
    Xtest and ytest shapes:
    (21, 4) (21,)
    
    ======
    
    KNN accuracy scores on test:
    [0.69565217391304346, 0.81818181818181823, 0.95238095238095233, 0.90476190476190477, 0.90476190476190477]
    KNN mean of accuracy scores on test:
    0.8551477508
    KNN mean of accuracy scores on train :
    0.872755633127
    
    ======
    
    Logistic Regression accuracy scores on test:
    [0.78260869565217395, 0.86363636363636365, 0.95238095238095233, 0.8571428571428571, 0.42857142857142855]
    Logistic Regression mean of accuracy scores on test:
    0.776868059477
    Logistic Regression mean of accuracy scores on train:
    0.937553658191
    
    ======
    
    Baseline accuracy:
      0.333333333333


### We found more accurate results using a stratified cross validation in both KNN and the logistic estimation


```python
'''the mean of the accuracy score on the test data has a significant increase from '''
print 'KNN mean of accuracy scores on test:\n', np.mean(knn_scores_test)
print 'KNN mean of accuracy scores on test PC:\n', np.mean(knn_scores_test_pc)
print "Increase of accuracy of:", (np.mean(knn_scores_test_pc) - np.mean(knn_scores_test))
```

    KNN mean of accuracy scores on test:
    0.610483719179
    KNN mean of accuracy scores on test PC:
    0.8551477508
    Increase of accuracy of: 0.244664031621


### We have a significant increase in accuracy doing the parallel estimation of KNN and PC

### Confusion Matrix for each of your classification methods.


```python
# Load Confusion Matrix 
from sklearn.metrics import confusion_matrix
```


```python
def confus_mat(ytrue, ypred_method, what_predict):
    what_predict = str(what_predict)
    confmat = confusion_matrix(y_true=ytrue, y_pred=ypred_method)
    confusion = pd.DataFrame(confmat, index=['is_not_' + what_predict, 'is_' + what_predict],
                         columns=['predicted_is_not_'+ what_predict, 'predicted_is_'+what_predict])
    return confusion
```


```python
# Load Classification Report
from sklearn.metrics import classification_report
```


```python
def class_report(ytrue, ypred):
    cls_rep = classification_report(yte, y_knn_predict)
    print cls_rep
```


```python
## Confuion Matrix for knn
confus_mat(yte, y_knn_predict, 'threshold')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted_is_not_threshold</th>
      <th>predicted_is_threshold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>is_not_threshold</th>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>is_threshold</th>
      <td>7</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Classification report for knn
class_report(yte, y_knn_predict)
```

                 precision    recall  f1-score   support
    
              0       0.65      0.93      0.76        14
              1       0.00      0.00      0.00         7
    
    avg / total       0.43      0.62      0.51        21
    



```python
## Confusion Matrix for logistic
confus_mat(yte, y_log_predict, 'threshold')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted_is_not_threshold</th>
      <th>predicted_is_threshold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>is_not_threshold</th>
      <td>14</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_threshold</th>
      <td>7</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Classification report for logistic
class_report(yte, y_log_predict)
```

                 precision    recall  f1-score   support
    
              0       0.65      0.93      0.76        14
              1       0.00      0.00      0.00         7
    
    avg / total       0.43      0.62      0.51        21
    



```python
## Confuion Matrix for knn with PC
confus_mat(yte, y_knn_predict_pc, 'threshold')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted_is_not_threshold</th>
      <th>predicted_is_threshold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>is_not_threshold</th>
      <td>12</td>
      <td>2</td>
    </tr>
    <tr>
      <th>is_threshold</th>
      <td>0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Classification report for knn with PC
class_report(yte, y_knn_predict_pc)
```

                 precision    recall  f1-score   support
    
              0       0.65      0.93      0.76        14
              1       0.00      0.00      0.00         7
    
    avg / total       0.43      0.62      0.51        21
    



```python
## Confuion Matrix for log with PC
confus_mat(yte, y_log_predict_pc, 'threshold')
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted_is_not_threshold</th>
      <th>predicted_is_threshold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>is_not_threshold</th>
      <td>2</td>
      <td>12</td>
    </tr>
    <tr>
      <th>is_threshold</th>
      <td>0</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Classification report for knn with PC
class_report(yte, y_log_predict_pc)
```

                 precision    recall  f1-score   support
    
              0       0.65      0.93      0.76        14
              1       0.00      0.00      0.00         7
    
    avg / total       0.43      0.62      0.51        21
    


### Our results from our confusion matrices suggest that it is better to use the parallel estimation of KNN and PC in order to have more accurate estimations. 
