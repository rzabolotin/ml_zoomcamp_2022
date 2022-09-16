## Churn prediction
It is a binary classification problem. We have to predict whether a customer will churn or not.

## Data preparation
We have a dataset with 7043 rows and 21 columns. The target variable is `Churn`. It is a binary variable. We have to predict whether a customer will churn or not.

```python
import pandas as pd
df = pd.read_csv('data/churn.csv')

df.head().T # transpose to see all columns

df.columns = df.columns.str.lower().str.replace(' ', '_')
df.dtypes # check data types
df.totalcharges = pd.to_numeric(df['totalcharges'], errors='coerce').fillna(0)

df.churn = (df.churn == 'Yes').astype(int) # convert to 0/1
```

## Validation
We have to split the data into train and test sets. We will use the `train_test_split` function from `sklearn.model_selection` module.

```python
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

```

## EDA
We will use the `pandas_profiling` library to generate a report.

```python
from pandas_profiling import ProfileReport
df_full_train.profile_report()
```
df_full_train.churn.value_counts(normalize=True) # 73% of customers did not churn
global_churn_rate = df_full_train.churn.mean() # the same

numeric = ['tenure', 'monthlycharges', 'totalcharges']
categorical = df.columns - numeric - ['customerid']

df_full_train[categorical].nunique() # check number of unique values

## Churn rate in categorical variables
```python

for col in categorical:
    df_group = (df_full_train.groupby(col).churn.agg(['mean','count']))
    df_group['diff'] = df_group['mean'] - global_churn_rate
    df_group['risk'] = df_group['mean'] / global_churn_rate
    df_group = df_group.sort_values(by='diff', ascending=False)
    display(df_group)
    print() 
    
```

## Feature importance

For categorical variables we use mutal info score

```python

from sklearn.metrics import mutual_info_score

mutual_info_score(df_full_train.churn, df_full_train.contract)
mutual_info_score(df_full_train.churn, df_full_train.gender)
mutual_info_score(df_full_train.churn, df_full_train.partner)

def mutal_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)

mi = df_full_train[categorical].apply(mutal_info_churn_score)
mi.sort_values(ascending=False)
 
```

Для числовых переменных используем корреляцию Спирмена

```python

df_full_train[numerical].corrwith(df_full_train.churn)

```

## One hot encoding

Для категориальных переменных делаем one hot encoding
Используем sklearn DictVectorizer

Переводим записи в словарь
А потом переводим словарь с матрицу


```python

from sklearn.feature_extraction import DictVectorizer

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)
X_columns_names = dv.get_feature_names()

```

## Logistic regression

Берем сигмоиду от линейной комбинации признаков.
Получаем предсказание вероятности принадлежности к классу 1

```python

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def linear_combination(X, w0, w):
    return X.dot(w) + w0

def probability(X, w0, w):
    z = linear_combination(X, w0, w)
    return sigmoid(z)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

model.intercept_
model.coef_

model.predict(x_val) # hard predictions
y_pred = model.predict_proba(x_val)[:, 1] # predict probabilities [0 and 1], soft predictions
churn_decision = (y_pred >= 0.5) 

df_pred = pd.DataFrame()
df['probability'] = y_pred
df['prediction'] = churn_decision.astype(int)
df['actual'] = y_val

df['correct'] = df.prediction == df.actual
df['correct'].mean() # accuracy



```
```




```



```