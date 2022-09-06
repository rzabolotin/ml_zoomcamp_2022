# Заметки по уроку 2. Регрессия

## EDA

`np.log1p()` - метод, который добавляет к каждому элементу массива 1 и применяет к нему логарифм
`np.expm1()` - обратный метод

### Как вывести список колонок, и краткую статистику по ним?

```python
for col in df.columns:
    print(col)
    print(df[col].unique()[:5]) 
    print(df[col].nunique())
```

sns.histplot(df.msrp, bins=50) - гистограмма


## Настройка валидации

### Разбиваем данные на тренировочные, тестовые и валидационные

```python 

n = len(df)

n_val = (int)(n * 0.2)
n_test = (int)(n * 0.2)
n_train = n - n_val - n_test
```

```python
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)

df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
df_val = df.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)
df_test = df.iloc[idx[n_train+n_val:]].reset_index(drop=True)

y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

del df_train['msrp']
del df_val['msrp']
del df_test['msrp']
```

### Использование sklearn

```python
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=n_test, random_state=42)
```

## Линейная регрессия
Линейная регрессия это алгоритм машинного обучения, которе предсказывает числовую переменную, используя линейную комбинацию других переменных.

```python
# Функция, которая предсказывает цену автомобиля
# x - признаки
# w0 - bias
# w - веса
def linear_regression(x, w0, w):
    return w0 + x.dot(w)

# В этой функции мы bias включили в w
def linear_regression2(x, w):
    x = np.hstack([np.ones((x.shape[0], 1)), x])
    return x.dot(w)
```

## Тренировка модели (матричный метод)

Иногда можно посчитать с помощью матричного разложения

```python
def train_linear_regression_reg(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]
```

## Baseline

Сделаем простую модельку

```python

def prepare_X(df):
    df_num = df.select_dtypes(include=['number'])
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train)

y_pred = w0 + X_train.dot(w)


# Можно нарисовать две гистограммы, одна поверх другой
sns.histplot(y_pred, bins=50, alpha=0.5, color='red', label='prediction')
sns.histplot(y_train, bins=50, alpha=0.5, color='blue', label='target')

```

## RMSE
Оцениваем качество модели
Среднее квадратичное отклонение

```python
def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)
```

## Оценка на валидационном наборе

```python
X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train)
X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)
```












