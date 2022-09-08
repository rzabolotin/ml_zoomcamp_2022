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


## Добавление новых фичей

- Добавим фичи из даты (преобразуем в число)
- Добавим фичи из категориальных переменных 
  - (label encoding) не было примера
  - one-hot encoding


```python

categories_list = ['make', 'model', 'condition', 'cylinders', 'fuel', 'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color']
categories = {}
for c in categories_list:
    categories[c] = list(df[c].value_counts().head().index)

def prepare_X(df):
    df = df.copy()
    df['age'] = datetime.now().year - df.year
    for v in [2,3,4]:
        df[f'doors_{v}'] = (df.number_of_doors == v).astype(int)
    
    for c, values in categories.items():
        for v in values:
            df[f'{c}_{v}'] = (df[c] == v).astype(int)
        
    df_num = df.select_dtypes(include=['number'])
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
```

## Регуляризация

Чтобы избежать переобучения, добавим регуляризацию
Бывает, когда колонки в матрице X линейно зависимы, тогда решение не единственно

Если решаем через линейное уравнение, то нужно добавить к XTX единичную матрицу с небольшим коэффициентом
Параметр регуляризации - это коэффициент перед единичной матрицей


```python

def train_linear_regression_reg(X, y, r=0.01):
    # r - коэффициент регуляризации
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]
```

## Ищем лучшее значение для параметра регуляризации

Он подбирается перебором, на валидационном наборе

```python
for r in [0, 0.0001, 0.001, 0.01, 0.1, 1, 10]:
    X_train = prepare_X(df_train)
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    print(r, rmse(y_val, y_pred))
``` 

# Тренируем модель на всех данных

- pd.concat([df_train, df_val]).reset_index(drop=True) - объединяет датафреймы по строкам, сбрасывает индексы
- np.concatenate([y_train, y_val]) - объединяет массивы по строкам

```python
df = pd.concat([df_train, df_val]).reset_index(drop=True) 
y = np.concatenate([y_train, y_val])

X = prepare_X(df)
w0, w = train_linear_regression_reg(X, y, r=0.001)

X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
print(rmse(y_test, y_pred))
```

# Используем модель

Сделаем вид, что данные нам пришли в виде словаря

```python
sample_x = df_test.iloc[20].to_dict()

df_small = pd.DataFrame([sample_x])
X_small = prepare_X(df_small)
y_pred = w0 + X_small.dot(w)
real_price = np.expm1(y_pred[0])
```








