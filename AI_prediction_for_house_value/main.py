import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('housing.csv')

nans = df.isna().sum()
desc = df.describe()

plt.scatter(df.median_income, df.median_house_value)
plt.show()

X = df.loc[:, ['median_income']]
y = df.loc[:, ['median_house_value']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Luodaan malli
model = LinearRegression()
model.fit(X_train, y_train)

# Selvitetään suoran yhtälö
coef = model.coef_
inter = model.intercept_
print('Suoran yhtälö on: ')
print(f'median_house_value = {coef[0]} * median_income + {inter}')

# Ennustetaan testidatalla
y_pred = model.predict(X_test)

# Visualisoidaan ennustettujen ja todellisten arvojen (talon arvo) ero histogrammilla
plt.hist(y_test - y_pred, bins=20)
plt.show()

# Metriikat
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print('\nMetriikka train datalla: ')
print(f'mae: {mae}')
print(f'rmse: {rmse}')
print(f'R2: {r2}')

# Ennustetaan kotitalouden arvo, kun vuositulot on 30 000 dollaria
income = [[30]]
print(f'Kotitalouden arvo 30 000 dollarin vuosituloilla: {model.predict(income)}')