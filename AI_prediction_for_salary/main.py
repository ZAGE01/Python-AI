import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('salary.csv')

nans = df.isna().sum()
desc = df.describe()

plt.scatter(df.YearsExperience, df.Salary)
plt.show()

X = df.loc[:, ['YearsExperience']]
y = df.loc[:, ['Salary']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Luodaan malli
model = LinearRegression()
model.fit(X_train, y_train)

# Ennustetaan testidatalla
y_pred = model.predict(X_test)

# Metriikat
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print('\nMetriikka train datalla: ')
print(f'mae: {mae}')
print(f'rmse: {rmse}')
print(f'R2: {r2}')

print(f'Uuden työntekijän palkka 7v kokemuksella on: {model.predict([[7]])}')

# Selvitetään suoran yhtälö
coef = model.coef_
inter = model.intercept_
print('Suoran yhtälö on: ')
print(f'Salary = {coef[0]} * YearsExperience + {inter}')

# Visualisoidaan testiaineiston ennusteet
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Palkka vs kokemus (testidata)')
plt.xlabel('Kokemus')
plt.ylabel('Palkka')
plt.show()
