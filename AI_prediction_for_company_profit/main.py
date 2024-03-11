import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Lue data sekä jaa X ja y
df = pd.read_csv('startup.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, [-1]]

# Pandas get_dummies():
    # dummies = pd.get_dummies(X['State'], drop_first=True)

X_org = X

# Parempi tapa dummy-muuttujille
ct = ColumnTransformer(transformers=[('encoder',
OneHotEncoder(drop='first'), ['State'])], remainder='passthrough')

X = ct.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scaling the features
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

# Scaling the target variable
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train)

# Training the Multiple Linear Regression model on the Training set
model = LinearRegression()
model.fit(X_train, y_train)

# Suoran yhtälön kertoimet
coef = model.coef_[0]  # Otetaan ensimmäinen rivi kertoimista
# Vakio osa
inter = model.intercept_[0]  # Otetaan ensimmäinen arvo vakiotermistä

print("Suora yhtälö:")
print(f"Profit = {inter:.2f} + ({coef[0]:.2f} * R&D Spend) + ({coef[1]:.2f} * Administration) + ({coef[2]:.2f} * Marketing Spend) + ({coef[3]:.2f} * State)")


# Predicting the Test set results
# y_pred = model.predict(X_test)
y_pred = y_scaler.inverse_transform(model.predict(X_test))

# Regression metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'r2: {round(r2,4)}')
print(f'mae: {round(mae,4)}')
print(f'rmse: {round(rmse,4)}')



## Tehtävä 2

df_new_company = pd.read_csv('new_company_ct.csv')
df_new_company = ct.transform(df_new_company)
df_new_company = X_scaler.transform(df_new_company)
y_comp = y_scaler.inverse_transform(model.predict(df_new_company))

print (f'Uuden yrityksen voitto: {y_comp[0]}, todellinen: ')
print (f'{df.iloc[0:1,-1].values[0]}')
