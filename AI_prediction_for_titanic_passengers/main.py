import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load data
df = pd.read_csv('titanic-class-age-gender-survived.csv')

X = df.loc[:, ['Age', 'Gender', 'PClass']]
y = df.loc[:, ['Survived']]

# dummyt
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'),
                                      ['Gender', 'PClass'])], remainder='passthrough')
X = ct.fit_transform(X)

# Opetus ja testidata
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Skaalataan data
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# Opetetaan malli
model = LogisticRegression()
model.fit(X_train,y_train)

# Ennustetaan testidatalla
y_pred = model.predict(X_test)
y_pred_pros = model.predict_proba(X_test)

# Lasketaan metriikat
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
pc = precision_score(y_test, y_pred)
rc = recall_score(y_test, y_pred)

# Tulostetaan metriikat
print (f'cm: \n{cm}')
print (f'acc: {acc}')
print (f'pc: {pc}')
print (f'rc: {rc}')

# Visualisoidaan confusion matrix
tn, fp, fn, tp = cm.ravel() # ravel palauttaa litistetyn taulukon
ax = plt.axes()
sns.heatmap(cm, ax = ax, annot=True, fmt='g')
ax.set_title(f'LogReg (acc: {acc:.02f}, recall: {rc:.02f}, precision: {pc:.02f})')
plt.show()

# Ennustetaan uusille matkustajille
Xnew = pd.read_csv('titanic-new.csv')
# Xnew = Xnew.loc[:, ['Age', 'Gender']]
Xnew = ct.transform(Xnew)
Xnew = scaler_x.transform(Xnew)
y_pred_new = model.predict(Xnew)
y_pred_new_pros = model.predict_proba(Xnew)

print(f'uudet: \n{y_pred_new_pros}')


