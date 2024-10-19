import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df=pd.read_csv('Blockbuster_Dataset_encoded.csv')
df=df.drop_duplicates()
y=df['Cycle time ']
x=df.drop('Cycle time ',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)
model=RandomForestRegressor(max_depth=4,random_state=35)
model.fit(x_train,y_train)
joblib.dump(model, 'model.pkl')
y_pre=model.predict(x_test)