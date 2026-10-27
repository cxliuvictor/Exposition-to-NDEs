# sklearn Modelling Emergency Reference

## Regression
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=.2,random_state=42)
baseline=np.repeat(y_train.mean(),len(y_valid))
rmse=mean_squared_error(y_valid,pred)**0.5
mae=mean_absolute_error(y_valid,pred)
r2=r2_score(y_valid,pred)
```
Ridge:
```python
ridge=Pipeline([("scale",StandardScaler()),("model",Ridge(alpha=1.0))])
```
Random forest:
```python
rf=RandomForestRegressor(n_estimators=200,min_samples_leaf=3,random_state=42,n_jobs=-1)
```

## Mixed numeric/categorical pipeline
```python
num_pipe=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
cat_pipe=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
pre=ColumnTransformer([("num",num_pipe,numeric_cols),("cat",cat_pipe,categorical_cols)])
model=Pipeline([("pre",pre),("reg",Ridge(alpha=1.0))])
```

## Classification
```python
clf=LogisticRegression(max_iter=2000)
prob=clf.predict_proba(X_valid)[:,1]
pred=(prob>=.3).astype(int)
```
Know accuracy, precision, recall, F1, ROC-AUC, PR-AUC. Select thresholds on validation, never test.

## Discipline
IID tabular: random split / K-fold can be fine. Time series: chronological split, no shuffling. Always baseline → simple model → nonlinear model → validation comparison → freeze → test once.

## Leakage examples
Future target-derived feature; scaler/imputer fit on all data; feature/threshold selection on test; post-event information used for a pre-event decision; incorrect time alignment.

## Error slicing
```python
tmp=pd.DataFrame({"y":y_valid,"pred":pred})
tmp["bucket"]=pd.qcut(tmp["y"],5,duplicates="drop")
bucket_rmse=tmp.groupby("bucket",observed=True).apply(lambda g: np.sqrt(np.mean((g["y"]-g["pred"])**2)))
```
