# Installation
```bash
pip install git+https://github.com/kickermeister/TaxiFareModel
```



# Usage

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data


df = get_data(aws=True)
df = clean_data(df)
y = df["fare_amount"]
X = df.drop("fare_amount", axis=1)

# hold out
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

# build pipeline
trainer = Trainer(X_train, y_train)

# train the pipeline
trainer.run()

# evaluate the pipeline
rmse = trainer.evaluate(X_val, y_val)
print(f'RMSE: {round(rmse, 2)}')
```
