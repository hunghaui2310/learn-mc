import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

data = pd.read_csv('data.csv', header=None)
print(data)
X = data.values
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(X)
result = imp.transform(X)
print(result)