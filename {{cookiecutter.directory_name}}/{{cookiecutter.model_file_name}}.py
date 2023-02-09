import pandas as pd
import spark
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
{%- if cookiecutter.given_model == "catboost" %}
import catboost
import zxc

model = CatBoost()
{%- elif cookiecutter.given_model == "logistic regression" %}
from sklearn.LinearModels import Ridge

model = Ridge()
{% endif %}

DF = # Your code here
X_train, y_train, X_test, y_test = train_test_split(DF)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Final accuracy is {np.mean(y_pred == y_test)})
