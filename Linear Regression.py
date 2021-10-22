import numpy as np
import matplotlib.pyplot as plt
from keras.losses import mean_squared_error
from sklearn import datasets, linear_model


# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename']
# here target is y and data is x we can take one weight or all
diabetes = datasets.load_diabetes()
diabetes_x = diabetes.data[:,np.newaxis,2]
diabetes_x_train = diabetes_x[:-30]
diabetes_x_test = diabetes_x[-20:]
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-20:]

model = linear_model.LinearRegression()
model.fit(diabetes_x_train,diabetes_y_train)
diabetes_y_predicted = model.predict(diabetes_x_test)

print("Mean squared error",mean_squared_error(diabetes_y_test,diabetes_y_predicted))
print("weights", model.coef_)
print("intercept", model.intercept_)

plt.scatter(diabetes_x_test,diabetes_y_test)
plt.plot(diabetes_x_test,diabetes_y_predicted)
plt.show()
