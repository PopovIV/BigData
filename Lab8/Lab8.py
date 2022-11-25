import numpy as np
import pandas as pd

from scipy import stats
from sklearn import linear_model
from matplotlib import pyplot as plt

import matplotlib as mpl

default_dpi = mpl.rcParamsDefault['figure.dpi']
mpl.rcParams['figure.dpi'] = default_dpi*2.5

plt.rcParams.update({"font.size": 3, "lines.linewidth": 1})

class PolyRegularRegression:

    def __init__(self, model, alfa: float, degree: int) -> None:
        
        self.model = model(alfa)
        self.degree = degree

    def fit(self, x: np.array, y: np.array):
        self.model.fit(self._polynomial(x), y)

    def predict(self, x: np.array):
        return self.model.predict(self._polynomial(x))

    def get_params(self):
        return self.model.get_params()

    def _polynomial(self, x: np.array) -> np.array:
        pol_x = np.array([x**(i + 1) for i in range(self.degree)]).T
        return pol_x

models = {
    "crest": linear_model.Ridge,
    "lasso": linear_model.Lasso
}

# Set things up
data = np.array([[-2, -7], [-1, 0], [0, 1], [1, 2], [2, 9]])
sigmas = [0.1, 0.2, 0.3]

noises = [stats.norm.rvs(size=len(data), scale=var) for var in sigmas]
opt_nois = noises[0]
x = data[:, 0]
y = data[:, 1]

degree = 11

alfas = [1, 0.1, 0.01, 0.001]
opt_alfa = 0.01

x_test = np.linspace(-2, 2, 100)

fig, axs = plt.subplots(len(models), len(alfas))
params = pd.DataFrame()

for i, alfa in enumerate(alfas):
    for j, (model_name, model_class) in enumerate(models.items()):
        name = f"{model_name}, alfa={alfa}"

        model = PolyRegularRegression(model=model_class, alfa=alfa, degree=degree)
        model.fit(x, y + opt_nois)
        
        axs[j, i].plot(x_test, model.predict(x_test))
        axs[j, i].plot(x, y, ".")
        axs[j, i].set_title(name)
        axs[j, i].grid()
        
        params[name] = model.model.coef_
plt.show()
for name in params:
    params[name].apply(lambda x: None if x == 0. else x)
print(params)
fig, axs = plt.subplots(len(models), len(noises))
params = pd.DataFrame()


for i, nois in enumerate(noises):
    for j, (model_name, model_class) in enumerate(models.items()):
        name = f"{model_name}, sigma={sigmas[i]}"

        model = PolyRegularRegression(model=model_class, alfa=opt_alfa, degree=degree)
        model.fit(x, y + nois)
        
        axs[j, i].grid()
        axs[j, i].plot(x_test, model.predict(x_test))
        axs[j, i].plot(x, y, ".")
        axs[j, i].set_title(name)
        params[name] = model.model.coef_
plt.show()
for name in params:
    params[name].apply(lambda x: None if x == 0. else x)
print(params)