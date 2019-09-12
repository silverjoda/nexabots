from matplotlib import pyplot as plt
import numpy as np

from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import torch
import robust_loss_pytorch.general#

class RegressionModel(torch.nn.Module):
    # A simple linear regression module.
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.linear = torch.nn.Linear(4, 1)
    def forward(self, x):
        return self.linear(x[:,None])[:,0]


np.random.seed(42)

X = np.random.normal(size=400)
y = np.sin(X)
# Make sure that it X is 2D
X = X[:, np.newaxis]
#xn = PolynomialFeatures(3, in).fit_transform(X)

X_test = np.random.normal(size=200)
y_test = np.sin(X_test)
X_test = X_test[:, np.newaxis]

y_errors = y.copy()
y_errors[::3] = 3

X_errors = X.copy()
X_errors[::3] = 3

y_errors_large = y.copy()
y_errors_large[::3] = 10

X_errors_large = X.copy()
X_errors_large[::3] = 10

estimators = [('OLS', LinearRegression()),
              ('Theil-Sen', TheilSenRegressor(random_state=42)),
              ('RANSAC', RANSACRegressor(random_state=42)),
              ('HuberRegressor', HuberRegressor()),
              ('AdaptiveRegressor', RegressionModel)]
colors = {'OLS': 'turquoise', 'Theil-Sen': 'gold', 'RANSAC': 'lightgreen', 'HuberRegressor': 'black', 'AdaptiveRegressor' : 'red'}
linestyle = {'OLS': '-', 'Theil-Sen': '-.', 'RANSAC': '--', 'HuberRegressor': '--', 'AdaptiveRegressor' : '--'}
lw = 3

y_pt = torch.tensor(y[:, np.newaxis], dtype=torch.float32)

x_plot = np.linspace(X.min(), X.max())
for title, this_X, this_y in [
        ('Modeling Errors Only', X, y),
        ('Corrupt X, Small Deviants', X_errors, y),
        ('Corrupt y, Small Deviants', X, y_errors),
        ('Corrupt X, Large Deviants', X_errors_large, y),
        ('Corrupt y, Large Deviants', X, y_errors_large)]:
    plt.figure(figsize=(5, 4))
    plt.plot(this_X[:, 0], this_y, 'b+')

    for name, estimator in estimators:
        if name == 'AdaptiveRegressor':
            model = estimator()
            tmp = PolynomialFeatures(3).fit_transform(this_X)
            this_X_PN = torch.tensor(PolynomialFeatures(3).fit_transform(this_X), dtype=torch.float32)
            X_test_PN = torch.tensor(PolynomialFeatures(3).fit_transform(X_test), dtype=torch.float32)
            x_plot_PN = torch.tensor(PolynomialFeatures(3).fit_transform(x_plot[:, np.newaxis]), dtype=torch.float32)

            adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
                num_dims=1, float_dtype=np.float32, device='cpu')
            params = list(model.parameters()) + list(adaptive.parameters())
            optimizer = torch.optim.Adam(params, lr=0.01)

            # Do the fitting
            for epoch in range(2000):

                y_i = model(this_X_PN)

                loss = torch.mean(adaptive.lossfun((y_i - y_pt)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if np.mod(epoch, 100) == 0:
                    print('{:<4}: loss={:03f}'.format(epoch, loss.data))

            mse = mean_squared_error(model(X_test_PN).detach().numpy(), y_test)
            y_plot = model(x_plot_PN).detach().numpy()
            plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
                     linewidth=lw, label='%s: error = %.3f' % (name, mse))

        else:
            model = make_pipeline(PolynomialFeatures(3), estimator)
            model.fit(this_X, this_y)
            mse = mean_squared_error(model.predict(X_test), y_test)
            y_plot = model.predict(x_plot[:, np.newaxis])
            plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
                     linewidth=lw, label='%s: error = %.3f' % (name, mse))

    legend_title = 'Error of Mean\nAbsolute Deviation\nto Non-corrupt Data'
    legend = plt.legend(loc='upper right', frameon=False, title=legend_title,
                        prop=dict(size='x-small'))
    plt.xlim(-4, 10.2)
    plt.ylim(-2, 10.2)
    plt.title(title)
plt.show()