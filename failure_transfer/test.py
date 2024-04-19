import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
import matplotlib.pyplot as plt

# Generate random data
n = 289
x = np.random.rand(n)
t = 0.5 + 0.3 * x + np.random.randn(n) * 0.1
weights = np.random.randint(100, 1000, n)

data = pd.DataFrame({'x': x, 't': t, 'n': weights})

# Fit the ecological regression models
ols_fit = sm.OLS(data['t'], sm.add_constant(data['x'])).fit()
wls_fit = sm.WLS(data['t'], sm.add_constant(data['x']), weights=data['n']).fit()
glm_fit = GLM(data['t'], sm.add_constant(data['x']), family=Binomial(), freq_weights=data['n']).fit()

# Compare the models
compare = pd.DataFrame({
    'OLS Estimate': ols_fit.params,
    'OLS Std. Error': ols_fit.bse,
    'OLS t value': ols_fit.tvalues,
    'WLS Estimate': wls_fit.params,
    'WLS Std. Error': wls_fit.bse,
    'WLS t value': wls_fit.tvalues,
    'GLM Estimate': glm_fit.params,
    'GLM Std. Error': glm_fit.bse,
    'GLM z value': glm_fit.tvalues
})
print(compare.round(3))

# Perform hypothesis tests
ols_test = ols_fit.pvalues
wls_test = wls_fit.pvalues
glm_test = glm_fit.pvalues

print("OLS p-values:", ols_test.round(3))
print("WLS p-values:", wls_test.round(3))
print("GLM p-values:", glm_test.round(3))

# Calculate and print R-squared values
ols_rsq = ols_fit.rsquared
wls_rsq = wls_fit.rsquared
glm_rsq = 1 - glm_fit.deviance / glm_fit.null_deviance

print("OLS R-squared:", round(ols_rsq, 3))
print("WLS R-squared:", round(wls_rsq, 3))
print("GLM Pseudo R-squared:", round(glm_rsq, 3))

# Print confidence intervals for the coefficients
ols_ci = ols_fit.conf_int()
wls_ci = wls_fit.conf_int()
glm_ci = glm_fit.conf_int()

print("OLS Confidence Intervals:")
print(ols_ci.round(3))
print("WLS Confidence Intervals:")
print(wls_ci.round(3))
print("GLM Confidence Intervals:")
print(glm_ci.round(3))

# Print model diagnostics
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].scatter(ols_fit.fittedvalues, ols_fit.resid)
axs[0, 0].set_xlabel("Fitted Values")
axs[0, 0].set_ylabel("Residuals")
axs[0, 0].set_title("Residuals vs. Fitted")

sm.qqplot(ols_fit.resid, line='45', ax=axs[0, 1])
axs[0, 1].set_title("QQ Plot")

axs[1, 0].scatter(ols_fit.fittedvalues, np.sqrt(np.abs(ols_fit.resid_pearson)))
axs[1, 0].set_xlabel("Fitted Values")
axs[1, 0].set_ylabel("Sqrt(|Standardized Residuals|)")
axs[1, 0].set_title("Scale-Location")

sm.graphics.influence_plot(ols_fit, ax=axs[1, 1])

plt.tight_layout()
plt.show()