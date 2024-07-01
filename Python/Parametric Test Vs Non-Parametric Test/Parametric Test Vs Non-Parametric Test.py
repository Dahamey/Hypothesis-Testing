#!/usr/bin/env python
# coding: utf-8

# <h1 style="text-align:center;">Parametric Test Vs Non-Parametric Test</h1>
# 

# <u>**Author</u> :** [Younes Dahami](https://www.linkedin.com/in/dahami/)

# What this video also : https://www.youtube.com/watch?v=wGPEy9ewq34

# <a id="top"></a>
# 
# <div class="list-group" id="list-tab" role="tablist">
# <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Table of contents</h3>
#     
# <font color="blue" size=+1><b>[0. Introduction](#0)</b></font>
#     
# <font color="blue" size=+1><b>[1. Parametric Tests](#1)</b></font>
#     
# <font color="blue" size=+1><b>[2. Non-Parametric Tests](#2)</b></font>
# 
# <font color="blue" size=+1><b>[3. Hypothesis Testing](#3)</b></font>
# * [3.1 Defining Hypotheses](#3.1)
# * [3.2 Assumption Check](#3.2)
# * [3.3 Checking for Normality](#3.3)
# * [3.4 Checking for Homoscedasticity](#3.4)    
# * [3.5 Selecting the Proper Test](#3.5)
# * [3.6 Decision and Conclusion](#3.6)
#     
#     
# <font color="blue" size=+1><b>[4. Tests Examples](#4)</b></font>
# * [4.1 Independent t-test / Two-sample t-test / Unpaired t-test](#4.1)
#     * [4.1.1 Definition](#4.1.1)
#     * [4.1.2 Example of Independent t-test](#4.1.2)
#     
# * [4.2 Analysis of Variance (ANOVA)](#4.2)
#     * [4.2.1 Definition](#4.2.1)
#     * [4.2.2 Example of ANOVA](#4.2.2)
#     
# * [4.3 Mann Whitney U / Wilcoxon rank-sum test](#4.3)
#     * [4.3.1 Definition](#4.3.1)
#     * [4.3.2 Example of Mann Whitney U / Wilcoxon rank-sum test](#4.3.2)
#     
#  
# * [4.4 Kruskal-Wallis test](#4.4)
#     * [4.4.1 Definition](#4.4.1)
#     * [4.4.2 Example of Kruskal-Wallis test](#4.4.2)
#     
#     
# * [4.5 Dependent t-Test / Paired t-Test](#4.5)
#     * [4.5.1 Definition](#4.5.1)
#     * [4.5.2 Example of Dependent t-Test (Paired t-Test)](#4.5.2)
#     
#     
# * [4.6 Wilcoxon Signed-Rank test](#4.6)
#     * [4.6.1 Definition](#4.6.1)
#     * [4.6.2 Example of Wilcoxon Signed-Rank test](#4.6.2)
#     
# * [4.7 Friedman $\chi$² (Chi-Square)](#4.7)
#     * [4.7.1 Definition](#4.7.1)
#     * [4.7.2 Example of Friedman $\chi$² (Chi-Square)](#4.7.2)
#     
# * [4.8 $\chi^2$ Test for Independence](#4.8)
#     * [4.8.1 Definition](#4.8.1)
#     * [4.8.2 Example of $\chi^2$ Test for Independence](#4.8.2) 
#     
# * [4.9 $\chi^2$ Goodness of Fit Test](#4.9)
#     * [4.9.1 Definition](#4.9.1)
#     * [4.9.2 Example of $\chi^2$ Goodness of Fit Test](#4.9.2)
# 
# * [4.10 Pearson Correlation](#4.10)
#     * [4.10.1 Definition](#4.10.1)
#     * [4.10.2 Example of Pearson Correlation](#4.10.2)
#       
# * [4.11 Spearman Correlation](#4.11)
#     * [4.11.1 Definition](#4.11.1)
#     * [4.11.2 Example of Spearman Correlation](#4.11.2)
#     

# <a id="0"></a>
# 
# # Introduction
# 
# You want to calculate a hypothesis test, but you don't know exactly what's the difference is between **parametric** and **non-parametric** tests, and you're wondering when to use each test.
# 
# If you want to calculate a hypothesis test, you must first check the assumptions. One of the most common assumptions is the data used must show a **certain distribution**, usually the normal distribution.
# 
# To simply put it :
# 
# * If your data is **normally distributed**, use the **parametric tests** e.g. : t-test, ANOVA or Pearson Correlation.
# 
# * If your data is **not normally distributed**, use the **non-parametric tests** e.g. : Mann-Whitney U test or a Spearman Correlation.
# 
# Of course you still have to check if there are further assumptions for the respective test. But in general, there are less assumptions for non-parametric tests than for parametric tests.
# 
# Parametric tests are generally more powerful than non-parametric tests.

# ## About Scale of Measurement
# 
# In statistics, the scale of measurement refers to the different ways in which variables can be quantified and categorized. There are four primary scales of measurement: **nominal, ordinal, interval,** and **ratio.**
# 
# * **Nominal Scale :** Data are categorized without any order or ranking. Examples: Gender (male, female), Marital status (single, married, divorced).
# 
# * **Ordinal Scale :** Data are categorized with a meaningful order, but the intervals between categories are not necessarily equal. Examples: Socioeconomic status (low, middle, high), Education level (high school, bachelor's, master's, Ph.D.).
# 
# * **Interval Scale :** Data are ordered, and the intervals between data points are equal and meaningful.
# However, there is no true zero point (i.e., zero does not indicate the absence of the quantity being measured). Examples: Temperature in Celsius or Fahrenheit, IQ scores.
# 
# * **Ratio Scale :** Data have all the properties of an interval scale, but also have a true zero point, which indicates the absence of the quantity being measured. Examples: Height, weight, age, income.
# 
# 

# <a id="1"></a>
# 
# # 1) Parametric Tests
# 
# **<u>Parametric tests</u>** **:** are statistical tests that make assumptions about the parameters of the population distribution from which the samples are drawn. Specifically, they assume that the data follows a certain distribution (usually a normal distribution) and often involve estimating parameters such as the mean and standard deviation.
# 
# Key Characteristics :
# 
# * **Assumption of Normality :** Data should be normally distributed or approximately normally distributed.
# * **Homogeneity of Variances :** Assumes equal variances among the groups being compared.
# * **Scale of Measurement :** Typically requires data that is at least interval scale, meaning that the data should have :
# 
#     * Equal intervals between values, ensuring that the difference between any two values is consistent.
#     * Meaningful numerical distances between data points.
# 
# In other words, the data should be quantitative and should reflect consistent intervals between measurements, allowing for meaningful addition, subtraction, and in some cases, more complex mathematical operations.
# 
# This requirement is crucial for parametric tests because these tests rely on calculations involving the mean and standard deviation, which assume that the differences between values are consistent and meaningful.
# 
# **<u>Examples</u>** **:** t-tests (independent and paired), ANOVA (Analysis of Variance), and linear regression.

# <a id="2"></a>
# 
# # 2) Non-Parametric Tests
# 
# **<u>Non-parametric tests</u>** **:** are statistical tests that do not assume a specific distribution for the data. They are used when the assumptions of parametric tests are not met, particularly when the data does not follow a normal distribution or when dealing with ordinal data or non-quantitative data.
# 
# Key Characteristics :
# 
# * **No Assumption of Distribution :** Does not require the data to follow a normal distribution.
# * **Flexible with Data Types :** Can handle ordinal data, ranks, and non-quantitative data.
# 
# **<u>Examples</u>** **:** Mann-Whitney U test, Wilcoxon signed-rank test, Kruskal-Wallis test, and Spearman's rank correlation.

# Parametric tests are more powerful when their assumptions are met because they use more information from the data, while non-parametric tests are more versatile and can be applied to a wider range of data types and distributions.
# 
# |                                    | Parametric Tests        | Non-Parametric Tests         | 
# | ---------------------------------- | ----------------------- | --------------------------   | 
# | **One Sample**                       | Simple t-Test           | Wilcoxon Test for One Sample |   
# | **Two dependent Samples**              | Paired Sample t-Test    | Wilcoxon Test                | 
# | **Two independent Samples**            | Unpaired Sample t-Test  | Mann-Whitney U Test          |   
# | **More than two independent Samples**  | One Factorial ANOVA     | Kruskal-Wallis Test          |   
# | **More than two dependent Samples**    | Repeated Measures ANOVA | Friedman Test                |   
# | **Correlation Between Two Variables**  | Pearson Correlation     | Spearman Correlation         |   
# 
# 

# <a id="3"></a>
# 
# # 3) Hypothesis Testing 
# 
# Hypotheses are claims (assertions) that can be verified or refuted using statistical methods. Hypothesis testing organizes these assertions into a framework, allowing us to use statistical evidence to evaluate their validity. This process enables us to determine whether the claims hold true or not.
# 
# In this notebook, I will demonstrate hypothesis testing with Python through several step-by-step examples. But first, let me briefly explain the hypothesis testing process. If you prefer, you can skip directly to the questions.
# 
# <a id="3.1"></a>
# ## 3.1) Defining Hypotheses
# 
# First, we need to identify the scientific question we aim to answer and formulate it as the Null Hypothesis $(H_0)$ and the Alternative Hypothesis ($H_{1}$ or $H_{a}$).
# 
# * **Null Hypothesis $H_0$ :** A default hypothesis that there is no effect or no difference. It is the hypothesis that the test seeks to reject.
# * **Alternative Hypothesis $H_{1}$ :** The hypothesis that there is an effect or a difference.
#  
# Remember that $H_0$ and $H_{1}$ **must be mutually exclusive,** and $H_{1}$ should not include equality :
# <center>
# $H_0 : \mu = x,\quad\quad H_{1}: μ \neq x$
# </center>
# <center>
# $H_0 : \mu \leq x,\quad\quad H_{1}: μ > x$
# </center>
# <center>
# $H_0 : \mu \geq x,\quad\quad H_{1}: μ < x$
# </center>
# 
# <a id="3.2"></a>
# ## 3.2) Assumption Check
# 
# To determine whether to use the **parametric** or **non-parametric** version of the test, we need to verify the following requirements :
# 
# * Observations in each sample are independent and identically distributed (iid).
# * Observations in each sample are normally distributed.
# * Observations in each sample have the same variance.
# 
# 
# <a id="3.3"></a>
# ## 3.3) Checking for Normality
# 
# ### For Small Samples
# For small sample sizes (typically $n \leq 50$), the Shapiro-Wilk test is often recommended due to its strong power in detecting deviations from normality. The Shapiro-Wilk test evaluates the null hypothesis that the data was drawn from a normally distributed population.
# 
# **Steps :**
# 1. **Shapiro-Wilk Test**:
#    - Conduct the Shapiro-Wilk test.
#    - The null hypothesis ($H_0$) states that the sample comes from a normal distribution.
#    - A low p-value (typically $\leq 0.05$) indicates that the null hypothesis can be rejected, suggesting the data is not normally distributed.
# 
# ### For Large Samples
# For larger sample sizes, other tests such as the Kolmogorov-Smirnov test, Anderson-Darling test, and Jarque-Bera test are more suitable due to their ability to handle larger datasets and focus on different aspects of the distribution.
# 
# **Steps:**
# 1. **Kolmogorov-Smirnov (K-S) Test**:
#    - Compares the sample distribution with a normal distribution.
#    - The null hypothesis ($H_0$) states that the sample comes from a specified distribution.
# 
# 2. **Anderson-Darling Test**:
#    - An enhancement of the K-S test, giving more weight to the tails of the distribution.
#    - Provides critical values for different significance levels to help assess normality.
# 
# 3. **Jarque-Bera Test**:
#    - Based on sample skewness and kurtosis.
#    - The null hypothesis ($H_0$) states that the data has skewness and kurtosis matching a normal distribution.
# 
# ### Visual Inspection
# In addition to formal tests, visual inspection methods can provide valuable insights into the distribution of the data:
# 1. **Histogram with KDE (Kernel Density Estimate)**:
#    - Provides a visual representation of the data distribution.
#    - Helps identify any skewness or kurtosis visually.
#    
# 2. **Q-Q Plot (Quantile-Quantile Plot)**:
#    - Plots the quantiles of the data against the quantiles of a normal distribution.
#    - If the data is normally distributed, the points should fall approximately along a straight line.
# 
# <a id="3.4"></a>
# ## 3.4) Checking for Homoscedasticity
# 
# Homoscedasticity refers to the assumption that the variance of errors or residuals is constant across all levels of the independent variable(s). It is a key assumption in regression analysis.
# 
# ### For Both Small and Large Samples
# There are three common tests for checking homoscedasticity: the Breusch-Pagan test, the White test, and Levene's test.
# 
# **Steps:**
# 1. **Breusch-Pagan Test**:
#    - Tests the null hypothesis ($H_0$) that the variance of the errors is constant.
#    - A low p-value (typically $\leq 0.05$) indicates heteroscedasticity (variance is not constant).
# 
# 2. **White Test**:
#    - Similar to the Breusch-Pagan test but more general, as it does not assume a specific form for the heteroscedasticity.
#    - Tests the null hypothesis ($H_0$) that the variance of the errors is constant.
# 
# 3. **Levene's Test**:
#    - Tests the null hypothesis ($H_0 $) that the variances are equal across groups.
#    - More robust to non-normal distributions compared to other tests.
#    - A low p-value (typically $\leq 0.05$) indicates that the variances are not equal (heteroscedasticity).
# 
# ### Visual Inspection
# Visual methods can also be employed to check for homoscedasticity:
# 1. **Residuals vs Fitted Plot**:
#    - Plots the residuals against the fitted values.
#    - If the variance is constant, the plot should show no clear pattern.
#    
# 2. **Histogram of Residuals**:
#    - Helps visualize the distribution of residuals.
#    - Ideally, the residuals should be normally distributed with constant variance.
# 
# By using both formal tests and visual inspection, you can effectively check for normality and homoscedasticity in your data, ensuring the assumptions for parametric tests are met.
# 

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest, normaltest, anderson, jarque_bera
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore") 

# Setting the style for plots
sns.set(style="whitegrid")

# Helper function to create example data
def create_example_data(size, mean=0, std=1):
    return np.random.normal(mean, std, size)

# Checking Normality
## For Small Samples
def check_normality_small_sample(data):
    print("Shapiro-Wilk Test:")
    stat, p = shapiro(data)
    print(f'Statistic: {stat}, p-value: {p}')
    return stat, p

## For Large Samples
def check_normality_large_sample(data):
    print("Kolmogorov-Smirnov Test:")
    stat, p = kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    print(f'Statistic: {stat}, p-value: {p}')
    
    print("\nAnderson-Darling Test:")
    result = anderson(data)
    print(f'Statistic: {result.statistic}')
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        print(f'Significance Level: {sl}, Critical Value: {cv}')
    
    print("\nJarque-Bera Test:")
    stat, p = jarque_bera(data)
    print(f'Statistic: {stat}, p-value: {p}')
    return stat, p

# Plotting Data for Visual Inspection
def plot_normality(data):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data, kde=True)
    plt.title("Histogram with KDE")
    
    plt.subplot(1, 2, 2)
    qqplot(data, line='s')
    plt.title("Q-Q Plot")
    
    plt.show()

# Checking Homoscedasticity
## For Both Small and Large Samples
def check_homoscedasticity(X, y):
    X = sm.add_constant(X)  # adding a constant for the intercept term
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    
    print("Breusch-Pagan Test:")
    bp_test = het_breuschpagan(residuals, X)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    print(dict(zip(labels, bp_test)))
    
    print("\nWhite Test:")
    white_test = het_white(residuals, X)
    print(dict(zip(labels, white_test)))
    
    # Plotting Residuals
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.residplot(x=X[:, 1], y=residuals, lowess=True, line_kws={'color': 'red'})
    plt.title("Residuals vs Fitted")
    
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True)
    plt.title("Histogram of Residuals")
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate example data
    small_sample_data = create_example_data(30)
    large_sample_data = create_example_data(500)
    
    # Check normality for small sample
    print("Normality Check for Small Sample:")
    check_normality_small_sample(small_sample_data)
    plot_normality(small_sample_data)
    
    # Check normality for large sample
    print("\nNormality Check for Large Sample:")
    check_normality_large_sample(large_sample_data)
    plot_normality(large_sample_data)
    
    # Generate example data for homoscedasticity check
    X = np.random.rand(100, 1) * 100
    y = 3 * X.squeeze() + np.random.randn(100) * 10
    
    # Check homoscedasticity
    print("\nHomoscedasticity Check:")
    check_homoscedasticity(X, y)


# In[2]:


# Import necessary libraries
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Example data
group1 = np.random.normal(50, 10, 100)  # Group 1 with mean 50 and std 10
group2 = np.random.normal(60, 15, 100)  # Group 2 with mean 60 and std 15
group3 = np.random.normal(55, 20, 100)  # Group 3 with mean 55 and std 20

# Perform Levene's test
stat, p = stats.levene(group1, group2, group3)

# Print the test statistic and p-value
print(f'Levene’s Test Statistic: {stat}')
print(f'p-value: {p}')

# Interpretation of the result
alpha = 0.05
if p < alpha:
    print("The variances are significantly different (reject H0)")
else:
    print("The variances are not significantly different (fail to reject H0)")

# Optional: Plotting the data
plt.figure(figsize=(10, 6))
plt.boxplot([group1, group2, group3], labels=['Group 1', 'Group 2', 'Group 3'])
plt.title("Boxplot of the Groups")
plt.ylabel("Values")
plt.show()


# ### Post-Hoc Test for Unequal Variances

# When Levene's test (or another test for equality of variances) indicates that the variances are significantly different, we can perform post-hoc tests to determine which specific groups have different variances. One such post-hoc test is the **Games-Howell post-hoc test,** which is designed for situations where variances are unequal and the sample sizes are unequal.
# 
# Here is an example of how to perform the Games-Howell post-hoc test in Python using the `pingouin` library :

# In[3]:


# Install necessary library
#!pip install pingouin


# In[4]:


# Import necessary libraries
import numpy as np
import pingouin as pg
import pandas as pd

# Example data
data = {
    'value': np.concatenate([np.random.normal(50, 10, 100), 
                             np.random.normal(60, 15, 100), 
                             np.random.normal(55, 20, 100)]),
    'group': np.concatenate([['Group 1'] * 100, 
                             ['Group 2'] * 100, 
                             ['Group 3'] * 100])
}

df = pd.DataFrame(data)

# Perform Levene's test
stat, p = stats.levene(df[df['group'] == 'Group 1']['value'], 
                       df[df['group'] == 'Group 2']['value'], 
                       df[df['group'] == 'Group 3']['value'])

print(f'Levene’s Test Statistic: {stat}')
print(f'p-value: {p}')

# Interpretation of the result
alpha = 0.05
if p < alpha:
    print("The variances are significantly different (reject H0)")
else:
    print("The variances are not significantly different (fail to reject H0)")

# Perform Games-Howell post-hoc test
posthoc = pg.pairwise_gameshowell(data=df, dv='value', between='group')
print(posthoc)

# Optional: Plotting the data
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.boxplot([df[df['group'] == 'Group 1']['value'], 
             df[df['group'] == 'Group 2']['value'], 
             df[df['group'] == 'Group 3']['value']], 
            labels=['Group 1', 'Group 2', 'Group 3'])
plt.title("Boxplot of the Groups")
plt.ylabel("Values")
plt.show()


# From the posthoc table, we can say that Group 2 and Group 3 variances are not significantly different.

# <a id="3.5"></a>
# ## 3.5) Selecting the Proper Test
# 
# Next, we need to choose the appropriate test. It's crucial to analyze the **number of groups being compared** and **whether the data are paired or not.** To determine if the data are matched, consider **whether the data were collected from the same individuals.** Based on these factors, you can select the appropriate test using the chart below.
# 
# 
# ![alt](tests_table.png)
# 
# <a id="3.6"></a>
# ## 3.6) Decision and Conclusion
# 
# After performing hypothesis testing, we obtain a **p-value** that indicates the significance of the test.
# 
# * **p-value :** is the probability of obtaining test results at least as extreme as the observed results, assuming that the null hypothesis is true. It quantifies how well the observed data fit the null hypothesis.
# 
# * **Threshold ($\alpha$-level)/The significance level ($\alpha$) :** is a pre-determined threshold set by the researcher, often 0.05.
# 
# If the p-value is smaller than the alpha level (the significance level $\alpha$), there is enough evidence to reject $H_0$. Otherwise, we fail to reject $H_0$. Remember, rejecting $H_0$ supports $H_{1}$, but failing to reject $H_0$ does not confirm $H_0$'s validity nor does it indicate that $H_{1}$ is incorrect.
# 
# ![](pvalue.jpg)

# <a id="4"></a>
# # 4) Tests Examples

# <a id="4.1"></a>
# 
# ## 4.1) Independent t-test / Two-sample t-test / Unpaired t-test
# 
# <a id="4.1.1"></a>
# 
# ### 4.1.1) Definition
# 
# * <u>**independent t-test**</u> **:** also known as the **two-sample t-test** or **unpaired t-test**, is a statistical test used to compare the means of two independent groups to determine if there is a statistically significant difference between them. This test is appropriate when you have two unrelated groups and you want to compare their means.
# 
# #### Assumptions
# 
# 1. **Independence :** The two samples must be independent of each other. This means that the data points in one group should not influence the data points in the other group.
# 2. **Normality :** The data in each group should be approximately normally distributed. This assumption can be relaxed with large sample sizes due to the Central Limit Theorem.
# 3. **Homogeneity of Variances (homoscedasticity) :**  The variances of the two groups should be approximately equal. This can be checked using tests such as Levene's test.
# 
# #### Hypotheses
# 
# - **Null Hypothesis $(H_0)$**: The means of the two groups are equal $(\mu_1 = \mu_2).$
# - **Alternative Hypothesis $(H_1)$**: The means of the two groups are not equal for a two-tailed test $(\mu_1 \neq \mu_2)$, or $(\mu_1 > \mu_2)$ or $(\mu_1 < \mu_2)$ for a one-tailed test.
# 
# #### Test Statistic
# 
# The test statistic for the independent t-test is calculated as follows:
# 
# $$
# t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
# $$
# 
# where :
# - $\bar{X}_1$ and $\bar{X}_2$ are the sample means,
# - $s_1^2$ and $s_2^2$ are the sample variances,
# - $n_1$ and $n_2$ are the sample sizes.
# 
# The degrees of freedom for the test are calculated using the following formula when assuming equal variances :
# $$
# df = n_1 + n_2 - 2
# $$

# <a id="4.1.2"></a>
# 
# ### 4.1.2) Example of Independent t-test 
# 
# ![](1.png)
# 
# A university professor transitioned from face-to-face classes to online lectures due to Covid-19. Subsequently, he uploaded recorded lectures to the cloud for students who chose to follow the course asynchronously. Despite this, he maintains a belief that students who attend the live classes and actively participate are more successful. To investigate this, he recorded the average grades of students at the end of the semester. The data is presented below.
# 
# synchronous = [93.2, 85.3, 82.9, 68.5, 79.9, 80.6, 80.9, 77.2, 81.5, 79.1, 72.3, 88.1, 86.8, 94.1, 83.6, 78.7, 77.4, 70.6, 89.1, 75.6, 73.9, 81.2]
# 
# asynchronous = [76.9, 71.3, 91.4, 71.5, 75.3, 84.9, 68.1, 70.5, 75.7, 71.2, 65.9, 72.4, 71.1, 78.5]
# 
# Let's perform hypothesis testing to determine the statistical significance of the professor's belief, using a significance level of 0.05 to assess the null and alternative hypotheses. But before conducting hypothesis testing, let's verify the relevant assumptions, and provide commentary on the obtained results.
# 
# #### <u>1. Defining Hypotheses</u>
# 
# Since the grades are obtained from different individuals, the data is unpaired.
# 
# * Null Hypothesis $(H_0) :$ The mean grade of students who attend live classes is less than or equal to the mean grade of students who watch recorded lectures.
# 
# $$H_0 : \mu_s \leq \mu_a$$
# 
# $s :$ synchronous
# 
# $a :$ asynchronous
# 
# 
# * Alternative Hypothesis $(H_1) :$ The mean grade of students who attend live classes is greater than the mean grade of students who watch recorded lectures.
# 
# $$H_1 : \mu_s > \mu_a$$
# 
# 
# #### <u>2. Assumption Check</u>
# 
# * **Normality :**
# 
#     * Null Hypothesis $(H_0) :$ The data is normally distributed.
#     * Alternative Hypothesis $(H_1) :$ The data is not normally distributed.
# 
# Assuming $\alpha=0.05,$ if the p-value is $>0.05,$ we can conclude that the data is normally distributed.
# For assessing normality, I employed the **Shapiro-Wilk test,** which is typically favored for smaller samples. However, other options such as the Kolmogorov-Smirnov and D’Agostino and Pearson’s tests are available. Please refer to [stats](https://docs.scipy.org/doc/scipy/reference/stats.html)'s documentation for further details.

# In[5]:


import numpy as np
from scipy import stats


# In[6]:


def normality_check(data):
    """Checks if the data is normally distributed"""
    
    test_stat_normality, p_value_normality=stats.shapiro(data)
    print("p value:%.4f" % p_value_normality)
    if p_value_normality < 0.05:
        print("Reject null hypothesis => The data is not normally distributed")
    else:
        print("Fail to reject null hypothesis => The data is normally distributed")


# In[7]:


sync = np.array([93.2, 85.3, 82.9, 68.5, 79.9, 80.6, 80.9, 77.2, 81.5, 79.1, 72.3, 88.1, 86.8,
                 94.1, 83.6, 78.7, 77.4, 70.6, 89.1, 75.6, 73.9, 81.2])
asyncr = np.array([76.9, 71.3, 91.4, 71.5, 75.3, 84.9, 68.1, 70.5, 75.7, 71.2, 65.9, 72.4, 71.1, 78.5])
normality_check(sync)
normality_check(asyncr)


# $H_0 :$ The variances of the samples are the same.
# 
# $H_1 :$ The variances of the samples are different.

# In[8]:


def variance_check_homogeneity(group1, group2):
    """Checks if two samples have the same variance"""
    
    test_stat_var, p_value_var = stats.levene(group1,group2)
    print("p value:%.4f" % p_value_var)
    if p_value_var < 0.05:
        print("Reject null hypothesis => The variances of the samples are different.")
    else:
        print("Fail to reject null hypothesis => The variances of the samples are same.")


# It tests the null hypothesis that the population variances are equal, a concept known as **homogeneity of variance** or **homoscedasticity.** If the resulting p-value from Levene’s test is less than the significance level (typically 0.05), it suggests that the observed differences in sample variances are unlikely to have occurred due to random sampling from a population with equal variances.
# 
# I used Levene’s test to check for variance homogeneity, but you can also use Bartlett’s test. For more information, visit: [scipy.stats.bartlett](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html#scipy.stats.bartlett)'s documentation.
# 

# In[9]:


variance_check_homogeneity(sync, asyncr)


# #### <u>3. Selecting the proper test</u>
# 
# Since the assumptions are satisfied, we can proceed with the parametric version of the test for two groups with unpaired data, which is **two-sample t-test**. For that, we use `scipy.stats`'s `ttest_ind`.

# In[10]:


ttest, p_value = stats.ttest_ind(sync, asyncr)
print("p value : %.8f" % p_value)
print("Since the hypothesis is one sided => use p-value/2 => p-value-one-sided : %.4f" %(p_value/2))
if p_value/2 < 0.05 :
    print("Reject the Null Hypothesis")
else :
    print("Fail to reject the Null Hypothesis")


# In[11]:


ttest


# #### <u>4. Decision and Conclusion</u>
# 
# At this significance level, there is sufficient evidence to conclude that the average grade of students who follow the course synchronously is higher than that of students who follow the course asynchronously.

# <a id="4.2"></a>
# 
# ## 4.2) Analysis of Variance (ANOVA)
# 
# <a id="4.2.1"></a>
# ### 4.2.1) Definition
# 
# * <u>**Analysis of Variance (ANOVA)**</u> **:** is a statistical method used to compare the means of three or more independent groups to determine if there are statistically significant differences between them. ANOVA tests the null hypothesis that the means of the groups are equal. The types of ANOVA are :
# 
# 1. **One-Way ANOVA :** Compares the means of one continuous dependent variable based on $\geq 3$ groups of one categorical variable.
# 2. **Two-Way ANOVA :** Compares the means of one continuous dependent variable based of $\geq 3$ groups of $2$ categorical variables.
# 3. **Repeated Measures ANOVA :** Analyzes the effects of one or more within-subjects factors (repeated measures) on a continuous dependent variable.
# 
# 
# #### Assumptions
# 
# 1. **Independence :** The observations within each group should be independent of each other.
# 2. **Normality :** The data in each group should be approximately normally distributed.
# 3. **Homogeneity of Variances (homoscedasticity) :** The variances of the groups should be approximately equal.
# 
# #### Hypotheses
# 
# - **Null Hypothesis $(H_0)$ :** The means of all groups are equal.
# - **Alternative Hypothesis $(H_1)$ :** At least one group mean is different from the others.
# 
# 
# #### Test Statistic
# 
# The test statistic for ANOVA follows an F-distribution. The F-statistic is calculated as the ratio of the between-group variance to the within-group variance :
# 
# $$
# F = \frac{{\text{Between-group variability}}}{{\text{Within-group variability}}}
# $$
# 
# #### Post Hoc Tests
# 
# After finding a significant result in ANOVA, post hoc tests are often performed to identify which specific group means differ from each other. Common post hoc tests include Tukey's HSD (Honestly Significant Difference), Bonferroni, Scheffe, and Dunnett's tests.

# <a id="4.2.2"></a>
# 
# ### 4.2.2) Example of ANOVA
# 
# ![](2.jpeg)
# 
# A pediatrician aims to examine the effect of formula consumption on the average monthly weight gain (in grams) of babies. To do this, he collected data from three different groups : 
# 
# * The first group consists of exclusively breastfed children (receiving only breast milk).
# * The second group includes children fed only with formula.
# * The third group comprises children who are both formula-fed and breastfed.
# 
# The data is presented below :
# 
# only_breast = [783.2, 720.1, 990.3, 725.4, 759.2, 905.6, 662.5, 691.4, 765.2, 718.6, 631.9, 732.8, 716.3, 808.9, 585.7, 682.3, 863.5]
# 
# only_formula = [900.1, 880.3, 938.4, 965.1, 958.7, 1060.2, 1047.8, 978.9, 894.1, 921.5, 1073.6, 953.4, 798.2, 858.7, 872.4, 1045.9, 918.2, 1161.8, 995.3]
# 
# both = [975.3, 658.1, 862.5, 708.9, 719.2, 718.4, 760.7, 893.1, 868.9, 806.3, 764.2, 802.7, 791.4, 874.5, 741.2, 797.8, 789.1, 794.3, 822.1, 820.3, 928.4, 793.5, 949.6]
# 
# Based on this information, let's conduct a hypothesis testing to determine whether there is a difference in the average monthly weight gain among the three groups, using a significance level of 0.05. If a significant difference is found, we'll perform further analysis to identify the cause of the difference. But before conducting hypothesis testing, let's verify the relevant assumptions.
# 
# #### <u>1. Defining Hypotheses</u>
# 
# * Null Hypothesis $H_0$ : The mean of the samples is the same
# 
# $$H_0 : \mu_1 = \mu_2 = \mu_3$$
# 
# * Alternative Hypothesis $H_1$ : At least one of them is different
# 
# $$H_1 : \mu_1 \neq \mu_2 \quad or \quad \mu_1 \neq \mu_3 \quad or \quad \mu_2 \neq \mu_3$$
# 
# #### <u>2. Assumptions Check</u>
# 
# * **Normality :**
# 
# $H_0 :$ "The data is normally distributed"
# 
# $H_1 :$ "The data is **NOT** normally distributed"
# 
# * **Homoscedasticity :**
# 
# $H_0 :$ "The variances of the samples are the same"
# 
# $H_1 :$ "The variances of the samples are different"

# In[12]:


breast_only = np.array([783.2, 720.1, 990.3, 725.4, 759.2, 905.6, 662.5, 691.4, 765.2, 718.6, 631.9, 732.8, 716.3, 808.9, 585.7, 682.3, 863.5])
formula_only = np.array([900.1, 880.3, 938.4, 965.1, 958.7, 1060.2, 1047.8, 978.9, 894.1, 921.5, 1073.6, 953.4, 798.2, 858.7, 872.4, 1045.9, 918.2, 1161.8, 995.3])
both = np.array( [975.3, 658.1, 862.5, 708.9, 719.2, 718.4, 760.7, 893.1, 868.9, 806.3, 764.2, 802.7, 791.4, 874.5, 741.2, 797.8, 789.1, 794.3, 822.1, 820.3, 928.4, 793.5, 949.6])


# In[13]:


# Normality Check :

normality_check(breast_only)
normality_check(formula_only)
normality_check(both)


# In[14]:


# Homoscedasticity check

stat, p_value_levene = stats.levene(breast_only, formula_only, both)
print("p-value : %.4f"%p_value_levene)
if p_value_levene < 0.05 :
    print("Reject null hypothesis => The variances of the samples are different")
else :
    print("Fail to reject the null hypothesis => The variances of the samples are the same")


# #### <u>3. Selecting the proper test</u>
# 
# Since the assumptions are satisfied, we can proceed with the parametric version of the test for **more than two groups** with **unpaired** data, which is ANOVA. For that, we use `scipy.stats`'s `f_oneway`.

# In[15]:


# Performing One-way ANOVA
F_test, p_value = stats.f_oneway(breast_only, formula_only, both)
print("p-value : %.6f" %p_value)
if p_value < 0.05 :
    print("Reject the null hypothesis")
else :
    print("Fail to reject the null hypothesis")


# #### <u>4. Decision and Conclusion</u>
# 
# At this significance level (of 0.05), it can be concluded that at least one of the groups has a different average monthly weight gain. To determine which group or groups are causing the difference, we need to conduct a **posthoc test** or **pairwise comparison.**
# 
# **Note :** To prevent family-wise p-value inflation, I applied the [Bonferroni adjustment](https://github.com/Dahamey/Hypothesis-Testing/blob/main/Python/Parametric%20Test%20Vs%20Non-Parametric%20Test/Family-Wise%20P-Value%20Inflation%20and%20Bonferroni%20Adjustment.ipynb). Alternatively, you can explore other options here: [scikit_posthocs.posthoc_ttest documentation](https://scikit-posthocs.readthedocs.io/en/latest/generated/scikit_posthocs.posthoc_ttest.html)
# 
# Pairwise T-test for multiple comparisons of independent groups, can be utilized after conducting a parametric ANOVA to perform pairwise comparisons.

# In[23]:


#pip install scikit-posthocs


# In[16]:


import scikit_posthocs as sp

df_posthoc = sp.posthoc_ttest([breast_only, formula_only, both],
                             equal_var = True,   # Homoscedasticity verified!
                             p_adjust = "bonferroni")

groups = ["Breast Only", "Formula Only", "Both"]
df_posthoc.columns = groups
df_posthoc.index = groups

df_posthoc.style.applymap(lambda x: "background-color:red" if x<0.05 else "background-color:white")


# At this significance level, it can be concluded that the "Formula only" group differs from both the "Breast Only" and "Both" groups.
# 

# <a id="4.3"></a>
# 
# ## 4.3) Mann Whitney U / Wilcoxon rank-sum test
# 
# <a id="4.3.1"></a>
# 
# ### 4.3.1) Definition
# 
# * <u>**The Mann-Whitney U test**</u> **:** also known as the **Wilcoxon rank-sum test,** is a non-parametric statistical test used to compare the means of two independent groups when the assumptions of the independent t-test are not met (e.g., when the data is not normally distributed or the variances are not equal).
# 
# 
# #### Assumptions
# 
# The Mann-Whitney U test does not assume normality or homogeneity of variances. However, it does **assume that the observations in each group are independent.**
# 
# #### Hypotheses
# 
# - **Null Hypothesis $(H_0)$ :** The distributions of both groups are equal.
# - **Alternative Hypothesis $(H_1)$ :** The distributions of both groups are not equal.
# 
# #### Test Statistic
# 
# The test statistic for the Mann-Whitney U test is based on the ranks of the observations in the combined sample of the two groups. The $U$ statistic is calculated as the sum of the ranks of the observations in one group, and it is compared to a critical value to determine statistical significance. Here's a step by step explanation :
# 
# 
# 1. **Combine the Samples :** First, combine all the observations from both groups into a single sample.
# 
# 2. **Rank the Observations :** Assign ranks to the combined sample. The smallest observation gets rank 1, the next smallest gets rank 2, and so on. If there are tied values (observations with the same value), each tied value is assigned the average of the ranks they would have received if they were not tied.
# 
# 3. **Sum of Ranks :** Calculate the sum of the ranks for the observations in each group. These are often referred to as $R_1$ and $R_2$.
# 
# 4. **U Statistic Calculation :** The U statistic is calculated using the following formulas:
# 
#    $$
#    U_1 = R_1 - \frac{n_1(n_1 + 1)}{2}
#    $$
# 
#    $$
#    U_2 = R_2 - \frac{n_2(n_2 + 1)}{2}
#    $$
# 
#    where $n_1$ and $n_2$ are the sample sizes of the two groups. The test statistic $U$ is the smaller of $U_1$ and $U_2$.
# 
# 5. **Determine the Significance :** The U statistic is then compared to a critical value from the U distribution, or a p-value is calculated. If the p-value is less than the chosen significance level (e.g., 0.05), you reject the null hypothesis that the two samples come from the same distribution.
# 
# 

# <a id="4.3.2"></a>
# 
# ### 4.3.2) Example of Mann Whitney U test
# 
# ![](3.jpg)
# 
# A human resource specialist at a technology company is interested in comparing the overtime hours of different teams. To explore whether there's a difference between the overtime hours of the software development team and the test team, the specialist randomly selected 17 employees from each team and recorded their weekly average overtime hours. The data is provided below.
# 
# tester_team = [6.5, 7.0, 1.8, 2.2, 2.4, 1.9, 1.7, 6.3, 2.6, 2.2, 12.1, 1.6, 5.1, 3.3, 9.0, 2.5, 4.0]
# 
# developer_team = [2.2, 2.0, 1.5, 2.1, 8.5, 2.3, 3.0, 4.0, 3.5, 2.4, 3.0, 6.0, 12.0, 4.0, 2.0, 1.3, 3.3]
# 
# Based on this data, we'll perform hypothesis testing to examine whether there exists a difference in the overtime hours between two teams, utilizing a significance level of 0.05. Buit before proceeding with hypothesis testing, let's ensure to verify the associated assumptions.
# 
# #### <u>1. Defining Hypotheses</u>
# 
# * Null Hypothesis $H_0$ : The mean of the two samples is the same
# 
# $$H_0 : \mu_t = \mu_d$$
# 
# * Alternative Hypothesis $H_1$ : The two samples have different means
# 
# $$H_1 : \mu_t \neq \mu_d$$
# 
# 
# #### <u>2. Assumptions Check</u>
# 
# * **Normality :**
# 
# $H_0 :$ "The data is normally distributed" (The overtime data for both teams should follow a normal distribution.)
# 
# $H_1 :$ "The data is **NOT** normally distributed"
# 
# * **Homoscedasticity :**
# 
# $H_0 :$ "The variances of the samples are the same" (The variance in overtime hours should be approximately equal between the two teams.)
# 
# $H_1 :$ "The variances of the samples are different"

# In[17]:


tester_team = np.array([6.5, 7.0, 1.8, 2.2, 2.4, 1.9, 1.7, 6.3, 2.6, 2.2, 12.1, 1.6, 5.1, 3.3, 9.0, 2.5, 4.0])
developer_team = np.array([2.2, 2.0, 1.5, 2.1, 8.5, 2.3, 3.0, 4.0, 3.5, 2.4, 3.0, 6.0, 12.0, 4.0, 2.0, 1.3, 3.3])

# Normality check
normality_check(tester_team)
normality_check(developer_team)

# Homoscedasticity check
variance_check_homogeneity(tester_team, developer_team)


# #### <u>3. Selecting the proper test</u>
# 
# Since there are two groups with data collected from different individuals, it's **unpaired.** However, the normality assumption is not met. Therefore, we should use the non-parametric version of a two-group comparison for unpaired data, which is the **Mann-Whitney U test.** For that, we use `scipy.stats`'s `mannwhitneyu` function.

# In[18]:


ttest, p_value = stats.mannwhitneyu(x = tester_team,
                                  y = developer_team,
                                  alternative = "two-sided")

print("p-value : %.4f"%p_value)
if p_value < 0.05 :
    print("Reject the null hypothesis")
else :
    print("Fail to reject the null hypothesis")


# #### <u>4. Decision and Conclusion</u>
# 
# At this significance level, there is no statistically significant difference between the average overtime hours of the two teams.

# <a id="4.4"></a>
# 
# ## 4.4) Kruskal-Wallis test
# 
# <a id="4.4.1"></a>
# 
# ### 4.4.1) Definition
# 
# * <u>**Kruskal-Wallis test**</u> **:** is a non-parametric statistical test used to compare the means of three or more independent groups when the assumptions of the one-way ANOVA are not met (e.g., when the data is not normally distributed or the variances are not equal). It is often referred to as the non-parametric equivalent of the one-way ANOVA.
# 
# #### Assumptions
# 
# The Kruskal-Wallis test does not assume normality or homogeneity of variances. However, it does **assume that the observations in each group are independent.**
# 
# #### Hypotheses
# 
# - **Null Hypothesis $(H_0)$ :** The distributions of all groups are equal.
# - **Alternative Hypothesis $(H_1)$ :** At least one group's distribution is different from the others.
# 
# #### Test Statistic
# 
# The test statistic for the Kruskal-Wallis test is based on the ranks of the observations in the combined sample of the groups. It compares the average ranks between groups to determine if there are significant differences. Here's a step by step explanation :
# 
# 1. **Combine the Samples :** Combine all the observations from all the groups into a single sample.
# 
# 2. **Rank the Observations :** Assign ranks to the combined sample. The smallest observation gets rank 1, the next smallest gets rank 2, and so on. If there are tied values (observations with the same value), each tied value is assigned the average of the ranks they would have received if they were not tied.
# 
# 3. **Sum of Ranks :** Calculate the sum of the ranks for the observations in each group. Denote the sum of ranks for the $i$-th group as $R_i$.
# 
# 4. **Kruskal-Wallis Statistic Calculation**: The Kruskal-Wallis test statistic $H$ is calculated using the formula:
# 
#    $$
#    H = \left( \frac{12}{N(N+1)} \sum_{i=1}^k \frac{R_i^2}{n_i} \right) - 3(N+1)
#    $$
# 
#    where:
#    - $N$ is the total number of observations across all groups.
#    - $k$ is the number of groups.
#    - $n_i$ is the number of observations in the $i$-th group.
#    - $R_i$ is the sum of ranks for the $i$-th group.
# 
# 5. **Determine the Significance :** The test statistic $H$ follows a chi-square distribution with $k-1$ degrees of freedom. Compare $H$ to the critical value from the chi-square distribution, or calculate the p-value. If the p-value is less than the chosen significance level (e.g., 0.05), you reject the null hypothesis that the groups have the same distribution.
# 
# 

# <a id="4.4.2"></a>
# 
# ### 4.4.2) Example of Kruskal-Wallis test
# 
# ![](4.jpg)
# 
# An e-commerce company frequently advertises its campaigns on YouTube, Instagram, and Facebook. However, the new manager is curious about whether there is any difference in the number of customers attracted by these platforms. Consequently, she began utilizing "Adjust", an application that provides insights to find out where the users came from. The daily numbers reported from Adjust for each platform are as follows :
# 
# Youtube = [1905, 1885, 1945, 2135, 2045, 2125, 2120, 2150, 2030, 1980, 1950, 2140, 2155, 1945, 2135]
# 
# Instagram = [2295, 2365, 2205, 2225, 2195, 2425, 2380, 2415, 2335, 2355, 2245, 2400, 2235, 2270, 2275]
# 
# Facebook = [2125, 2530, 2110, 2555, 2285, 2375, 2450, 2315, 2180, 2120, 2055, 2435, 2270, 2105, 2535]
# 
# Based on this data, we'll perform hypothesis testing to examine whether there are differences in the average customer acquisition across the three platforms, using a significance level of 0.05. Should a significant difference be observed, we'll conduct additional analysis to detect the underlying cause. But prior to hypothesis testing, let's ensure to validate the relevant assumptions.
# 
# #### <u>1. Defining Hypotheses</u>
# 
# * Null Hypothesis $H_0$ : The mean of the samples is the same
# 
# $$H_0 : \mu_Y = \mu_I = \mu_F$$
# 
# * Alternative Hypothesis $H_1$ : At least one of them is different
# 
# $$H_1 : \mu_Y \neq \mu_I \quad or \quad \mu_Y \neq \mu_F \quad or \quad \mu_I \neq \mu_F$$
# 
# 
# #### <u>2. Assumptions Check</u>
# 
# * **Normality :**
# 
# $H_0 :$ "The data is normally distributed" (The number of customers acquired from each platform should follow a normal distribution.)
# 
# $H_1 :$ "The data is **NOT** normally distributed"
# 
# * **Homoscedasticity :**
# 
# $H_0 :$ "The variances of the samples are the same" (The variance in the number of customers acquired should be approximately equal between the three platforms.)
# 
# $H_1 :$ "The variances of the samples are different"

# In[19]:


youtube = np.array([1905, 1885, 1945, 2135, 2045, 2125, 2120, 2150, 2030, 1980, 1950, 2140, 2155, 1945, 2135])
instagram = np.array([2295, 2365, 2205, 2225, 2195, 2425, 2380, 2415, 2335, 2355, 2245, 2400, 2235, 2270, 2275])
facebook = np.array([2125, 2530, 2110, 2555, 2285, 2375, 2450, 2315, 2180, 2120, 2055, 2435, 2270, 2105, 2535])

# Normality check
normality_check(youtube)
normality_check(instagram)
normality_check(facebook)

# Homoscedasticity check
stat, p_value_levene = stats.levene(youtube, instagram, facebook)
print("p-value : %.4f"%p_value_levene)
if p_value_levene < 0.05 :
    print("Reject the null hypothesis => The variances of the samples are different.")
else :
    print("Fail to reject the null hypothesis => The variances of the samples are the same.")


# #### <u>3. Selecting the proper test</u>
# 
# Since the assumptions of normality and variance homogeneity are not met, we should employ the non-parametric version of ANOVA for unpaired data, as the data is collected from different sources, which is the **Kruskal-Wallis test.** For that, we use `scipy.stats`'s `kruskal` function.

# In[20]:


F_test, p_value = stats.kruskal(youtube, instagram, facebook)
print("p-value : %.6f"%p_value)
if p_value <0.05 :
    print("Reject the null hypothesis")
else : 
    print("Fail to reject the null hypothesis")


# #### <u>4. Decision and Conclusion</u>
# 
# At this significance level, there is evidence to suggest that at least one of the average customer acquisition numbers differs across the platforms.
# 
# **Note :** Due to the non-normality of the data, the **non-parametric version of the posthoc test** is employed. For that, we use `scikit_posthocs`'s `posthoc_mannwhitney` function.

# In[21]:


import scikit_posthocs as sp

df_posthoc = sp.posthoc_mannwhitney([youtube, instagram, facebook],
                                   p_adjust = "bonferroni")

company_names = ["Youtube", "Instagram", "Facebook"]
df_posthoc.columns = company_names
df_posthoc.index = company_names

df_posthoc.style.applymap(lambda x : "background-color:red" if x<0.05 else "background-color:white")


# At this significance level, it can be concluded that the average number of customers coming from YouTube is significantly different from the average number of customers coming from Instagram or Facebook, being notably smaller.

# 
# <a id="4.5"></a>
# 
# ## 4.5) Dependent t-Test / Paired t-Test
# 
# <a id="4.5.1"></a>
# 
# ### 4.5.1) Definition
# 
# * <u>**Dependent t-test**</u> **:** also known as the **paired t-test,** is a statistical test used to compare the means of two related groups to determine if there is a statistically significant difference between them. This test is appropriate when the same subjects are measured before and after an intervention or treatment, or when the measurements are matched in some way.
# 
# #### Assumptions
# 
# 1. **Dependent Observations :** The observations in each group should be dependent or paired in some way. Typically, this means that each subject is measured under two different conditions.
# 2. **Normality :** The differences between the paired observations (i.e., the difference scores) should be approximately normally distributed. This can be assessed using statistical tests such as the Shapiro-Wilk test or by visual inspection of a Q-Q plot.
# 3. **Homogeneity of Differences :** The differences between pairs of observations should have a constant variance. . This assumption is less commonly mentioned but ensures that the differences do not show a systematic pattern.
# 
# **NOTE :** Homoscedasticity (equal variances) is not an assumption for the Dependent t-Test because it compares the means of two related groups, and thus, between-group variability is not a factor. This is in contrast to the Independent t-Test, where homoscedasticity is an important assumption.
# 
# #### Hypotheses
# 
# - **Null Hypothesis $(H_0)$ :** The mean difference between paired observations is zero $(\mu_d =0)$.
# - **Alternative Hypothesis $(H_1)$ :** The mean difference between paired observations is not zero $(\mu_d \neq 0)$ for a two-tailed test, or $(\mu_d > 0)$ or $(\mu_d < 0)$ for a one-tailed test.
# 
# #### Test Statistic
# 
# The test statistic for the dependent t-test is calculated as follows :
# 
# $$
# t = \frac{\bar{d}}{\frac{s_d}{\sqrt{n}}}
# $$
# 
# where:
# - $\bar{d} = \bar{X_2}-\bar{X_1}$ is the mean difference between paired observations,
# - $s_d$ is the standard deviation of the differences between paired observations,
# - $n$ is the number of pairs.
# 

# <a id="4.5.2"></a>
# 
# ### 4.5.2) Example of a Dependent t-Test (Paired t-Test)
# 
# ![](5.jpg)
# 
# Eighteen students diagnosed with high cholesterol at the University Health Center were advised by healthcare personnel about the risks associated with high cholesterol and were prescribed a diet program. After one month, the patients returned for a follow-up, during which their cholesterol levels were reevaluated to test whether there is a change in the cholesterol levels of the patients after the diet intervention.
# 
# Based on this information, we'll perform hypothesis testing to determine whether there is a reduction in the cholesterol levels of the patients following the diet, using a significance level of 0.05. but prior to hypothesis testing, let's assess the relevant assumptions, and provide commentary on the results.
# 
# results_before_diet = [226, 233, 220, 252, 254, 220, 242, 227, 258, 222, 240, 238, 237, 230, 274, 252, 236, 228]
# 
# results_after_diet = [196, 197, 215, 189, 245, 207, 224, 198, 212, 209, 187, 204, 201, 219, 191, 197, 193, 220]
# 
# Since the same individuals are being measured before and after the diet, the observations are paired.
# 
# #### <u>1. Defining Hypotheses</u>
# 
# * Null Hypothesis $H_0$ : The true mean difference $\mu_d = \mu_{\text{after}} - \mu_{\text{before}}$ is equal to or bigger than zero.
# 
# $$H_0 : \mu_d \geq 0$$
# 
# * Alternative Hypothesis $H_1$ : The true mean difference is smaller than zero
# 
# $$H_1 : \mu_d <0 $$
# 
# 
# #### <u>2. Assumptions Check</u>
# 
# * **Continuity :**
# 
# The dependent variable must be **continuous (interval/ratio).**
# 
# * **Independence :**
# 
# The observations are **independent of one another.**
# 
# * **Normality :**
# 
# The dependent variable should be approximately **normally distributed**
# 
# $H_0 :$ "The data is normally distributed" (The cholesterol level data should follow a normal distribution.)
# 
# $H_1 :$ "The data is **NOT** normally distributed"
# 

# In[23]:


before_diet = np.array([226, 233, 220, 252, 254, 220, 242, 227, 258, 222, 240, 238, 237, 230, 274, 252, 236, 228])
after_diet = np.array([196, 197, 215, 189, 245, 207, 224, 198, 212, 209, 187, 204, 201, 219, 191, 197, 193, 220])

# normality check

normality_check(before_diet)
normality_check(after_diet)


# #### <u>3. Selecting the proper test</u>
# 
# The data is paired as it is collected from the same individuals, and the assumptions are satisfied. Therefore, we can employ the **dependent t-test.** For that we use `scipy.stats`'s `ttest_rel` function.

# In[25]:


# ttest_rel : Calculate the t-test on TWO RELATED samples of scores, a and b.
test_stat, p_value_paired = stats.ttest_rel(a = before_diet, b = after_diet)

#  since by default we have : alternative='two-sided'
print("p-value : %.6f"%p_value_paired, "one-tailed p-value : %.6f"%(p_value_paired/2))



# 'greater': the mean of the distribution underlying the first sample
# is greater than the mean of the distribution underlying the second sample.
test_stat, p_value_paired = stats.ttest_rel(a = before_diet, b = after_diet, alternative="greater")
print("One-sided p-value : %.6f"%p_value_paired)

if p_value < 0.05 :
    print("Reject null hypotheses")
else :
    print("Fail to reject null hypothesis")


# #### <u>4. Decision and Conclusion</u>
# 
# At this significance level, there is sufficient evidence to conclude that the mean cholesterol level of patients has decreased after the diet.

# <a id="4.6"></a>
# 
# ## 4.6) Wilcoxon Signed-Rank test
# 
# <a id="4.6.1"></a>
# 
# ### 4.6.1) Definition
# 
# * <u>**Wilcoxon Signed-Rank test**</u> **:** is a non-parametric statistical test used to compare the means of two related groups to determine if there is a statistically significant difference between them. This test is appropriate when the same subjects are measured before and after an intervention or treatment, or when the measurements are matched in some way.
# 
# #### Assumptions
# 
# 1. **Continuous Data :** The data should be measured on at least an ordinal scale.
# 2. **Dependent Observations :** The observations in each group should be dependent or paired in some way.
# 3. **Random Sampling :** The data should be collected through a random sampling method.
# 
# 
# #### Hypotheses
# 
# - **Null Hypothesis $(H_0)$ :** There is no difference between the paired observations $(\mu_d =0)$.
# - **Alternative Hypothesis $(H_1)$ :** There is a difference between the paired observations $(\mu_d \neq 0)$ for a two-tailed test, or $(\mu_d > 0)$ or $(\mu_d < 0)$ for a one-tailed test.
# 
# #### Test Statistic
# 
# The test statistic for the Wilcoxon Signed-Rank test is based on the ranks of the absolute differences between paired observations. It compares the sum of the positive ranks to the sum of the negative ranks. Here's a step by step explanation :
# 
# 1. **Paired Observations :** First, identify paired observations from your dataset. Each pair consists of two related measurements or observations taken under different conditions, treatments, or times.
# 
# 2. **Compute Differences :** Calculate the differences between paired observations. Let $ D_i $ denote the difference between the ith pair of observations.
# 
# 3. **Absolute Differences and Ranks :** Compute the absolute differences $|D_i|$ for each pair. Then, rank these absolute differences from smallest to largest, assigning ranks based on their magnitude.
# 
# 4. **Sum of Ranks :** Calculate the sum of ranks for the positive and negative signed ranks separately. This involves summing the ranks of $|D_i|$ where $ D_i > 0 $ (positive differences) and where $ D_i < 0 $ (negative differences).
# 
# 5. **Test Statistic :** The test statistic $ T^+ $ (for positive differences) or $ T^- $ (for negative differences) is computed based on the smaller of these two sums of ranks. The decision on which test statistic to use depends on whether you are testing for a one-sided or two-sided alternative hypothesis.
# 
# 6. **Null Hypothesis :** Formulate the null hypothesis $ H_0 $ and alternative hypothesis $ H_1 $ based on whether the mean/median of the differences is equal to zero (or a specified value) versus not equal to zero.
# 
# 7. **Critical Value or p-value :** Determine the critical value from the Wilcoxon Signed-Rank table or compute the p-value using the distribution of $ T^+ $ or $ T^- $ under the null hypothesis.
# 
# 8. **Decision :** Compare the computed test statistic with the critical value (for a given significance level) or the p-value. If the test statistic exceeds the critical value or if the p-value is less than the significance level (commonly $ \alpha = 0.05 $), reject the null hypothesis. Otherwise, fail to reject the null hypothesis.
# 

# <a id="4.6.2"></a>
# 
# ### 4.6.2) Example of Wilcoxon Signed-Rank test
# 
# ![](6.png)
# 
# A venture capitalist is considering investing in a startup that offers data compression without any loss in quality. However, two competitors, "Pied Piper" and "EndFrame", are also in the market. Initially, he speculated that EndFrame's performance might be superior but wished to verify this before making any investment decisions. Consequently, he provided the same files to each company for compression and recorded their performance scores. The data is provided below.
# 
# pied_piper = [4.55, 4.53, 5.45, 4.65, 5.39, 5.53, 5.51, 5.61, 3.84, 3.95, 5.42, 3.91, 5.29, 5.15, 4.37, 4.26, 5.23]
# 
# endframe = [4.25, 3.91, 3.99, 4.05, 3.85, 3.98, 3.98, 3.70, 4.14, 4.08, 3.88, 3.95, 4.06, 3.94, 3.94, 3.75, 4.07]
# 
# Based on this data, we will perform the relevant hypothesis testing using a significance level of 0.05. But prior to conducting the hypothesis testing, let's ensure to validate the relevant assumptions, and provide commentary on the results afterward.
# 
# Since the performance scores are derived from the same set of files, the data is paired.
# 
# #### <u>1. Defining Hypotheses</u>
# 
# * Null Hypothesis $H_0$ : The true mean difference $\mu_d = \mu_{\text{EndFrame}} - \mu_{\text{Pied Paper}}$ is equal to or bigger than zero.
# 
# $$H_0 : \mu_d \geq 0$$
# 
# * Alternative Hypothesis $H_1$ : The true mean difference is smaller than zero
# 
# $$H_1 : \mu_d <0 $$
# 
# 
# #### <u>2. Assumptions Check</u>
# 
# * **Continuity :**
# 
# The dependent variable must be **continuous (interval/ratio).**
# 
# * **Independence :**
# 
# The observations are **independent of one another.** (The performance scores for each company should be independent of each other.)
# 
# * **Normality :**
# 
# The dependent variable should be approximately **normally distributed**
# 
# $H_0 :$ "The data is normally distributed" (The performance scores for both companies should follow a normal distribution.)
# 
# $H_1 :$ "The data is **NOT** normally distributed"

# In[26]:


pied_piper = np.array([4.55, 4.53, 5.45, 4.65, 5.39, 5.53, 5.51, 5.61, 3.84, 3.95, 5.42, 3.91, 5.29, 5.15, 4.37, 4.26, 5.23])
endframe = np.array([4.25, 3.91, 3.99, 4.05, 3.85, 3.98, 3.98, 3.70, 4.14, 4.08, 3.88, 3.95, 4.06, 3.94, 3.94, 3.75, 4.07])

# normality check
normality_check(pied_piper)
normality_check(endframe)


# #### <u>3. Selecting the proper test</u>
# 
# Since the normality assumption is not met, we should use the non-parametric version of the paired test, which is the **Wilcoxon Signed-Rank test.** For that we use `scipy.stats`'s `wilcoxon` function.

# In[27]:


# alternative ="two-sided" by default in stats.wilcoxon
test, p_value = stats.wilcoxon(x = endframe, y = pied_piper)
print("p-value : %.6f"%p_value, "=> one-tailed p-value : %.6f"%(p_value/2))


# In[28]:


# `d` represent the difference between the paired samples: `d = x - y`
# 'less': the distribution underlying ``d`` is stochastically less than a distribution symmetric about zero.

test, p_value = stats.wilcoxon(x = endframe, y = pied_piper, alternative = "less")
print("One-sided p-value : %.6f"%p_value)
if p_value < 0.05 :
    print("Reject null hypotheses")
else :
    print("Fail to reject null hypothesis")


# #### <u>4. Decision and Conclusion</u>
# 
# At this significance level, there is sufficient evidence to conclude that the performance of "Pied Piper" is superior to that of "EndFrame".

# <a id="4.7"></a>
# 
# ## 4.7) Friedman $\chi$² (Chi-Square)
# 
# <a id="4.7.1"></a>
# 
# ### 4.7.1) Definition
# 
# * <u>**Friedman's Chi-Square Test**</u> **:** also known simply as **Friedman Test** generally refer to the same statistical test. The Friedman Test is a non-parametric alternative to the repeated measures ANOVA (Analysis of Variance) test, used to determine whether there are statistically significant differences between the means of three or more paired groups. It's named after the statistician Milton Friedman. It is often used when the data is ordinal and the observations are dependent.
# 
# #### Assumptions
# 
# 1. **Ordinal Data :** The data should be measured on an ordinal scale. This means that the measurements can be ranked, but the intervals between the ranks may not be equal.
# 2. **Dependent Samples :** The test requires that you have a single group of participants or objects that are measured under three or more conditions or treatments. Each participant/object must have been measured under all conditions.
# 3. **Related Samples Independence :** The observations within each group must be independent of each other, and the groups should be independent of each other as well.
# 3. **Random Sampling**: The data should be collected through a random sampling method.
# 
# 
# #### Hypotheses
# 
# - **Null Hypothesis $(H_0)$ :** There is no difference between the paired groups.
# - **Alternative Hypothesis $(H_1)$ :** There is a difference between the paired groups.
# 
# #### Test Statistic
# 
# The test statistic for Friedman's Chi-Square Test is based on the ranks of the observations within each group. It compares the mean ranks of the groups to determine if there are significant differences. Here's a step by step explanation :
# 
# 
# 1. **Ranking Observations :** First, rank the observations within each group separately. The ranks are assigned based on the order of the observations from smallest to largest across all groups.
# 
# 2. **Calculate Average Ranks :** Compute the average rank for each group. This is the mean of the ranks assigned to observations within each group.
# 
# 3. **Test Statistic :** Calculate the Friedman test statistic $\chi^2$ using the formula:
#    $$
#    \chi^2 = \frac{12}{N(k+1)} \left[ \sum_{j=1}^{k} R_j^2 - \frac{N(k+1)^2}{4} \right]
#    $$
#    where:
#    - $N$ is the total number of observations (across all groups).
#    - $ k$ is the number of groups.
#    - $ R_j $ is the sum of ranks for group $ j$.
# 
# 4. **Degrees of Freedom :** The degrees of freedom $df$ for Friedman's Chi-Square test is $ k-1 $, where $ k$ is the number of groups.
# 
# 5. **Null Hypothesis :** Formulate the null hypothesis $ H_0$ and alternative hypothesis $H_1$ based on whether the mean ranks of the groups are equal or not.
# 
# 6. **Critical Value or p-value :** Determine the critical value from the Chi-Square distribution table or compute the p-value associated with the Chi-Square statistic under the null hypothesis.
# 
# 7. **Decision :** Compare the computed Chi-Square statistic with the critical value (for a given significance level) or compare the p-value with the significance level (commonly $\alpha = 0.05 $). If the test statistic exceeds the critical value or if the p-value is less than $\alpha$, reject the null hypothesis. Otherwise, fail to reject the null hypothesis.
# 
# 
# 

# <a id="4.7.2"></a>
# 
# ### 4.7.2) Example of Friedman $\chi$² (Chi-Square)
# 
# ![](7.png)
# 
# The researchers wanted to investigate whether there is a difference between the performance of the methodology they developed, $C,$ and the baseline methods $A$ and $B.$ To do this, the researchers conducted various experiments and recorded the achieved accuracy of each method on test sets. The table below displays the accuracy achieved by each method. It's important to note that the same train and test sets were used for each method.
# 
# | Experiment |    A    |    B    |    C    |
# |------------|---------|---------|---------|
# | E1         |  89.7   |  90.0   |  91.5   |
# | E2         |  89.8   |  90.1   |  90.7   |
# | E3         |  88.9   |  88.8   |  90.3   |
# | E4         |  88.6   |  88.9   |  90.4   |
# | E5         |  89.5   |  89.9   |  90.2   |
# | E6         |  89.6   |  90.0   |  90.3   |
# | E7         |  89.3   |  89.0   |  90.2   |
# | E8         |  89.4   |  89.2   |  90.3   |
# 
# 
# Based on this data, we will perform hypothesis testing to assess whether there exists a discrepancy in the performance of the methods, utilizing a significance level of 0.05. In case of a significant difference, we will conduct additional analysis to pinpoint which method is responsible for the disparity. Bu prior to hypothesis testing, let's validate the relevant assumptions, and provide commentary on the outcomes.
# 
# Since the accuracy scores are obtained from the same test sets, the data is paired.
# 
# #### <u>1. Defining Hypotheses</u>
# 
# * Null Hypothesis $H_0$ : The mean of the samples is the same
# 
# $$H_0 : \mu_A = \mu_B = \mu_C$$
# 
# * Alternative Hypothesis $H_1$ : At least one of them is different
# 
# $$H_1 : \mu_A \neq \mu_B \quad or \quad \mu_A \neq \mu_C \quad or \quad \mu_B \neq \mu_C$$
# 
# 
# #### <u>2. Assumptions Check</u>
# 
# * **Normality :**
# 
# $H_0 :$ "The data is normally distributed" (The accuracy data for each method should follow a normal distribution.)
# 
# $H_1 :$ "The data is **NOT** normally distributed"
# 
# * **Homoscedasticity :**
# 
# $H_0 :$ "The variances of the samples are the same" (The variances in accuracy should be approximately equal between the methods.)
# 
# $H_1 :$ "The variances of the samples are different"

# In[29]:


import pandas as pd
from io import StringIO

data = """
| Experiment |    A    |    B    |    C    |
|------------|---------|---------|---------|
| E1         |  89.7   |  90.0   |  91.5   |
| E2         |  89.8   |  90.1   |  90.7   |
| E3         |  88.9   |  88.8   |  90.3   |
| E4         |  88.6   |  88.9   |  90.4   |
| E5         |  89.5   |  89.9   |  90.2   |
| E6         |  89.6   |  90.0   |  90.3   |
| E7         |  89.3   |  89.0   |  90.2   |
| E8         |  89.4   |  89.2   |  90.3   |

"""

df = pd.read_csv(StringIO(data), sep="|", skipinitialspace=True, usecols=['Experiment ', 'A    ', 'B    ', 'C    '])[1:]
df.columns = ['Experiment', 'A', 'B', 'C']
df


# In[30]:


method_A = df["A"].astype("float")
method_B = df["B"].astype("float")
method_C = df["C"].astype("float")


# check normality
normality_check(method_A)
normality_check(method_B)
normality_check(method_C)


# In[26]:


# Homoscedasticity check
stat, p_value_levene = stats.levene(method_A, method_B, method_C)
print("p-value : %.4f"%p_value_levene)
if p_value_levene<0.05 :
    print("Reject the null hypothesis => The variances of the sampkes are different")
else:
    print("Fail to reject the null hypothesis => The variances of the samples are the same")


# #### <u>3. Selecting the proper test</u>
# 
# There are three groups, but the normality assumption is violated. Therefore, we should use the non-parametric version of ANOVA for paired data, as the accuracy scores are obtained from the same test sets, which is **Friedman $\chi ^2$ test.** For that, we use `scipy.stats`'s `friedmanchisquare` function.

# In[31]:


test_stat, p_value = stats.friedmanchisquare(method_A, method_B, method_C)
print("p-value : %.4f"%p_value)

if p_value < 0.05:
    print("Reject the null hypothesis")
else :
    print("Fail to reject the null hypothesis")
    
print(np.round(np.mean(method_A), 2),"\n", np.round(np.mean(method_B), 2),"\n", np.round(np.mean(method_C), 2))


# #### <u>4. Decision and Conclusion</u>
# 
# At this significance level, it is evident that at least one of the methods exhibits a different performance.
# 
# **Note :** Due to the non-normality of the data, the **non-parametric version of the posthoc test** is employed. For that we use `scikit_posthocs`'s `posthoc_wilcoxon` function.

# In[32]:


import scikit_posthocs as sp

data = np.array([method_A, method_B, method_C])
df_posthoc = sp.posthoc_wilcoxon(a = data, p_adjust = "holm")
#df_posthoc = sp.posthoc_nemenyi_friedman(data.T)  # another option for the posthoc test

method_names = ["Method A", "Method B", "Method C"]
df_posthoc.columns = method_names
df_posthoc.index = method_names

df_posthoc.style.applymap(lambda x : "background-color:red" if x<0.05 else "background-color:white")


# At this significance level, it can be concluded that the method $C$ outperformed the others, achieving higher accuracy scores than both methods $A$ and $B.$

# <a id="4.8"></a>
# 
# # 4.8) $\chi^2$ Test for Independence
# 
# <a id="4.8.1"></a>
# 
# ## 4.8.1) Definition
# 
# * <u>**$\chi^2$ Test for Independence**</u> **:** also called **test for homogeneity,** is a statistical test used to determine whether there is a significant association between **two categorical variables.** It assesses whether the observed frequencies of the categories differ from the expected frequencies under the assumption of independence between the variables.
# 
# * **Observed Frequencies (O) :** These are the actual counts observed in each category of the contingency table.
# 
# * **Expected Frequencies (E) :** These are the counts that would be expected if there were no association between the variables, calculated under the assumption of independence.
# 
# * **Contingency Table :** also known as a **cross-tabulation** or **crosstab,** is a tabular method in statistics used to summarize and display the relationships between categorical variables. It organizes data into rows and columns where each cell represents the frequency count of a combination of categories from the variables being studied. Contingency tables are especially useful for analyzing and visualizing the association or independence between variables in categorical data analysis.
# 
# 
# #### Assumptions
# 
# Chi-Square Test for Independence does have certain assumptions that need to be satisfied for the test to be valid. Here are the key assumptions:
# 
# 1. **Random Sampling :** The data should be collected through a random sampling method. Each observation should be independent of the others.
# 
# 2. **Categorical Data :** The variables being tested should be categorical. This means that the data should be in the form of frequencies or counts for each category.
# 
# 3. **Large Sample Size :** The sample size should be sufficiently large. Specifically, the expected frequency in each cell of the contingency table should be at least 5 for the test to be reliable. If the expected frequencies are too small, the Chi-Square test may not be appropriate, and an alternative test, such as Fisher's Exact Test, should be considered.
# 
# 4. **Independence of Observations :** The observations should be independent. This means that the occurrence of one observation should not influence the occurrence of another. Each subject or item should contribute to only one cell in the contingency table.
# 
# **NOTE :** Sufficiently Large Expected Frequencies: Generally, the rule of thumb is that all expected cell frequencies should be 5 or more. If more than 20% of expected frequencies are less than 5, the results of the test may not be reliable.
# 
# #### Hypotheses
# 
# - **Null Hypothesis $(H_0)$ :**: There is no association between the two categorical variables (they are independent).
# - **Alternative Hypothesis $(H_0)$ :** There is an association between the two categorical variables (they are dependent).
# 
# #### Test Statistic
# 
# The test statistic for the $\chi^2$ Test for Independence is calculated based on the differences between observed and expected frequencies in each cell of the contingency table. It follows a $\chi^2$ distribution :
# 
# $$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$
# 
# where $O_i$ is the observed frequency for the $i$-th cell, and $E_i$ is the expected frequency for the $i$-th cell.
# 
# The expected frequency for each cell $(i,j)$ is calculated using :
# 
# $$E_ij = \frac{R_i C_j}{T}$$
# 
# where $R_i$ is the total of row $i$, $C_j$ is the total of column $j$, and $T$ the total count in the 2x2 table.
# 
# #### Degrees of Freedom
# 
# The degrees of freedom for the test are calculated as :
# 
# $$df=(r−1)(c−1)$$
# 
# where $r$ is the number of rows and $c$ is the number of columns in the contingency table.
# 

# <a id="4.8.2"></a>
# 
# ## 4.8.2) Example of $\chi^2$ Test for Independence
# 
# ![](risk.jpg) 
# 
# 
# An analyst at a financial investment company is interested in examining the relationship between gender and risk appetite.
# 
# **Risk appetite :** is the amount of risk an organization or investor (like in this example) is willing to take in pursuit of objectives it deems have value.
# 
# A random sample of 660 customers was taken from the database. These customers were categorized based on their gender and risk appetite. The results are presented in the table below.
# 
# | Gender/Risk Appetite |    very Low    |    Low    |    Medium    |High|Very High|Total|
# |----------------------|---------|---------|---------|----|-----|-----------------------|
# | **Female**        |  53   |  23  |  30   |     36   |88   |230  |
# | **Male**        |  71   |  48  |  51   |57|203|430|
# | **Total**        |  124   |  71  |  81   |93|291|660|
# 
# We want to test the hypothesis that the risk appetite of the customers in this company is independent of their gender. We use $\alpha = 0.01.$
# 
# #### <u>1. Defining Hypotheses</u>
# 
# * Null Hypothesis $H_0$ : "Gender and risk appetite are independent" 
# 
# * Alternative Hypothesis $H_1$ : "Gender and risk appetite are dependent"
# 
# 
# 
# #### <u>2. Assumption Check and Selecting the proper test</u>
# 
# The chi-squared test should be used for this question. This test is known as the Chi-squared test for independence. The assumption for this test, which requires that each expected frequency be at least 5 $(E_i\geq 5)$ in at least 80% of the cells, is satisfied. For that, we use `scipy.stats`'s `chi2_contingency`.

# In[29]:


from scipy.stats import chi2_contingency

obs = np.array([[53,23,30,36,88], [71,48,51,57,203]])
chi2, p_value, degrees_of_freedom, expected_freq = chi2_contingency(obs, correction = False)

print("Expected frequencies :\n", np.round(expected_freq, 2))
print("Degrees of freedom :", degrees_of_freedom)
print("Test statistic : %.4f"%chi2)
print("p-value : %.4f"%p_value)


# Let's calculate the **critical statistic**

# In[30]:


from scipy.stats import chi2

# 2 rows and 5 columns
alpha = 0.01
dof = (2-1)*(5-1)
critical_stat = chi2.ppf((1-alpha), dof)
print("Critical statistic : %.4f"%critical_stat)


# #### <u>3. Decision and Conclusion</u>
# 
# Since the p-value is larger than $\alpha=0.01$ (or the calculated statistic of 7.09 is smaller than the critical statistic of 13.28), we fail to reject the null hypothesis $H_0$. At this significance level, it can be concluded that gender and risk appetite are independent.

# <a id="4.9"></a>
# 
# # 4.9) $\chi^2$ Goodness of Fit Test
# 
# <a id="4.9.1"></a>
# 
# ## 4.9.1) Definition
# 
# * <u>**$\chi^2$ Goodness of Fit Test**</u> **:** This test is used to determine if a sample data matches a population with a specific distribution. It assesses whether the observed frequencies of categories differ from the expected frequencies under the specified theoretical distribution.
# 
# * **Observed Frequencies (O) :** These are the actual counts observed in each category of the sample data.
# 
# * **Expected Frequencies (E) :** These are the counts that would be expected under the specified theoretical distribution.
# 
# #### Assumptions
# 
# The Chi-Square Goodness of Fit Test has certain assumptions that need to be satisfied for the test to be valid. Here are the key assumptions:
# 
# 1. **Random Sampling :** The data should be collected through a random sampling method. Each observation should be independent of the others.
# 
# 2. **Categorical Data :** The variable being tested should be categorical. This means that the data should be in the form of frequencies or counts for each category.
# 
# 3. **Large Sample Size :** The sample size should be sufficiently large. Specifically, the expected frequency in each category should be at least 5 for the test to be reliable. If the expected frequencies are too small, the Chi-Square test may not be appropriate, and an alternative test should be considered.
# 
# #### Hypotheses
# 
# - **Null Hypothesis $(H_0)$ :** The sample data fits the specified theoretical distribution.
# - **Alternative Hypothesis $(H_1)$ :** The sample data does not fit the specified theoretical distribution.
# 
# #### Test Statistic
# 
# The test statistic for the $\chi^2$ Goodness of Fit Test is calculated based on the differences between observed and expected frequencies in each category. It follows a $\chi^2$ distribution :
# 
# $$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$
# 
# where $O_i$ is the observed frequency for the $i$-th category, and $E_i$ is the expected frequency for the $i$-th category.
# 
# #### Degrees of Freedom
# 
# The degrees of freedom for the test are calculated as :
# 
# $$df = k - 1$$
# 
# where $k$ is the number of categories.
# 
# <a id="4.9.2"></a>
# 
# ## 4.9.2) Example of $\chi^2$ Goodness of Fit Test
# 
# ![](colors.jpg)
# 
# An analyst wants to test if the colors of candies in a bag follow a specified distribution. The expected distribution is:
# 
# - Red: 30%
# - Blue: 20%
# - Green: 20%
# - Yellow: 15%
# - Orange: 15%
# 
# A random sample of 100 candies from the bag is taken, and the observed frequencies are recorded as follows:
# 
# - Red: 35
# - Blue: 15
# - Green: 20
# - Yellow: 10
# - Orange: 20
# 
# We want to test the hypothesis that the colors of candies in the bag follow the specified distribution. We use $\alpha = 0.05.$
# 
# #### <u>1. Defining Hypotheses</u>
# 
# * Null Hypothesis $H_0$ : "The colors of candies follow the specified distribution" 
# 
# * Alternative Hypothesis $H_1$ : "The colors of candies do not follow the specified distribution"
# 
# #### <u>2. Assumption Check and Selecting the proper test</u>
# 
# The chi-squared goodness of fit test should be used for this question. The assumption for this test, which requires that each expected frequency be at least 5 $(E_i\geq 5)$, is satisfied. For that, we use `scipy.stats`'s `chisquare`.
# 

# In[1]:


from scipy.stats import chisquare

# Observed frequencies
observed = [35, 15, 20, 10, 20]

# Expected frequencies based on the specified distribution
expected = [30, 20, 20, 15, 15]

chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)

print("Test statistic : %.4f" % chi2_stat)
print("p-value : %.4f" % p_value)


# #### <u>3. Decision and Conclusion</u>
# 
# Since the p-value is larger than $\alpha=0.05$, we fail to reject the null hypothesis $H_0$. At this significance level, it can be concluded that the colors of candies in the bag follow the specified distribution.

# <a id="4.10"></a>
# 
# ## 4.10) Pearson Correlation
# 
# <a id="4.10.1"></a>
# ### 4.10.1) Definition
# 
# * <u>**Pearson Correlation Coefficient**</u> **:** is a measure of the strength and direction of the linear relationship between **two continuous variables.** It is denoted by $r$ and ranges from -1 to 1.
# 
# #### Interpretation
# 
# - $r = 1$ : Perfect positive linear relationship
# - $r = -1$ : Perfect negative linear relationship
# - $r = 0$ : No linear relationship
# - $0 < r < 1$ : Positive linear relationship
# - $-1 < r < 0$ : Negative linear relationship
# 
# #### Assumptions
# 
# 1. **Linearity :** The relationship between the two variables should be linear.
# 2. **Normality :** The variables should be approximately normally distributed.
# 3. **Homoscedasticity :** The variance of the errors should be constant across all levels of the independent variable.
# 4. **Independence :** The observations should be independent of each other.
# 
# #### Hypotheses
# 
# - **Null Hypothesis $(H_0)$ :** There is no linear relationship between the two variables ($\rho = 0$).
# - **Alternative Hypothesis $(H_1)$ :** There is a linear relationship between the two variables ($\rho \neq 0$).
# 
# **NOTE :** $r$ is the sample correlation coefficient, which measures the strength and direction of the linear relationship between two variables in a **sample.** Meanwhile $\rho$ (the Greek letter rho) is the population correlation coefficient, which measures the strength and direction of the linear relationship between two variables in the **entire population.**
# 
# #### Test Statistic
# 
# The test statistic for the Pearson correlation is calculated as:
# 
# $$
# r = \frac{{n \sum xy - (\sum x)(\sum y)}}{{\sqrt{[n \sum x^2 - (\sum x)^2][n \sum y^2 - (\sum y)^2]}}}
# $$
# 
# where $n$ is the number of pairs, $x$ and $y$ are the variables.
# 
# #### p-Value
# 
# To test the significance of the Pearson correlation, the t-distribution is used:
# 
# $$
# t = \frac{r \sqrt{n-2}}{\sqrt{1-r^2}}
# $$
# 
# The degrees of freedom (df) are calculated as:
# 
# $$
# df = n - 2
# $$
# 
# <a id="4.10.2"></a>
# ### Example of Pearson Correlation
# 
# ![](studies.jpg)
# 
# Suppose a researcher wants to investigate the relationship between the number of hours studied and the scores obtained in an exam by a group of students. The data collected is as follows:
# 
# | Hours Studied | Exam Score |
# |---------------|------------|
# |       1       |     50     |
# |       2       |     55     |
# |       3       |     60     |
# |       4       |     65     |
# |       5       |     70     |
# |       6       |     75     |
# |       7       |     80     |
# |       8       |     85     |
# |       9       |     90     |
# |      10       |     95     |
# 
# Based on this information, let's conduct a hypothesis test to determine whether there is a significant linear relationship between the number of hours studied and the exam scores.
# 
# #### <u>1. Defining Hypotheses</u>
# 
# * Null Hypothesis $H_0$ : "There is no linear relationship between the number of hours studied and the exam scores" 
# 
# $$H_0 : \rho = 0$$
# 
# * Alternative Hypothesis $H_1$ : "There is a linear relationship between the number of hours studied and the exam scores"
# 
# $$H_1 : \rho \neq 0$$
# 
# #### <u>2. Assumption Check</u>
# 
# * **Linearity :**
# 
# $H_0 :$ "The relationship between the variables is linear"
# 
# $H_1 :$ "The relationship between the variables is **NOT** linear"
# 
# * **Normality :**
# 
# $H_0 :$ "The data is normally distributed"
# 
# $H_1 :$ "The data is **NOT** normally distributed"
# 
# * **Homoscedasticity :**
# 
# $H_0 :$ "The variance of the errors is constant across all levels of the independent variable"
# 
# $H_1 :$ "The variance of the errors is **NOT** constant across all levels of the independent variable"
# 

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, shapiro, bartlett

# Data
hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
exam_scores = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])

# Linearity Check
plt.scatter(hours_studied, exam_scores)
plt.title('Scatter plot of Hours Studied vs Exam Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Scores')
plt.show()

# Normality Check
stat, p_value_normality = shapiro(exam_scores)
print("p-value for Normality : %.4f" % p_value_normality)
if p_value_normality < 0.05 :
    print("Reject the null hypothesis => The data is not normally distributed")
else :
    print("Fail to reject the null hypothesis => The data is normally distributed")

# Homoscedasticity Check
stat, p_value_homoscedasticity = bartlett(hours_studied, exam_scores)
print("p-value for Homoscedasticity : %.4f" % p_value_homoscedasticity)
if p_value_homoscedasticity < 0.05 :
    print("Reject the null hypothesis => The variances are not constant")
else :
    print("Fail to reject the null hypothesis => The variances are constant")


# #### <u>3. Conducting the Pearson Correlation Test</u>
# Since the assumptions are satisfied, we can proceed with the Pearson correlation test. For that, we use `scipy.stats`'s `pearsonr`.

# In[5]:


# Performing Pearson Correlation Test
r, p_value = pearsonr(hours_studied, exam_scores)
print("Pearson correlation coefficient : %.4f" % r)
print("p-value : %.4f" % p_value)
if p_value < 0.05 :
    print("Reject the null hypothesis")
else :
    print("Fail to reject the null hypothesis")


# #### <u>4. Decision and Conclusion</u>
# At this significance level (of 0.05), it can be concluded that there is a significant linear relationship between the number of hours studied and the exam scores.

# <a id="4.11"></a>
# 
# ## 4.11) Spearman Correlation
# 
# <a id="4.11.1"></a>
# ### 4.11.1) Definition
# 
# * <u>**Spearman's Rank Correlation Coefficient**</u> **:** is a non-parametric measure of the strength and direction of the association between two ranked variables. It assesses how well the relationship between two variables can be described using a monotonic function. Spearman's correlation coefficient is denoted by $\rho$ (rho) or $r_s$. Here's a step-by-step explanation:
# 
# 1. **Combine the Samples :** First, combine all the observations from both variables into a single sample.
# 
# 2. **Rank the Observations :** Assign ranks to the combined sample. The smallest observation gets rank 1, the next smallest gets rank 2, and so on. If there are tied values (observations with the same value), each tied value is assigned the average of the ranks they would have received if they were not tied.
# 
# 3. **Calculate the Differences :** For each pair of observations, calculate the difference between their ranks.
# 
# 4. **Square the Differences :** Square each of the rank differences.
# 
# 5. **Sum of Squared Differences :** Calculate the sum of the squared rank differences, denoted as $ d_i^2 $.
# 
# 6. **Spearman's Rank Correlation Coefficient Formula :** Use the formula to calculate the Spearman's rank correlation coefficient $ r_s $:
# 
# $$
# r_s = 1 - \frac{{6 \sum d_i^2}}{{n(n^2 - 1)}}
# $$
# 
# where $d_i$ is the difference between ranks for each pair of observations, and $n$ is the number of observations.
# 
# 7. **Interpret the Result :** The value of $ r_s$ will range between -1 and 1. An $ r_s $ close to 1 indicates a strong positive monotonic relationship, while an $r_s$ close to -1 indicates a strong negative monotonic relationship. An $ r_s $ close to 0 suggests no monotonic relationship.
# 
# Spearman correlation is the non-parametric counterpart to the Pearson correlation and is used when the assumptions of Pearson correlation are not met, such as non-normality or ordinal data.
# 
# #### Interpretation
# 
# - $r_s = 1$ : Perfect positive monotonic relationship
# - $ r_s = -1 $ : Perfect negative monotonic relationship
# - $ r_s = 0 $ : No monotonic relationship
# - $ 0 < r_s < 1 $ : Positive monotonic relationship
# - $ -1 < r_s < 0 $ : Negative monotonic relationship
# 
# #### Assumptions
# 
# 1. **Monotonicity :** The relationship between the two variables should be monotonic (either consistently increasing or consistently decreasing).
# 2. **Ordinal, Interval, or Ratio Data :** The data can be ordinal, interval, or ratio, but the relationship must be monotonic.
# 
# #### Hypotheses
# 
# - **Null Hypothesis $(H_0)$ :** There is no monotonic relationship between the two variables ($\rho = 0$).
# - **Alternative Hypothesis $(H_1)$ :** There is a monotonic relationship between the two variables ($ \rho \neq 0 $).
# 
# #### Test Statistic
# 
# The test statistic for Spearman correlation is calculated as:
# 
# $$
# \rho = 1 - \frac{{6 \sum d_i^2}}{{n(n^2 - 1)}}
# $$
# 
# where $d_i $ is the difference between the ranks of corresponding values and $ n$ is the number of pairs.
# 
# #### p-Value
# 
# To test the significance of the Spearman correlation, the t-distribution is used:
# 
# $$
# t = \frac{r_s \sqrt{n-2}}{\sqrt{1-r_s^2}}
# $$
# 
# The degrees of freedom (df) are calculated as:
# 
# $$
# df = n - 2
# $$
# 
# <a id="4.11.2"></a>
# ### 4.11.2) Example of Spearman Correlation
# 
# ![](customer.jpg)
# 
# Suppose a researcher wants to examine the relationship between the ranks of customer satisfaction and the ranks of customer loyalty in a study of various companies. The following ranks are collected:
# 
# | Company | Satisfaction Rank | Loyalty Rank |
# |---------|-------------------|--------------|
# | A       | 8                 | 7            |
# | B       | 6                 | 5            |
# | C       | 3                 | 2            |
# | D       | 9                 | 8            |
# | E       | 5                 | 4            |
# | F       | 7                 | 6            |
# | G       | 2                 | 1            |
# | H       | 4                 | 3            |
# | I       | 10                | 9            |
# | J       | 1                 | 10           |
# 
# #### <u>1. Defining Hypotheses</u>
# 
# * Null Hypothesis $ H_0 $ : "There is no monotonic relationship between customer satisfaction ranks and customer loyalty ranks"
# 
# $$H_0 : \rho = 0$$
# 
# * Alternative Hypothesis $ H_1 $ : "There is a monotonic relationship between customer satisfaction ranks and customer loyalty ranks"
# 
# $$H_1 : \rho \neq 0$$
# 
# #### <u>2. Assumption Check</u>
# 
# * **Monotonicity :**
# 
#   $H_0 :$ "The relationship between customer satisfaction and customer loyalty ranks is monotonic"
# 
#   $H_1 :$ "The relationship between customer satisfaction and customer loyalty ranks is **NOT** monotonic"
# 
# #### <u>3. Python Implementation</u>
# 
# Let's calculate the Spearman correlation coefficient using Python:
# 
# 
# 

# In[8]:


import numpy as np
from scipy.stats import spearmanr

# Data
satisfaction_ranks = np.array([8, 6, 3, 9, 5, 7, 2, 4, 10, 1])
loyalty_ranks = np.array([7, 5, 2, 8, 4, 6, 1, 3, 9, 10])

# Calculate Spearman correlation
rho, p_value = spearmanr(satisfaction_ranks, loyalty_ranks)

print(f"Spearman correlation coefficient: {rho:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")


# #### <u>4. Decision and Conclusion</u>
# Based on the calculated Spearman correlation coefficient and p-value:
# 
# Spearman correlation coefficient: 0.4424
# 
# P-value: 0.1869
# 
# At a significance level of 0.05, with a Spearman correlation coefficient of 0.4424, we fail to reject the null hypothesis ($H_0: \rho = 0$). Therefore, there is insufficient evidence to conclude a significant monotonic relationship between the ranks of customer satisfaction and customer loyalty in this study.

# # Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By      | Change Description      |
# | ----------------- | ------- | -------------   | ----------------------- |
# | 2024-02-16       | 1.0     | Younes Dahami   |  initial version |
# | 2024-05-20       | 1.1     | Younes Dahami   |  details   |
# | 2024-07-1       | 1.2     | Younes Dahami   |  some definitions additions   |

# In[ ]:




