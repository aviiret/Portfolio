import scipy.stats as stats
from sklearn.model_selection import train_test_split
import pandas as pd

xl = pd.ExcelFile("WAIS_t_expanded.xlsx")
subtest = 'chi-square'
dataset = xl.parse(subtest)
array = dataset.values
X = array[:, :-1]
y = array[:, -1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

X_traindf = pd.DataFrame(X_train)
X_validationdf = pd.DataFrame(X_validation)
Y_traindf = pd.DataFrame(Y_train)
Y_validationdf = pd.DataFrame(Y_validation)

print('sugu')

# 1 - train
data = [[1, 1] for i in range(280)] + \
        [[1, 2] for i in range(332)] + \
        [[2, 1] for i in range(59)] + \
        [[2, 2] for i in range(95)]

df = pd.DataFrame(data, columns=['group', 'sex'])
crosstab = pd.crosstab(df['sex'], df['group'])
print('counts')
print(crosstab)
print(pd.crosstab(df['sex'], df['group'], normalize='columns'))
print(stats.chi2_contingency(crosstab))
print('haridus')

data = [[1, 1] for i in range(27)] + \
        [[1, 2] for i in range(143)] + \
        [[1, 3] for i in range(138)] + \
        [[1, 4] for i in range(138)] + \
        [[1, 5] for i in range(121)] + \
        [[2, 1] for i in range(8)] + \
        [[2, 2] for i in range(39)] + \
        [[2, 3] for i in range(28)] + \
        [[2, 4] for i in range(48)] + \
        [[2, 5] for i in range(31)]

df = pd.DataFrame(data, columns=['group', 'edu'])
crosstab = pd.crosstab(df['edu'], df['group'])
print('counts')
print(crosstab)
print(pd.crosstab(df['edu'], df['group'], normalize='columns'))
print(stats.chi2_contingency(crosstab))
