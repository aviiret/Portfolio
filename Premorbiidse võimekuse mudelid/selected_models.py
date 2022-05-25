import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
from scipy.stats import ttest_ind


def pearsonr_pval(x, y):
    return pearsonr(x, y)[1]


# Load data
xl = pd.ExcelFile("AGE_4ST.xlsx")
print(xl.sheet_names)
r2 = []
mae = []
corr = []
pval = []
acc = []

for subtest in xl.sheet_names:
    validClasses = pd.DataFrame()
    predClasses = pd.DataFrame()
    dataset = xl.parse(subtest)

    # Split-out validation dataset
    array = dataset.values
    X = array[:, :-1]
    y = array[:, -1]
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    # Assign classes to Y_validation
    Y_valid64 = Y_validation.astype('float64')
    Y_validDf = pd.DataFrame(Y_valid64, columns=['FSIQ'])

    conditions = [
        (Y_validDf['FSIQ'] < 70),
        (Y_validDf['FSIQ'] >= 70) & (Y_validDf['FSIQ'] < 80),
        (Y_validDf['FSIQ'] >= 80) & (Y_validDf['FSIQ'] < 90),
        (Y_validDf['FSIQ'] >= 90) & (Y_validDf['FSIQ'] < 110),
        (Y_validDf['FSIQ'] >= 110) & (Y_validDf['FSIQ'] < 120),
        (Y_validDf['FSIQ'] >= 120) & (Y_validDf['FSIQ'] < 130),
        (Y_validDf['FSIQ'] >= 130)
    ]
    values = [1, 2, 3, 4, 5, 6, 7]
    validClasses['class'] = np.select(conditions, values)

    # Fit model
    model = SVR(C=3, coef0=10, kernel='poly')
    model.fit(X_train, Y_train)

    # Get predictions and evaluations
    predictions = model.predict(X_validation)
    score = r2_score(Y_validation, predictions)
    error = mean_absolute_error(Y_validation, predictions)
    r2.append(score)
    mae.append(error)

    # Descriptives
    predDf = pd.DataFrame(predictions, columns=['FSIQ'])
    print(subtest)
    print(predDf.describe())
    print('')

    # T-test
    print('T-test')
    print(ttest_ind(Y_validation, predictions))
    print('')

    # Correlation
    corr.append(predDf['FSIQ'].corr(Y_validDf['FSIQ']))
    pval.append(predDf['FSIQ'].corr(Y_validDf['FSIQ'], method=pearsonr_pval))

    # Assign classes to predictions
    conditions = [
        (predDf['FSIQ'] < 70),
        (predDf['FSIQ'] >= 70) & (predDf['FSIQ'] < 80),
        (predDf['FSIQ'] >= 80) & (predDf['FSIQ'] < 90),
        (predDf['FSIQ'] >= 90) & (predDf['FSIQ'] < 110),
        (predDf['FSIQ'] >= 110) & (predDf['FSIQ'] < 120),
        (predDf['FSIQ'] >= 120) & (predDf['FSIQ'] < 130),
        (predDf['FSIQ'] >= 130)
    ]

    predClasses['class'] = np.select(conditions, values)

    # Evaluate classification
    print('Difference descriptives')
    noAbsPointDistance = pd.DataFrame(predDf-Y_validDf)
    print(noAbsPointDistance.describe())
    print('Point Distance:')
    pointDistance = pd.DataFrame(abs(predDf-Y_validDf))
    pointDistance5 = pointDistance.apply(lambda x: True
                                         if x['FSIQ'] <= 5 else False, axis=1)
    print('+-5: ', len(pointDistance[pointDistance5 == True].index)/154)
    pointDistance10 = pointDistance.apply(lambda x: True
                                          if x['FSIQ'] <= 10 else False, axis=1)
    print('+-10: ', len(pointDistance[pointDistance10 == True].index)/154)
    classDistance = pd.DataFrame(abs(predClasses - validClasses))
    acc.append(accuracy_score(validClasses['class'], predClasses['class']))
    print('\nClass distance:')
    print(classDistance['class'].value_counts(), '\n', classDistance['class'].value_counts()/154)
    print('Within 1 cat: {:.1f}'.format(len(classDistance['class'][classDistance['class'] <= 1]) / 154 * 100))
    print('Within 1 cat: {:.1f}'.format(len(classDistance['class'][classDistance['class'] <= 2]) / 154 * 100))
    cf_matrix = confusion_matrix(validClasses['class'], predClasses['class'], labels=values)
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n" for v1, v2 in
              zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(7, 7)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title(subtest + ' Confusion matrix\n\n')
    ax.set_xlabel('\nPredicted FSIQ Category')
    ax.set_ylabel('Actual FSIQ Category ')

    # Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(values)
    ax.yaxis.set_ticklabels(values)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Display the visualization of the Confusion Matrix.
    plt.show()

    # Accuracy by category
    catAccList = []
    catAccDf = pd.DataFrame(dtype='float64', columns=['Category', 'Accuracy'])
    catAccDf['Category'] = ['Extremely Low',
                            'Borderline',
                            'Low Average',
                            'Average',
                            'High Average',
                            'Superior',
                            'Very Superior'
                            ]

    for i in range(7):
        catAcc = cf_matrix[i, i]/sum(cf_matrix[i,:])
        catAccList.append(catAcc)

    catAccDf['Accuracy'] = catAccList
    print('\nAccuracy by category')
    print(catAccDf)
    print('\n+-5 by category')

    for i in values:
        print(i, ':', len(pointDistance[pointDistance5 == True][validClasses['class'][pointDistance5 == True] == i])
              / len(validClasses['class'][validClasses['class'] == i]))

    print('\n+-10 by category')

    for i in values:
        print(i, ':', len(pointDistance[pointDistance10 == True][validClasses['class'][pointDistance10 == True] == i])
              / len(validClasses['class'][validClasses['class'] == i]))

print('\nDescriptives of validation FSIQ')
print(Y_validDf.describe(include='all'))

# Create sorted dataframe
zipped = list(zip(xl.sheet_names, r2, mae, corr, pval, acc))
df = pd.DataFrame(zipped, columns=['model', 'r2', 'mae', 'corr', 'pval', 'acc'])
df_sorted = df.sort_values('r2')
print('')
print(df_sorted)

# Plot sorted data
plt.barh('model', 'r2', data=df_sorted[['model', 'r2']])
plt.title('SVR Mudelite võrdlus')
plt.show()
