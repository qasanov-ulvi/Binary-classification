"""
Optimal Binning, Normalization, Standardization
1. Dəyişənləri qruplandırmaq üçün istifadə olunur, müəyyən məhdudiyyətlərlə verilənlərin İnformasiya dəyərini (Information Value - IV)
maksimuma çatdırmağa çalışır. Bir hissə data itkisinə səbəb olsa da model üçün datanı daha asan öyrənilən hala salır, modelin doğruluğunu
artır. Səhv parametrlər Overfitting'ə səbəb ola bilər.
Digər iki metod dəyişənlərin bir birinə nisbi gücü qoruyur, məqsəd modelin daha düzgün öyrənməsidir.
Binning verilən datasetdəki kimi bir birinə çox yaxın kəsilməz dəyişənləri ən məntiqli formada qruplaşdırmağa kömək edir.
2. 0-1
3. Z score

"""

import pandas as pd
import math
import matplotlib.pyplot as plt

df = pd.read_csv('bank-direct-marketing-campaigns.csv')

df.info()

#print(df.isna().sum())

print(df.describe())

"""
for x in df.columns:
    print(df[x].value_counts())
"""


#age distribution
plt.figure(figsize=(10, 6))
plt.hist(df['age'], bins=30, color='skyblue', edgecolor='black')
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#jobTittles
plt.figure(figsize=(12, 6))
df['job'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Attendance by Job tittles')
plt.xlabel('JOb')
plt.xticks(rotation=25, ha='right')
plt.ylabel('Frequency')
plt.show()

#education
plt.figure(figsize=(12, 6))
df['education'].value_counts().plot(kind='bar', color='lightcoral', edgecolor='black')
plt.title('Attendance by Education Level')
plt.xlabel('education')
plt.xticks(rotation=15, ha='right')
plt.ylabel('Freq')
plt.show()

#totaljoining
plt.figure(figsize=(8, 6))
plt.boxplot(df['campaign'])
plt.title('Each customer\'s total campaign joining')
plt.xlabel('Customer')
plt.ylabel('Count')
plt.show()


##
plt.figure(figsize=(8, 6))
df['poutcome'].value_counts().plot(kind='bar', color='orange', edgecolor='black')
plt.title('Result of Previous Campaign')
plt.xlabel('Result')
plt.xticks(rotation=15, ha='right')
plt.ylabel('Frequency')
plt.show()


##
plt.figure(figsize=(8, 6))
df['y(dependent _variable)'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightpink'])
plt.title('Percentage of Attendance')
plt.show()

#comparison of week days
df2 = df[df['y(dependent _variable)']=='yes']
plt.figure(figsize=(10, 6))
plt.hist(df["day_of_week"], bins=30, color='skyblue', edgecolor='black',alpha=0.5)
plt.hist(df2["day_of_week"], bins=30, color='red', edgecolor='white',alpha=0.5)
plt.title("Comparison for Week Days")
plt.xlabel('Days')
plt.show()

#to geting insights quickly
"""
for c in df.columns:
    plt.figure(figsize=(10, 6))
    plt.hist(df[c], bins=30, color='skyblue', edgecolor='black')
    plt.title(c)
    plt.show()
"""


"""********/////////////////////////******************************************************************************************************//////////////////////******************************"""
"""********/////////////////////////******************************************************************************************************//////////////////////******************************"""



#unneccasary column - uniform distribution
del df['day_of_week']

#in my mind, we can use the outliers instead of removing
#for 1.5IQR and 3STD Outliers

def fix_outliers(df, col_list, filltype):
    for column_name in col_list:
        if filltype == 1:
            Q1 = df[column_name].quantile(0.25)
            Q3 = df[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = math.floor(Q1 - 1.5 * IQR)
            upper_bound = math.ceil(Q3 + 1.5 * IQR)
            
            df[column_name] = df[column_name].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
        
        elif filltype == 2:
            mean = df[column_name].mean()
            std = df[column_name].std()
            lower_bound = math.floor(mean - 2 * std)
            upper_bound = math.ceil(mean + 2 * std)
            
            df[column_name] = df[column_name].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
        
        else:
            raise ValueError("Invalid filltype. Use 1 for IQR or 2 for mean ± 2*std.")
    
    return df

list4outliers = ['age','campaign','previous']
fix_outliers(df,list4outliers,2)


print(df.describe())

unknown_columns = [col for col in df.columns if df[col].astype(str).str.contains('unknown').any()]
unknown_counts = df[unknown_columns].apply(lambda x: x.str.count('unknown').sum())
print(unknown_counts)

for x in unknown_columns:
    print(df[x].value_counts())

#we have 3 'yes' in default column, thats why it is so difficult find logical way to fill the a lot of unknown cells, to answer the question Which one is Yes or No
del df['default']

#contacted the customer last 30 days, because the max value of this column is 30
df['last30'] = df['pdays'].apply(lambda x : 1 if x<30 else 0)
del df['pdays']


def fill_missing_values(df, col_mean=None, col_median=None, col_mode=None):
    if col_mean is not None:
        for col in col_mean:
            mean_value = df[col].mean()
            df[col].replace('unknown', mean_value, inplace=True)
    
    if col_median is not None:
        for col in col_median:
            median_value = df[col].median()
            df[col].replace('unknown', median_value, inplace=True)
    
    if col_mode is not None:
        for col in col_mode:
            mode_value = df[col].mode()[0]
            df[col].replace('unknown', mode_value, inplace=True)

    return df

list4fill=['job','marital','education','loan', 'housing']

fill_missing_values(df,col_mode=list4fill)

for x in list4fill:
    print(df[x].value_counts())



"""********/////////////////////////******************************************************************************************************//////////////////////******************************"""
"""********/////////////////////////******************************************************************************************************//////////////////////******************************"""



numeric_columns = ['age', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

df['y(dependent _variable)'] = df['y(dependent _variable)'].map({'no': 0, 'yes': 1})

from optbinning import OptimalBinning

numeric_columns = ['age', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

#for dummies
binned_columns = []


for col in numeric_columns:
    optb = OptimalBinning(name=col, dtype="numerical", solver="cp")
    optb.fit(df[col], df['y(dependent _variable)'])
    binning_table = optb.binning_table.build()

    df[f'{col}_binned'] = optb.transform(df[col], metric="bins")

    binned_columns.append(f'{col}_binned')

    del df[col]



from sklearn.preprocessing import OrdinalEncoder

def ordinal_encode_education(df):
    education_order = [
        'illiterate', 'basic.4y', 'basic.6y', 'basic.9y', 
        'high.school', 'professional.course', 'university.degree'
    ]
    encoder = OrdinalEncoder(categories=[education_order])
    df['education_encoded'] = encoder.fit_transform(df[['education']])
    df.drop('education', axis=1, inplace=True)
    return df

df = ordinal_encode_education(df)


#dummies for other columns
categorical_columns = ['job', 'marital', 'loan', 'contact', 'month', 'poutcome','housing']

df = pd.get_dummies(df, columns=categorical_columns, drop_first=False, dtype=int)

df = pd.get_dummies(df, columns=binned_columns, drop_first=False, dtype=int)

df.to_csv('12346.csv')

"""********/////////////////////////******************************************************************************************************//////////////////////******************************"""
"""********/////////////////////////******************************************************************************************************//////////////////////******************************"""

y = df['y(dependent _variable)']
X = df.drop('y(dependent _variable)', axis=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


class ClassifierComparison:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, train_size=0.7, random_state=42)
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB()
        }
        self.best_model_name = None
        self.best_model = None
        self.best_score = 0

    def fit_and_evaluate(self):
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"{name} Accuracy: {accuracy:.4f}")

            if accuracy > self.best_score:
                self.best_score = accuracy
                self.best_model_name = name
                self.best_model = model

        print(f"\nBest Model: {self.best_model_name} with Accuracy: {self.best_score:.4f}")

classifier_comparison = ClassifierComparison(X, y)
classifier_comparison.fit_and_evaluate()

