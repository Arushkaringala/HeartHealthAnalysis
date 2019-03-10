#Importing the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#Importing the dataset
heart_data = pd.read_csv('heart.csv')

#Renaming the columns
heart_data.rename(columns={'cp': 'chest_pain_type',
                           'trestbps': 'resting_BP',
                           'chol': 'Cholesterol',
                           'fbs': 'fasting_blood_sugar>120',
                           'restecg': 'resting_ecg',
                           'thalach': 'max_hr',
                           'exang': 'exercise_induced_angina',
                           'oldpeak': 'std_depression',
                           'slope': 'std_slope',
                           'ca': 'num_major_vessels',
                           'thal': 'thalassemia'}, inplace=True)

#Replace categorical labels 0,1,2,... to actual string values
heart_data['chest_pain_type'] = heart_data['chest_pain_type'].apply(lambda x: 'typical' if x == 0 else ('atypical' if x == 1 else ('non-anginal' if x == 2 else 'asymptomatic')))
heart_data['resting_ecg'] = heart_data['resting_ecg'].apply(lambda x: 'normal' if x == 0 else ('st-t_wave_abnormality' if x == 1 else 'left_ventricular_hypertrophy'))
heart_data['std_slope'] = heart_data['std_slope'].apply(lambda x: 'upsloping' if x == 0 else ('flat' if x == 1 else 'downsloping'))
heart_data['thalassemia'] = heart_data['thalassemia'].apply(lambda x: 'reversable_defect' if x == 3 else ('fixed_defect' if x == 2 else 'normal'))

#Splitting the Independent and Dependent variables
X = heart_data.drop(['target'], axis=1)
Y = pd.Series(heart_data['target'])

#Convert categorical to dummy
X= pd.get_dummies(X)

#Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state = 0)


#Applying the classification algorithm
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

#Prediction
Y_pred = classifier.predict(X_test)

#Calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(Y_test, Y_pred))

#precision recall curve
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

precision, recall, _ = precision_recall_curve(Y_test, Y_pred)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))



