#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize

#Importing the dataset
data = pd.read_csv('heart.csv')

#Analysing the dataset
data.info()

#Analysing each attribute
plt.hist(data.age)
plt.hist(data.sex)
plt.hist(data.trestbps)
plt.hist(data.chol)
plt.hist(data.fbs)
plt.hist(data.restecg)
plt.hist(data.thalach)
plt.hist(data.exang)
plt.hist(data.oldpeak)
plt.hist(data.slope)
plt.hist(data.ca)
plt.hist(data.thal)

#Analysing and Removing Outliers( ex. for one example )
plt.figure(figsize = (15,25))
plt.subplots_adjust(hspace = 0.5)
plt.subplot(623)
plt.boxplot(data.trestbps)
plt.title('Resting blood pressure(trestbps)')
plt.subplot(624)
plt.boxplot(winsorize(data.trestbps, 0.03))
plt.title('Winsorized Resting blood pressure')

#Analysing each attribute of the dataset

#Age
print('Mean age of people', data.age.mean())
print('Mean age of risky people',data[data['target']==1].age.mean())
print('Mean age of not risky people', data[data['target']==0].age.mean())
print('Patients over 54 diagnosis ratio: {} %'.format(round(data[data['age']>54]['target'].mean()*100,2)))
print('Patients under 54 diagnosis ratio: {} %'.format(round(data[data['age']<=54]['target'].mean()*100,2)))

#Sex
plt.hist(data[data['target']==1].sex)
    
#cholestrol
plt.figure(figsize=(16,19))
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.subplot(423)
plt.title('Diagnosed People \n\n\n', size = 16,color ='darkred')
sns.regplot(data[data['target']==1]['age'], data[data['target']==1]['chol'])
plt.xlabel('Age')
plt.ylabel('Cholesterol Serum')
plt.subplot(424)
plt.title('Not Diagnosed People \n\n\n', size = 16,color ='darkred')
sns.regplot(data[data['target']==0]['age'], data[data['target']==0]['chol'])
plt.xlabel('Age')
plt.ylabel('Cholesterol Serum')    
    
#Exercised induced angina(exang)    
plt.hist(data[data['target']==1].exang)
plt.hist(data[data['target']==0].exang)

#Resting blood presssure
plt.figure(figsize=(16,19))
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.subplot(425)
plt.title('Diagnosed People \n\n\n', size = 16,color ='darkred')
sns.regplot(data[data['target']==1]['age'], data[data['target']==1]['trestbps'])
plt.xlabel('Age')
plt.ylabel('Resting Blood Pressure')
plt.subplot(426)
plt.title('Not Diagnosed People \n\n\n', size = 16,color ='darkred')
sns.regplot(data[data['target']==0]['age'], data[data['target']==0]['trestbps'])
plt.xlabel('Age')
plt.ylabel('Resting Blood Pressure')


#oldpeak
plt.figure(figsize=(16,19))
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.title('Diagnosed People \n\n\n', size = 16,color ='darkred')
plt.subplot(427)
sns.regplot(data[data['target']==1]['age'], data[data['target']==1]['oldpeak'])
plt.xlabel('Age')
plt.ylabel('ST Depression')
plt.subplot(428)
sns.regplot(data[data['target']==0]['age'], data[data['target']==0]['oldpeak'])
plt.xlabel('Age')
plt.ylabel('ST Depression')
plt.show()

#thalac
plt.figure(figsize=(16,19))
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.subplot(421)
plt.title('Diagnosed People \n\n\n', size = 16,color ='darkred')
sns.regplot(data[data['target']==1]['age'], data[data['target']==1]['thalach'])
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate Achieved')

plt.subplot(422)
plt.title('Not Diagnosed People \n\n\n', size = 16,color ='darkred')
sns.regplot(data[data['target']==0]['age'], data[data['target']==0]['thalach'])
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate Achieved')


#Chest pain type
sns.set(style="darkgrid")
baslik_font = {'family': 'arial', 'color': 'darkred', 'weight': 'bold', 'size' : 13}
eksen_font = {'family':'arial', 'color':'darkred', 'weight' : 'bold', 'size':13}
plt.figure(figsize = (14,6))
sns.countplot(y = 'cp', hue = 'sex', data = data, palette = 'Greens_d')
plt.title('Chest Pain Distribution \n', fontdict = baslik_font)
plt.ylabel('Chest Pain Type\n 0:Typical Ang., 1:Atypical Ang.\n 2:Non anginal, 3:Asypmtomatic \n', fontdict = eksen_font)
plt.xlabel(('\n Number of People \n0:Female, 1:Male'), fontdict = eksen_font)
plt.show()




