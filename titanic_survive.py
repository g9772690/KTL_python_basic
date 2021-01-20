import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

titanic=sns.load_dataset("titanic")
titanic.to_csv('E:/강의안/파이썬/My_python/titanic.csv',index=False)
#print(titanic.isnull().sum())
titanic['age']=titanic['age'].fillna(titanic['age'].median())
#print(titanic['embarked'].value_counts())
titanic['embarked']=titanic['embarked'].fillna('S')
#print(titanic['embark_town'].value_counts())
titanic['embark_town']=titanic['embark_town'].fillna('Southampton')
#print(titanic['deck'].value_counts())
titanic['deck']=titanic['deck'].fillna('C')
#print(titanic.isnull().sum())                                            # 데이터 수집
f,ax=plt.subplots(1,2,figsize=(10,5))
titanic['survived'][titanic['sex']=='male'].value_counts().plot.pie(explode=[0,0.1],autopct \
    ='%1.1f%%',ax=ax[0], shadow=True)
titanic['survived'][titanic['sex']=='female'].value_counts().plot.pie(explode=[0,0.1],autopct \
    ='%1.1f%%',ax=ax[1], shadow=True)  
ax[0].set_title('Servived(Male)')      
ax[1].set_title('Servived(Female)')      
#plt.show()                                                                # 성별에 따른 생존율 차트 생성
sns.countplot(x='pclass',hue='survived', data=titanic)
plt.title('Pclass vs Survived')
#plt.show()                                                                # 객실등급에 따른 생존자 수, 데이터 탐색
titanic_corr=titanic.corr(method='pearson')                                # 상관계수 생성 
#print(titanic_corr)
#titanic_corr.to_csv('E:/강의안/파이썬/My_python/titanic_corr.csv',index=False)
titanic['survived'].corr(titanic['adult_male'])                            # survived 와 adult_male 간의 상관계수 계산
#print(titanic['survived'].corr(titanic['fare']))                           # 데이터 모델링
sns.pairplot(titanic,hue='survived') 
#plt.show()
sns.catplot(x='pclass',y='survived',hue='sex',data=titanic, kind='point')    # 객실등급과 생존의 상과관계를 catplot 으로 시각화
plt.show() 
def category_age(x):
    if x<10:
        return 0
    elif x<20:
        return 1
    elif x<30:
        return 2
    elif x<40:
        return 3
    elif x<50:
        return 4
    elif x<60:
        return 5  
    elif x<70:
        return 6                     
    else :
        return 7

titanic['age2']=titanic['age'].apply(category_age)
titanic['sex']=titanic['sex'].map({'male':1,'female':0})
titanic['family']=titanic['sibsp']+titanic['parch']+1
#titanic.to_csv('E:/강의안/파이썬/My_python/titanic3.csv',index=False)
