import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols, glm                      # pip install statsmodels 필요         
import matplotlib.pyplot as plt                                   
import seaborn as sns                                             # pip install seaborn 필요
import statsmodels.api as sm

red_df=pd.read_csv('E:\강의안\파이썬\My_python\winequality-red2.csv')
white_df=pd.read_csv('E:\강의안\파이썬\My_python\winequality-white.csv',sep=';',header=0,engine='python')
#red_df.to_csv('E:\강의안\파이썬\My_python\winequality-red2.csv',index=False)
white_df.to_csv('E:\강의안\파이썬\My_python\winequality-white2.csv',index=False)
red_df.insert(0,column='type', value='red')
white_df.insert(0,column='type', value='white')
wine=pd.concat([red_df,white_df])
wine.to_csv('E:\강의안\파이썬\My_python\wine.csv', index=False)    # 여기까지 분석에 필요한 데이터 수집 완료
#print(wine.info())
wine.columns=wine.columns.str.replace(' ','_')
#print(wine.head())
#print(wine.describe())
#print(sorted(wine.quality.unique()))
#print(wine.quality.value_counts())                                # 여기까지 데이터에 대한 데이터 탐색 
#print(wine.groupby('type')['quality'].describe())                 # type 별 그룹핑 후, quality 값에 대한 기술통계 정리, describe()함수로 그룹 비교
#print(wine.groupby('type')['quality'].mean())                     # 기술통계 전부가 아닌, 하나의 값만 필요할 경우
#print(wine.groupby('type')['quality'].agg(['mean','std']))         # 기술통계 전부가 아닌, mean 함수와 std 함수를 묶어서 한 번에 사용
red_wine_quality=wine.loc[wine['type']=='red','quality']
white_wine_quality=wine.loc[wine['type']=='white','quality']
#print(stats.ttest_ind(red_wine_quality,white_wine_quality,equal_var=False))      # t-검정 결과확인
Rformula='quality~fixed_acidity + volatile_acidity+citric_acid + \
    residual_sugar+chlorides+free_sulfur_dioxide+total_sulfur_dioxide+\
        density+pH+sulphates+alcohol'
regression_result=ols(Rformula,data=wine).fit()
#print(regression_result.summary())                                               # 회귀분석
sample1=wine[wine.columns.difference(['quality','type'])] 
sample1=sample1[0:5][:]
sample1_predict=regression_result.predict(sample1)
print(sample1_predict)                                                            # 샘플에 대한 예측값
print(wine[0:5]['quality'])                                                       # 등급 확인
sns.set_style('dark')
sns.distplot(red_wine_quality,kde=True, color="red", label='red wine')
sns.distplot(white_wine_quality,kde=True, label='white wine')
plt.title("quality of wine type")
plt.legend()
#plt.show()                                                                        # 여기까지 와인유형에 따른 품질 등급 히스토그램 그리기
others=list(set(wine.columns).difference(set(["quality","fixed_acidity"])))
p,resids=sm.graphics.plot_partregress("quality","fixed_acidity", others, data=wine, ret_coords=True)
plt.show()
fig=plt.figure(figsize=(8,13))
sm.graphics.plot_partregress_grid(regression_result, fig=fig)
plt.show()                                                                         # 부분 회귀 플롯


                                                       
