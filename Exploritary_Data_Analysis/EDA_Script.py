import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import skew

def Overview_Missing_Values(df):

    m = df.shape[0]
    null_values = df.isnull().sum()
    percentage_null_values = (null_values/m)*100

    Overview_Missing_Values = pd.DataFrame({'Null Values': null_values, 'Percentage Null Values': percentage_null_values})
    Overview_Missing_Values= Overview_Missing_Values.sort_values(by='Null Values', ascending=False)

    OMV_null_values = Overview_Missing_Values['Null Values']
    head = OMV_null_values[OMV_null_values > 0].count()

    fig = px.imshow(df.isnull(), x=df.columns, y=df.index, color_continuous_scale=px.colors.sequential.Reds)
    fig.show()

    return Overview_Missing_Values.head(head)

def Analyse_Features(train,test,target,feature):
    if train[feature].dtype!="O":
        fig = px.scatter(train, x=feature, y=target, trendline="ols")
        fig.show()
    else:
        fig = px.box(train, x=feature, y=target)
        fig.show()
    print("\n****INFO****")
    print(train[feature].describe())
    print("\n****VALUE COUNTS****")
    print(train[feature].value_counts())
    print("\n****VALUE AVG SALE PRICE****")
    print(train.groupby(feature)[target].mean())
    if train[feature].dtype!="O":
        print("\nSkewness:",str(skew(train[feature])))

    print("\n****TEST INFO****")
    print(test[feature].describe())
    print("\n****VALUE COUNTS****")
    print(test[feature].value_counts())
    
    print("\nOnly in Train: "+str(list(set(train[feature].value_counts().index.values) - set(test[feature].value_counts().index.values))))
    print("Only in Test: "+ str(list(set(test[feature].value_counts().index.values) - set(train[feature].value_counts().index.values))))
    
def Impute_Missing_Values(column,value):
    column = column.fillna(value)
    return df