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
def Overview_Correlation(df,target):
    #use only numerical values of df
    df = df.select_dtypes(exclude=['object'])
    corr = df.corr()
    target_corr = corr["SalePrice"]
    corr_df = pd.DataFrame(target_corr, df.columns).sort_values(by="SalePrice", ascending=False)
    return corr_df.head(len(target_corr))

def Analyse_Features(train,test,target,feature):
    if train[feature].dtype!="O":
        fig = px.scatter(train, x=feature, y=target, trendline="ols")
        fig.show()
    else:
        fig = px.box(train, x=feature, y=target)
        fig.show()
    print("\n****INFO****")
    print(train[feature].describe())
    print("\n****Missing Values****")
    print(train[feature].isnull().sum())
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

def Overview_Inconsistencies(train,test,target):
    for i in train.columns:
        if i == 'SalePrice':
            continue
        train_l = train[i].value_counts().index.values
        test_l = test[i].value_counts().index.values
        inconst_train = list(set(train_l) - set(test_l))
        inconst_test = list(set(test_l) - set(train_l))
        if len(inconst_train) > 0 or len(inconst_test) > 0:
            print('------------------------------------------------------------------')
            print(f"Only in train[{i}]: "+str(inconst_train[:5]))
            print(f"Only in test[{i}]: "+ str(inconst_test[:5]))
            print("\n****TRAIN VALUE COUNTS****")
            print(train[i].value_counts())
            print("\n****TEST VALUE COUNTS****")
            print(test[i].value_counts())
            print("\n****VALUE AVG SALE PRICE****")
            print(train.groupby(i)[target].mean())
            print('------------------------------------------------------------------')
      

def Overview_Unique_Values(df,percentage=0.85):
    for i in df.columns:
        sum_null = df[i].isnull().sum()
        max_value = df[i].value_counts().max()
        if max_value > sum_null:
            if max_value > df.shape[0]*percentage:
                print(f'VALUE: {df[i].value_counts().index.max()} = {(max_value/df.shape[0])*100}%')
                print(df[i].value_counts())
                print('------------------------------------------------------------------')
        else:
            if sum_null > df.shape[0]*percentage:
                print(f'Value: Null = {(sum_null/df.shape[0])*100}%')
                print(df[i].value_counts())
                print('------------------------------------------------------------------')

def Check_Outliers(df,target,feature, num, operator):
    #Before removing outliers
    fig = px.scatter(df, x=feature, y=target, trendline="ols")
    fig.show()
    before_correlation = df[feature].corr(df[target])
    #Remove outliers
    a_df = df.copy()
    if operator == 'greater':
        a_df.loc[a_df[feature] > num, feature] = num
    elif operator == 'smaller':
        a_df.loc[a_df[feature] < num, feature] = num
    #After remove outliers
    fig = px.scatter(a_df, x=feature, y=target, trendline="ols")
    fig.show()
    after_correlation = a_df[feature].corr(a_df[target])
    print(f'Before: {before_correlation}')
    print(f'After: {after_correlation}')
    print(f'Difference: {(after_correlation-before_correlation)/before_correlation*100}%')

def Overview_Categories(df):
    #Categorical features
    df_categories = df.select_dtypes(include=['object'])
    #Create Dataframe
    overview_categories = pd.DataFrame(df_categories.columns, columns=['Feature'])
    #Add unique values and counts
    unique_values = []
    unique_counts = []
    for col in df_categories.columns:
        unique_values.append(df[col].unique())
        unique_counts.append(len(df[col].unique()))
    overview_categories["Categories"] = unique_values
    overview_categories["Number"] = unique_counts

    return overview_categories.head(len(overview_categories))

def Check_Skewness(df):
    sk_df = df.copy()
    skewded_features = df.apply(lambda x: x.skew()).sort_values(ascending=False)
    skewness_table = pd.DataFrame({'Skew':skewded_features})
    return skewness_table.head(len(skewness_table))