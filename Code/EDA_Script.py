import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import skew
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

def Overview_Missing_Values(df):
    '''
    Input: Dataframe
    Output: 
        - Dataframe with the number of missing values and the percentage of missing values
        - Heatmap of missing values
    '''
    # Get the null values
    null_values = df.isnull().sum()

    # Get the percentage of null values
    percentage_null_values = (null_values/df.shape[0])*100

    # Create a dataframe with the null values and the percentage of null values
    o_w_m = pd.DataFrame({'Null Values': null_values, 'Percentage Null Values': percentage_null_values})
    o_w_m= o_w_m.sort_values(by='Null Values', ascending=False)

    # Create a heatmap of missing values
    fig = px.imshow(df.isnull(), x=df.columns, y=df.index, color_continuous_scale=px.colors.sequential.Reds)
    fig.show()

    # Only return the rows with null values > 0
    head = o_w_m['Null Values'][o_w_m['Null Values'] > 0].count()
    return o_w_m.head(head)

def Overview_Correlation(df,target):
    '''
    Input: Dataframe
    Output: Dataframe with the correlation of the target with the other features
    '''
    #Get the correlation of the target with the other features
    df = df.select_dtypes(exclude=['object'])
    corr = df.corr()
    target_corr = corr["SalePrice"]

    # Create a dataframe with the correlation of the target with the other features
    corr_df = pd.DataFrame(target_corr, df.columns).sort_values(by="SalePrice", ascending=False)

    return corr_df.head(len(target_corr))

def Overview_Inconsistencies(train,test,target):
    '''
    Input:
        - train: train dataframe
        - test: test dataframe
        - target: target column
    Output:
        - Print the inconsistencies between the train and test dataframe
        - Print the value counts in train
        - Print the value counts in test
        - Print the average sale price of the inconsistencies
    '''
    for i in train.columns:
        if i == 'SalePrice':
            continue
        # Get the unique values
        train_l = train[i].value_counts().index.values
        test_l = test[i].value_counts().index.values
        # Check if there are inconsistencies
        inconst_train = list(set(train_l) - set(test_l))
        inconst_test = list(set(test_l) - set(train_l))
        if len(inconst_train) > 0 or len(inconst_test) > 0:
            print(f"{i}")
            print(f"Only in train[{i}]: "+str(inconst_train[:5]))
            print(f"Only in test[{i}]: "+ str(inconst_test[:5]))
            print("----------------------------------------------------")
            print("Train - Value Counts")
            print(train[i].value_counts())
            print("----------------------------------------------------")
            print("Train - Value Counts")
            print(test[i].value_counts())
            print("----------------------------------------------------")
            print("Average Sale Price")
            print(train.groupby(i)[target].mean())
            print("///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
      
def Overview_Unique_Values(df,percentage=0.85):
    """
    Input:  Dataframe
    Output: Print the unique values and the percentage of unique values
    """
    for i in df.columns:
        # We get the number of null values also, because it is not counted as a unique value
        sum_null = df[i].isnull().sum()
        max_value = df[i].value_counts().max()
        if max_value > sum_null:
            #Check if the unique value is more than 85% of the total values
            if max_value > df.shape[0]*percentage:
                print(f'{i} consist {(max_value/df.shape[0])*100}% out of the unique value: {df[i].value_counts().index.max()}')
                print(df[i].value_counts())
                print("///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
        else:
            #Check if the null values is more than 85% of the total values
            if sum_null > df.shape[0]*percentage:
                print(f'{i} consist {(max_value/df.shape[0])*100}% out of null values')
                print(df[i].value_counts())
                print("///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")

def Check_Outliers(df,target,feature, num, operator):
    """
    Input:
        - df: Dataframe
        - target: Target column
        - feature: Feature column
        - num: Number
        - operator: Operator
    Output:
        - Scatter plot before removing outliers
        - Scatter plot after removing outliers
        - Correlation before removing outliers
        - Correlation after removing outliers
        - Difference between the correlation before and after removing outliers
    """
    # Scatter plot before removing outliers
    fig = px.scatter(df, x=feature, y=target, trendline="ols")
    fig.show()
    before_correlation = df[feature].corr(df[target])
    # Scatter plot after removing outliers
    a_df = df.copy()
    if operator == 'greater':
        a_df.loc[a_df[feature] > num, feature] = num
    elif operator == 'smaller':
        a_df.loc[a_df[feature] < num, feature] = num
    fig = px.scatter(a_df, x=feature, y=target, trendline="ols")
    fig.show()
    # Difference between the correlation before and after removing outliers
    after_correlation = a_df[feature].corr(a_df[target])
    print(f'Before: {before_correlation}')
    print(f'After: {after_correlation}')
    print(f'Difference: {(after_correlation-before_correlation)/before_correlation*100}%')

def Overview_Categories(df):
    """
    Input: Dataframe
    Output: Dataframe with only categorical features, the unique values and the number of unique values
    """
    # Categorical features
    df_categories = df.select_dtypes(include=['object'])
    # Create Dataframe
    overview_categories = pd.DataFrame(df_categories.columns, columns=['Feature'])
    # Add unique values and counts
    unique_values = []
    unique_counts = []
    for col in df_categories.columns:
        unique_values.append(df[col].unique())
        unique_counts.append(len(df[col].unique()))
    overview_categories["Categories"] = unique_values
    overview_categories["Number"] = unique_counts

    return overview_categories.head(len(overview_categories))

def Check_Skewness(df):
    """
    Input: Dataframe
    Output: Dataframe with the skewness of the features
    """
    # Get the skewness of the features
    skewded_features = df.apply(lambda x: x.skew()).sort_values(ascending=False)
    # Create Dataframe
    skewness_table = pd.DataFrame({'Skew':skewded_features})
    return skewness_table.head(30)

def Log_Transform(train,test,target,threshold):
    """
    Input:
        - train: Training set
        - test: Test set
        - target: Target column
        - skewness: Skewness
    Output:
        - train: Training set with log transformation
        - test: Test set with log transformation
    """
    # Get the skewness of the features
    sk_features = train.apply(lambda x: x.skew()).sort_values(ascending=False)
    # Create Dataframe
    sk_table = pd.DataFrame({'Skew':sk_features})
    # Get the features with skewness greater than the threshold
    skewness = sk_table[abs(sk_table) > threshold]
    # Create new Dataframes
    train_skew = train.copy()
    test_skew = test.copy()
    # Log transformation
    for feat in skewness.index:
        if feat == target:
            train_skew[feat] = np.log1p(train_skew[feat])
            continue
        train_skew[feat] = np.log1p(train_skew[feat])
        test_skew[feat] = np.log1p(test_skew[feat])
    return train_skew,test_skew

def Check_INF_values(df):
    """
    Input: Dataframe
    Output: Print the features with infinite values
    """
    # Check for infinite values
    for i in df.columns:
        if ((df[i] == np.inf) | (df[i] == -np.inf)).sum() > 0:
            print(i)
    print('Done')

def hypertuning(X_train, y_train, X_cv, y_cv, ml_model, model_param1, model_param2, param_increase, param_end, param_start=0.00000000001):
    """
    Input:
        - X_train: Training set
        - y_train: Training set target
        - X_cv: Cross validation set
        - y_cv: Cross validation set target
        - ml_model: Machine learning model
        - model_param1: Model parameter 1
        - model_param2: Model parameter 2
        - param_increase: Parameter increase
        - param_start: Parameter start
        - param_end: Parameter end
    Output:
        - Plot with the MSE for the training set and the cross validation set
        - Print minimum CV MSE and the corresponding parameter
        - Print minimum MSE for the training set and the cross validation set
    """
    parameter = param_start
    # Create empty lists
    train_mse = np.array([])
    cv_mse = np.array([])
    params_list = []
    while parameter < param_end:
        # Fit model
        model = ml_model(**{model_param1: parameter})
        model.set_params(**model_param2)
        model.fit(X_train, y_train)
        # Predict
        y_train_pred_1 = model.predict(X_train)
        y_cv_pred_1 = model.predict(X_cv)
        # Calculate MSE
        train_mse= np.append(train_mse, mean_squared_error(y_train, y_train_pred_1))
        cv_mse = np.append(cv_mse,mean_squared_error(y_cv, y_cv_pred_1))
        params_list.append(parameter)
        parameter += param_increase
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=params_list, y=train_mse, name='Train MSE', mode='lines+markers'))  
    fig.add_trace(go.Scatter(x=params_list, y=cv_mse, name='CV MSE' , mode='lines+markers'))
    fig.update_layout(title=f'{ml_model.__name__}', xaxis_title='Alpha', yaxis_title='MSE')
    fig.show()
    # Calculate the minimum difference between CV and Train MSE
    m_d = cv_mse - train_mse
    print(f"Minimum CV MSE is {cv_mse.min()} at alpha = {params_list[cv_mse.argmin()]}")
    print(f"Minimum difference between CV and Train MSE is {m_d.min()} at alpha = {params_list[m_d.argmin()]}")