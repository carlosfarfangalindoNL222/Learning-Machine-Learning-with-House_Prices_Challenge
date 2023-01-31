import pandas as pd
import numpy as np
import plotly.express as px

def Overview_Missing_Values(df,head):
    m = df.shape[0]
    null_values = df.isnull().sum()
    percentage_null_values = (null_values/m)*100
    Overview_Missing_Values = pd.DataFrame({'Null Values': null_values, 'Percentage Null Values': percentage_null_values})
    Overview_Missing_Values= Overview_Missing_Values.sort_values(by='Null Values', ascending=False)
    fig = px.imshow(df.isnull(), x=df.columns, y=df.index, color_continuous_scale=px.colors.sequential.Reds)
    fig.show()
    return Overview_Missing_Values.head(head)