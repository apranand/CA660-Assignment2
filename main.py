# Importing Libraries First
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import math
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Import Regression Libraries
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

# loading csv file stores.csv, features.csv train.csv test.csv
stores_csv = pd.read_csv("/Users/aprajitanand/Desktop/walmart-recruiting-store-sales-forecasting/stores.csv")
features_csv = pd.read_csv("/Users/aprajitanand/Desktop/walmart-recruiting-store-sales-forecasting/features.csv")
train_csv = pd.read_csv("/Users/aprajitanand/Desktop/walmart-recruiting-store-sales-forecasting/train.csv")
test_csv = pd.read_csv("/Users/aprajitanand/Desktop/walmart-recruiting-store-sales-forecasting/test.csv")

# merge train with stores and features
# merge test with stores and features
train_merged = train_csv.merge(stores_csv, how = 'left').merge(features_csv, how = 'left')
test_merged = test_csv.merge(stores_csv, how = 'left').merge(features_csv, how = 'left')


# split the Date Col in Plural Forms as Columns
def sDate(aDataFrame):
    aDataFrame['Date'] = pd.to_datetime(aDataFrame['Date'])
    aDataFrame['Year'] = aDataFrame.Date.dt.year
    aDataFrame['Month'] = aDataFrame.Date.dt.month
    aDataFrame['Day'] = aDataFrame.Date.dt.day
    aDataFrame['WeekOfYear'] = (aDataFrame.Date.dt.isocalendar().week) * 1.0
sDate(train_merged)
sDate(test_merged)
# #  ------------------------------------------------------------------------------------------------------------------------------------------------
# # Out 3 types of stores A/B/C, which one is more popular
# popular_store_type_count = train_merged.Type.value_counts().to_dict() # key(Type): value(Count)
# popular_store = pd.DataFrame(list(popular_store_type_count.items()), columns=['store_type', 'type_count'])
# popular_store_piechartFig = px.pie(popular_store, values='type_count', names='store_type',
#              title='Popularity of Store Types by its Count',labels='store_type')
# popular_store_piechartFig.update_traces(textposition='inside', textinfo='percent+label')
# # popular_store_piechartFig.show()
# #  ------------------------------------------------------------------------------------------------------------------------------------------------
# avg_weekly_sales = train_merged.groupby('Type')['Weekly_Sales'].mean().to_dict()
# avg_weekly_sales_df = pd.DataFrame(list(avg_weekly_sales.items()), columns=['store_type', 'avg_sales'])
# avg_weekly_sales_fig = px.bar(avg_weekly_sales_df,
#              x="store_type",
#              y="avg_sales",
#              title="Avg sale by store type",
#              color_discrete_sequence=["#DC143C"])
# # avg_weekly_sales_fig.show()
#  ------------------------------------------------------------------------------------------------------------------------------------------------
# wSales2010 = train_merged[train_merged.Year==2010].groupby('WeekOfYear')['Weekly_Sales'].mean()
# wSales2011 = train_merged[train_merged.Year==2011].groupby('WeekOfYear')['Weekly_Sales'].mean()
# wSales2012 = train_merged[train_merged.Year==2012].groupby('WeekOfYear')['Weekly_Sales'].mean()
# # plot in the axis
# plt.figure(figsize=(15,5))
# plt.plot(wSales2010.index, wSales2010.values, 's-b')
# plt.plot(wSales2011.index, wSales2011.values, 'o-r')
# plt.plot(wSales2012.index, wSales2012.values, '*-g')
# plt.legend(['2010', '2011', '2012'], fontsize=10);
# plt.xlabel('The Week of specific year Year', fontsize=10)
# plt.ylabel('Sales', fontsize=10)
# plt.title("Average Weekly Sales Per Year", fontsize=10)
# plt.show()
#  -------  -----------------------------------------------------------------------------------------------------------------------------------------
# # data frames for the store sales scattered year wise (3 dataframes)
# storeSales10 = train_merged[train_merged.Year==2010].groupby('Store')['Weekly_Sales'].mean().to_dict() #to dictionary and then into DF
# storeSales10_df = pd.DataFrame(list(storeSales10.items()), columns=['Store_id', 'Avg_Sales_2010'])
# # print(storeSales10_df)
# #***
# storeSales11 = train_merged[train_merged.Year==2011].groupby('Store')['Weekly_Sales'].mean().to_dict() #to dictionary and then into DF
# storeSales11_df = pd.DataFrame(list(storeSales11.items()), columns=['Store_id', 'Avg_Sales_2011'])
# #***
# storeSales12 = train_merged[train_merged.Year==2012].groupby('Store')['Weekly_Sales'].mean().to_dict() #to dictionary and then into DF
# # print(list(storeSales12.items()))
# storeSales12_df = pd.DataFrame(list(storeSales12.items()), columns=['Store_id', 'Avg_Sales_2012'])
# #***
# #some more exploration with data visualization and stacking with pyplot
# # df_dummy = storeSales10_df.merge(storeSales11_df, how = 'left').merge(storeSales12_df, how = 'left')
# # Row_list=[]
# # for index, rows in df_dummy.iterrows():
# #     my_list =[rows.Avg_Sales_2010, rows.Avg_Sales_2011, rows.Avg_Sales_2012]
# #     Row_list.append(my_list)
# # # print(Row_list)
# # df_dummy['combine'] = Row_list
# # print(df_dummy) #cool!!
# fig_pyplot = make_subplots(rows=3, cols=1, subplot_titles=("Sales 2010", "Sales 2011", "Sales 2012"))
# fig_pyplot.add_trace(go.Bar(x=storeSales10_df.Store_id, y=storeSales10_df.Avg_Sales_2010,),1, 1)
# fig_pyplot.add_trace(go.Bar(x=storeSales11_df.Store_id, y=storeSales11_df.Avg_Sales_2011,),2, 1)
# fig_pyplot.add_trace(go.Bar(x=storeSales12_df.Store_id, y=storeSales12_df.Avg_Sales_2012,),3, 1)
# fig_pyplot.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False, height=600)
# fig_pyplot.update_xaxes(title_text="Store ID", row=1, col=1)
# fig_pyplot.update_xaxes(title_text="Store ID", row=2, col=1)
# fig_pyplot.update_xaxes(title_text="Store ID", row=3, col=1)
# fig_pyplot.update_yaxes(title_text="Average Sales", row=1, col=1)
# fig_pyplot.update_yaxes(title_text="Average Sale", row=2, col=1)
# fig_pyplot.update_yaxes(title_text="Average Sale", row=3, col=1)
# fig_pyplot.update_xaxes(tick0=1, dtick=1)
# # fig_pyplot.show()
# #  ------------------------------------------------------------------------------------------------------------------------------------------------
# # Exploring relationship between size of the stores and sales relavancy
# plt.figure(figsize=(8,5))
# sns.scatterplot(x=train_merged.Size, y=train_merged.Weekly_Sales, hue=train_merged.Type, s=80);
# plt.xticks( fontsize=5)
# plt.yticks( fontsize=5)
# plt.xlabel('Size', fontsize=5, labelpad=5)
# plt.ylabel('Sales', fontsize=5, labelpad=5)
# # plt.show()
# #  ------------------------------------------------------------------------------------------------------------------------------------------------
# corelation between all indicator of the merged dataframe
# storetype_values = {'A':3, 'B':2, 'C':1}
# train_merged['Type_Numeric'] = train_merged.Type.map(storetype_values)
# test_merged['Type_Numeric'] = test_merged.Type.map(storetype_values)
# plt.figure(figsize=(10,10))
# plt.xticks( fontsize=7)
# plt.yticks( fontsize=7)
# sns.heatmap(train_merged.corr(), cmap='Reds', annot=True, annot_kws={'size':7})
# plt.title('Correlation of merged df', fontsize=7);
# plt.show()
# #  ------------------------------------------------------------------------------------------------------------------------------------------------
# trace1 = go.Bar(
#                 x = storeSales10_df.Store_id,
#                 y = storeSales10_df.Avg_Sales_2010,
#                 name = "year 2010")
# trace2 = go.Bar(
#                 x = storeSales11_df.Store_id,
#                 y = storeSales11_df.Avg_Sales_2011,
#                 name = "year2011")
# trace3 = go.Bar(
#                 x = storeSales12_df.Store_id,
#                 y = storeSales12_df.Avg_Sales_2012,
#                 name = "year2012")
# data = [trace1, trace2, trace3]
# layout = go.Layout(barmode = "group",
#                    xaxis_title="store_id",
#                    yaxis_title="Average Sales")
# fig = go.Figure(data = data, layout = layout)
# # fig.update_xaxes( dtick=1)
# fig.show()
# #  ------------------------------------------------------------------------------------------------------------------------------------------------
train_merged_ts = train_csv.merge(features_csv, on=['Store','Date'], how = 'inner').merge(stores_csv, on = ['Store'], how = 'inner')
train_merged_ts.Date = pd.to_datetime(train_merged_ts.Date,format='%Y-%m-%d')
train_merged_ts.index = train_merged_ts.Date
train_merged_ts = train_merged_ts.drop('Date', axis=1)
train_merged_ts = train_merged_ts.resample('MS').mean()
train_data_1 = train_merged_ts[:int(0.7*(len(train_merged_ts)))]
test_data_1 = train_merged_ts[int(0.7*(len(train_merged_ts))):]
train_data_1 = train_data_1['Weekly_Sales']
test_data_1 = test_data_1['Weekly_Sales']
# train_data_1.plot(figsize=(20,8), title= 'Weekly_Sales', fontsize=14)
# test_data_1.plot(figsize=(20,8), title= 'Weekly_Sales', fontsize=14)
# # plt.show()
# #  ------------------------------------------------------------------------------------------------------------------------------------------------
from pmdarima import auto_arima
model_auto_arima = auto_arima(train_data_1, trace=True, error_action='ignore', suppress_warnings=True)
model_auto_arima = auto_arima(test_data_1, trace=True,start_p=0, start_q=0, start_P=0, start_Q=0, max_p=10, max_q=10, max_P=10, max_Q=10, seasonal=True,stepwise=False, suppress_warnings=True, D=1, max_D=10,error_action='ignore',approximation = False)
model_auto_arima.fit(train_data_1)
#  ------------------------------------------------------------------------------------------------------------------------------------------------
# Predicting the test values using predict function.
forecast = model_auto_arima.predict(n_periods=len(test_data_1))
forecast = pd.DataFrame(forecast,index = test_data_1.index,columns=['Prediction'])
plt.figure(figsize=(20,6))
plt.plot(train_data_1, label='Train')
plt.plot(test_data_1, label='Test')
plt.plot(forecast, label='Prediction')
plt.legend(loc='best')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Weeksales', fontsize=14)
plt.show()
print('MSE- ARIMA: ', mean_squared_error(test_data_1, forecast))
print('RMSE-ARIMA: ', math.sqrt(mean_squared_error(test_data_1, forecast)))
print('MAD- ARIMA: ', mean_absolute_error(test_data_1, forecast))
# #  ------------------------------------------------------------------------------------------------------------------------------------------------
# data = pd.concat([train_merged['Type'], train_merged['Weekly_Sales']], axis=1)
# f, ax = plt.subplots(figsize=(8, 6))
# fig = sns.boxplot(x='Type', y='Weekly_Sales', data=data)
# print(fig)
