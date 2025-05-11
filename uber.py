'''
Using the provided dataset, answer the following questions:
'''

import pandas as pd
import numpy as np

# read dataset
df = pd.read_csv("./dataset_1.csv")
df

df.head(10)

df.columns

'''
Question 1:
Which date had the most completed trips during the two week period?

'''
# forward fill empty Dates
df = df.fillna(method="ffill")
df


# aggregate on Date since the question asks the completed trips by Date
df_agg_date = df.groupby('Date').sum().reset_index()
df_agg_date.nlargest(1, 'Completed Trips ')

df_agg_date.nlargest(1, 'Completed Trips ')['Date']

'''
What was the highest number of completed trips within a 24 hour period?


# to be able to use resample function, create timestamp and use as index
def create_timestamp(date, time):
    return pd.to_datetime(f"{date} {time}:00")
df['Timestamp'] = df.apply(lambda row: create_timestamp(row['Date'], row['Time (Local)']), axis=1)

# calculate rollings sums with 24 hours period
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=24)
df_agg_twentyfour_hrs = df.set_index("Timestamp").rolling(window=indexer, min_periods=1).sum()
df_agg_twentyfour_hrs.nlargest(1, 'Completed Trips ')

# parse time interval to output
time_format = "%Y/%m/%d %H:%M"
df_agg_twentyfour_hrs['Time Interval']  = df_agg_twentyfour_hrs.index.strftime(time_format) +  "-" + (df_agg_twentyfour_hrs.index + pd.Timedelta('24 hours')).strftime(time_format)
df_agg_twentyfour_hrs


completed_trips = df_agg_twentyfour_hrs.nlargest(1, 'Completed Trips ')['Completed Trips '].values[0]
time_interval = df_agg_twentyfour_hrs.nlargest(1, 'Completed Trips ')['Time Interval'].values[0]
print("Number of completed trips:", completed_trips)
print("Time Interval:", time_interval)
'''

'''
Which hour of the day had the most requests during the two week period?
'''

# aggregate on hour
df_agg_time = df.groupby('Time (Local)').sum().reset_index()
df_agg_time.nlargest(1, 'Requests ')


df_agg_time.nlargest(1, 'Requests ')['Time (Local)']


'''
What percentages of all zeroes during the two week period occurred on weekend (Friday at 5 pm to Sunday at 3 am)?
'''

# all zeroes in the dataset
total_zeroes = df['Zeroes '].sum()
total_zeroes

# all zeroes in weekend based on given condition
df['Day'] = pd.to_datetime(df['Date']).dt.dayofweek
weekend_zeroes = df[((df['Day'] == 4) & (df['Time (Local)'] >= 17)) 
                          | (df['Day'] == 5) |
                          ((df['Day'] == 6) & (df['Time (Local)'] < 3))]['Zeroes '].sum()
weekend_zeroes

# calculate the percentage
weekend_zeroes_pct = weekend_zeroes / total_zeroes * 100
print(weekend_zeroes_pct,'%')

'''
What is the weighted average ratio of completed trips per driver during the two week period?
'''

# calculate the ratio
df['completed_trip_ratio_per_driver'] = df['Completed Trips '] / df['Unique Drivers']
df

# drop rows with nan 
dataset_without_zero_unique_driver = df[df['Unique Drivers'] > 0]
# calculate weighted average by giving weight as day's completed trip ratio all completed trips
weighted_average_ratio = np.average(dataset_without_zero_unique_driver['completed_trip_ratio_per_driver'], weights=dataset_without_zero_unique_driver['Completed Trips '])
weighted_average_ratio

# check it is not same with normal average
np.average(dataset_without_zero_unique_driver['completed_trip_ratio_per_driver'])


'''
In drafting a driver schedule in terms of 8 hours shifts, when are the busiest 8 consecutive hours over the two week period in terms of unique requests? A new shift starts in every 8 hours. Assume that a driver will work same shift each day.


# resample with 8 hours period                               
df_agg_eight_hrs = df.set_index("Timestamp").resample('8H').sum()

time_format = "%Y/%m/%d %H:%M"
df_agg_eight_hrs['Time Interval']  = df_agg_eight_hrs.index.strftime(time_format) +  "-" + (df_agg_eight_hrs.index + pd.Timedelta('8 hours')).strftime(time_format)
df_agg_eight_hrs

df_agg_eight_hrs.nlargest(1, 'Requests ')

# output time interval


df_agg_eight_hrs.nlargest(1, 'Requests ')['Time Interval']

'''

'''
True or False: Driver supply always increases when demand increases during the two week period.
'''



# create empty dataframe with only timestamps
difference_df = pd.DataFrame(df['Timestamp'])
# calculate request differences to catch request increases
difference_df['request_diff'] = df['Requests '].shift(-1) - df['Requests ']
# do same of supply side
difference_df['supply_diff'] = df['Unique Drivers'].shift(-1) - df['Unique Drivers']
difference_df

# check if request increases, supply also increases 
(difference_df[difference_df['request_diff'] > 0]['supply_diff'] > 0).all().item()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
sns.lineplot(data=df[['Requests ', 'Unique Drivers', 'Timestamp']])
plt.show()



'''
In which 72 hour period is the ratio of Zeroes to Eyeballs the highest?
'''
# calculate rolling sums by 3 days/ 72 hours 
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=72)
df_agg_three_days = df.set_index("Timestamp").rolling(window=indexer, min_periods=1).sum().reset_index()
df_agg_three_days['zeros_to_eyeballs_ratio'] = df_agg_three_days['Zeroes '] / df_agg_three_days['Eyeballs ']

df_agg_three_days = df_agg_three_days[df_agg_three_days['Timestamp'] <= (df_agg_three_days['Timestamp'].max() - pd.Timedelta(days=3))]

time_format = "%Y/%m/%d %H:%M"
df_agg_three_days['Time Interval']  = df_agg_three_days['Timestamp'].dt.strftime(time_format) +  "-" + (df_agg_three_days['Timestamp'] + pd.Timedelta('3 days')).dt.strftime(time_format)
df_agg_three_days

df_agg_three_days.nlargest(1, 'zeros_to_eyeballs_ratio')['zeros_to_eyeballs_ratio']

df_agg_three_days.nlargest(1, 'zeros_to_eyeballs_ratio')['Time Interval']


'''
If you could add 5 drivers to any single hour of every day during the two week period, which hour should you add them to?
'''
df_agg_time = df.groupby('Time (Local)').sum().reset_index()
df_agg_time['eyeball_to_driver_ratio'] = df_agg_time['Eyeballs '] / df_agg_time['Unique Drivers'] 
df_agg_time.nlargest(1, 'eyeball_to_driver_ratio')

df_agg_time.nlargest(1, 'eyeball_to_driver_ratio')['Time (Local)']


'''
True or False: There is exactly two weeks of data in this analysis
'''
# check time difference between the beginning and end
(df['Timestamp'][df.shape[0]-1]-df['Timestamp'][0])
# compare it with timedelta 14 days to see if the difference exactly matches as 14 days
pd.Timedelta('14 days') == df['Timestamp'][df.shape[0]-1]-df['Timestamp'][0]
# the difference is less than 14 days
pd.Timedelta('14 days') > df['Timestamp'][df.shape[0]-1]-df['Timestamp'][0]


'''
Looking at the data from all two weeks, which time might make the most sense to consider a true "end day" instead of midnight? (i.e when are supply and demand at both their natural minimums)
'''

# find min supply and demand per day
df_min_supply_per_day = df.groupby(['Date'])['Unique Drivers'].min().reset_index().rename(columns={'Unique Drivers':'min_supply_per_day'})
df_min_demand_per_day = df.groupby(['Date'])['Requests '].min().reset_index().rename(columns={'Requests ':'min_demand_per_day'})

# merge demand and supply based on date
supply_demand = pd.merge(df_min_supply_per_day, df_min_demand_per_day, on="Date")
supply_demand


# merge found min values with initial dataset on Date
dataset_w_min_supply_demand = pd.merge(df, supply_demand, on="Date")
dataset_w_min_supply_demand

# check the exact matches with natural minimums
min_point = dataset_w_min_supply_demand[(dataset_w_min_supply_demand.min_supply_per_day == dataset_w_min_supply_demand['Unique Drivers']) & 
                           (dataset_w_min_supply_demand.min_demand_per_day == dataset_w_min_supply_demand['Requests '])]['Time (Local)'].mode()
min_point

# minimum hour is 4 am
min_point[0]

# plot and annotate the minimum hour
df_agg_time = df.groupby('Time (Local)').sum().reset_index()
min_point_value = df_agg_time['Requests '][min_point[0]]
plt.figure(figsize=(12,8))
fig = sns.lineplot(data=df_agg_time[['Requests ', 'Unique Drivers']], markers=True, dashes=False)
fig.set_xticks(df_agg_time['Time (Local)'])
plt.annotate('minimum supply/demand', xy=(min_point[0], min_point_value), color='r', fontsize=16)
plt.show()

