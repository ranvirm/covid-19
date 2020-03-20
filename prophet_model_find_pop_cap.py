from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_plotly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import data_fetchers
import fbprophet
from datetime import datetime
import os

# set plot style
plt.style.use('ggplot')

# constants
COUNTRY = 'South Africa'
POP_CAP = 10000  # population ceiling that model can grow to
#  - i found that making this lower than the actual pop gave better predictions
INITIAL_PERIOD = '5 Days'
PROJECTION_HORIZON = '5 Days'  # number of days into the future to forecast (projection period)
PROJECTION_PERIOD = '1 days' 	# number of days to skip before starting a new projection period
N_CHANGEPOINTS = 3
FORECAST_PERIODS = 14


def forecast(df, country, pop_cap, forecast_periods, n_changepoints):
	ts_data = df[df.Country == country].drop(columns='Country')
	# pivot into ts format prophet requires
	ts_data = pd.melt(ts_data)
	ts_data.iloc[55, 1:] = 84  # manual fix for emergency run
	# format time
	ts_data['ds'] = ts_data['variable'].apply(lambda x: pd.datetime.strptime(x, '%m/%d/%y')).drop(columns=['variable'])
	# rename cols
	ts_data = ts_data.rename(columns={"value": "y"})
	# group by country and sum obs - due to some countries being divided by region/ state/ province
	ts_data = ts_data.groupby(by='ds')['y'].sum().reset_index()
	ts_data = ts_data[ts_data['y'] > 0]
	# create population cap col for prophet growth model
	ts_data['cap'] = pop_cap
	# sort df
	ts_data = ts_data.sort_values(by='ds')

	# basic model
	model = Prophet(growth='logistic', interval_width=0.95)#, n_changepoints=n_changepoints)  # , changepoint_prior_scale=0.0008)
	model.fit(ts_data[:-3])

	# df for forecasts
	future = model.make_future_dataframe(periods=forecast_periods)
	future['cap'] = pop_cap
	forecast_df = model.predict(future)

	# add actual values
	forecast_df = pd.merge(forecast_df, ts_data, on='ds', how='left')

	# fig = model.plot(forecast_df)

	# # simple plot
	# fig = plt.figure()
	# ax = plt.axes()
	# x = cv_df['ds']
	# ax.plot(x, cv_df['y'], color='red')
	# ax.plot(x, cv_df['yhat'], color='blue', linestyle='dashdot')
	# # ax.plot(x, cv_df['yhat_lower'], color='green', linestyle='dotted')
	# # ax.plot(x, cv_df['yhat_upper'], color='green', linestyle='dotted')
	return forecast_df  #, fig


# init data api
kaggle_data_fetcher = data_fetchers.KaggleDataFetcher()

# get confirmed cases data
confirmed_cases_df = kaggle_data_fetcher.get_confirmed_time_series_data().drop(
	columns=['Province/State', 'Country/Region', 'Lat', 'Long']
)

# run model
runtime = datetime.now().strftime('%Y-%m-%d_%H:%M')

if os.path.exists('prophet_output/runtime_%s/' % runtime):
	os.rmdir('prophet_output/runtime_%s/' % runtime)
	os.mkdir('prophet_output/runtime_%s/' % runtime)
else:
	os.mkdir('prophet_output/runtime_%s/' % runtime)


# run iteratively to find which pop cap minimizes the error
pop_caps = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 50000, 100000]
errors = []

for pop_cap in pop_caps:

	forecast_data = forecast(
		df=confirmed_cases_df[:-3],
		country=COUNTRY,
		pop_cap=pop_cap,
		forecast_periods=0,
		n_changepoints=N_CHANGEPOINTS
	)

	forecast_data = forecast_data[-3:]


	forecast_data['sq_error'] = (forecast_data['y']-forecast_data['yhat']) ** 2

	sum_error = forecast_data['sq_error'].sum()

	errors.append(sum_error)






