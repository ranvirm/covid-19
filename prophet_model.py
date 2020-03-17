from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_plotly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helpers import data_api
import fbprophet

# set plot style
plt.style.use('ggplot')

# constants
COUNTRY = 'China'
POP_CAP = 50000  # population ceiling that model can grow to
#  - i found that making this lower than the actual pop gave better predictions
INITIAL_PERIOD = '5 Days'
PROJECTION_HORIZON = '5 Days'  # number of days into the future to forecast (projection period)
PROJECTION_PERIOD = '1 days' 	# number of days to skip before starting a new projection period
N_CAHNGEPOINTS = 3
FORECAST_PERIODS = 5


# define models
def cv_forecast(df, country, pop_cap, projection_horizon, projection_period, initial_period=None):
	ts_data = df[df.Country==country].drop(columns='Country')
	# pivot into ts format prophet requires
	ts_data = pd.melt(ts_data)
	# format time
	ts_data['ds'] = ts_data['variable'].apply(lambda x: pd.datetime.strptime(x, '%m/%d/%y')).drop(columns=['variable'])
	# rename cols
	ts_data = ts_data.rename(columns={"value": "y"})
	# group by country and sum obs - due to some countries being divided by region/ state/ province
	ts_data = ts_data.groupby(by='ds')[ 'y'].sum().reset_index()
	ts_data = ts_data[ts_data['y'] > 0]
	# create population cap col for prophet growth model
	ts_data['cap'] = pop_cap
	# sort df
	ts_data = ts_data.sort_values(by='ds')

	# basic model
	model = Prophet(growth='logistic', n_changepoints=3)#, changepoint_prior_scale=0.0008)
	model.fit(ts_data)

	# generate predictions
	if initial_period:
		cv_df = cross_validation(model, initial=initial_period, period=projection_period, horizon=projection_horizon)
	else:
		cv_df = cross_validation(model, period=projection_period, horizon=projection_horizon)
	cv_df['error'] = cv_df['y'] - cv_df['yhat']
	cv_df['percent_error'] = ((cv_df['yhat'] - cv_df['y'])/cv_df['y']) * 100
	cv_df = cv_df.rename(columns={'cutoff': 'reference_date'})
	cv_df.head()

	# simple plot
	fig = plt.figure()
	ax = plt.axes()
	x = cv_df['ds']
	ax.plot(x, cv_df['y'], color='red')
	ax.plot(x, cv_df['yhat'], color='blue', linestyle='dashdot')
	# ax.plot(x, cv_df['yhat_lower'], color='green', linestyle='dotted')
	# ax.plot(x, cv_df['yhat_upper'], color='green', linestyle='dotted')
	return cv_df, fig


def forecast(df, country, pop_cap, forecast_periods, n_changepoints):
	ts_data = df[df.Country == country].drop(columns='Country')
	# pivot into ts format prophet requires
	ts_data = pd.melt(ts_data)
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
	model = Prophet(growth='logistic', n_changepoints=n_changepoints)  # , changepoint_prior_scale=0.0008)
	model.fit(ts_data)

	# df for forecasts
	future = model.make_future_dataframe(periods=forecast_periods)
	future['cap'] = pop_cap
	forecast_df = model.predict(future)

	fig = model.plot(forecast_df)

	# # simple plot
	# fig = plt.figure()
	# ax = plt.axes()
	# x = cv_df['ds']
	# ax.plot(x, cv_df['y'], color='red')
	# ax.plot(x, cv_df['yhat'], color='blue', linestyle='dashdot')
	# # ax.plot(x, cv_df['yhat_lower'], color='green', linestyle='dotted')
	# # ax.plot(x, cv_df['yhat_upper'], color='green', linestyle='dotted')
	return forecast_df, fig


# init data api
kaggle_api = data_api.KaggleDataApi()

# get confirmed cases data
confirmed_cases_df = kaggle_api.get_confirmed_time_series_data().drop(
	columns=['Province/State', 'Country/Region', 'Lat', 'Long']
)

# uncomment to do cv forecasting - to measure model performance
# (forecast_data, fig) = cv_forecast(
# 	df=confirmed_cases_df,
# 	country=COUNTRY,
# 	pop_cap=POP_CAP,
# 	projection_horizon=PROJECTION_HORIZON,
# 	projection_period=PROJECTION_PERIOD,
# 	initial_period=INITIAL_PERIOD
# )
#
# forecast_data.to_csv('prophet_output/predictions_daysfuture_%_popcap_%.csv')
# forecast_data.head(100)
# fig.show()


# generate forecasts for forecast_periods days into the future
(forecast_data, fig) = forecast(
	df=confirmed_cases_df,
	country=COUNTRY,
	pop_cap=POP_CAP,
	forecast_periods=5,
	n_changepoints=N_CAHNGEPOINTS
)

forecast_data.to_csv('prophet_output/predictions_country_%s_daysfuture_%s_popcap_%s.csv' % (COUNTRY, FORECAST_PERIODS, POP_CAP))
fig.savefig('prophet_output/predictions_country_%s_daysfuture_%s_popcap_%s.png' % (COUNTRY, FORECAST_PERIODS, POP_CAP))
forecast_data.head(100)



