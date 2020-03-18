from helpers import data_fetchers
import pandas as pd

# init api's
kaggle_data_fetcher = data_fetchers.KaggleDataFetcher()
world_bank_data_fetcher = data_fetchers.WorldBankDataFetcher()

# get confirmed cases data - time series across countries
# also avail is deaths and recoveries
confirmed_cases_df = kaggle_data_fetcher.get_confirmed_time_series_data().drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])

deaths_cases_df = kaggle_data_fetcher.get_deaths_time_series_data().drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])

recoveries_cases_df = kaggle_data_fetcher.get_recovered_time_series_data().drop(columns=['Province/State', 'Country/Region', 'Lat', 'Long'])

# get country data
country_df= world_bank_data_fetcher.get_all_data()

# full data
confirmed_cases_df = pd.merge(confirmed_cases_df, country_df, on='Country', how='left')

confirmed_cases_df.head()

