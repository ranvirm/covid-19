from helpers import data_api
import pandas as pd

# init api's
kaggle_api = data_api.KaggleDataApi()
world_bank_api = data_api.WorldBankDataApi()

# get confirmed cases data
confirmed_cases_df = kaggle_api.get_confirmed_time_series_data()

# get country data
country_df= world_bank_api.get_all_data()

# full data
confirmed_cases_df = pd.merge(confirmed_cases_df, country_df, on='Country', how='left')

confirmed_cases_df.head()

