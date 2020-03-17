import os
import pandas as pd
import world_bank_data as wb


class KaggleDataApi(object):
	def __init__(self, api_token=None):
		self.api_token = api_token if api_token else {"username": "ranvir", "key": "c6e75c4f84ec7e43374893e4a4de349e"}

		# export api credentials
		os.environ['KAGGLE_USERNAME'] = self.api_token['username']
		os.environ['KAGGLE_KEY'] = self.api_token['key']

	@staticmethod
	def get_daily_level_data():
		os.system('kaggle datasets download -f covid_19_data.csv -p data/ sudalairajkumar/novel-corona-virus-2019-dataset')
		df = pd.read_csv('data/covid_19_data.csv')
		df['Country'] = df['Country/Region']
		return df

	@staticmethod
	def get_confirmed_time_series_data():
		# os.system('kaggle datasets download -f time_series_covid_19_confirmed.csv -p data/ sudalairajkumar/novel-corona-virus-2019-dataset')
		# df = pd.read_csv('data/time_series_covid_19_confirmed.csv')
		df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
		df['Country'] = df['Country/Region']
		return df

	@staticmethod
	def get_deaths_time_series_data():
		# os.system('kaggle datasets download -f time_series_covid_19_deaths.csv -p data/ sudalairajkumar/novel-corona-virus-2019-dataset')
		# df = pd.read_csv('data/time_series_covid_19_deaths.csv')
		df = pd.read_csv(
			'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')

		df['Country'] = df['Country/Region']
		return df

	@staticmethod
	def get_recovered_time_series_data():
		# os.system('kaggle datasets download -f time_series_covid_19_recovered.csv-p data/ sudalairajkumar/novel-corona-virus-2019-dataset')
		# df = pd.read_csv('data/time_series_covid_19_recovered.csv')
		df = pd.read_csv(
			'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')

		df['Country'] = df['Country/Region']
		return df


class WorldBankDataApi(object):

	@staticmethod
	def indicators():
		indicators = [
			'UNDP.HDI.XD',  # human development index HDI
			'2.0.hoi.Cel',  # Mobile phone
			'2.0.hoi.Ele',  # electricity
			'2.0.hoi.FPS',	# HOI: Finished Primary School'
			'2.0.hoi.Int',  # HOI: Internet '
			'2.0.hoi.San',  #	HOI: Sanitation'
			'2.0.hoi.Wat',  #	HOI: Water'
			'NE.IMP.GNFS.ZS',  #	Imports of goods and services (% of GDP)'
			'NE.EXP.GNFS.ZS',  #	Exports of goods and services (% of GDP)'
			'NY.GDP.MKTP.PP.CD',  #	GDP, PPP (current international $)
			'NY.GNP.MKTP.PC.CD',  #	GNI per capita (current US$)'
			'SI.POV.GINI',  #	GINI index (World Bank estimate)'
			'HF.DYN.IMRT.IN',  #	Mortality rate, infant (per 1,000 live births)'
			'SE.XPD.TOTL.GB.ZS',  #	Government expenditure on education, total (% of government expenditure)'
			'SG.H2O.PRMS.HH.ZS',  #	Households with water on the premises (%)'
			'SH.DYN.AIDS',  #	Adults (ages 15+) living with HIV'
			'SH.DYN.AIDS.ZS',  #	Prevalence of HIV, total (% of population ages 15-49)'
			'SH.MED.BEDS.ZS',  #	Hospital beds (per 1,000 people)
			'SH.MED.CMHW.P3',  #	Community health workers (per 1,000 people)
			'SH.MED.NUMW.P3',  #	Nurses and midwives (per 1,000 people)
			'SH.MED.PHYS.ZS',  #	Physicians (per 1,000 people)
			'SH.PRV.SMOK',  #	Smoking prevalence, total (ages 15+)'
			'SH.STA.BASS.RU.ZS',  #	People using at least basic sanitation services, rural (% of rural population)'
			'SH.STA.BASS.UR.ZS',  #	People using at least basic sanitation services, urban (% of urban population)
			'SH.STA.BASS.ZS',  #	People using at least basic sanitation services (% of population)
			'SH.STA.HYGN.ZS',  #	People with basic handwashing facilities including soap and water (% of population)'
			'SH.TBS.MORT',  #	Tuberculosis death rate (per 100,000 people)'
			'SH.XPD.CHEX.GD.ZS',  #	Current health expenditure (% of GDP)
			'SH.XPD.CHEX.PC.CD',  #	Current health expenditure per capita (current US$)
			'SP.POP.65UP.TO.ZS',  #	Population ages 65 and above (% of total population)'
			'SP.POP.TOTL',  #	Population, total'
			'SP.URB.TOTL.ZS'  #	Percentage of Population in Urban Areas (in % of Total Population)'
		]
		return indicators

	@staticmethod
	def get_series_data(series, date='2019'):
		df = wb.get_series(series, mrv=1, date=date)
		df = df.to_frame()
		df = df.reset_index()
		return df

	def get_all_data(self):
		indicators = self.indicators()
		i = 0
		for indicator in indicators:
			df_tmp = self.get_series_data(series=indicator)
			df_tmp = df_tmp[['Country', indicator]]
			if i > 0:
				df = pd.merge(df, df_tmp, how="outer")

			else:
				df = df_tmp
				i += 1

		return df



