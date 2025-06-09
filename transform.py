import pandas as pd

df = pd.read_csv("HICP - inflation rate - monthly - 1997 - 2025.csv")

df['Year'] = pd.to_datetime(df['Year'], format='%Y-%m')
df['Year'] = df['Year'].dt.year

yearly_avg_df = df.groupby(['GeoCode','Country','Year'])['Inflation'].mean().reset_index()

print(yearly_avg_df.head(15))

yearly_avg_df.to_csv("yearly_avg_inflation.csv", index=False)

# simplified = df[['geo', 'Geopolitical entity (reporting)', 'TIME_PERIOD', 'OBS_VALUE']]

# simplified.columns = ['GeoCode', 'Country', 'Year', 'Debt Change']

# simplified = simplified.dropna(subset=['Debt Change'])


# #simplified = simplified.sort_values(by=['Country', 'Year'])


# simplified.to_csv("debt_change.csv", index=False)

