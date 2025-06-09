import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

START_YEAR = 2000 
END_YEAR = 2024   
RECENT_START_YEAR = 2020  

poland_df = pd.read_csv('./inflacja_w_polsce_a_w_najwszekszych_gospodarkach/poland_inflation_yearly.csv')
china_df = pd.read_csv('./inflacja_w_polsce_a_w_najwszekszych_gospodarkach/china_inflation_yearly.csv')
germany_df = pd.read_csv('./inflacja_w_polsce_a_w_najwszekszych_gospodarkach/germany_inflation_yearly.csv')
usa_df = pd.read_csv('./inflacja_w_polsce_a_w_najwszekszych_gospodarkach/usa_inflation_yearly.csv')
india_df = pd.read_csv('./inflacja_w_polsce_a_w_najwszekszych_gospodarkach/india_inflation_yearly.csv')
eu_df = pd.read_csv('./inflacja_w_polsce_a_w_najwszekszych_gospodarkach/eu_inflation_yearly.csv')
world_df = pd.read_csv('./inflacja_w_polsce_a_w_najwszekszych_gospodarkach/world_inflation_yearly.csv')

common_years = range(START_YEAR, END_YEAR + 1)
data_dict = {
    'Year': common_years,
    'Poland': [poland_df[poland_df['Year'] == year]['Inflation'].values[0] if year in poland_df['Year'].values else np.nan for year in common_years],
    'China': [china_df[china_df['Year'] == year]['Inflation'].values[0] if year in china_df['Year'].values else np.nan for year in common_years],
    'Germany': [germany_df[germany_df['Year'] == year]['Inflation'].values[0] if year in germany_df['Year'].values else np.nan for year in common_years],
    'USA': [usa_df[usa_df['Year'] == year]['Inflation'].values[0] if year in usa_df['Year'].values else np.nan for year in common_years],
    'India': [india_df[india_df['Year'] == year]['Inflation'].values[0] if year in india_df['Year'].values else np.nan for year in common_years],
    'EU': [eu_df[eu_df['Year'] == year]['Inflation'].values[0] if year in eu_df['Year'].values else np.nan for year in common_years],
    'World': [world_df[world_df['Year'] == year]['Inflation'].values[0] if year in world_df['Year'].values else np.nan for year in common_years]
}
combined_df = pd.DataFrame(data_dict).set_index('Year')

def manual_linear_regression(df, country_name, start_year=START_YEAR, end_year=END_YEAR):
    df_subset = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)][['Year', 'Inflation']]
    X = df_subset['Year'].values
    y = df_subset['Inflation'].values
    
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    
    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)
    slope = numerator / denominator if denominator != 0 else 0

    intercept = y_mean - slope * x_mean

    y_pred = slope * X + intercept
    
    return X, y_pred, slope, intercept

poland_X, poland_y_pred, poland_slope, poland_intercept = manual_linear_regression(poland_df, 'Polska')
china_X, china_y_pred, china_slope, china_intercept = manual_linear_regression(china_df, 'Chiny')
germany_X, germany_y_pred, germany_slope, germany_intercept = manual_linear_regression(germany_df, 'Niemcy')
usa_X, usa_y_pred, usa_slope, usa_intercept = manual_linear_regression(usa_df, 'USA')
india_X, india_y_pred, india_slope, india_intercept = manual_linear_regression(india_df, 'Indie')
eu_X, eu_y_pred, eu_slope, eu_intercept = manual_linear_regression(eu_df, 'UE')
world_X, world_y_pred, world_slope, world_intercept = manual_linear_regression(world_df, 'Świat')

poland_filtered = poland_df[(poland_df['Year'] >= START_YEAR) & (poland_df['Year'] <= END_YEAR)]
china_filtered = china_df[(china_df['Year'] >= START_YEAR) & (china_df['Year'] <= END_YEAR)]
germany_filtered = germany_df[(germany_df['Year'] >= START_YEAR) & (germany_df['Year'] <= END_YEAR)]
usa_filtered = usa_df[(usa_df['Year'] >= START_YEAR) & (usa_df['Year'] <= END_YEAR)]
india_filtered = india_df[(india_df['Year'] >= START_YEAR) & (india_df['Year'] <= END_YEAR)]
eu_filtered = eu_df[(eu_df['Year'] >= START_YEAR) & (eu_df['Year'] <= END_YEAR)]
world_filtered = world_df[(world_df['Year'] >= START_YEAR) & (world_df['Year'] <= END_YEAR)]

plt.figure(figsize=(16, 10))
plt.plot(poland_filtered['Year'], poland_filtered['Inflation'], label='Polska', marker='o', linewidth=2.2, markersize=4, color='#1f77b4')
plt.plot(china_filtered['Year'], china_filtered['Inflation'], label='Chiny', marker='s', linewidth=2.2, markersize=4, color='#ff7f0e')
plt.plot(germany_filtered['Year'], germany_filtered['Inflation'], label='Niemcy', marker='^', linewidth=2.2, markersize=4, color='#2ca02c')
plt.plot(usa_filtered['Year'], usa_filtered['Inflation'], label='USA', marker='d', linewidth=2.2, markersize=4, color='#d62728')
plt.plot(india_filtered['Year'], india_filtered['Inflation'], label='Indie', marker='x', linewidth=2.2, markersize=6, color='#9467bd')
plt.plot(eu_filtered['Year'], eu_filtered['Inflation'], label='UE', marker='v', linewidth=2.2, markersize=4, color='#8c564b')
plt.plot(world_filtered['Year'], world_filtered['Inflation'], label='Świat', marker='*', linewidth=2.8, markersize=6, color='#e377c2')


plt.plot(poland_X, poland_y_pred, '--', label=f'Polska (Regresja: y={poland_slope:.3f}x+{poland_intercept:.1f})', color='blue', alpha=0.7)
plt.plot(china_X, china_y_pred, '--', label=f'Chiny (Regresja: y={china_slope:.3f}x+{china_intercept:.1f})', color='orange', alpha=0.7)
plt.plot(germany_X, germany_y_pred, '--', label=f'Niemcy (Regresja: y={germany_slope:.3f}x+{germany_intercept:.1f})', color='green', alpha=0.7)
plt.plot(usa_X, usa_y_pred, '--', label=f'USA (Regresja: y={usa_slope:.3f}x+{usa_intercept:.1f})', color='red', alpha=0.7)
plt.plot(india_X, india_y_pred, '--', label=f'Indie (Regresja: y={india_slope:.3f}x+{india_intercept:.1f})', color='purple', alpha=0.7)
plt.plot(eu_X, eu_y_pred, '--', label=f'UE (Regresja: y={eu_slope:.3f}x+{eu_intercept:.1f})', color='brown', alpha=0.7)
plt.plot(world_X, world_y_pred, '--', label=f'Świat (Regresja: y={world_slope:.3f}x+{world_intercept:.1f})', color='black', alpha=0.7)

plt.title(f'Inflacja w Polsce, Chinach, Niemczech, USA, Indiach, UE i na Świecie ({START_YEAR}–{END_YEAR}) z regresją liniową', fontsize=14)
plt.xlabel('Rok')
plt.ylabel('Inflacja (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

recent_years = range(RECENT_START_YEAR, END_YEAR + 1)
bar_data = {
    'Poland': [poland_df[poland_df['Year'] == year]['Inflation'].values[0] if year in poland_df['Year'].values else np.nan for year in recent_years],
    'China': [china_df[china_df['Year'] == year]['Inflation'].values[0] if year in china_df['Year'].values else np.nan for year in recent_years],
    'Germany': [germany_df[germany_df['Year'] == year]['Inflation'].values[0] if year in germany_df['Year'].values else np.nan for year in recent_years],
    'USA': [usa_df[usa_df['Year'] == year]['Inflation'].values[0] if year in usa_df['Year'].values else np.nan for year in recent_years],
    'India': [india_df[india_df['Year'] == year]['Inflation'].values[0] if year in india_df['Year'].values else np.nan for year in recent_years],
    'EU': [eu_df[eu_df['Year'] == year]['Inflation'].values[0] if year in eu_df['Year'].values else np.nan for year in recent_years],
    'World': [world_df[world_df['Year'] == year]['Inflation'].values[0] if year in world_df['Year'].values else np.nan for year in recent_years]
}
bar_df = pd.DataFrame(bar_data, index=[str(year) for year in recent_years])

plt.figure(figsize=(12, 7))
bar_df.plot(kind='bar', figsize=(12, 7), width=0.8)
plt.title(f'Porównanie inflacji w latach {RECENT_START_YEAR}–{END_YEAR}')
plt.xlabel('Rok')
plt.ylabel('Inflacja (%)')
plt.legend(title='Kraj/Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, axis='y', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

correlation_matrix = combined_df.corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title(f'Macierz korelacji inflacji ({START_YEAR}–{END_YEAR})')
plt.tight_layout()
plt.show()

print(f"\n=== STATYSTYKI INFLACJI DLA OKRESU {START_YEAR}-{END_YEAR} ===")
print(f"{'Kraj/Region':<10} {'Średnia':<8} {'Mediana':<8} {'Odch.std':<8} {'Min':<6} {'Max':<6}")
print("-" * 50)

countries_data = {
    'Polska': poland_filtered['Inflation'],
    'Chiny': china_filtered['Inflation'],
    'Niemcy': germany_filtered['Inflation'],
    'USA': usa_filtered['Inflation'],
    'Indie': india_filtered['Inflation'],
    'UE': eu_filtered['Inflation'],
    'Świat': world_filtered['Inflation']
}

for country, data in countries_data.items():
    if not data.empty:
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        min_val = data.min()
        max_val = data.max()
        print(f"{country:<10} {mean_val:<8.2f} {median_val:<8.2f} {std_val:<8.2f} {min_val:<6.2f} {max_val:<6.2f}")

print(f"\n=== WSPÓŁCZYNNIKI REGRESJI LINIOWEJ ({START_YEAR}-{END_YEAR}) ===")
print(f"{'Kraj/Region':<10} {'Nachylenie':<12} {'Trend':<15}")
print("-" * 40)

regression_data = [
    ('Polska', poland_slope),
    ('Chiny', china_slope),
    ('Niemcy', germany_slope),
    ('USA', usa_slope),
    ('Indie', india_slope),
    ('UE', eu_slope),
    ('Świat', world_slope)
]

for country, slope in regression_data:
    trend = "Rosnący" if slope > 0 else "Malejący" if slope < 0 else "Stały"
    print(f"{country:<10} {slope:<12.4f} {trend:<15}")

#Regresja liniowa: Polska vs Niemcy

valid_pg = combined_df[['Poland', 'Germany']].dropna()
X = valid_pg['Germany'].values
y = valid_pg['Poland'].values

x_mean = np.mean(X)
y_mean = np.mean(y)

numerator = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean) ** 2)
slope_pg = numerator / denominator if denominator != 0 else 0
intercept_pg = y_mean - slope_pg * x_mean

y_pred_pg = slope_pg * X + intercept_pg

ss_tot = np.sum((y - y_mean) ** 2)
ss_res = np.sum((y - y_pred_pg) ** 2)
r_squared_pg = 1 - (ss_res / ss_tot)


plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Dane rzeczywiste', alpha=0.8, color='#1f77b4', edgecolors='black')  # niebieski z obwódką
plt.plot(X, y_pred_pg, color='#d62728', linewidth=2.5, label=f'Regresja: y = {slope_pg:.3f}x + {intercept_pg:.3f}')
plt.xlabel('Inflacja w Niemczech (%)')
plt.ylabel('Inflacja w Polsce (%)')
plt.title(f'Regresja liniowa: Inflacja Polska vs Niemcy ({START_YEAR}-{END_YEAR})\n', fontsize=13)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print(f"\n=== REGRESJA (BEZ SKLEARN): POLSKA vs NIEMCY ({START_YEAR}–{END_YEAR}) ===")
print(f"y = {slope_pg:.4f} * inflacja_niemcy + {intercept_pg:.4f}")
print(f"R² (dopasowanie modelu): {r_squared_pg:.4f}")
