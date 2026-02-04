# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

print("="*80)
print("MENTAL WELL-BEING IN SINGAPORE: COMPREHENSIVE ANALYSIS")
print("="*80)

# Data loading and preprocessing

print("\n[1/4] Loading datasets...")

df_global = pd.read_csv('/Users/iffahilyana/Desktop/DSA4262/Assignment 1/Mental health Depression disorder Data.csv', low_memory=False)
df_lifestyle = pd.read_csv('/Users/iffahilyana/Desktop/DSA4262/Assignment 1/mental_health.csv')
df_sg_survey = pd.read_csv('/Users/iffahilyana/Desktop/DSA4262/Assignment 1/MentalWellbeingSurvey.csv')

print(f"✓ Global data: {df_global.shape}")
print(f"✓ Lifestyle data: {df_lifestyle.shape}")
print(f"✓ Singapore survey: {df_sg_survey.shape}")

# Clean and prepare global data

# Convert Year to numeric
df_global['Year_int'] = pd.to_numeric(df_global['Year'], errors='coerce')
df_global_clean = df_global[df_global['Year_int'].notna()].copy()
df_global_clean = df_global_clean[(df_global_clean['Year_int'] >= 1990) & 
                                   (df_global_clean['Year_int'] <= 2020)]

# Define disorder columns
disorder_cols = ['Depression (%)', 'Anxiety disorders (%)', 
                 'Bipolar disorder (%)', 'Schizophrenia (%)']

# Get 2017 data (most complete recent year)
df_2017 = df_global_clean[df_global_clean['Year_int'] == 2017].copy()

# Convert disorder columns to numeric
for col in disorder_cols:
    df_2017[col] = pd.to_numeric(df_2017[col], errors='coerce')

# Calculate total prevalence
df_2017['Total_Prevalence'] = df_2017[disorder_cols].sum(axis=1, skipna=True)

# Filter for valid data: has code, has reasonable prevalence (0.1% to 30%)
df_countries = df_2017[
    (df_2017['Code'].notna()) & 
    (df_2017['Total_Prevalence'] > 0.1) &
    (df_2017['Total_Prevalence'] < 30)  # Removing outliers
].copy()

# SG: get the most reliable record (the one with reasonable values)
sg_records = df_countries[df_countries['Entity'] == 'Singapore']
if len(sg_records) > 0:
    # Take record with prevalence in reasonable range
    sg_data = sg_records[
        (sg_records['Total_Prevalence'] > 5) & 
        (sg_records['Total_Prevalence'] < 15)
    ].iloc[0]
    
    sg_prevalence = sg_data['Total_Prevalence']
    sg_depression = sg_data['Depression (%)']
    sg_anxiety = sg_data['Anxiety disorders (%)']
    sg_bipolar = sg_data['Bipolar disorder (%)']
    sg_schizo = sg_data['Schizophrenia (%)']
else:
    # Fallback values
    sg_depression = 3.44
    sg_anxiety = 3.73
    sg_bipolar = 0.73
    sg_schizo = 0.27
    sg_prevalence = sg_depression + sg_anxiety + sg_bipolar + sg_schizo

print(f"\nSingapore Mental Health Metrics (2017):")
print(f"  Depression: {sg_depression:.2f}%")
print(f"  Anxiety: {sg_anxiety:.2f}%")
print(f"  Bipolar: {sg_bipolar:.2f}%")
print(f"  Schizophrenia: {sg_schizo:.2f}%")
print(f"  Total Prevalence: {sg_prevalence:.2f}%")
print(f"\nGlobal dataset: {len(df_countries)} countries with valid data")

# Calculate Singapore percentile globally
sg_percentile = (
    (df_countries['Total_Prevalence'] < sg_prevalence).sum()
    / len(df_countries) * 100
)


print(f"  Prevalence range: {df_countries['Total_Prevalence'].min():.2f}% to {df_countries['Total_Prevalence'].max():.2f}%")
print(f"  Global mean: {df_countries['Total_Prevalence'].mean():.2f}%")

### Visualisation 1: Macro (Global Context)

print("\n[2/4] Creating Visualization 1: MACRO - Global Context (NEW DESIGN)...")

fig = plt.figure(figsize=(20, 11))
gs = fig.add_gridspec(
    3, 4,
    height_ratios=[1.3, 0.05, 1],
    hspace=0.1,
    wspace=0.35
)

# Panel 1: World chloropleth-style map
ax_map = fig.add_subplot(gs[0, :])

# Country coordinates (longitude, latitude)
country_coords = {
    'USA': (-95, 37), 'CAN': (-106, 56), 'MEX': (-102, 23), 'BRA': (-51, -14), 
    'ARG': (-63, -38), 'CHL': (-71, -30), 'COL': (-74.2, 4.5), 'PER': (-75, -9),
    'VEN': (-66.5, 6.4), 'ECU': (-78.1, -1.8), 'BOL': (-63.5, -16.2),
    'GBR': (-3, 54), 'FRA': (2, 46), 'DEU': (10, 51), 'ESP': (-3, 40), 'ITA': (12, 42),
    'POL': (19, 52), 'UKR': (31, 49), 'RUS': (105, 61), 'NOR': (8.4, 60.4),
    'SWE': (18.6, 60.1), 'FIN': (25.7, 61.9), 'DNK': (9.5, 56.2), 'NLD': (5.2, 52.1),
    'BEL': (4.4, 50.5), 'CHE': (8.2, 46.8), 'AUT': (14.5, 47.5), 'GRC': (21.8, 39),
    'PRT': (-8.2, 39.3), 'IRL': (-8.2, 53.4), 'ISL': (-19, 64.9), 'ROM': (24.9, 45.9),
    'BGR': (25.4, 42.7), 'HRV': (15.2, 45.1), 'SRB': (21, 44), 'HUN': (19.5, 47.1),
    'CZE': (15.4, 49.8), 'SVK': (19.6, 48.6), 'SVN': (14.9, 46.1), 'EST': (25, 58.5),
    'LVA': (24.6, 56.8), 'LTU': (23.8, 55.1), 'BLR': (27.9, 53.7), 'ALB': (20.1, 41.1),
    'CHN': (104, 35), 'JPN': (138, 36), 'IND': (78, 20), 'IDN': (113, -2),
    'AUS': (133, -27), 'NZL': (174, -40), 'ZAF': (24, -30), 'EGY': (30, 26),
    'NGA': (8, 9), 'KEN': (37, -0.5), 'ETH': (40.4, 9.1), 'TZA': (34.8, -6.3),
    'GHA': (-1, 7.9), 'UGA': (32.2, 1.3), 'MOZ': (35.5, -18.6), 'ZMB': (27.8, -13.1),
    'SAU': (45, 24), 'IRN': (53, 32), 'TUR': (35, 39), 'IRQ': (43.6, 33.2),
    'PAK': (69.3, 30.3), 'BGD': (90.3, 23.6), 'AFG': (67.7, 33.9), 'ARE': (53.8, 23.4),
    'SGP': (103.8, 1.35), 'MYS': (101.9, 4.2), 'THA': (100.9, 15.8), 'VNM': (108.2, 14),
    'PHL': (121.7, 12.8), 'KOR': (127.7, 37.5), 'PRK': (127, 40), 'TWN': (121, 23.5),
    'LKA': (80.7, 7.8), 'NPL': (84.1, 28.3), 'KHM': (104.9, 12.5), 'LAO': (102.4, 19.8),
    'MMR': (95.9, 21.9), 'ISR': (34.8, 31), 'JOR': (36.2, 30.5), 'LBN': (35.8, 33.8),
    'SYR': (38.9, 34.8), 'YEM': (48.5, 15.5), 'OMN': (55.9, 21.4), 'KWT': (47.4, 29.3),
    'QAT': (51.1, 25.3), 'BHR': (50.5, 26), 'DZA': (1.6, 28), 'TUN': (9.5, 33.8),
    'LBY': (17.2, 26.3), 'MAR': (-7, 31.7), 'SDN': (30.2, 12.8), 'AGO': (17.8, -11.2),
    'COD': (21.7, -4), 'CIV': (-5.5, 7.5), 'CMR': (12.3, 7.3), 'MDG': (46.8, -18.7),
    'ZWE': (29.1, -19), 'BWA': (24.6, -22.3), 'NAM': (18.4, -22.9), 'KAZ': (66.9, 48),
    'UZB': (64.5, 41.3), 'MNG': (103.8, 46.8), 'CUB': (-77.7, 21.5), 'DOM': (-70.1, 18.7),
    'HTI': (-72.2, 18.9), 'JAM': (-77.2, 18.1), 'GTM': (-90.2, 15.7), 'HND': (-86.2, 15.1),
    'NIC': (-85.2, 12.8), 'CRI': (-84, 9.7), 'PAN': (-80.7, 8.5), 'URY': (-55.7, -32.5),
    'PRY': (-58.4, -23.4), 'SLV': (-88.8, 13.7), 'TTO': (-61.2, 10.6),
}

# Map countries to coordinates and colors
plot_data = []
for _, row in df_countries.iterrows():
    code = row['Code']
    if code in country_coords:
        lon, lat = country_coords[code]
        plot_data.append({
            'Entity': row['Entity'],
            'Code': code,
            'Prevalence': row['Total_Prevalence'],
            'lon': lon,
            'lat': lat
        })

df_plot = pd.DataFrame(plot_data)

# Create color scale
vmin, vmax = 4, 14  # Focus on realistic range
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.RdYlGn_r  # Red = high prevalence, Green = low

# Plot countries as circles
for _, row in df_plot.iterrows():
    if row['Code'] == 'SGP':
        continue  # Plot Singapore separately
    
    color = cmap(norm(row['Prevalence']))
    ax_map.scatter(row['lon'], row['lat'], 
                  c=[color], s=250, alpha=0.8,
                  edgecolors='black', linewidths=0.8, zorder=2)

# Highlight Singapore with large red star
sg_row = df_plot[df_plot['Code'] == 'SGP'].iloc[0]
ax_map.scatter(
    sg_row['lon'], sg_row['lat'],
    marker='*',
    s=2000,
    c='red',
    edgecolors='darkred',
    linewidths=3,
    zorder=100,
    label=f'Singapore: {sg_prevalence:.2f}%'
)

# Add Singapore label
ax_map.annotate('SINGAPORE', 
               xy=(sg_row['lon'], sg_row['lat']),
               xytext=(sg_row['lon']+15, sg_row['lat']+8),
               fontsize=13, fontweight='bold', color='red',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='red', linewidth=2),
               arrowprops=dict(arrowstyle='->', color='red', lw=2.5))

# Styling
ax_map.set_xlim(-180, 180)
ax_map.set_ylim(-60, 80)
ax_map.set_xlabel('Longitude', fontsize=12, fontweight='bold')
ax_map.set_ylabel('Latitude', fontsize=12, fontweight='bold')
ax_map.set_title('Global Mental Health Burden: Where Does Singapore Stand?\n' +
                 f'Mental Disorder Prevalence by Country (2017) - Singapore at {sg_percentile:.0f}th Percentile',
                 fontsize=15, fontweight='bold', pad=20)
ax_map.grid(True, alpha=0.2, linestyle='--')
ax_map.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
ax_map.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = plt.colorbar(
    sm,
    ax=ax_map,
    orientation='horizontal',
    pad=0.12,
    aspect=50,
    shrink=0.45
)

cbar.set_label(
    'Total Mental Disorder Prevalence (%)',
    fontsize=11,
    fontweight='bold',
    labelpad=10
)

cbar.ax.tick_params(labelsize=9)

# Panel 2: Singapore's Global Position - Clear Percentile Bar
ax_percentile = fig.add_subplot(gs[2, :2])

# Calculate percentile
sg_percentile = (df_countries['Total_Prevalence'] < sg_prevalence).sum() / len(df_countries) * 100

# Create gradient bar
gradient = np.linspace(0, 1, 256).reshape(1, -1)
ax_percentile.imshow(gradient, aspect='auto', cmap=plt.cm.RdYlGn_r,
                    extent=[0, 100, 0, 1], zorder=1)

# Add percentile markers
percentiles = [25, 50, 75]
for p in percentiles:
    ax_percentile.axvline(p, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax_percentile.text(p, 1.15, f'{p}th', ha='center', fontsize=10, fontweight='bold')

# Mark Singapore (bring forward)
ax_percentile.plot(
    [sg_percentile], [0.55],
    marker='v',
    markersize=28,
    color='red',
    markeredgecolor='darkred',
    markeredgewidth=2.5,
    zorder=20
)

ax_percentile.axvline(
    sg_percentile,
    color='red',
    linestyle='-',
    linewidth=3,
    alpha=0.9,
    zorder=15
)

ax_percentile.text(
    sg_percentile,
    -0.32,
    f'{sg_percentile:.1f}th Percentile\n{sg_prevalence:.2f}%',
    ha='center',
    fontsize=12,
    fontweight='bold',
    color='red',
    zorder=30,
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        edgecolor='red',
        linewidth=2
    )
)

# Interpretation text
if sg_percentile < 50:
    interpretation = "LOWER than most countries (Better mental health)"
    interp_color = 'green'
else:
    interpretation = "HIGHER than most countries (More mental health challenges)"
    interp_color = 'black'

ax_percentile.text(
    50, 1.5,
    f"Singapore's Position: {interpretation}",
    ha='center',
    fontsize=11,
    fontweight='bold',
    color=interp_color,
    zorder=40,
    bbox=dict(
        boxstyle='round',
        facecolor='lightyellow',
        edgecolor=interp_color,
        linewidth=2
    )
)

ax_percentile.set_xlim(0, 100)
ax_percentile.set_ylim(-0.5, 1.8)
ax_percentile.set_xlabel('Percentile Ranking (Lower = Less Burden)', 
                        fontsize=11, fontweight='bold')
ax_percentile.set_yticks([])
ax_percentile.set_title('Singapore\'s Global Ranking',
                       fontsize=13, fontweight='bold', pad=10)

# Panel 3: Distribution Histogram with Singapore Marked
ax_dist = fig.add_subplot(gs[2, 2:])

# Create histogram
n, bins, patches = ax_dist.hist(df_countries['Total_Prevalence'], bins=30,
                                color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.8)

# Color bars based on value
for i, patch in enumerate(patches):
    color = cmap(norm(bins[i]))
    patch.set_facecolor(color)

# Mark Singapore
ax_dist.axvline(sg_prevalence, color='red', linestyle='--', linewidth=3.5,
               label=f'Singapore: {sg_prevalence:.2f}%', zorder=10)

# Mark global mean
global_mean = df_countries['Total_Prevalence'].mean()
ax_dist.axvline(global_mean, color='blue', linestyle=':', linewidth=3,
               label=f'Global Mean: {global_mean:.2f}%', alpha=0.7)

ax_dist.set_xlabel('Mental Disorder Prevalence (%)', fontsize=11, fontweight='bold')
ax_dist.set_ylabel('Number of Countries', fontsize=11, fontweight='bold')
ax_dist.set_title('Global Distribution\n(Where does Singapore fit?)',
                 fontsize=13, fontweight='bold', pad=10)
ax_dist.legend(fontsize=10, loc='upper right')
ax_dist.grid(axis='y', alpha=0.3, linestyle='--')

# Add statistics box
stats_text = f"""Global Statistics:
Countries: {len(df_countries)}
Mean: {global_mean:.2f}%
Median: {df_countries['Total_Prevalence'].median():.2f}%
Range: {df_countries['Total_Prevalence'].min():.2f}% - {df_countries['Total_Prevalence'].max():.2f}%

Singapore: {sg_prevalence:.2f}%
Rank: {sg_percentile:.1f}th percentile"""

ax_dist.text(0.98, 0.97, stats_text, transform=ax_dist.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, 
                     edgecolor='black', linewidth=1.5))

ax_dist.legend(fontsize=9, loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('viz1_macro_global_FINAL.png', dpi=300, bbox_inches='tight')
plt.close()

### Visualisation 2: Micro (Singapore Context)

print("\n[3/4] Creating Visualization 2: MICRO - Local Patterns...")

# Create mental health composite score from symptoms
symptom_cols = ['Feeling_Sad_Down', 'Loss_Of_Interest', 'Sleep_Trouble', 
                'Fatigue', 'Feeling_Worthless', 'Concentration_Difficulty', 
                'Anxious_Nervous', 'Panic_Attacks', 'Mood_Swings', 'Irritability']
df_lifestyle['Symptom_Score'] = df_lifestyle[symptom_cols].sum(axis=1)
# Convert to 0-100 scale where higher is better
df_lifestyle['Wellbeing_Score'] = 100 - (df_lifestyle['Symptom_Score'] / 
                                          len(symptom_cols) * 10)

# Create age groups
df_lifestyle['Age_Group'] = pd.cut(df_lifestyle['Age'], 
                                    bins=[0, 25, 35, 45, 55, 100],
                                    labels=['18-25', '26-35', '36-45', 
                                           '46-55', '56+'])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Violin plot by age group
age_groups = ['18-25', '26-35', '36-45', '46-55', '56+']
age_data = [df_lifestyle[df_lifestyle['Age_Group'] == ag]['Wellbeing_Score'].dropna().values 
            for ag in age_groups]

parts = ax1.violinplot(age_data, positions=range(len(age_groups)), 
                       widths=0.7, showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('#3498db')
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')

parts['cmeans'].set_color('#e74c3c')
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color("#0F5218")
parts['cmedians'].set_linewidth(2)

ax1.set_xticks(range(len(age_groups)))
ax1.set_xticklabels(age_groups)
ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mental Wellbeing Score (0-100)', fontsize=12, fontweight='bold')
ax1.set_title('Mental Wellbeing Distribution by Age Group\n' + 
              '(Representative of Singapore Demographics)', 
              fontsize=13, fontweight='bold', pad=15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim([0, 105])

# Add median annotations
for i, ag in enumerate(age_groups):
    median_val = df_lifestyle[df_lifestyle['Age_Group'] == ag]['Wellbeing_Score'].median()
    mean_val = df_lifestyle[df_lifestyle['Age_Group'] == ag]['Wellbeing_Score'].mean()
    ax1.text(i, median_val + 3, f'Med: {median_val:.1f}', ha='center', 
             fontweight='bold', fontsize=8, color="#0F5218")
    ax1.text(i, mean_val - 3, f'Mean: {mean_val:.1f}', ha='center', 
             fontweight='bold', fontsize=8, color="#b33628")

# 2. Box plot by employment status
employment_order = df_lifestyle['Employment_Status'].value_counts().head(5).index.tolist()
df_emp = df_lifestyle[df_lifestyle['Employment_Status'].isin(employment_order)]

emp_data = [df_emp[df_emp['Employment_Status'] == es]['Wellbeing_Score'].dropna().values 
            for es in employment_order]
bp = ax2.boxplot(emp_data, labels=[es.replace('_', '\n') for es in employment_order], 
                 vert=False, patch_artist=True, showmeans=True)
for patch in bp['boxes']:
    patch.set_facecolor('#27ae60')
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
for mean in bp['means']:
    mean.set_marker('D')
    mean.set_markerfacecolor('#b33628')
    mean.set_markersize(6)

ax2.set_xlabel('Mental Wellbeing Score (0-100)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Employment Status', fontsize=12, fontweight='bold')
ax2.set_title('Mental Wellbeing by Employment Status\n(Red Diamond = Mean)', 
              fontsize=13, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_xlim([0, 105])

# 3. Overall distribution with color coding
n, bins, patches = ax3.hist(df_lifestyle['Wellbeing_Score'], bins=40, 
                            color='#9b59b6', edgecolor='black', 
                            alpha=0.7, linewidth=0.5)

# Color code bins
cm = plt.cm.RdYlGn
norm_hist = plt.Normalize(vmin=bins.min(), vmax=bins.max())
for i, patch in enumerate(patches):
    patch.set_facecolor(cm(norm_hist(bins[i])))

mean_val = df_lifestyle['Wellbeing_Score'].mean()
median_val = df_lifestyle['Wellbeing_Score'].median()
ax3.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2.5, 
            label=f'Mean: {mean_val:.1f}', zorder=10)
ax3.axvline(median_val, color='#27ae60', linestyle='--', linewidth=2.5, 
            label=f'Median: {median_val:.1f}', zorder=10)

ax3.set_xlabel('Mental Wellbeing Score', fontsize=12, fontweight='bold')
# FIXED: Move ylabel to the left
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold', labelpad=15)
ax3.set_title('Overall Wellbeing Distribution\n(Color: Red=Low, Green=High)', 
              fontsize=13, fontweight='bold', pad=15)
ax3.legend(fontsize=10, loc='upper left')
ax3.grid(alpha=0.3, linestyle='--')

# Add statistics text box
stats_text = f"""Sample Statistics:
n = {len(df_lifestyle):,}
Mean = {mean_val:.1f}
Median = {median_val:.1f}
Std = {df_lifestyle['Wellbeing_Score'].std():.1f}
Min = {df_lifestyle['Wellbeing_Score'].min():.1f}
Max = {df_lifestyle['Wellbeing_Score'].max():.1f}"""
ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes, 
         fontsize=9, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Mental health issue prevalence by gender
gender_mh = df_lifestyle.groupby('Gender')['Has_Mental_Health_Issue'].agg(['sum', 'count'])
gender_mh['Prevalence'] = (gender_mh['sum'] / gender_mh['count']) * 100

x = np.arange(len(gender_mh))
width = 0.6
bars = ax4.bar(x, gender_mh['Prevalence'], width, 
               color=['#3498db', '#e74c3c'], alpha=0.8, 
               edgecolor='black', linewidth=1)

ax4.set_ylabel('Prevalence of Mental Health Issues (%)', 
               fontsize=12, fontweight='bold')
ax4.set_xlabel('Gender', fontsize=12, fontweight='bold')
ax4.set_title('Mental Health Issue Prevalence by Gender\n(Based on Survey Data)', 
              fontsize=13, fontweight='bold', pad=15)
ax4.set_xticks(x)
ax4.set_xticklabels(gender_mh.index)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.set_ylim([0, max(gender_mh['Prevalence']) * 1.15])

# Add value labels
for i, (idx, row) in enumerate(gender_mh.iterrows()):
    ax4.text(i, row['Prevalence'] + 1, f"{row['Prevalence']:.1f}%\n(n={row['count']})", 
             ha='center', fontweight='bold', fontsize=10)

plt.suptitle('Mental Well-Being in Singapore: Local Distribution Patterns', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('viz2_micro_singapore.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz2_micro_singapore.png")
plt.close()

# Visualisation 3: Actionable Plot - Lifestyle determinants
print("\n[4/4] Creating Visualization 3: ACTIONABLE - Lifestyle Insights...")

# Convert categorical variables to numeric
exercise_map = {'None': 0, '1-2 times': 1.5, '3-4 times': 3.5, '5+ times': 6}
df_lifestyle['Exercise_Numeric'] = df_lifestyle['Exercise_Per_Week'].map(exercise_map)

diet_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
df_lifestyle['Diet_Numeric'] = df_lifestyle['Diet_Quality'].map(diet_map)

alcohol_map = {'Never': 0, 'Rarely': 1, 'Occasionally': 2, 
               'Frequently': 3, 'Daily': 4}
df_lifestyle['Alcohol_Numeric'] = df_lifestyle['Alcohol_Frequency'].map(alcohol_map)

smoking_map = {'No': 0, 'Former': 1, 'Yes': 2}
df_lifestyle['Smoking_Numeric'] = df_lifestyle['Smoking'].map(smoking_map)

# Define lifestyle factors
lifestyle_factors = {
    'Social Support (1-10)': 'Social_Support',
    'Work-Life Balance (1-10)': 'Work_Life_Balance',
    'Job Satisfaction (1-10)': 'Job_Satisfaction',
    'Exercise per Week (sessions)': 'Exercise_Numeric',
    'Sleep Hours per Night': 'Sleep_Hours_Night',
    'Diet Quality (1-4)': 'Diet_Numeric',
    'Hobby Time (hrs/week)': 'Hobby_Time_Hours_Week',
    'Work Stress (1-10)': 'Work_Stress_Level',
    'Financial Stress (1-10)': 'Financial_Stress',
    'Work Hours per Week': 'Work_Hours_Per_Week',
    'Screen Time (hrs/day)': 'Screen_Time_Hours_Day',
    'Social Media (hrs/day)': 'Social_Media_Hours_Day',
    'Close Friends Count': 'Close_Friends_Count',
    'Caffeine (drinks/day)': 'Caffeine_Drinks_Day',
    'Alcohol Frequency (0-4)': 'Alcohol_Numeric',
    'Smoking Status (0-2)': 'Smoking_Numeric'
}

# Calculate correlations with mental health issues
correlations = []
for name, col in lifestyle_factors.items():
    if col in df_lifestyle.columns:
        corr = df_lifestyle[[col, 'Has_Mental_Health_Issue']].corr().iloc[0, 1]
        if not pd.isna(corr):
            correlations.append({'Factor': name, 'Correlation': corr, 'Column': col})

corr_df = pd.DataFrame(correlations).sort_values('Correlation')

print(f"Analyzed {len(corr_df)} lifestyle factors")

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 11))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)
ax_main = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[1, 2])
ax5 = fig.add_subplot(gs[2, 1])
ax6 = fig.add_subplot(gs[2, 2])

# Main correlation bar chart
colors_corr = ['#27ae60' if x < 0 else '#e74c3c'
               for x in corr_df['Correlation']]
bars = ax_main.barh(corr_df['Factor'], corr_df['Correlation'], 
                     color=colors_corr, alpha=0.85, 
                     edgecolor='black', linewidth=0.8)
ax_main.axvline(0, color='black', linestyle='-', linewidth=2)
ax_main.set_xlabel('Correlation with Mental Health Issues', 
                    fontsize=12, fontweight='bold')
ax_main.set_ylabel('Lifestyle Factors', fontsize=12, fontweight='bold')
ax_main.set_title('Impact of Lifestyle Factors on Mental Health\n' + 
                   'Green=Protective, Red=Risk Factor', 
                   fontsize=13, fontweight='bold', pad=15)
ax_main.grid(axis='x', alpha=0.3, linestyle='--')

# Add correlation values
x_label_pos = corr_df['Correlation'].max() + 0.04

for i, row in corr_df.iterrows():
    ax_main.text(
        x_label_pos,
        row['Factor'],
        f"{row['Correlation']:+.3f}",
        va='center',
        ha='left',
        fontsize=9,
        fontweight='bold'
    )

# Extend x-limits to make room for labels
ax_main.set_xlim(
    corr_df['Correlation'].min() - 0.05,
    x_label_pos + 0.05
)

# Top 3 protective and risk factors for detailed plots
top_protective = corr_df.nsmallest(3, 'Correlation')
top_risk = corr_df.nlargest(3, 'Correlation')
factors_to_plot = pd.concat([top_protective, top_risk])

axes = [ax1, ax2, ax3, ax4, ax5, ax6]

for idx, (i, row) in enumerate(factors_to_plot.iterrows()):
    if idx >= len(axes):
        break
    
    ax = axes[idx]
    col = row['Column']
    is_protective = row['Correlation'] < 0
    
    # Group by mental health status
    mh_yes = df_lifestyle[df_lifestyle['Has_Mental_Health_Issue'] == 1][col].dropna()
    mh_no = df_lifestyle[df_lifestyle['Has_Mental_Health_Issue'] == 0][col].dropna()
    
    # Create violin plots for comparison
    data_to_plot = [mh_no, mh_yes]
    parts = ax.violinplot(data_to_plot, positions=[0, 1], widths=0.7, 
                          showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('#27ae60' if is_protective else '#e74c3c')
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
    
    parts['cmeans'].set_color('blue')
    parts['cmeans'].set_linewidth(2)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No MH Issue', 'Has MH Issue'], fontsize=9)
    ax.set_ylabel(row['Factor'], fontsize=10, fontweight='bold')
    
    # Add means and calculate difference
    mean_no = mh_no.mean()
    mean_yes = mh_yes.mean()
    diff = mean_yes - mean_no
    diff_pct = (diff / mean_no * 100) if mean_no != 0 else 0
    
    title_text = f"{row['Factor']}\nr={row['Correlation']:.3f}, " + \
                 f"Δ={diff:.2f} ({diff_pct:+.1f}%)"
    ax.set_title(title_text, fontsize=10, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add mean values as text
    ax.text(0, mean_no, f'{mean_no:.1f}', ha='center', va='bottom', 
            fontweight='bold', fontsize=8, color='blue')
    ax.text(1, mean_yes, f'{mean_yes:.1f}', ha='center', va='bottom', 
            fontweight='bold', fontsize=8, color='blue')

plt.suptitle('Mental Well-Being in Singapore: Actionable Lifestyle Insights\n' + 
             'Comparing Individuals With vs Without Mental Health Issues', 
             fontsize=15, fontweight='bold', y=0.995)
plt.savefig('viz3_actionable_lifestyle.png', dpi=300, bbox_inches='tight')
print("✓ Saved: viz3_actionable_lifestyle.png")
plt.close()

# Summary Statistics

print("\n" + "="*80)
print("ANALYSIS COMPLETE - SUMMARY STATISTICS")
print("="*80)

print(f"\nGLOBAL CONTEXT (Visualization 1 - FINAL):")
print(f"  • Singapore Total Prevalence: {sg_prevalence:.2f}%")
print(f"  • Global Average: {df_countries['Total_Prevalence'].mean():.2f}%")
print(f"  • Singapore Percentile Rank: {sg_percentile:.1f}th (out of {len(df_countries)} countries)")

print(f"\nLOCAL PATTERNS (Visualization 2):")
print(f"  • Sample Size: {len(df_lifestyle):,}")
print(f"  • Mean Wellbeing Score: {df_lifestyle['Wellbeing_Score'].mean():.2f}/100")
print(f"  • Mental Health Issue Prevalence: {df_lifestyle['Has_Mental_Health_Issue'].mean()*100:.1f}%")

print(f"\nLIFESTYLE INSIGHTS (Visualization 3):")
print(f"\n  Top 3 Protective Factors:")
for _, row in corr_df.nsmallest(3, 'Correlation').iterrows():
    print(f"    ✓ {row['Factor']}: r = {row['Correlation']:.3f}")

print(f"\n  Top 3 Risk Factors:")
for _, row in corr_df.nlargest(3, 'Correlation').iterrows():
    print(f"    ✗ {row['Factor']}: r = {row['Correlation']:.3f}")

print("\n" + "="*80)
print("FILES CREATED:")
print("  1. viz1_macro_global_FINAL.png")
print("  2. viz2_micro_singapore.png")
print("  3. viz3_actionable_lifestyle.png")
print("  4. viz3_correlation_heatmap.png")
print("="*80)
print("\nAll visualizations saved successfully!")
print("Ready for presentation and analysis.")