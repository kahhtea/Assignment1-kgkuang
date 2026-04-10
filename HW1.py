import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://bana290-assignment1.netlify.app/"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

table = soup.find("table", class_="directory-table")
rows = table.find_all("tr")

headers = [td.get_text(strip=True) for td in rows[0].find_all("td")]

data = []
for row in rows[1:]:
    cells = row.find_all("td")
    if not cells:
        continue
    row_data = []
    for i, td in enumerate(cells):
        if i == 0:
            row_data.append(td.find("strong").get_text(strip=True))
        else:
            row_data.append(td.get_text(strip=True))
    data.append(row_data)

df = pd.DataFrame(data, columns=headers)

print(df.head(10))
print(df.shape)
print(df.info())

#Cleaning the data
import numpy as np

df.columns = ['Firm', 'Segment', 'HQ_Region', 'Founded', 'Team_Size', 
              'Annual_Rev', 'Rev_Growth', 'RD_Spend', 'AI_Status', 
              'Cloud_Stack', 'Digital_Sales', 'Compliance_Tier', 
              'Fraud_Exposure', 'Funding_Stage', 'Customer_Accts']

def clean_revenue(val):
    val = str(val).lower().strip().replace('$', '').replace('usd', '').replace(',', '')
    try:
        if 'm' in val:
            return float(val.replace('million', '').replace('mn','').replace('m','').strip()) * 1000000
        else:
            return float(val)
    except:
        return np.nan
    
# Clean R&D Spend
def clean_rd(val):
    val = str(val).lower().strip().replace('$','').replace('usd','').replace(',','').strip()
    if val in ['--', 'unknown', 'n/a', 'na', 'none', '']:
        return np.nan
    try:
        if 'm' in val:
            return float(val.replace('million','').replace('mn','').replace('m','').strip()) * 1_000_000
        else:
            return float(val)
    except:
        return np.nan

# Clean AI Status → binary 1/0
def clean_ai(val):
    val = str(val).lower().strip()
    if val in ['adopted', 'ai enabled', 'yes']:
        return 1
    elif val in ['--', 'unknown', 'n/a', '']:
        return np.nan
    else:
        return 0
    
def clean_customer_accts(val):
    val = str(val).strip().replace(',', '')
    try:
        if 'K' in val or 'k' in val:
            return float(val.replace('K', '').replace('k', '')) * 1000
        else:
            return float(val)
    except:
        return np.nan



# Apply all cleaning
df['Annual_Rev'] = df['Annual_Rev'].apply(clean_revenue)
df['Rev_Growth'] = pd.to_numeric(df['Rev_Growth'].str.replace('%','').str.replace('+',''), errors='coerce')
df['RD_Spend'] = df['RD_Spend'].apply(clean_rd)
df['AI_Status'] = df['AI_Status'].apply(clean_ai)
df['Digital_Sales'] = pd.to_numeric(df['Digital_Sales'].str.replace('%',''), errors='coerce')
df['Team_Size'] = pd.to_numeric(df['Team_Size'].str.replace(',',''), errors='coerce')
df['Founded'] = pd.to_numeric(df['Founded'], errors='coerce')
df['Customer_Accts'] = df['Customer_Accts'].apply(clean_customer_accts)

# Drop incomplete rows
df.dropna(inplace=True)

# Print
print(df.dtypes)
print(df.head(10))
print(df['AI_Status'].value_counts())
print(f"Rows remaining: {len(df)}")


#3. Analyze the Data
import statsmodels.api as sm

# Naive OLS - ignoring selection bias
X = sm.add_constant(df['AI_Status'])
y = df['Rev_Growth']

ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())
print(f"Naive AI coefficient: {ols_model.params['AI_Status']:.4f}")

#Test Assumptions
covariates = ['Team_Size', 'Founded', 'Annual_Rev', 'RD_Spend', 
              'Digital_Sales', 'Customer_Accts']
#Propensity Score Model
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler

X_psm = df[covariates]
y_psm = df['AI_Status']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_psm)

# Fit logistic regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_scaled, y_psm)

# Add propensity scores to dataframe
df['pscore'] = lr.predict_proba(X_scaled)[:, 1]
print(df['pscore'].describe())

import matplotlib.pyplot as plt

treated = df[df['AI_Status'] == 1]['pscore']
control = df[df['AI_Status'] == 0]['pscore']

plt.figure(figsize=(8,5))
plt.hist(treated, bins=20, alpha=0.5, label='AI Adopted (Treated)', color='blue')
plt.hist(control, bins=20, alpha=0.5, label='No AI (Control)', color='red')
plt.xlabel('Propensity Score')
plt.ylabel('Count')
plt.title('Common Support: Propensity Score Distribution')
plt.legend()
plt.savefig('common_support.png')
plt.show()
print("Plot saved!")

#SMD
def calculate_smd(treated, control):
    mean_diff = treated.mean() - control.mean()
    pooled_std = np.sqrt((treated.std()**2 + control.std()**2) / 2)
    return mean_diff / pooled_std

print("\nSMD Before Matching:")
for col in covariates:
    smd = calculate_smd(
        df[df['AI_Status']==1][col],
        df[df['AI_Status']==0][col]
    )
    print(f"  {col}: {smd:.4f}")

from sklearn.neighbors import NearestNeighbors

# Separate treated and control
treated_df = df[df['AI_Status'] == 1].copy()
control_df = df[df['AI_Status'] == 0].copy()

# Match each treated firm to nearest control by propensity score
nn = NearestNeighbors(n_neighbors=1)
nn.fit(control_df[['pscore']])
distances, indices = nn.kneighbors(treated_df[['pscore']])

# Build matched dataset
matched_control = control_df.iloc[indices.flatten()].copy()
matched_df = pd.concat([treated_df, matched_control])

# SMD After Matching
print("\nSMD After Matching:")
for col in covariates:
    smd = calculate_smd(
        matched_df[matched_df['AI_Status']==1][col],
        matched_df[matched_df['AI_Status']==0][col]
    )
    print(f"  {col}: {smd:.4f}")

# Re-estimate AI effect on matched sample
X_matched = sm.add_constant(matched_df['AI_Status'])
y_matched = matched_df['Rev_Growth']

psm_model = sm.OLS(y_matched, X_matched).fit()
print(psm_model.summary())
print(f"\nNaive OLS coefficient:  {ols_model.params['AI_Status']:.4f}")

print(f"PSM matched coefficient: {psm_model.params['AI_Status']:.4f}")