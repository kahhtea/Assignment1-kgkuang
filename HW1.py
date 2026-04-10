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

# Apply all cleaning
df['Annual_Rev'] = df['Annual_Rev'].apply(clean_revenue)
df['Rev_Growth'] = pd.to_numeric(df['Rev_Growth'].str.replace('%','').str.replace('+',''), errors='coerce')
df['RD_Spend'] = df['RD_Spend'].apply(clean_rd)
df['AI_Status'] = df['AI_Status'].apply(clean_ai)
df['Digital_Sales'] = pd.to_numeric(df['Digital_Sales'].str.replace('%',''), errors='coerce')
df['Team_Size'] = pd.to_numeric(df['Team_Size'].str.replace(',',''), errors='coerce')
df['Founded'] = pd.to_numeric(df['Founded'], errors='coerce')

# Drop incomplete rows
df.dropna(inplace=True)

# ---- Verify ----
print(df.dtypes)
print(df.head(10))
print(df['AI_Status'].value_counts())
print(f"Rows remaining: {len(df)}")



