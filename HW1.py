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

