
import pandas as pd

data = pd.read_excel('Estaciones/estaciones.xlsx')
print(data)
# df_mask=data[(data['Latitud']>=5.0) & (data['Latitud']<=8.0)]
df_mask=data[(data['Canal']=="BHE") | (data['Canal']=="BHN") | (data['Canal']=="BHZ")]

print(df_mask)