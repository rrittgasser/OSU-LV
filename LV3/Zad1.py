import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

print(f"Broj mjerenja: {len(data)}")
print(f"Tipovi veličina: {data.dtypes}")
print(f"Izostale vrijednosti: {data.isnull().sum()}")
print(f"Duplicirane vrijednosti: {data.duplicated().sum()}")
data['Make'] = data['Make'].astype('category')
data['Model'] = data['Model'].astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data['Transmission'] = data['Transmission'].astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')
print(f"Tipovi veličina: {data.dtypes}")

print(f"3 automobila sa najvecom potrosnjom:\n{data.sort_values(by=['Fuel Consumption City (L/100km)'], ascending=False).head(3)[['Make','Model','Fuel Consumption City (L/100km)']]}")
print(f"3 automobila sa najmanjom potrosnjom:\n{data.sort_values(by=['Fuel Consumption City (L/100km)'],ascending=False).tail(3)[['Make','Model','Fuel Consumption City (L/100km)']]}")


print('Broj vozila sa veličinom motora između 2.5 i 3.5 L: ', data[(data["Engine Size (L)"] > 2.5) & (data["Engine Size (L)"] < 3.5)].__len__()) 
print('Prosječna a C02 emisija plinova: ', data[(data["Engine Size (L)"] > 2.5) & (data["Engine Size (L)"] < 3.5)]["CO2 Emissions (g/km)"].mean())

print('Broj vozila proizvodača Audi:', len(data[data["Make"] == "Audi"]))
print('Prosječna emisija C02 plinova automobila proizvodača Audi koji imaju 4 cilindara: ', data[(data["Make"] == "Audi") & (data["Cylinders"] == 4)]["CO2 Emissions (g/km)"].mean())

broj_vozila = data['Cylinders'].value_counts().sort_index()
print("Broj vozila po cilindru", broj_vozila)

prosjecna_emisija = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print("Prosječna emisija", prosjecna_emisija)

print('Prosječna gradska potrošnja u slučaju vozila koja koriste dizel: ', data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].mean())
print('Prosječna gradska potrošnja u slučaju vozila koja koriste benzin:', data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].mean())
print('Median vrijednost potrošnje u slučaju vozila koja koriste dizel:', data[data["Fuel Type"] == "D"]["Fuel Consumption City (L/100km)"].median())
print('Median vrijednost potrošnje u slučaju vozila koja koriste benzin: ', data[data["Fuel Type"] == "X"]["Fuel Consumption City (L/100km)"].median())

print('Vozilo s 4 cilindra koje koristi dizelski motor I ima najvecu gradsku potrošnju goriva:\n ', data[(data["Cylinders"] == 4) & (data["Fuel Type"] == "D")].sort_values(by=["Fuel Consumption City (L/100km)"], ascending=False).head(1))

print("Broj vozila sa ručnim mjenjačem:", data[data["Transmission"].str.startswith("M")].__len__())

print("Korelacija:", data.corr(numeric_only=True))