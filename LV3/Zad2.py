import pandas as pd
import numpy as np
import matplotlib . pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist').set_title('CO2 Emissions (g/km)')

data['Fuel Type'] = data['Fuel Type'].astype('category')
colors = {'Z': 'brown', 'X': 'red', 'E': 'blue', 'D': 'black'}
data.plot.scatter(x="Fuel Consumption City (L/100km)", y="CO2 Emissions (g/km)", c=data["Fuel Type"].map(colors), s=50)

data.boxplot(column=['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')

plt.figure()
data.groupby(by=['Fuel Type'])['Make'].count().plot(kind='bar').set_title('Number of cars by fuel type')

plt.figure()
data.groupby(by=['Cylinders'])['CO2 Emissions (g/km)'].mean().plot(kind='bar')

plt.show()
