import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Données
Surface = np.array([30,50,60,80,100,120,150]).reshape(-1,1)
prix = [95,130,155,180,200,230,275]

# Modèle
model = LinearRegression()
model.fit(Surface, prix)

# Prédictions pour toutes les surfaces
price_predict = model.predict(Surface)

# Prédiction pour 90 m²
surface_test = np.array([[90]])
prix_90 = model.predict(surface_test)

# Graphique
plt.scatter(Surface, prix, color='blue', label='Données réelles')
plt.plot(Surface, price_predict, color='red', label='Régression linéaire')
plt.scatter(surface_test, prix_90, color='green', label=f'Prédiction 90m²: {prix_90[0]:.2f}k€')

plt.title('Prix des maisons en fonction de la surface')
plt.xlabel('Surface (m²)')
plt.ylabel('Prix (k€)')
plt.legend()
plt.grid(True)
plt.show()
