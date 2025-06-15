import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
# Données
Surface,prix = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
prix= prix.reshape(prix.shape[0],1)
print(prix.shape)
# Modèle
model = LinearRegression()
model.fit(Surface, prix)

# Prédictions pour toutes les surfaces
price_predict = model.predict(Surface)

# Prédiction pour 90 m²
surface_test = np.array([[2]])
new_price = model.predict(surface_test)
print(new_price)
# Graphique
plt.scatter(Surface, prix, color='blue', label='Données réelles')
plt.plot(Surface, price_predict, color='red', label='Régression linéaire')

plt.title('Prix des maisons en fonction de la surface')
plt.xlabel('Surface (m²)')
plt.ylabel('Prix (k€)')
plt.legend()
plt.grid(True)
plt.show()
