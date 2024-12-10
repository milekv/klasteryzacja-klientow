
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

dane = {
    "ID_Klienta": range(1, 101),
    "Roczny_Dochod": np.random.randint(20000, 120000, 100),
    "Wynik_Zakupowy": np.random.randint(1, 100, 100)
}
df = pd.DataFrame(dane)

df.to_csv("dane_klientow.csv", index=False)

df = pd.read_csv("dane_klientow.csv")
print("Przykładowe dane:")
print(df.head())

X = df[["Roczny_Dochod", "Wynik_Zakupowy"]]

inercja = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inercja.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inercja, marker="o")
plt.title("Metoda łokcia")
plt.xlabel("Liczba klastrów")
plt.ylabel("Inercja")
plt.savefig("metoda_lokcia.png")  
plt.show()

liczba_klastrow = 3  
kmeans = KMeans(n_clusters=liczba_klastrow, random_state=42)
df["Klaster"] = kmeans.fit_predict(X)

df.to_csv("klienci_z_klastrami.csv", index=False)
# wizualizacja klastrow tutaj
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="Roczny_Dochod",
    y="Wynik_Zakupowy",
    hue="Klaster",
    data=df,
    palette="viridis",
    s=100
)
plt.title("Klasteryzacja Klientów")
plt.xlabel("Roczny Dochód")
plt.ylabel("Wynik Zakupowy")
plt.legend(title="Klaster")
plt.savefig("klasteryzacja_klientow.png")  
plt.show()
