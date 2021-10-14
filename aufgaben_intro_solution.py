import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Erstellen Sie eine Liste mit Werten unterschiedlicher Datentypen
x1 = [1, 2.0, 'A', [8, 9, 10], {'A': 1, 'B': 2}]

# 2) Wandeln Sie die Liste in ein Numpy-Array um. Funktioniert das?
x2 = np.array(x1)
# Ja, es funktioniert, aber es kommt eine Warnung, es ist zukünftig nicht mehr zulässig: "VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray"

x2o = np.array(x1, dtype=object)
# Das funktioniert ohne Probleme

# 3) Iterieren Sie mit einer Schleife über die Liste und geben Sie Index, Typ und Wert der Elemente aus
for i in range(len(x1)):
    print(i, type(x1[i]), x1[i])

# 4) Erstellen Sie ein Numpy-Array mit 20 beliebigen Float-Werten und geben Sie Minimum, Maximum, Mittelwert und Standardabweichung aus
x4 = np.array([1.0, 2.0, 4.0, 3.0, 2.0, 2.2, 1.8, 4.6, 2.8, 3.9, 2.5, 3.3, 5.5, 7.7, 8.8, 9.9, 0.1, 0.2, 0.3, 0.4])
print(np.min(x4), np.max(x4), np.mean(x4), np.std(x4))

# 5) Erstellen Sie einen Pandas Dataframe aus einem Dictionary (denken Sie dabei an die Struktur eines Dataframes)
d = {'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'col2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'col3': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
df = pd.DataFrame(d)
print(df)
# Die Listen im Dictionary müssen die gleiche Länge besitzen, sonst tritt ein Fehler auf: "ValueError: arrays must all be same length"

# 6) Sortieren Sie den Dataframe absteigend nach einer Spalte Ihrer Wahl
df = df.sort_values('col1', ascending=False)
print(df)

# 7) Speichern Sie den Dataframe als CSV ab
df.to_csv('data/mydf.csv')

# 8) Erstellen Sie einen Linien-Plot in Matplotlib mit drei Werte-Linien und einer Legende
plt.plot(x4, label='x4')
plt.plot(sorted(x4), label='x4_sorted')
plt.plot(sorted(x4, reverse=True), label='x4_sorted_reverse')
plt.legend()
plt.show()

# 9) Weisen Sie den Linien explizit eine Farbe zu
plt.plot(x4, label='x4', c='r')
plt.plot(sorted(x4), label='x4_sorted', c='g')
plt.plot(sorted(x4, reverse=True), label='x4_sorted_reverse', c='b')
plt.legend()
plt.show()

# 10) Fügen Sie Achsenbeschriftungen und eine Titel-Überschrift Ihrer Wahl zum Plot hinzu
plt.plot(x4, label='x4', c='r')
plt.plot(sorted(x4), label='x4_sorted', c='g')
plt.plot(sorted(x4, reverse=True), label='x4_sorted_reverse', c='b')
plt.legend()
plt.title('MyTitle')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.show()

# 11) Speichern Sie den Plot (ohne in vorher anzuzeigen!) als PDF ab

plt.plot(x4, label='x4', c='r')
plt.plot(sorted(x4), label='x4_sorted', c='g')
plt.plot(sorted(x4, reverse=True), label='x4_sorted_reverse', c='b')
plt.legend()
plt.title('MyTitle')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.savefig('data/myplot.pdf')