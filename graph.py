import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ace_tools as tools

# Données extraites
data = [
    (1, 201.17), (1.5, 202.97), (0, 188.58), (1, 201.17),
    (2.5, 203.79), (2.5, 203.79), (2.5, 203.79), (2.5, 203.79),
    (4, 194.12), (5, 189.71), (6, 210.61), (7.5, 193.83),
    (7, 202.41), (8.5, 194.89), (8, 196.12), (9, 191.88), (10, 175.85)
]

# Création du DataFrame
df = pd.DataFrame(data, columns=["Gamma", "Durée (s)"])

# Agrégation pour éviter les duplications
df_grouped = df.groupby("Gamma", as_index=False).agg({"Durée (s)": "mean"}).sort_values(by="Gamma")

# Affichage dans le tableau
tools.display_dataframe_to_user(name="Durée moyenne par Gamma", dataframe=df_grouped)
