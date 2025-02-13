import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

def shirley_correction(raman_shift, intensity):
    """
    Aplica la corrección de Shirley para un espectro.

    Parameters:
        raman_shift (array-like): Valores del eje X (Raman Shift).
        intensity (array-like): Intensidades del espectro (eje Y).

    Returns:
        corrected_intensity (array-like): Intensidades corregidas.
    """
    if len(raman_shift) != len(intensity):
        raise ValueError("La longitud de 'Ramanshift' y la intensidad no coincide.")

    corrected_intensity = intensity.copy()
    start = corrected_intensity[0]
    end = corrected_intensity[-1]

    for _ in range(100):  # Máximo 100 iteraciones
        background = start + (end - start) * np.cumsum(corrected_intensity) / np.sum(corrected_intensity)
        corrected_intensity = intensity - background
        corrected_intensity[corrected_intensity < 0] = 0  # Evitar valores negativos

    return corrected_intensity


df = pd.read_csv('limpio.csv', delimiter = ',')
# Extraer Raman Shift
raman_shift = df['Ramanshift'].values
num_points = len(raman_shift)  # Número esperado de puntos en cada espectro 

# Identificar categorías de manera dinámica sin duplicados
categories = {}
column_names = df.columns[1:].unique()  # Obtener nombres únicos

for col in column_names:
    match = re.match(r"([a-zA-Z]+)", col)  # Extraer solo las letras al inicio,
    #aca porque se agregaron subfijos de su qlo
    if match:
        category = match.group(1)
        if category not in categories:
            categories[category] = []
        categories[category].append(col)
    else:
        if "other" not in categories:
            categories["other"] = []
        categories["other"].append(col)

# Función de corrección de Shirley
def shirley_correction(raman_shift, intensity):
    """
    Aplica la corrección de Shirley para un espectro.

    Parameters:
        raman_shift (array-like): Valores del eje X (Raman Shift).
        intensity (array-like): Intensidades del espectro (eje Y).

    Returns:
        corrected_intensity (array-like): Intensidades corregidas.
    """
    if len(raman_shift) != len(intensity):
        raise ValueError(f"Dimensiones no coinciden: {len(raman_shift)} vs {len(intensity)}")

    corrected_intensity = intensity.copy()
    start = corrected_intensity[0]
    end = corrected_intensity[-1]

    for _ in range(100):  # Máximo 100 iteraciones
        background = start + (end - start) * np.cumsum(corrected_intensity) / np.sum(corrected_intensity)
        corrected_intensity = intensity - background
        corrected_intensity[corrected_intensity < 0] = 0  # Evitar valores negativos

    return corrected_intensity

# Aplicar corrección de Shirley y almacenar datos corregidos
corrected_spectra = {}

for category, cols in categories.items():
    for col in cols:
        intensity = df[col].values
        
        # Ajustar longitud si es necesario
        if len(intensity) < num_points:
            print(f"⚠️ Advertencia: La columna {col} tiene menos datos ({len(intensity)}). Se rellenará con ceros.")
            intensity = np.pad(intensity, (0, num_points - len(intensity)), constant_values=0)
        elif len(intensity) > num_points:
            print(f"⚠️ Advertencia: La columna {col} tiene más datos ({len(intensity)}). Se recortará.")
            intensity = intensity[:num_points]

        corrected_spectra[col] = shirley_correction(raman_shift, intensity)

# Convertir a DataFrame
df_corrected = pd.DataFrame(corrected_spectra)
df_corrected.insert(0, 'Ramanshift', raman_shift)

# Asignar colores dinámicamente usando Matplotlib
unique_categories = list(categories.keys())
color_map = plt.colormaps.get_cmap("tab10")
category_colors = {category: color_map(i / max(1, len(unique_categories) - 1)) for i, category in enumerate(unique_categories)}

# Graficar con colores por tipo detectado
plt.figure(figsize=(14, 10))

for category, cols in categories.items():
    for col in cols:
        plt.plot(df_corrected['Ramanshift'], df_corrected[col], alpha=0.6, 
                 label=category if col == cols[0] else "", color=category_colors[category])

plt.title("Espectros Raman Corregidos (Shirley)", fontsize=16)
plt.xlabel("Raman Shift (cm⁻¹)", fontsize=14)
plt.ylabel("Intensidad Corregida", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()