# main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

bd_name = 'limpio.csv' #Este archivo contiene los datos espectroscópicos que serán leídos
df = pd.read_csv(bd_name, delimiter = ',', header = None)
#print(df)


#GRAFICAMOS LOS ESPECTROS SIN NORMALIZAR#

raman_shift = df.iloc[:, 0]  # EXTRAEMOS TODA LA PRIMERA COLUMNA
#print(raman_shift)

intensity = df.iloc[1:, 1:] # EXTRAEMOS TODAS DEMAS COLUMNAS EXCEPTO LA PRIMERA FILA Y PRIMERA COLUMNA
#print(intensity)

tipos = df.iloc[0, 1:] # EXTRAEMOS LA PRIMERA FILA MENOS DE LA PRIMERA COLUMNA
#print(type)

cant_tipos = tipos.nunique() # PARA EL EJEMPLO DE LIMPIO.CSV CANT_TIPOS TENDRA VALOR 4 YA QUE HAY 4 TIPOS (collagen,lipids,glycogen,DNA)
#print(cant_tipos)

tipos_nombres = df.iloc[0, :].unique() # OBTENEMOS LOS NOMBRES DE LOS TIPOS
#print(tipos_nombres)


# Generar una paleta de colores con el número de colores igual a cant_tipos
colores = plt.cm.get_cmap('hsv', cant_tipos)

# Crear el diccionario de asignación de colores
asignacion_colores = {tipo: mcolors.to_hex(colores(i)) for i, tipo in enumerate(tipos_nombres)}

# Mostrar el diccionario de colores
print("Diccionario de asignación de colores:")
print(asignacion_colores)


#AHORA QUE YA TENGO ASIGNADO UN COLOR POR CADA TIPO TENGO QUE GRAFICAR LOS ESPECTROS#

# Graficar los espectros
plt.figure(figsize=(10, 6))

# Iterar sobre cada columna de intensidad y graficar con el color correspondiente
for col_idx, tipo in enumerate(tipos):
    plt.plot(raman_shift, intensity.iloc[:, col_idx], label=tipo, color=asignacion_colores[tipo])

# Personalizar la gráfica
plt.xlabel('Raman Shift (cm⁻¹)')
plt.ylabel('Intensity (a.u.)')
plt.title('Espectros sin Normalizar')
plt.legend(loc='upper right')
plt.show()