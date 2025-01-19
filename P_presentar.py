 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:15:28 2024

@author: diego
"""


# main.py
import os
import numpy as np
import pandas as pd
import csv # PARA ENCONTRAR EL TIPO DE DELIMITADOR DEL ARCHIVO .CSV
import re # PARA LA EXPRECION REGULAR DE LOS SUFIJOS
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler # PARA LA NORMALIZACION POR LA MEDIA 
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter # Para suavizado de Savitzky Golay
from scipy.ndimage import gaussian_filter # PARA EL FILTRO GAUSSIANO
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram

def archivo_existe(ruta_archivo):
    return os.path.isfile(ruta_archivo)

#nombre = input("Por favor, ingresa tu nombre: ")
#print(f"Hola, {nombre}!")

existe = False
archivo_nombre = input("Ingrese el nombre del archivo: ")
 

# Función para detectar el delimitador automáticamente por que los archivos pueden estar ceparados por , o ; etc
def identificar_delimitador(archivo):
    with open(archivo, 'r') as file:
        muestra_csv = file.read(4096)  # Lee una muestra de 4096 bytes
        #print("LA MUESTRA DEL CSV ES:")
        #print(muestra_csv)
        caracter = csv.Sniffer()
        delimitador = caracter.sniff(muestra_csv).delimiter
    return delimitador




def detectar_labels(df): #Detecta si los labels están en la fila o en la columna para ver si hacemos la transpuesta  o no

    # Verificar la primera fila (si contiene strings)
    if df.iloc[0].apply(lambda x: isinstance(x, str)).all():
        return "fila" #si los labels están en la primera fila
    
    # Verificar la primera columna (si contiene strings)
    elif df.iloc[:, 0].apply(lambda x: isinstance(x, str)).all():
        return "columna" #si los labels están en la primera columna
    
    # Si no hay etiquetas detectadas
    return "ninguno" #si no se detectan labels.



while existe == False:   
    if archivo_existe(archivo_nombre):   
        bd_name = archivo_nombre #Este archivo contiene los datos espectroscópicos que serán leídos
        delimitador = identificar_delimitador(bd_name)
        print("EL DELIMITADOR ES: ", delimitador)
        df = pd.read_csv(bd_name, delimiter = delimitador , header=None)
        existe = True
        if detectar_labels(df) == "columna" :
            print("SE HIZO LA TRASPUESTA")
            df = df.T
        else:
            print("NO SE HIZO LA TRANSPUESTA")
    else:
        print("El archivo no existe.")
        archivo_nombre = input("Ingrese el nombre del archivo: ")

print("DF ANTES DEL CORTE")
print(df)
print(df.shape)

print("LOGRO LEER EL ARCHIVO")




menor_cant_filas = df.dropna().shape[0] # Buscamos la columna con menor cantidad de intensidades
#print("menor cantidad de filas:", menor_cant_filas)

df_truncado = df.iloc[:menor_cant_filas] # Hacemos los cortes para igualar las columnas

df = df_truncado

#print("DF DESPUES DEL CORTE")
#print(df)
print(df.shape)


# renombramos la celda [0,0]

print("Cambiar a cero: ",df.iloc[0,0])

df.iloc[0,0] = float(0)

print("Cambiar a cero: ",df.iloc[0,0])

print(df)



# HACEMOS LA ELIMINACION DE LOS SUFIJOS EN CASO DE TENER


for col in df.columns:
    valor = re.sub(r'[_\.]\d+$', '', str(df.at[0, col]).strip())  # Eliminar sufijos con _ o .
    try:
        df.at[0, col] = float(valor)  # Convertir de nuevo a float si es posible
    except ValueError:
        df.at[0, col] = valor  # Mantener como string si no es convertible


print("Luego de eliminar los sufijos")
print(df)





'''
    PREPARAMOS EL SIGUIENTE MENU
'''


def main():
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción: ")
        
        if opcion == '1':
            metodo = 1
            print("Como deseas ver el espectro")
            print("1-Grafico completo")
            print("2-Grafico acotado")
            print("3-Grafico por tipo")
            print("4-Grafico acotado por tipo")
            metodo_grafico = int(input("Opcion: "))
            if metodo_grafico == 1:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                mostrar_espectros(df2,raman_shift,metodo,0,0)
            elif metodo_grafico == 2:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                espectro_acotado(df2, 0,1)
            elif metodo_grafico == 3:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                grafico_tipo(df2,raman_shift,metodo,0,0)
            # else
            #     #grafico_tipo_acotado()
            print("Procesando los datos")
            print("Por favor espere un momento...")
        elif opcion == '2':
            metodo = 2
            print("Como deseas ver el espectro")
            print("1-Grafico completo")
            print("2-Grafico acotado")
            print("3-Grafico por tipo")
            print("4-Grafico acotado por tipo")
            metodo_grafico = int(input("Opcion: "))
            if metodo_grafico == 1:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                mostrar_espectros(df_media_pca,raman_shift,metodo,0,0)
            elif metodo_grafico == 2:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                espectro_acotado(df_media_pca, 0,2)
            elif metodo_grafico == 3:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                grafico_tipo(df_media_pca,raman_shift,metodo,0,0)
            # else
            #     #grafico_tipo_acotado()
            print("Procesando los datos")
            print("Por favor espere un momento...")
        elif opcion == '3':
            metodo = 3
            print("Como deseas ver el espectro")
            print("1-Grafico completo")
            print("2-Grafico acotado")
            print("3-Grafico por tipo")
            print("4-Grafico acotado por tipo")
            metodo_grafico = int(input("Opcion: "))
            if metodo_grafico == 1:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                mostrar_espectros(df_concatenado_cabecera_nueva_area,raman_shift,metodo,0,0)
            elif metodo_grafico == 2:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                espectro_acotado(df_concatenado_cabecera_nueva_area, 0,3)
            elif metodo_grafico == 3:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                grafico_tipo(df_concatenado_cabecera_nueva_area,raman_shift,metodo,0,0)
            # else
            #     #grafico_tipo_acotado()
            print("Procesando los datos")
            print("Por favor espere un momento...")           
        elif opcion == '4':
            metodo = 4
            print("Como deseas ver el espectro")
            print("1-Grafico completo")
            print("2-Grafico acotado")
            print("3-Grafico por tipo")
            print("4-Grafico acotado por tipo")
            metodo_grafico = int(input("Opcion: "))
            if metodo_grafico == 1:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                suavizado_saviztky_golay(0,0)
            elif metodo_grafico == 2:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                espectro_acotado(suavizado_saviztky_golay(0,1), 2,4)
            elif metodo_grafico == 3:
                print("NORMALIZAR POR:")
                print("1-Media")
                print("2-Area")
                print("3-Sin normalizar")
                opcion = int(input("Selecciona una opción: "))
                if opcion == 1:
                    print("Procesando los datos")
                    print("Por favor espere un momento...")
                    grafico_tipo(suavizado_saviztky_golay(0,2),raman_shift,metodo,'1',0)
                elif opcion == 2:
                    print("Procesando los datos")
                    print("Por favor espere un momento...")
                    grafico_tipo(suavizado_saviztky_golay(0,3),raman_shift,metodo,'2',0)
                else: 
                    print("Procesando los datos")
                    print("Por favor espere un momento...")
                    grafico_tipo(suavizado_saviztky_golay(0,4),raman_shift,metodo,'3',0)
            # else
            #     #grafico_tipo_acotado()                  
        elif opcion == '5':
            metodo = 5
            print("Como deseas ver el espectro")
            print("1-Grafico completo")
            print("2-Grafico acotado")
            print("3-Grafico por tipo")
            print("4-Grafico acotado por tipo")
            metodo_grafico = int(input("Opcion: "))
            if metodo_grafico == 1:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                suavizado_filtroGausiano(0,0)
            elif metodo_grafico == 2:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                espectro_acotado(suavizado_filtroGausiano(0,1),2,5)
            elif metodo_grafico == 3: 
                 print("NORMALIZAR POR:")
                 print("1-Media")
                 print("2-Area")
                 print("3-Sin normalizar")
                 opcion = int(input("Selecciona una opción: "))
                 if opcion == 1:
                     print("Procesando los datos")
                     print("Por favor espere un momento...")
                     grafico_tipo(suavizado_filtroGausiano(0,2),raman_shift,metodo,'1',0)
                 elif opcion == 2:
                     print("Procesando los datos")
                     print("Por favor espere un momento...")
                     grafico_tipo(suavizado_filtroGausiano(0,3),raman_shift,metodo,'2',0)
                 else: 
                     print("Procesando los datos")
                     print("Por favor espere un momento...")
                     grafico_tipo(suavizado_filtroGausiano(0,4),raman_shift,metodo,'3',0)
            # else
            #     #grafico_tipo_acotado()                
        elif opcion == '6':
            metodo = 5
            print("Como deseas ver el espectro")
            print("1-Grafico completo")
            print("2-Grafico acotado")
            print("3-Grafico por tipo")
            print("4-Grafico acotado por tipo")
            metodo_grafico = int(input("Opcion: "))
            if metodo_grafico == 1:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                suavizado_mediamovil(0,0)
            elif metodo_grafico == 2:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                espectro_acotado(suavizado_mediamovil(0,1), 2,6)
            elif metodo_grafico == 3:
                 print("NORMALIZAR POR:")
                 print("1-Media")
                 print("2-Area")
                 print("3-Sin normalizar")
                 opcion = int(input("Selecciona una opción: "))
                 if opcion == 1:
                     print("Procesando los datos")
                     print("Por favor espere un momento...")
                     grafico_tipo(suavizado_mediamovil(0,2),raman_shift,metodo,'1',0)
                 elif opcion == 2:
                     print("Procesando los datos")
                     print("Por favor espere un momento...")
                     grafico_tipo(suavizado_mediamovil(0,3),raman_shift,metodo,'2',0)
                 else: 
                     print("Procesando los datos")
                     print("Por favor espere un momento...")
                     grafico_tipo(suavizado_mediamovil(0,4),raman_shift,metodo,'3',0)                                            
            # else
            #     #grafico_tipo_acotado()                    
        elif opcion == '7':
            print("Procesando los datos")
            print("Por favor espere un momento...")
            mostrar_pca(0)       
        elif opcion == '8':
            metodo = 7
            print("Como deseas ver el espectro")
            print("1-Grafico completo")
            print("2-Grafico acotado")
            print("3-Grafico por tipo")
            print("4-Grafico acotado por tipo")
            metodo_grafico = int(input("Opcion: "))
            if metodo_grafico == 1:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                primera_derivada(0,0)
            elif metodo_grafico == 2:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                espectro_acotado(primera_derivada(0,3), 2,7)
            elif metodo_grafico == 3:
                print("NORMALIZAR POR:")
                print("1-Media")
                print("2-Area")
                print("3-Sin normalizar")
                opcion = int(input("Selecciona una opción: "))
                if opcion == 1:
                    print("Procesando los datos")
                    print("Por favor espere un momento...")
                    grafico_tipo(primera_derivada(df2,4),raman_shift,8,'1',0)
                elif opcion == 2:
                    print("Procesando los datos")
                    print("Por favor espere un momento...")
                    grafico_tipo(primera_derivada(df_media_pca,4),raman_shift,8,'2',0)
                else: 
                    print("Procesando los datos")
                    print("Por favor espere un momento...")
                    grafico_tipo(primera_derivada(df_concatenado_cabecera_nueva_area,4),raman_shift,8,'3',0)   
            # else
            #     #grafico_tipo_acotado()  
               
        elif opcion == '9':
            metodo = 8
            print("Como deseas ver el espectro")
            print("1-Grafico completo")
            print("2-Grafico acotado")
            print("3-Grafico por tipo")
            print("4-Grafico acotado por tipo")
            metodo_grafico = int(input("Opcion: "))
            if metodo_grafico == 1:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                segunda_derivada(0,0)
            elif metodo_grafico == 2:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                espectro_acotado(segunda_derivada(0,3), 2,9)
            elif metodo_grafico == 3:
                print("NORMALIZAR POR:")
                print("1-Media")
                print("2-Area")
                print("3-Sin normalizar")
                opcion = int(input("Selecciona una opción: "))
                if opcion == 1:
                    print("Procesando los datos")
                    print("Por favor espere un momento...")
                    grafico_tipo(primera_derivada(df2,4),raman_shift,9,'1',0)
                elif opcion == 2:
                    print("Procesando los datos")
                    print("Por favor espere un momento...")
                    grafico_tipo(primera_derivada(df_media_pca,4),raman_shift,9,'2',0)
                else: 
                    print("Procesando los datos")
                    print("Por favor espere un momento...")
                    grafico_tipo(primera_derivada(df_concatenado_cabecera_nueva_area,4),raman_shift,9,'3',0)   
            # else
            #     #grafico_tipo_acotado()              
        elif opcion == '10':
            metodo = 8
            print("Como deseas ver el espectro")
            print("1-Grafico completo")
            print("2-Grafico acotado")
            print("3-Grafico por tipo")
            print("4-Grafico acotado por tipo")
            metodo_grafico = int(input("Opcion: "))
            if metodo_grafico == 1:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                correcion_LineaB(0,0)
            elif metodo_grafico == 2:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                espectro_acotado(correcion_LineaB(0,5), 2,9)
            elif metodo_grafico == 3:
                print("NORMALIZAR POR:")
                print("1-Media")
                print("2-Area")
                print("3-Sin normalizar")
                opcion = int(input("Selecciona una opción: "))
                if opcion == 1:
                    print("Procesando los datos")
                    print("Por favor espere un momento...")
                    grafico_tipo(correcion_LineaB(df2,4),raman_shift,9,'1',0)
                elif opcion == 2:
                    print("Procesando los datos")
                    print("Por favor espere un momento...")
                    grafico_tipo(correcion_LineaB(df_media_pca,4),raman_shift,9,'2',0)
                else: 
                    print("Procesando los datos")
                    print("Por favor espere un momento...")
                    grafico_tipo(correcion_LineaB(df_concatenado_cabecera_nueva_area,4),raman_shift,9,'3',0)   
            # else
            #     #grafico_tipo_acotado()              
             
        elif opcion == '11':
             print("Procesando los datos")
             print("Por favor espere un momento...")
             correcion_shirley(0,0)
        elif opcion == '12':
            print("Procesando los datos")
            print("Por favor espere un momento...")
            graficar_loadings(wavelengths=None, n_components=2)
        elif opcion == '13':
             print("Procesando los datos")
             print("Por favor espere un momento...")
             espectro_acotado(0,0,1)
        elif opcion == '14': 
             print("Procesando los datos")
             print("Por favor espere un momento...")
             hca()
        elif opcion == '15':
             print("Saliendo del programa...")
             break
        else:
            print("Opción no válida. Inténtalo de nuevo.")

def mostrar_menu():
      print("\n--- Menú Principal ---")
      print("1. MOSTRAR ESPECTROS")
      print("2. NORMALIZAR POR MEDIA")
      print("3. NORMALIZAR POR AREA")
      print("4. SUAVIZADO POR SAVIZTKY-GOLAY")
      print("5. SUAVIZADO POR FILTRO GAUSIANO")
      print("6. SUAVIZADO POR MEDIA MOVIL")
      print("7. PCA")
      print("8. PRIMERA DERIVADA")
      print("9. SEGUNDA DERIVADA")
      print("10. CORRECCION BASE LINEAL")
      print("11. CORRECION SHIRLEY")
      print("12. GRAFICO DE LOADINGS")
      print("13. ESPECTRO ACOTADO")
      print("14. GRAFICO HCA")
      print("15. Salir")
      


 
#GRAFICAMOS LOS ESPECTROS SIN NORMALIZAR#

raman_shift = df.iloc[1:, 0].reset_index(drop=True)  # EXTRAEMOS TODA LA PRIMERA COLUMNA, reset_index(drop=True) SIRVE PARA QUE EL INDICE COMIENCE EN 0 Y NO EN 1
#print("gbdgb")
print(raman_shift)

intensity = df.iloc[1:, 1:] # EXTRAEMOS TODAS DEMAS COLUMNAS EXCEPTO LA PRIMERA FILA Y PRIMERA COLUMNA
#print(intensity)   

tipos = df.iloc[0, 1:] # EXTRAEMOS LA PRIMERA FILA MENOS DE LA PRIMERA COLUMNA
#print(tipos)
types=tipos.tolist() #OJO AUN NO AGREGAMOS ESTA LINEA A ULTIMO.PY
#print(types)

cabecera = df.iloc[[0]].copy() # EXTRAEMOS LA PRIMERA FILA 
print(cabecera)
#print(cabecera.shape)


cant_tipos = tipos.nunique() # PARA EL EJEMPLO DE LIMPIO.CSV CANT_TIPOS TENDRA VALOR 4 YA QUE HAY 4 TIPOS (collagen,lipids,glycogen,DNA)
#print(cant_tipos)

tipos_nombres = df.iloc[0, 1:].unique() # OBTENEMOS LOS NOMBRES DE LOS TIPOS
#print(tipos_nombres)


# Seleccionar un colormap distintivo
cmap = plt.cm.Spectral  # Puedes probar con "hsv", "Set3", "Spectral", etc.

#cmap = plt.colormaps['hsv']  # Usamos solo el nombre del colormap, Obtenemos el colormap sin especificar el número de colores

# Nos aseguramos de que `colores` es una lista
colores = [cmap(i) for i in np.linspace(0, 1, len(tipos_nombres))]  # Genera una lista de colores

# Crear el diccionario de asignación de colores
asignacion_colores = {tipo: mcolors.to_hex(colores[i]) for i, tipo in enumerate(tipos_nombres)}
#print(asignacion_colores)

diccionario=pd.DataFrame(asignacion_colores.items())
 #print(diccionario)

#AHORA QUE YA TENGO ASIGNADO UN COLOR POR CADA TIPO TENGO QUE GRAFICAR LOS ESPECTROS#



"""
VARIABLES DE MOSTRAR ESPECTROS
"""
df2 = df.copy()
df2.columns = df2.iloc[0]
#print(df2)
df2 = df2.drop(0).reset_index(drop=True) #eliminamos la primera fila
df2 = df2.drop(df2.columns[0], axis=1) #eliminamos la primera columna el del rama_shift
#print(df2) # aca ya tenemos la tabla de la manera que necesitamos, fila cero es la cabecera con los nombres de los tipos anteriormente eran indice numericos consecutivos
df2 = df2.apply(pd.to_numeric, errors='coerce') #CONVERTIMOS A NUMERICO
#print("EL DATAFRAME DEL ESPECTRO SIN NORMALIZAR ES")
print(df2) #ESTA VARIABLE SE USA PARA EL PCA TAMBIEN
#print(df2.shape)



"""
VARIABLES DE NORMALIZAR POR LA MEDIA    tratar de hacer por la forma del ejemplo y no por z-core para ver si se soluciona lo de la raya
"""

global df_media_pca
scaler = StandardScaler() 
cal_nor = scaler.fit_transform(intensity) #calcula la media y desviación estándar
#print(cal_nor)
dato_normalizado = pd.DataFrame(cal_nor, columns=intensity.columns) # lo convertimos de vuelta en un DataFrame
#print(dato_normalizado)
df_concatenado = pd.concat([cabecera,dato_normalizado], axis=0, ignore_index=True)
#print(df_concatenado)
#  Convertimos la primera fila en cabecera
df_concatenado.columns = df_concatenado.iloc[0]  # Asigna la primera fila como nombres de columna
# Eliminamos la primera fila (ahora es la cabecera) y reseteamos el índice
df_concatenado_cabecera_nueva = df_concatenado[1:].reset_index(drop=True)
#print(df_concatenado_cabecera_nueva.head(50))
df_media_pca= pd.DataFrame(df_concatenado_cabecera_nueva.iloc[:,1:])
#print("EL ESPECTRO NORMALIZADO POR LA MEDIA ES")
#print(df_media_pca) #ESTA VARIABLE SE USA PARA EL PCA TAMBIEN
#print('normalizacion media')
#df_media_pca.to_csv("nor_media_df.csv", index=False)


"""
 VARIABLES DE NORMALIZAR POR AREA    tratar de hacer otro sin np.trap para ver si se soluciona lo de la raya
"""

global df_concatenado_cabecera_nueva_area
df3 = pd.DataFrame(intensity)
#print("DataFrame de Intensidades:")
#print(df3)
df3 = df3.apply(pd.to_numeric, errors='coerce')  # Convierte a numérico, colocando NaN donde haya problemas
#print(df3)
np_array = raman_shift.astype(float).to_numpy() #CONVERTIMOS INTENSITY AL TIPO NUMPY POR QUE POR QUE NP.TRAPZ UTILIZA ESE TIPO DE DATOS
print("valor de np_array: ")
print(np_array)

df3_normalizado = df3.copy()
print("EL VALOR DE DF3 ES :")
print(df3)
# Cálculamos el área bajo la curva para cada columna
#print("\nÁreas bajo la curva para cada columna:")
for col in df3.columns:
    #print(df3[col])
    #print(df3_normalizado[col])
    
    # np.trapz para hallar el area bajo la curva por el metodo del trapecio
    area = (np.trapz(df3[col], np_array)) *-1  #MULTIPLIQUE POR -1 PARA QUE EL GRAFICO SALGA TODO HACIA ARRIBA ESTO SE DEBE A QUE EL RAMAN_SHIFT ESTA EN FORMA DECRECIENTE
    if area != 0:
        df3_normalizado[col] = df3[col] / area
    else:
        print(f"Advertencia: El área de la columna {col} es cero y no se puede normalizar.") #seguro contra errores de división por cero 
#print(df3_normalizado)
df_concatenado_area = pd.concat([cabecera,df3_normalizado], axis=0, ignore_index=True)
#print(df_concatenado_area)
# Paso 1: Convertir la primera fila en cabecera
df_concatenado_area.columns = df_concatenado_area.iloc[0]  # Asigna la primera fila como nombres de columna
# Paso 2: Eliminar la primera fila (ahora es la cabecera) y resetear el índice
df_concatenado_cabecera_nueva_area = df_concatenado_area[1:].reset_index(drop=True)
# AHORA ELIMINAMOS LA COLUMNA CON VALORES NAN
df_concatenado_cabecera_nueva_area = df_concatenado_cabecera_nueva_area.dropna(axis=1, how='all')
#print("ESPECTRO NORMALIZADO POR EL AREA")
#print(df_concatenado_cabecera_nueva_area) #ESTA VARIABLE SE USA PARA EL PCA TAMBIEN
#print('entro 10')
#df_concatenado_cabecera_nueva_area.to_csv("nor_area_df.csv", index=False)




# MOSTRAMOS LA LEYENDA PARA CADA TIPO
plt.figure(figsize=(2,2))    
for index, row in diccionario.iterrows():
    #print('entro 15')
    tipo = row[0]   # Nombre del tipo (por ejemplo, 'collagen')
    color = row[1]  # Color asociado (por ejemplo, '#ff0000')
    plt.plot([], [], color=color, label=tipo) 
# Mostrar la leyenda y el gráfico
#print('entro 20')
plt.legend(loc='center')
plt.grid(False)
plt.title(f'Cantidad de tipos encontrados {cant_tipos}')
plt.axis('off')
plt.show()




def mostrar_espectros(datos,raman_shift,metodo,opcion,m_suavi):
    
    
    
    #print("ENTRO EN MOSTRAR ESPECTROS")
    #print(datos)
    
    # Graficar los espectros
    plt.figure(figsize=(10, 6))
 
  
    leyendas_tipos = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
    pos_y=0
    for col in datos.columns :
        #print('entro normal')
        #col agarrar el nombre de los tipos de cada columna
        for tipo in asignacion_colores:
            #print("wwwwwww")
            if tipo == col :
                color_actual= asignacion_colores[tipo] #ACA YA ENCONTRAMOS EL COLOR CORRESPONDIENTE A ESE TIPO   
                if isinstance(col, str):  #Verifica que el nombre de la columna sea un string
                    if tipo in leyendas_tipos:
                        plt.plot(raman_shift , datos[col], color=color_actual, alpha=0.5, linewidth = 0.1,label=col)   
                        break
                    else:
                        plt.plot(raman_shift , datos[col], color=color_actual, alpha=0.5, linewidth = 0.1) 
                        leyendas_tipos.add(tipo) 
                pos_y+=1 
    
    # TODO ESTE CONDICIONAL ES SOLO PARA QUE EL TITULO DEL GRAFICO MUESTRE POR CUAL METODO SE NORMALIZO
    if metodo == 1:
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectros del archivo {bd_name}')
        plt.show()
    elif metodo == 2:
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectros del archivo {bd_name} Normalizado por la Media')
        plt.show()
    elif metodo == 3:
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectros del archivo {bd_name} Normalizado por Area')
        plt.show()
    elif metodo == 4:
        if opcion == '1':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por la media')
            plt.show()   
        elif opcion == '2':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado Area')
            plt.show() 
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Saviztky_golay y sin Normalizar ')
            plt.show()          
    elif metodo == 5:
        if opcion == '1':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Filtro Gaussiano y Normalizado por la media')
            plt.show()   
        elif opcion == '2':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Filtro Gaussiano y Normalizado Area')
            plt.show() 
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Filtro Gaussiano y sin Normalizar ')
            plt.show() 
    elif metodo == 6:
        if opcion == '1':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Media Movil y Normalizado por la media')
            plt.show()   
        elif opcion == '2':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Media Movil y Normalizado Area')
            plt.show() 
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Suavizado por Media Movil y sin Normalizar ')
            plt.show() 
    elif metodo == 7:
            print("hola PCA")
    elif metodo == 10:
            if opcion == '1':
                if m_suavi == 1:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por SAVIZTKY-GOLAY y Normalizado por la media')
                    plt.show()
                elif m_suavi == 2:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por la media')
                    plt.show()
                elif m_suavi == 3:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por la media')
                    plt.show()
                else:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} sin suavizar y Normalizado por la media')
                    plt.show()
            elif opcion == '2':
                if m_suavi == 1:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por SAVIZTKY-GOLAY y Normalizado por Area')
                    plt.show()
                elif m_suavi == 2:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por Area')
                    plt.show()
                elif m_suavi == 3:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por Area')
                    plt.show()
                else:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} sin Suavizar y Normalizado por Area')
                    plt.show()
            else:
                if m_suavi == 1:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por SAVIZTKY-GOLAY y Sin Normalizar')
                    plt.show()
                elif m_suavi == 2:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Sin Normalizar')
                    plt.show()
                elif m_suavi == 3:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por MEDIA MOVIL y Sin Normalizar')
                    plt.show()
                else:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} sin Suavizar y Sin Normalizar')
                    plt.show()
    # elif metodo == 11:
    # elif metodo == 12:
    elif metodo == 13:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Acotado ')
            plt.show() 
    else:
        print("NO HAY GRAFICA DISPONIBLE PARA ESTA OPCION")




# SUAVIZADO POR SAVIZTKY-GOLAY

def suavizado_saviztky_golay(normalizado_pca, pca_op):  #acordarse que se puede suavizar por la media, area y directo
    if pca_op == 0 or pca_op == 1:
        print("NORMALIZAR POR: xdf")
        print("1-MEDIA")
        print("2-AREA")
        print("3-SIN NORMALIZAR")
        opcion = input("Selecciona una opción: ")
        ventana = int(input("INGRESE EL VALOR DE LA VENTANA: "))
        orden = int(input("INGRESE EL VALOR DEL ORDEN: "))
        print("Procesando los datos")
        print("Por favor espere un momento...")
     
        if opcion == '1'  :
            normalizado_pca = df_media_pca
            #print(normalizado_pca)
        elif opcion == '2' :
            normalizado_pca = df_concatenado_cabecera_nueva_area
            #print(normalizado_pca)
        elif opcion == '3' :
            normalizado_pca = df2
            #print(normalizado_pca)
        else:
            print("OPCION NO VALIDA")
            print("SALIR...")
            #mostrar_menu()
    else:
        print("entro 3")
        ventana = int(input("INGRESE EL VALOR DE LA VENTANA: "))
        orden = int(input("INGRESE EL VALOR DEL ORDEN: "))
        print("Procesando los datos")
        print("Por favor espere un momento...")
        #ESTAS CONDICIONALES FUNCIONA PARA CUANDO ENTRE EN GRAFICAR POR TIPOS
        if pca_op == 2:
            normalizado_pca = df_media_pca
        elif pca_op == 3:
            normalizado_pca = df_concatenado_cabecera_nueva_area
        else:
            normalizado_pca = df2
     
      
    
    dato = normalizado_pca.to_numpy() #PASAMOS LOS DATOS A NUMPY POR QUE SAVGOL_FILTER USA SOLO NUMPY COMO PARAMETRO (PIERDE LA CABECERA DE TIPOS AL HACER ESTO)

    suavizado = savgol_filter(dato, window_length=ventana, polyorder=orden)
    suavizado_pd = pd.DataFrame(suavizado) # PASAMOS SUAVIZADO A PANDAS Y GUARDAMOS EN SUAVIZADO_PD
    suavizado_pd.columns = normalizado_pca.columns # AGREGAMOS LA CABECERA DE TIPOS
    
    #print(suavizado_pd)
    
    if pca_op == 0 or pca_op == 1:
        #print("Entro aca 2")
        #print(suavizado_pd) 
        if pca_op == 0:
            mostrar_espectros(suavizado_pd,raman_shift,4,opcion,0)
        else:
            return suavizado_pd
    else:
        #print("ESPECTRO SUAVIZADO POR SAVITZKY GOLAY")
        #print("suavizado savitkz golay:",suavizado_pd.shape) 
        #print(suavizado_pd)
        #print("aca si entro xD")
        return suavizado_pd
    




 
# SUAVIZADO POR FILTRO GAUSIANO

def suavizado_filtroGausiano(normalizado_pca, pca_op):  #acordarse que se puede suavizar por la media, area y directo
    if pca_op == 0 or pca_op == 1:
        print("NORMALIZAR POR:")
        print("1-Media")
        print("2-Area")
        print("3-Sin normalizar")
        opcion = input("Selecciona una opción: ")
        sigma = int(input("INGRESE EL VALOR DE SIGMA: ")) #Un valor mayor de sigma produce un suavizado más fuerte
        print("Procesando los datos")
        print("Por favor espere un momento...")
        
     
        if opcion == '1'  :
            normalizado = df_media_pca
            print("entro 1")
        elif opcion == '2' :
            normalizado = df_concatenado_cabecera_nueva_area
        elif opcion == '3' :
            normalizado = df2
        else:
            print("OPCION NO VALIDA")
            print("Salir...")
    else:
        sigma = int(input("INGRESE EL VALOR DE SIGMA: ")) #Un valor mayor de sigma produce un suavizado más fuerte
        normalizado = normalizado_pca
        print("Procesando los datos")
        print("Por favor espere un momento...")
        if pca_op == 2:
            print("entro en pca_op = 2")
            normalizado = df_media_pca
        elif pca_op == 3:
            print("entro en pca_op = 3")
            normalizado = df_concatenado_cabecera_nueva_area
        else:
            print("entro en pca_op = 4")
            normalizado = df2
     
        
    #print("avanzo")
    #print(pca_op)
    #print(type(normalizado))  
    #print(normalizado)
    dato = normalizado.to_numpy() #PASAMOS LOS DATOS A NUMPY (PIERDE LA CABECERA DE TIPOS AL HACER ESTO)
    #print(dato)
    #print(type(dato))
    #print(dato.dtype)  # me tira que es  Object, eso quiere decir que el array numpy contiene datos que no son de un tipo numerico uniforme
    # por lo que tendremos que forza su conversion con astype(float)
    dato = np.array(dato, dtype=float)
    #print(dato)
    #print(dato.dtype)
    suavizado_gaussiano = gaussian_filter(dato,sigma=sigma)
    #print(suavizado_gaussiano)
    suavizado_gaussiano_pd = pd.DataFrame(suavizado_gaussiano)
    #print("xxxxxxxxxxxxxxxxxxxxxxxxxx")
    #print(suavizado_gaussiano_pd)
     
    suavizado_gaussiano_pd.columns = cabecera.iloc[0,1:].values #agregamos la cabecera 
    #print(suavizado_gaussiano_pd)
    #print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRrr")
    if pca_op == 0 or pca_op == 1:
        if pca_op == 0:
            #print("ESPECTRO SUAVIZADO POR FILTRO GAUSSIANO")
            #print(suavizado_gaussiano_pd)
            mostrar_espectros(suavizado_gaussiano_pd,raman_shift,5,opcion,0)
        else:
            #print("risas58")
            #print(suavizado_gaussiano_pd)
            return suavizado_gaussiano_pd
    else:
        #print("ESPECTRO SUAVIZADO POR FILTRO GAUSSIANO")
        #print(suavizado_gaussiano_pd)
        #print("hizo bien su suavizado xDDDDDDDD")
        return suavizado_gaussiano_pd








def suavizado_mediamovil(normalizado_pca, pca_op):
    if pca_op == 0 or pca_op == 1:
        print("NORMALIZAR POR:")
        print("1-Media")
        print("2-Area")
        print("3-Sin normalizar")
        opcion = input("Selecciona una opción: ")
        ventana = int(input("INGRESE EL VALOR DE LA VENTANA: "))
        print("Procesando los datos")
        print("Por favor espere un momento...")
        
     
        if opcion == '1'  :
            normalizado = df_media_pca
        elif opcion == '2' :
            normalizado = df_concatenado_cabecera_nueva_area
        elif opcion == '3' :
            normalizado = df2
        else:
            print("OPCION NO VALIDA")
            print("Salir...")
    else:
        ventana = int(input("INGRESE EL VALOR DE LA VENTANA: "))
        normalizado = normalizado_pca
        print("Procesando los datos")
        print("Por favor espere un momento...")
        if pca_op == 2:
            normalizado = df_media_pca
        elif pca_op == 3:
            normalizado = df_concatenado_cabecera_nueva_area
        else:
            normalizado = df2
      
    suavizado_media_movil = pd.DataFrame()
    
    
    suavizado_media_movil = normalizado.rolling(window=ventana, center=True).mean() # mean() es para hallar el promedio
    
    if pca_op == 0 or pca_op == 1:
        if pca_op == 0:
            #print("ESPECTRO SUAVIZADO POR FILTRO GAUSSIANO")
            #print(suavizado_media_movil)
            mostrar_espectros(suavizado_media_movil,raman_shift,6,opcion,0)
        else:
            #print("risas2")
            #print(suavizado_media_movil)
            return suavizado_media_movil
        #print("ESPECTRO SUAVIZADO POR MEDIA MOVIL")
        #print(suavizado_media_movil)
        #mostrar_espectros(suavizado_media_movil,6,opcion,0)
    else:
        #print("ESPECTRO SUAVIZADO POR MEDIA MOVIL")
        #print("risas3")
        #print(suavizado_media_movil)
        return suavizado_media_movil
    
    
   # print(suavizado_media_movil)








def espectro_acotado(datos, pca_op,nor_op):
    
    #print("ENTRO EN EL ESPECTRO ACOTADO")
    #print(datos)
    
    if pca_op == 0:
        df_aux = df.iloc[1:,1:].to_numpy()
        print("df_aux")
        print(df_aux)
    else:
        print("entro en el else")
        df_aux = datos.to_numpy()
        
    #print("ENTRO EN EL ESPECTRO ACOTADO222")
    #print(df_aux)
    #print(df_aux.shape)
    
    cabecera_np = df.iloc[0,:].to_numpy()   # la primera fila contiene los encabezados
    cabecera_np = cabecera_np[1:]
    #print("la cabeceras son:")
    #print(cabecera_np)
    #print(cabecera_np.shape)
    
    
    intensidades_np = df_aux[: , :] # apartamos las intensidades
    #print("intensidades_np son:")
    #print(intensidades_np)
    #print(intensidades_np.shape)
    
    
    raman =  df.iloc[:, 0].to_numpy().astype(float)  # Primera columna (Raman Shift)
    raman = raman[1:]
    intensidades =  intensidades_np[:, 1:].astype(float)  # Columnas restantes (intensidades)
    # print("RAMAN:")
    # print(raman)
    # print(raman.shape)
    # print("INTENSIDADES:")
    # print(intensidades)
    # print(intensidades.shape)
    # Filtrado del rango de las intensidades
    min_rango = int(input("Rango minimo: "))  # Cambia según lo que necesites
    max_rango = int(input("Rango maximo: "))  # Cambia según lo que necesites
    
    
    indices_acotados = (raman >= min_rango) & (raman <= max_rango) #retorna false o true para los que estan en el rango
    #print("Indices acotados")
    #print(indices_acotados)
    #print(indices_acotados.shape)
    
    raman_acotado = raman[indices_acotados]
    intensidades_acotadas = intensidades[indices_acotados,:]

    
        
    # # Imprimir resultados
    # print("Raman Shift Acotado:")
    # print(raman_acotado)
    # print("\nIntensidades Acotadas:")
    # print(intensidades_acotadas)
        
    
    # Crear un DataFrame a partir de las dos variables
    df_acotado = pd.DataFrame(
    data=np.column_stack([raman_acotado, intensidades_acotadas]),
    columns=["Raman Shift"] + list(cabecera_np[1:])  # Encabezados para el DataFrame
    )

    # Mostrar el DataFrame resultante
    #print(df_acotado)
    # df_acotado = pd.DataFrame(df_acotado)
    # print(df_acotado)
    if pca_op == 0 or pca_op == 2:
   
        # Graficar los espectros
        plt.figure(figsize=(10, 6))
    
        #print("entro en el graficador")
        #DESCOMENTAR EL CODIGO DE ABAJO ESE ES MIO, EL DE ARRIBA ES CHATGPT  
         
        leyendas = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
        pos_y=0
        for col in df_acotado.columns :
                #print('entro normal')
              for tipo in asignacion_colores:
    
                    if tipo == col :
                      color_actual= asignacion_colores[tipo] #ACA YA ENCONTRAMOS EL COLOR CORRESPONDIENTE A ESE TIPO   
    
                      if isinstance(col, str):  #Verifica que el nombre de la columna sea un string
    
                            if tipo in leyendas:
                                
                                #print("error 1")
                                plt.plot(raman_acotado , df_acotado[col], color=color_actual, alpha=0.6, linewidth = 0.1,label=col) 
                                '''raman_shift:LE PASAMOS TODAS LAS INTENSIDADES , df2[col]= LE PASAMOS TODAS LAS COLUMNAS CON EL MISMO TIPO'''
      
                                break
                            else:
                                #print("error 2")
                                #print(raman)
                                #print(df_acotado[col])
                                plt.plot(raman_acotado , df_acotado[col], color=color_actual, alpha=0.6, linewidth = 0.1) 
                                leyendas.add(tipo) 
                      pos_y+=1 
           
    
        if nor_op == 1:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} en el rango de {min_rango} a {max_rango}')
            plt.show()
        elif nor_op == 2:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} normalizado por la media en el rango de {min_rango} a {max_rango}')
            plt.show()
        elif nor_op == 3:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} normalizado por la area en el rango de {min_rango} a {max_rango}')
            plt.show() 
        elif nor_op == 4:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name}  SUAVIZADO POR SAVIZTKY-GOLAY en el rango de {min_rango} a {max_rango}')
            plt.show() 
        elif nor_op == 5:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name}  SUAVIZADO POR FILTRO GAUSIANO en el rango de {min_rango} a {max_rango}')
            plt.show() 
        elif nor_op == 6:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name}  SUAVIZADO POR MEDIA MOVIL en el rango de {min_rango} a {max_rango}')
            plt.show()
        elif nor_op == 8:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Primera derivada del archivo {bd_name} en el rango de {min_rango} a {max_rango}')
            plt.show()
        elif nor_op == 9:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Segunda derivada del archivo {bd_name} en el rango de {min_rango} a {max_rango}')
            plt.show()
    else: 
        return df_acotado , raman_acotado # creo que no hace falta retornarn nada ya que si una funcion le llama seria solamente para graficarla y retorna tiene quw retornar tambien su raman_shift acotado






def primera_derivada(normalizado, pca_op):
            
    #print("entro en la funcion der")
            
    if pca_op == 0 or pca_op == 1 or pca_op == 3:
        print("NORMALIZAR POR:")
        print("1-Media")
        print("2-Area")
        print("3-Sin normalizar")
        opcion = input("Selecciona una opción: ")
           
        if opcion == '1'  :
            normalizado = df_media_pca
        elif opcion == '2' :
            normalizado = df_concatenado_cabecera_nueva_area
        elif opcion == '3' :
            normalizado = df2
        else:
            print("OPCION NO VALIDA")
            print("Salir...")
       
        #print("EL normalizado.SHAPE antes ES:", normalizado.shape)
        
        print("DESEA SUAVIZAR")
        print("1. SI")
        print("2. NO")
        opcion_s =  int(input("OPCION: "))
        if opcion_s == 1:
            print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
            print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
            print("2. SUAVIZADO POR FILTRO GAUSIANO")
            print("3. SUAVIZADO POR MEDIA MOVIL")
            metodo_suavizado = input("OPCION: ")
            if metodo_suavizado == '1':
                aux = suavizado_saviztky_golay(normalizado,2)
                #print("EL aux.SHAPE antes ES:", aux.shape)
                #normalizado = suavizado_saviztky_golay(normalizado,1)
            elif metodo_suavizado == '2':
                #print("normalizado shape por FG = ", normalizado.shape)
                #normalizado = suavizado_filtroGausiano(normalizado,1)
                aux = suavizado_filtroGausiano(normalizado,2)
                #print("normalizado shape despues por FG = ", normalizado.shape)
            elif metodo_suavizado == '3':
                #normalizado = suavizado_mediamovil(normalizado,1) 
                aux = suavizado_mediamovil(normalizado,2) 
        else:
                metodo_suavizado = '4'         
                #print("No se va a normalizar, directamente la derivada")
                 
                  # si viene 0 que haga todo eso pero si viene 1 desde la funcion del PCA que haller directo la primera derivada con esos parametros
           
    else:
       #daba error por que no retornaba nada
       normalizado_pca = normalizado

    if (pca_op == 0 or pca_op == 3) and opcion_s == 1  :
        #print("entro aca")
        normalizado_f = aux
        #print("EL DF NORMALIZADO_F ES:")
        #print(normalizado_f)
    elif (pca_op == 0 or pca_op == 3) and opcion_s == 2:
        normalizado_f = normalizado
    else:
        normalizado_f = normalizado_pca #este es para cuando sea la funcion derivada sea llamado por la funcion del PCA
        #print("aca va lo del PCA")
    
    
    df_derivada = normalizado_f.apply(pd.to_numeric, errors='coerce') # PASAMOS A NUMERICO SI ES NECESARIO
    #print("xXXXXXXXxxXXXX")
    #print(df_derivada)
    
    # Calcular la primera derivada
    df_derivada_diff = df_derivada.diff()
    
    # Imprimir la primera derivada
    #print("Primera Derivada:")
    #print(df_derivada_diff)
    
    
    plt.figure(figsize=(10, 6))
    

    if pca_op == 0 and pca_op != 3:
        #print("LA PRIMERA DERIVADA ES:")
        #print(df_derivada_diff.shape)
        #mostrar_espectros(df_derivada_diff, 8,opcion,metodo_suavizado)
        leyendas_tipos = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
        pos_y=0
        for col in df_derivada_diff.columns :
            #print('entro normal')
            for tipo in asignacion_colores:
                #print("wwwwwww")
                if tipo == col :
                    color_actual= asignacion_colores[tipo] #ACA YA ENCONTRAMOS EL COLOR CORRESPONDIENTE A ESE TIPO   
                    #print(tipo,'==',col,'color=',color_actual) 
                    #print("ccccccc")
                    if isinstance(col, str):  #Verifica que el nombre de la columna sea un string
                        #print('RAMAN SHIFT')
                        #print(raman_shift)
                        #print('INTENSIDADES')
                        #print(df2.iloc[:,pos_y]) 
                        #print(col)
                        #print("xdxdxdxdxdxd")
                        if tipo in leyendas_tipos:
                            plt.plot(raman_shift , df_derivada_diff[col], color=color_actual, alpha=0.3, linewidth = 0.1,label=col) 
                            '''raman_shift:LE PASAMOS TODAS LAS INTENSIDADES , df2[col]= LE PASAMOS TODAS LAS COLUMNAS CON EL MISMO TIPO'''
                            #print('entro4')
                            #print(pos_y)   
                            break
                        else:
                            plt.plot(raman_shift , df_derivada_diff[col], color=color_actual, alpha=0.3, linewidth = 0.1) 
                            leyendas_tipos.add(tipo)
                            #print(leyendas_tipos)
                            #print("entro 5")
                    pos_y+=1 
                        
        if opcion == '1':
            if metodo_suavizado == '1':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por la media ')
                plt.show()
            elif metodo_suavizado == '2':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por la media ')
                plt.show()
            elif metodo_suavizado == '3':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada  del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por la media ')
                plt.show()
            else:
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada  del archivo {bd_name} Sin Suavizar y Normalizado por la media ')
                plt.show()
        elif opcion == '2':
            if metodo_suavizado == '1':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada  del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por Area ')
                plt.show()
            elif metodo_suavizado == '2':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por Area ')
                plt.show()
            elif metodo_suavizado == '3':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada  del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por Area')
                plt.show()
            else:
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada  del archivo {bd_name} Sin Suavizar y Normalizado por Area')
                plt.show()
        else:
            if metodo_suavizado == '1':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por Saviztky_golay y sin Normalizar')
                plt.show()
            elif metodo_suavizado == '2':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y sin Normalizar ')
                plt.show()
            elif metodo_suavizado == '3':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Suavizado por MEDIA MOVIL y sin Normalizar')
                plt.show()
            else:
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Primera Derivada del archivo {bd_name} Sin Suavizar y sin Normalizar')
                plt.show()
                
    else:
        #print("LA PRIMERA DERIVADA ES:")
        #print("ahora va a retornar el df de la 1der")
        return df_derivada_diff
        #para la llamada del PCA





def segunda_derivada(normalizado, pca_op):
            
    #print("entro en la funcion")
            
    if pca_op == 0 or pca_op == 1 or pca_op == 3:
        print("NORMALIZAR POR:")
        print("1-Media")
        print("2-Area")
        print("3-Sin normalizar")
        opcion = input("Selecciona una opción: ")
           
        if opcion == '1'  :
            normalizado = df_media_pca
        elif opcion == '2' :
            normalizado = df_concatenado_cabecera_nueva_area
        elif opcion == '3' :
            normalizado = df2
        else:
            print("OPCION NO VALIDA")
            print("Salir...")
       
        #print("EL normalizado.SHAPE antes ES:", normalizado.shape)
        
        print("DESEA SUAVIZAR")
        print("1. SI")
        print("2. NO")
        opcion_s =  int(input("OPCION: "))
        if opcion_s == 1:
            print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
            print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
            print("2. SUAVIZADO POR FILTRO GAUSIANO")
            print("3. SUAVIZADO POR MEDIA MOVIL")
            metodo_suavizado = input("OPCION: ")
            if metodo_suavizado == '1':
                aux = suavizado_saviztky_golay(normalizado,1)
                #print("EL aux.SHAPE antes ES:", aux.shape)
                #normalizado = suavizado_saviztky_golay(normalizado,1)
            elif metodo_suavizado == '2':
                #print("normalizado shape por FG = ", normalizado.shape)
                #normalizado = suavizado_filtroGausiano(normalizado,1)
                aux = suavizado_filtroGausiano(normalizado,1)
                #print("normalizado shape despues por FG = ", normalizado.shape)
            elif metodo_suavizado == '3':
                #normalizado = suavizado_mediamovil(normalizado,1) 
                aux = suavizado_mediamovil(normalizado,1) 
        else:
                metodo_suavizado = '4'         
                #print("No se va a normalizar, directamente la derivada")
                 
                  # si viene 0 que haga todo eso pero si viene 1 desde la funcion del PCA que haller directo la primera derivada con esos parametros
           
    else:
       #daba error por que no retornaba nada
       normalizado_pca = normalizado
    
    if (pca_op == 0 or pca_op == 3) and opcion_s == 1  :
        #print("entro aca")
        normalizado_f = aux
        #print("EL DF NORMALIZADO_F ES:")
        #print(normalizado_f)
    elif (pca_op == 0 or pca_op == 3) and opcion_s == 2:
        normalizado_f = normalizado
    else:
        normalizado_f = normalizado_pca #este es para cuando sea la funcion derivada sea llamado por la funcion del PCA
        #print("aca va lo del PCA")
    
    
    df_derivada = normalizado_f.apply(pd.to_numeric, errors='coerce') # PASAMOS A NUMERICO SI ES NECESARIO
    #print("xXXXXXXXxxXXXX")
    #print(df_derivada)
    
    # Calcular la primera derivada
    df_derivada_diff = df_derivada.diff()
    #print("primera derivada")
    #print(df_derivada_diff)
    # Calculamos la segunda derivada
    df_derivada_diff = df_derivada_diff.diff()
    #print("segunda derivada")
    #print(df_derivada_diff)
    # Imprimir la primera derivada
    #print("Primera Derivada:")
    #print(df_derivada_diff)
    
    
    plt.figure(figsize=(10, 6))
    

    if pca_op == 0 and pca_op != 3:
        #print("LA PRIMERA DERIVADA ES:")
        #print(df_derivada_diff.shape)
        #mostrar_espectros(df_derivada_diff, 8,opcion,metodo_suavizado)
        leyendas_tipos = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
        pos_y=0
        for col in df_derivada_diff.columns :
            #print('entro normal')
            for tipo in asignacion_colores:
                #print("wwwwwww")
                if tipo == col :
                    color_actual= asignacion_colores[tipo] #ACA YA ENCONTRAMOS EL COLOR CORRESPONDIENTE A ESE TIPO   
                    #print(tipo,'==',col,'color=',color_actual) 
                    #print("ccccccc")
                    if isinstance(col, str):  #Verifica que el nombre de la columna sea un string
                        #print('RAMAN SHIFT')
                        #print(raman_shift)
                        #print('INTENSIDADES')
                        #print(df2.iloc[:,pos_y]) 
                        #print(col)
                        #print("xdxdxdxdxdxd")
                        if tipo in leyendas_tipos:
                            plt.plot(raman_shift , df_derivada_diff[col], color=color_actual, alpha=0.3, linewidth = 0.1,label=col) 
                            '''raman_shift:LE PASAMOS TODAS LAS INTENSIDADES , df2[col]= LE PASAMOS TODAS LAS COLUMNAS CON EL MISMO TIPO'''
                            #print('entro4')
                            #print(pos_y)   
                            break
                        else:
                            plt.plot(raman_shift , df_derivada_diff[col], color=color_actual, alpha=0.3, linewidth = 0.1) 
                            leyendas_tipos.add(tipo)
                            #print(leyendas_tipos)
                            #print("entro 5")
                    pos_y+=1 
                        
        if opcion == '1':
            if metodo_suavizado == '1':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por la media ')
                plt.show()
            elif metodo_suavizado == '2':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por la media ')
                plt.show()
            elif metodo_suavizado == '3':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada  del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por la media ')
                plt.show()
            else:
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada  del archivo {bd_name} Sin Suavizar y Normalizado por la media ')
                plt.show()
        elif opcion == '2':
            if metodo_suavizado == '1':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada  del archivo {bd_name} Suavizado por Saviztky_golay y Normalizado por Area ')
                plt.show()
            elif metodo_suavizado == '2':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por Area ')
                plt.show()
            elif metodo_suavizado == '3':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada  del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por Area')
                plt.show()
            else:
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada  del archivo {bd_name} Sin Suavizar y Normalizado por Area')
                plt.show()
        else:
            if metodo_suavizado == '1':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por Saviztky_golay y sin Normalizar')
                plt.show()
            elif metodo_suavizado == '2':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por FILTRO GAUSIANO y sin Normalizar ')
                plt.show()
            elif metodo_suavizado == '3':
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Suavizado por MEDIA MOVIL y sin Normalizar')
                plt.show()
            else:
                plt.xlabel('Longitud de onda / Frecuencia')
                plt.ylabel('Intensidad')
                plt.title(f'Segunda Derivada del archivo {bd_name} Sin Suavizar y sin Normalizar')
                plt.show()
                
    else:
        #print("LA PRIMERA DERIVADA ES:")
        #print(df_derivada_diff)
        return df_derivada_diff
        #para la llamada del PCA
   


# POR EL METODO DE REGRESION LINEAL
def correcion_LineaB(normalizado, pca_op):
        if pca_op == 0 or pca_op == 5:
            print("NORMALIZAR POR:")
            print("1-Media")
            print("2-Area")
            print("3-Sin normalizar")
            opcion = int(input("Selecciona una opción: "))
               
            if opcion == 1  :
                normalizado = df_media_pca
            elif opcion == 2 :
                normalizado = df_concatenado_cabecera_nueva_area
            elif opcion == 3 :
                normalizado = df2
            else:
                print("OPCION NO VALIDA")
                print("Salir...")
           
            #print("EL normalizado.SHAPE antes ES:", normalizado.shape)
            
            print("DESEA SUAVIZAR")
            print("1. SI")
            print("2. NO")
            opcion_s =  int(input("OPCION: "))
            if opcion_s == 1:
                print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
                print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
                print("2. SUAVIZADO POR FILTRO GAUSIANO")
                print("3. SUAVIZADO POR MEDIA MOVIL")
                metodo_suavizado = int(input("OPCION: "))
                if metodo_suavizado == 1:
                    aux = suavizado_saviztky_golay(normalizado,2)
                    #print("EL aux.SHAPE antes ES:", aux.shape)
                    #normalizado = suavizado_saviztky_golay(normalizado,1)
                elif metodo_suavizado == 2:
                    #print("normalizado shape por FG = ", normalizado.shape)
                    #normalizado = suavizado_filtroGausiano(normalizado,1)
                    aux = suavizado_filtroGausiano(normalizado,2)
                    #print("normalizado shape despues por FG = ", normalizado.shape)
                elif metodo_suavizado == 3:
                    #normalizado = suavizado_mediamovil(normalizado,1) 
                    aux = suavizado_mediamovil(normalizado,2) 
            else:
                    metodo_suavizado = 4         
                    #print("No se va a suavizar, directamente la correccion")
                     
                      # si viene 0 que haga todo eso pero si viene 1 desde la funcion del PCA que haller directo la primera derivada con esos parametros
               
        else:
           #daba error por que no retornaba nada
           #print("entro en el else de correccion base")
           normalizado_correccion = normalizado
        
        if (pca_op == 0 or pca_op == 3 or pca_op == 5) and opcion_s == 1:
            #print("entro aca")
            normalizado_f = aux
            #print("EL DF NORMALIZADO_F ES:")
            #print(normalizado_f)
        elif (pca_op == 0 or pca_op == 3 or pca_op == 5) and opcion_s == 2:
            normalizado_f = normalizado
        else:
            normalizado_f = normalizado_correccion #este es para cuando sea la funcion derivada sea llamado por la funcion del PCA
            #print("aca va lo del PCA")
        
        #print("NORMALIZADO-F")
        #print(normalizado_f)
        cabecera_aux = normalizado_f.columns
        #print("XDDDDDDDDDD")
        #print(cabecera_aux)
        np_corregido = normalizado_f.to_numpy() # pasamos a numpy para borrar la cabecera de tipos
        #print("DF_CORREGIDO")
        #print(np_corregido)
        #print("DF_CORREGIDO")
        df_corregido = pd.DataFrame(np_corregido) # pasamos a panda para tener de vuelta el DF original pero sin cabecera
        #print(df_corregido)
        
        
      
        pendientes = {}   # Crear un diccionario para almacenar las pendientes
        intersecciones = {}
        y_ajustados = {}
        #dic_prueba = {}

        pos = 0  
        # Asignar la primera columna como Raman shift
        #raman_shift = df_corregido.iloc[:, 0]  # Primera columna
        #print("Raman shift")
        #print(raman_shift)
        #print("Cant filas = ", len(raman_shift))

        #y_ajustados = pd.DataFrame(raman_shift)
        #print(y_ajustados)

        #raman_shift = pd.to_numeric(raman_shift, errors='coerce')      # Aseguramos que Raman Shift sea numérico
        #print("RAMAN SHIFT")
        #print(raman_shift)
        
        # Iterar sobre las demás columnas
        for col in df_corregido.columns:
            intensidades = df_corregido[col]  # Extraer la columna actual
            intensidades = pd.to_numeric(intensidades, errors='coerce') #ASEGURAMOS QUE LAS INTENSIDADES SEAN NUMERICOS           
            #print(intensidades)
            # print(cont)                        
            # Calcularmos la pendiente 
            coef = np.polyfit(raman_shift, intensidades, 1)  # Grado 1 para línea recta , coef = coeficiente de Y=mx+b , coef[0] es la pendiente m, y coef[1] es la intersección b.
            # SE PONE DE GRADO 1 POR QUE QUEREMOS AJUSTAR UN POLINOMIO DE LA FORMA Y=MX+B 
            pendiente = coef[0]  # La pendiente (m) está en el índice 0 de los coeficientes
            interseccion = coef[1] # La interseccion (b) esta en el indice 1 
            # Guardamos la pendiente y intersecciones en el diccionario
            pendientes[col] = pendiente
            intersecciones[col] = interseccion
            #print(raman_shift[pos])
            #print("xD")
            y_ajustado = []
            for pos, intensidad in enumerate(intensidades):
                y = pendiente * raman_shift[pos] + interseccion  
                #print()
                #print(pendiente, "*",raman_shift[pos] , "+", interseccion , "=", y_ajustado, intensidades)
                y_ajustado.append(y)
                #dic_prueba.append(y)
            y_ajustados[col] = intensidades - y_ajustado
            #dic_prueba[col] = y_ajustado
            #print(intensidades, "-", dic_prueba[col] , "=", y_ajustados[col])
            #print("XDDDDDDDDDDDD")
            #print(y_ajustado)
            # if pos == len(raman_shift)-1: # la funcion len() sirve para saber la cantidad de filas
            #     pos = 0
            #     #print("entro")
                
  
            # else:
            #     pos += 1 
            #     #print("emtro2")
                
                
        df_y_ajustados = pd.DataFrame(y_ajustados)
        #df_y_ajustados.index.name = "Raman Shift"  
        #print(len(df_y_ajustados.columns))
        #print(len(cabecera))
        df_y_ajustados.columns = cabecera_aux
        #print(df_y_ajustados)    
        
        #print()
        #print("LA PENDIENTE SON:", pendientes)
        # print(type(pendientes))
        #print()
        #print("LA INTERSECCIONES SON:", intersecciones)
        # print(type(intersecciones))
        #print()
        #print("Y AJUSTADO =", y_ajustados)
        # print(type(y_ajustados))
        
        #   CORROBORAR LOS RESULTADOS 
        # SI TODO ESTA BIEN CREAR EL DATAFRAME Y CON LOS RESULTADOS Y VOLVER A UNIR LA CABECERA
        if pca_op == 0 and pca_op != 5:
            if opcion_s == 1:
                mostrar_espectros(df_y_ajustados,raman_shift, 10, opcion, metodo_suavizado)
            else:
                mostrar_espectros(df_y_ajustados,raman_shift, 10, opcion,4)
        else:
            #print("entro en el else para retornar el df")
            return df_y_ajustados   #FALTA HACER QUE LA FUNCION PCA LLAME A ESTA FUNCION



def correcion_shirley(normalizado, pca_op) :
    print("FALTA IMPLEMENTEAR")
    




#COMENZAR CON EL PCA

def  mostrar_pca(op_load):
    print("NORMALIZAR POR:")
    print("1-MEDIA")
    print("2-AREA")
    print("3-SIN NORMALIZAR")
    
    opcion = input("Selecciona una opción: ")
    
 
    if opcion == '1'  :
        normalizado_pca = df_media_pca
        print("Deseas Suavizar?")
        print("1- SI")
        print("2- NO")
        suavizar = int(input("OPCION: "))
        if suavizar == 1:
            print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
            print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
            print("2. SUAVIZADO POR FILTRO GAUSIANO")
            print("3. SUAVIZADO POR MEDIA MOVIL")
            metodo_suavizado = int(input("OPCION: "))
            if metodo_suavizado == 1:
                normalizado_pca = suavizado_saviztky_golay(normalizado_pca,2)
                # print("volvio")
                #print(metodo_suavizado)
            elif metodo_suavizado == 2:
                normalizado_pca = suavizado_filtroGausiano(normalizado_pca,2)
            else:
                normalizado_pca = suavizado_mediamovil(normalizado_pca,2) 
            
        print("\n--- DESEAS REALIZAR ALGUNA DERIVADA ---")  
        print("1- SI")
        print("2- NO")
        opcion = int(input("OPCION: "))
        if opcion == 1:
            print("1- PRIMERA DERIVADA")
            print("2- SEGUNDA DERIVADA")
            op_der= int(input("OPCION: "))
            if op_der == 1:
                 normalizado_pca = primera_derivada(normalizado_pca,4)
            else:
                 normalizado_pca = segunda_derivada(normalizado_pca,4)
        
    elif opcion == '2' :
        #print("entor op 2")
        normalizado_pca = df_concatenado_cabecera_nueva_area
        print("Deseas Suavizar?")
        print("1- SI")
        print("2- NO")
        suavizar = int(input("OPCION: "))
        if suavizar == 1:
            print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
            print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
            print("2. SUAVIZADO POR FILTRO GAUSIANO")
            print("3. SUAVIZADO POR MEDIA MOVIL")
            metodo_suavizado = int(input("OPCION: "))
            if metodo_suavizado == 1:
                normalizado_pca = suavizado_saviztky_golay(normalizado_pca,2)
            elif metodo_suavizado == 2:
                normalizado_pca = suavizado_filtroGausiano(normalizado_pca,2)
            else:
                normalizado_pca = suavizado_mediamovil(normalizado_pca,2) 
        #print("no suavizar xdd")
        print("\n--- DESEAS REALIZAR ALGUNA DERIVADA ---")  
        print("1- SI")
        print("2- NO")
        opcion = int(input("OPCION: "))
        if opcion == 1:
           print("1- PRIMERA DERIVADA")
           print("2- SEGUNDA DERIVADA")
           op_der= int(input("OPCION: "))
           if op_der == 1:
                print("si quiero derivar")
                #print(normalizado_pca)
                normalizado_pca = primera_derivada(normalizado_pca,4)
           else:
                normalizado_pca = segunda_derivada(normalizado_pca,4)      


    elif opcion == '3' :
        normalizado_pca = df2
        print("Deseas Suavizar?")
        print("1- SI")
        print("2- NO")
        suavizar = int(input("OPCION: "))
        if suavizar == 1:
            print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
            print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
            print("2. SUAVIZADO POR FILTRO GAUSIANO")
            print("3. SUAVIZADO POR MEDIA MOVIL")
            metodo_suavizado = int(input("OPCION: "))
            if metodo_suavizado == 1:
                normalizado_pca = suavizado_saviztky_golay(normalizado_pca,2)
            elif metodo_suavizado == 2:
                normalizado_pca = suavizado_filtroGausiano(normalizado_pca,2)
            else:
                normalizado_pca = suavizado_mediamovil(normalizado_pca,2) 
        print("\n--- DESEAS REALIZAR ALGUNA DERIVADA ---")  
        print("1- SI")
        print("2- NO")
        opcion = int(input("OPCION: "))
        if opcion == 1:
            print("1- PRIMERA DERIVADA")
            print("2- SEGUNDA DERIVADA")
            op_der= int(input("OPCION: "))
            if op_der == 1:
                normalizado_pca = primera_derivada(normalizado_pca,4)
            else:
                normalizado_pca = segunda_derivada(normalizado_pca,4)      

    else:
        print("OPCION NO VALIDA")
        print("SAlir...")
        #mostrar_menu()

    datos = pd.DataFrame(normalizado_pca)
    
    #print("DATOS:")
    #print(datos)
    datos = datos.dropna() #eliminamos las filas con valores NAN
    #datos2 = datos.copy()
    # print("DATOS sin NaN:")
    # print(datos)
    
    datos_df = datos.transpose() #PASAMOS LA CABECERA DE TIPOS A LA COLUMNA
    #print('prueba')
    #print(datos_df)
     
    # Escalar los datos originales
    escalado = StandardScaler() #Escalar los datos para que cada columna (intensidades en cada longitud de onda) tenga una media de 0 y una desviación estándar de 1.
    dato_escalado = escalado.fit_transform(datos_df)
    print("DATOS ESCALADOS")  
    print(dato_escalado)
       
    #datos_np = datos_df.to_numpy() # PASAMOS DE UN DATAFRAME PANDAS A UN ARRAY NUMPY
    #print(datos_np.shape)
    
    pca = PCA(n_components=2)
    # Ajustar y transformar los datos
    dato_pca = pca.fit_transform(dato_escalado) # fit_transform ya hace el calcilo de los eigenvectores y eigenvalores y matriz de covarianza
    print("DATOS PCA")
    print(dato_pca)
    #print(dato_pca.shape)

    
    
    #print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
    colores_pca_original = [asignacion_colores.get(type_, 'black') for type_ in types]
    plt.figure(figsize=(10, 6))
    plt.scatter(dato_pca[:, 0], dato_pca[:, 1], c=colores_pca_original, alpha=0.7)
    
    
    
    if opcion == '1':   
        plt.xlabel('Raman_shift')
        plt.ylabel('Intensidad')
        plt.title('PCA NORMALIZADA POR LA MEDIA')
        plt.grid()
        plt.show()
    elif opcion == '2':
        plt.xlabel('Raman_shift')
        plt.ylabel('Intensidad')
        plt.title('PCA NORMALIZADA POR AREA')
        plt.grid()
        plt.show()
    else:
        plt.xlabel('Raman_shift')
        plt.ylabel('Intensidad')
        plt.title('PCA SIN NORMALIZAR')
        plt.grid()
        plt.show()        




def grafico_tipo(datos,raman_shift,metodo,opcion,m_suavi):

    
    #print("ENTRO EN MOSTRAR ESPECTROS")
    #print(datos)
    
    mostrar_tipo = input("Ingrese el tipo que deseas visualizar: ")
    
    # Graficar los espectros
    plt.figure(figsize=(10, 6))
 
  
    leyendas_tipos = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
    pos_y=0
    for col in datos.columns :
        if col == mostrar_tipo:
            #print("tipo seleccionado:", col)
            for tipo in asignacion_colores:
                #print("wwwwwww")
                if tipo == col :
                    color_actual= asignacion_colores[tipo] #ACA YA ENCONTRAMOS EL COLOR CORRESPONDIENTE A ESE TIPO   
                    if isinstance(col, str):  #Verifica que el nombre de la columna sea un string
                        if tipo in leyendas_tipos:
                            plt.plot(raman_shift , datos[col], color=color_actual, alpha=0.3, linewidth = 0.1,label=col)   
                            break
                        else:
                            plt.plot(raman_shift , datos[col], color=color_actual, alpha=0.3, linewidth = 0.1) 
                            leyendas_tipos.add(tipo) 
                    pos_y+=1 
    
    # TODO ESTE CONDICIONAL ES SOLO PARA QUE EL TITULO DEL GRAFICO MUESTRE POR CUAL METODO SE NORMALIZO
    if metodo == 1:
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectro del tipo: {mostrar_tipo} archivo {bd_name}')
        plt.show()
    elif metodo == 2:
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectro del tipo: {mostrar_tipo} del archivo {bd_name} Normalizado por la Media')
        plt.show()
    elif metodo == 3:
        plt.xlabel('Longitud de onda / Frecuencia')
        plt.ylabel('Intensidad')
        plt.title(f'Espectro del tipo: {mostrar_tipo} del archivo {bd_name} Normalizado por Area')
        plt.show()
    elif metodo == 4:
        if opcion == '1':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Saviztky_golay y Normalizado por la media')
            plt.show()   
        elif opcion == '2':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo} Suavizado por Saviztky_golay y Normalizado Area')
            plt.show() 
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Saviztky_golay y sin Normalizar ')
            plt.show()          
    elif metodo == 5:
        if opcion == '1':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Filtro Gaussiano y Normalizado por la media')
            plt.show()   
        elif opcion == '2':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Filtro Gaussiano y Normalizado Area')
            plt.show() 
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo} Suavizado por Filtro Gaussiano y sin Normalizar ')
            plt.show() 
    elif metodo == 6:
        if opcion == '1':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Media Movil y Normalizado por la media')
            plt.show()   
        elif opcion == '2':
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Media Movil y Normalizado Area')
            plt.show() 
        else:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  Suavizado por Media Movil y sin Normalizar ')
            plt.show() 
    elif metodo == 7:
            print("hola PCA")
    elif metodo == 8:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  por la primera derivada ')
            plt.show() 
    elif metodo == 9:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectro del tipo: {mostrar_tipo}  por la primera derivada ')
            plt.show() 
    elif metodo == 10:
            if opcion == '1':
                if m_suavi == 1:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por SAVIZTKY-GOLAY y Normalizado por la media')
                    plt.show()
                elif m_suavi == 2:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por la media')
                    plt.show()
                elif m_suavi == 3:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por la media')
                    plt.show()
                else:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} sin suavizar y Normalizado por la media')
                    plt.show()
            elif opcion == '2':
                if m_suavi == 1:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por SAVIZTKY-GOLAY y Normalizado por Area')
                    plt.show()
                elif m_suavi == 2:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Normalizado por Area')
                    plt.show()
                elif m_suavi == 3:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por MEDIA MOVIL y Normalizado por Area')
                    plt.show()
                else:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} sin Suavizar y Normalizado por Area')
                    plt.show()
            else:
                if m_suavi == 1:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por SAVIZTKY-GOLAY y Sin Normalizar')
                    plt.show()
                elif m_suavi == 2:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por FILTRO GAUSIANO y Sin Normalizar')
                    plt.show()
                elif m_suavi == 3:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} Suavizado por MEDIA MOVIL y Sin Normalizar')
                    plt.show()
                else:
                    plt.xlabel('Longitud de onda / Frecuencia')
                    plt.ylabel('Intensidad')
                    plt.title(f'Espectros del archivo {bd_name} sin Suavizar y Sin Normalizar')
                    plt.show()
    # elif metodo == 11:
    # elif metodo == 12:
    elif metodo == 13:
            plt.xlabel('Longitud de onda / Frecuencia')
            plt.ylabel('Intensidad')
            plt.title(f'Espectros del archivo {bd_name} Acotado ')
            plt.show() 
    else:
        print("NO HAY GRAFICA DISPONIBLE PARA ESTA OPCION")






def graficar_loadings(wavelengths=None, n_components=2):
    print("NORMALIZAR POR:")
    print("1-MEDIA")
    print("2-AREA")
    print("3-SIN NORMALIZAR")
    
    opcion = input("Selecciona una opción: ")
    
 
    if opcion == '1'  :
        normalizado_pca = df_media_pca
        print("Deseas Suavizar?")
        print("1- SI")
        print("2- NO")
        suavizar = int(input("OPCION: "))
        if suavizar == 1:
            print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
            print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
            print("2. SUAVIZADO POR FILTRO GAUSIANO")
            print("3. SUAVIZADO POR MEDIA MOVIL")
            metodo_suavizado = int(input("OPCION: "))
            if metodo_suavizado == 1:
                normalizado_pca = suavizado_saviztky_golay(normalizado_pca,2)
                # print("volvio")
                #print(metodo_suavizado)
            elif metodo_suavizado == 2:
                normalizado_pca = suavizado_filtroGausiano(normalizado_pca,2)
            else:
                normalizado_pca = suavizado_mediamovil(normalizado_pca,2) 
            
        print("\n--- DESEAS REALIZAR ALGUNA DERIVADA ---")  
        print("1- SI")
        print("2- NO")
        opcion = int(input("OPCION: "))
        if opcion == 1:
            print("1- PRIMERA DERIVADA")
            print("2- SEGUNDA DERIVADA")
            op_der= int(input("OPCION: "))
            if op_der == 1:
                 normalizado_pca = primera_derivada(normalizado_pca,4)
            else:
                 normalizado_pca = segunda_derivada(normalizado_pca,4)
        
    elif opcion == '2' :
        #print("entor op 2")
        normalizado_pca = df_concatenado_cabecera_nueva_area
        print("Deseas Suavizar?")
        print("1- SI")
        print("2- NO")
        suavizar = int(input("OPCION: "))
        if suavizar == 1:
            print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
            print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
            print("2. SUAVIZADO POR FILTRO GAUSIANO")
            print("3. SUAVIZADO POR MEDIA MOVIL")
            metodo_suavizado = int(input("OPCION: "))
            if metodo_suavizado == 1:
                normalizado_pca = suavizado_saviztky_golay(normalizado_pca,2)
            elif metodo_suavizado == 2:
                normalizado_pca = suavizado_filtroGausiano(normalizado_pca,2)
            else:
                normalizado_pca = suavizado_mediamovil(normalizado_pca,2) 
        #print("no suavizar xdd")
        print("\n--- DESEAS REALIZAR ALGUNA DERIVADA ---")  
        print("1- SI")
        print("2- NO")
        opcion = int(input("OPCION: "))
        if opcion == 1:
           print("1- PRIMERA DERIVADA")
           print("2- SEGUNDA DERIVADA")
           op_der= int(input("OPCION: "))
           if op_der == 1:
                print("si quiero derivar")
                #print(normalizado_pca)
                normalizado_pca = primera_derivada(normalizado_pca,4)
           else:
                normalizado_pca = segunda_derivada(normalizado_pca,4)      


    elif opcion == '3' :
        normalizado_pca = df2
        print("Deseas Suavizar?")
        print("1- SI")
        print("2- NO")
        suavizar = int(input("OPCION: "))
        if suavizar == 1:
            print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
            print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
            print("2. SUAVIZADO POR FILTRO GAUSIANO")
            print("3. SUAVIZADO POR MEDIA MOVIL")
            metodo_suavizado = int(input("OPCION: "))
            if metodo_suavizado == 1:
                normalizado_pca = suavizado_saviztky_golay(normalizado_pca,2)
            elif metodo_suavizado == 2:
                normalizado_pca = suavizado_filtroGausiano(normalizado_pca,2)
            else:
                normalizado_pca = suavizado_mediamovil(normalizado_pca,2) 
        print("\n--- DESEAS REALIZAR ALGUNA DERIVADA ---")  
        print("1- SI")
        print("2- NO")
        #print(normalizado_pca)
        opcion = int(input("OPCION: "))
        if opcion == 1:
            print("1- PRIMERA DERIVADA")
            print("2- SEGUNDA DERIVADA")
            op_der= int(input("OPCION: "))
            if op_der == 1:
                normalizado_pca = primera_derivada(normalizado_pca,4)
            else:
                normalizado_pca = segunda_derivada(normalizado_pca,4)      

    else:
        print("OPCION NO VALIDA")
        print("SAlir...")
        #mostrar_menu()
        
        
    print(normalizado_pca)
        
    # Escalar los datos
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(normalizado_pca)

    # Realizar PCA
    pca = PCA(n_components=n_components)
    pca.fit(scaled_data)
    loadings = pca.components_
    
    # Si no se proporcionan longitudes de onda, usar índices de las columnas
    if wavelengths is None:
        wavelengths = np.arange(1, normalizado_pca.shape[1] + 1)

    # Graficar los loadings
    plt.figure(figsize=(10, 5))
    for i in range(n_components):
        plt.plot(wavelengths, loadings[i], label=f"PC{i+1}", linestyle='--' if i else '-')

    # Línea de referencia en y = 0
    plt.axhline(0, color='gray', linewidth=0.8)
    print("mostrar loading")
    # Personalizar la gráfica
    plt.xlabel("Wavelength (nm)" if wavelengths is not None else "Características")
    plt.ylabel("PCA Loading Weight")
    plt.title("PCA Loading Weights para Datos de Espectroscopia")
    plt.legend()
    plt.grid()
    plt.show()
    
       
       
def hca():
   print("NORMALIZAR POR:")
   print("1-MEDIA")
   print("2-AREA")
   print("3-SIN NORMALIZAR")
   
   opcion = input("Selecciona una opción: ")
   

   if opcion == '1'  :
       normalizado_hca = df_media_pca
       print("Deseas Suavizar?")
       print("1- SI")
       print("2- NO")
       suavizar = int(input("OPCION: "))
       if suavizar == 1:
           print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
           print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
           print("2. SUAVIZADO POR FILTRO GAUSIANO")
           print("3. SUAVIZADO POR MEDIA MOVIL")
           metodo_suavizado = int(input("OPCION: "))
           if metodo_suavizado == 1:
               normalizado_hca = suavizado_saviztky_golay(normalizado_hca,2)
               # print("volvio")
               #print(metodo_suavizado)
           elif metodo_suavizado == 2:
               normalizado_hca = suavizado_filtroGausiano(normalizado_hca,2)
           else:
               normalizado_hca = suavizado_mediamovil(normalizado_hca,2) 
           
       print("\n--- DESEAS REALIZAR ALGUNA DERIVADA ---")  
       print("1- SI")
       print("2- NO")
       opcion = int(input("OPCION: "))
       if opcion == 1:
           print("1- PRIMERA DERIVADA")
           print("2- SEGUNDA DERIVADA")
           op_der= int(input("OPCION: "))
           if op_der == 1:
                normalizado_hca = primera_derivada(normalizado_hca,4)
           else:
                normalizado_hca = segunda_derivada(normalizado_hca,4)
       
   elif opcion == '2' :
       #print("entor op 2")
       normalizado_hca = df_concatenado_cabecera_nueva_area
       print("Deseas Suavizar?")
       print("1- SI")
       print("2- NO")
       suavizar = int(input("OPCION: "))
       if suavizar == 1:
           print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
           print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
           print("2. SUAVIZADO POR FILTRO GAUSIANO")
           print("3. SUAVIZADO POR MEDIA MOVIL")
           metodo_suavizado = int(input("OPCION: "))
           if metodo_suavizado == 1:
               normalizado_hca = suavizado_saviztky_golay(normalizado_hca,2)
           elif metodo_suavizado == 2:
               normalizado_hca = suavizado_filtroGausiano(normalizado_hca,2)
           else:
               normalizado_hca = suavizado_mediamovil(normalizado_hca,2) 
       #print("no suavizar xdd")
       print("\n--- DESEAS REALIZAR ALGUNA DERIVADA ---")  
       print("1- SI")
       print("2- NO")
       opcion = int(input("OPCION: "))
       if opcion == 1:
          print("1- PRIMERA DERIVADA")
          print("2- SEGUNDA DERIVADA")
          op_der= int(input("OPCION: "))
          if op_der == 1:
               print("si quiero derivar")
               #print(normalizado_pca)
               normalizado_hca = primera_derivada(normalizado_hca,4)
          else:
               normalizado_hca = segunda_derivada(normalizado_hca,4)      


   elif opcion == '3' :
       normalizado_hca = df2
       print("Deseas Suavizar?")
       print("1- SI")
       print("2- NO")
       suavizar = int(input("OPCION: "))
       if suavizar == 1:
           print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
           print("1. SUAVIZADO POR SAVIZTKY-GOLAY")
           print("2. SUAVIZADO POR FILTRO GAUSIANO")
           print("3. SUAVIZADO POR MEDIA MOVIL")
           metodo_suavizado = int(input("OPCION: "))
           if metodo_suavizado == 1:
               normalizado_hca = suavizado_saviztky_golay(normalizado_hca,2)
           elif metodo_suavizado == 2:
               normalizado_hca = suavizado_filtroGausiano(normalizado_hca,2)
           else:
               normalizado_hca = suavizado_mediamovil(normalizado_hca,2) 
       print("\n--- DESEAS REALIZAR ALGUNA DERIVADA ---")  
       print("1- SI")
       print("2- NO")
       opcion = int(input("OPCION: "))
       if opcion == 1:
           print("1- PRIMERA DERIVADA")
           print("2- SEGUNDA DERIVADA")
           op_der= int(input("OPCION: "))
           if op_der == 1:
               normalizado_hca = primera_derivada(normalizado_hca,4)
           else:
               normalizado_hca = segunda_derivada(normalizado_hca,4)      

   else:
       print("OPCION NO VALIDA")
       print("SAlir...") 
       
     
   # # Calcular HCA con diferentes métodos
   # Z_single = linkage(df, method='single')   # Single linkage
   # Z_complete = linkage(df, method='complete')  # Complete linkage
   # Z_average = linkage(df, method='average')  # Average linkage
   # Z_ward = linkage(df, method='ward')  # Ward linkage (minimiza varianza)
   
   normalizado_hca = normalizado_hca.dropna()  # Eliminamos filas con NaN ya que el algoritmo de linkage no lee esos tipos de datos
     
   Z_ward = linkage(normalizado_hca, method='ward')    
   
   plt.figure(figsize=(10, 6))
   dendrogram(Z_ward)
   plt.title('Dendrograma usando Ward linkage (HCA)')
   plt.xlabel('Muestras')
   plt.ylabel('Distancia')
   plt.show()
   
   
   
     
   print("DATAFRAME INICIAL DEL NORMALIZADO_HCA")
   print(normalizado_hca)
      


if __name__ == "__main__":
     main()

















 




