#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 08:04:08 2024

@author: diego
"""


# main.py
import os
import numpy as np
import pandas as pd
import csv # PARA ENCONTRAR EL TIPO DE DELIMITADOR DEL ARCHIVO .CSV
import re # PARA LA EXPRECION REGULAR DE LOS SUFIJOS
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler # PARA LA NORMALIZACION POR LA MEDIA 
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter # Para suavizado de Savitzky Golay
from scipy.ndimage import gaussian_filter # PARA EL FILTRO GAUSSIANO
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, cut_tree, dendrogram


pila = []

def archivo_existe(ruta_archivo):
    # print("Buscando archivo.")
    # time.sleep(3)
    # print("Buscando archivo..")
    # time.sleep(2)
    print("Buscando archivo...")
    time.sleep(1)
    return os.path.isfile(ruta_archivo)

nombre = input("Por favor, ingresa tu nombre: ")
print(f"Hola, {nombre}!")

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
        # print("Encontrado!.")
        # print("Analizando archivo.")
        # time.sleep(3)
        # print("Analizando archivo..")
        # time.sleep(2)
        # print("Analizando archivo...")
        time.sleep(1)
        bd_name = archivo_nombre #Este archivo contiene los datos espectroscópicos que serán leídos
        delimitador = identificar_delimitador(bd_name)
        print("EL DELIMITADOR ES: ", delimitador)
        df = pd.read_csv(bd_name, delimiter = delimitador , header=None)
        pila.append(df.copy())
        existe = True
        if detectar_labels(df) == "columna" :
            print("SE HIZO LA TRASPUESTA")
            df = df.T
        else:
            print("NO SE HIZO LA TRANSPUESTA")
    else:
        print("El archivo no existe.")
        archivo_nombre = input("Ingrese el nombre del archivo: ")

# print("DF ANTES DEL CORTE")
# print(df)
# print(df.shape)

# print("LOGRO LEER EL ARCHIVO")


def columna_con_menor_filas(df):
    
    # Calcular el número de valores no nulos en cada columna
    valores_no_nulos = df.notna().sum()
    
    # Encontrar la columna con la menor cantidad de valores no nulos
    columna_menor = valores_no_nulos.idxmin()
    cantidad_menor = valores_no_nulos.min()
    
    return columna_menor, cantidad_menor


col,fil = columna_con_menor_filas(df)
if len(df) == fil:
    print("EL DATASET TIENE LA MISMA CANTIDAD DE FILAS EN CADA COLUMNA")
else:
    print("LA COLUMNA CON MENOR CANTIDAD DE DATOS ES:")
    print("TIPO: ",df.iloc[0,col+1])
    print("FILA: ",fil)
    print("COLUMNA: ",col+1)   
    print("DIMENSION DEL DATAFRAME", df.shape)
    opcion = 0
    while opcion != 6:
        print("COMO DESEAS ARREGLAR EL DATAFRAME")
        print("1- ELIMINAR TODAS LAS FILAS HASTA IGUALAR A LA MENOR")
        print("2- ELIMINAR LA COLUMNA CON MENOR NUMERO DE FILAS")
        print("3- VER DATAFRAME ACTUAL")
        print("4- VOLVER ATRAS")
        print("5- GENERAR .CSV")
        print("6- SALIR")
        opcion= int(input("OPCION: "))
        
        if opcion == 1:
            
            menor_cant_filas = df.dropna().shape[0] # Buscamos la columna con menor cantidad de intensidades
            # print("menor cantidad de filas:", menor_cant_filas)
    
            df_truncado = df.iloc[:menor_cant_filas] # Hacemos los cortes para igualar las columnas
    
            df = df_truncado
            
            pila.append(df.copy())
            # print(df.shape)
        elif opcion == 2:
            # print(df.shape)
            df.drop(columns=[col], inplace=True)
            # print(df.shape)
            pila.append(df.copy())
        elif opcion == 3:
            print(df)
        elif opcion == 4:
             if pila:
                 # Recuperar el último estado del DataFrame
                 df = pila.pop()
                 print("Se ha revertido al estado anterior.")
             else:
                 print("No hay acciones para deshacer.")
    
        elif opcion == 5:
            df.to_csv('output.csv', index=False, header=0)
        else:
            print("Saliendo")
        



# renombramos la celda [0,0]

# print("Cambiar a cero: ",df.iloc[0,0])

df.iloc[0,0] = float(0)

# print("Cambiar a cero: ",df.iloc[0,0])

#print(df)



# HACEMOS LA ELIMINACION DE LOS SUFIJOS EN CASO DE TENER


for col in df.columns:
    valor = re.sub(r'[_\.]\d+$', '', str(df.at[0, col]).strip())  # Eliminar sufijos con _ o .
    try:
        df.at[0, col] = float(valor)  # Convertir de nuevo a float si es posible
    except ValueError:
        df.at[0, col] = valor  # Mantener como string si no es convertible


# print("Luego de eliminar los sufijos")
print(df)



##### PENSAR EN COMO HACER LA OPCION DE IR HACIA ATRAS Y DE GENERAR .CSV PARA DESCARGAR

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



'''
    PREPARAMOS EL SIGUIENTE MENU
'''



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
print(df_media_pca) #ESTA VARIABLE SE USA PARA EL PCA TAMBIEN
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
            print("5-Volver")
            metodo_grafico = int(input("Opcion: "))
            if metodo_grafico == 1:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                mostrar_espectros(df2,raman_shift,metodo,0,0)
            elif metodo_grafico == 2:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                #espectro_acotado(df2, 0,1)
            elif metodo_grafico == 3:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                #grafico_tipo(df2,raman_shift,metodo,0,0)
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
            print("5- Descargar .csv normalizado por la media")
            metodo_grafico = int(input("Opcion: "))
            if metodo_grafico == 1:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                mostrar_espectros(df_media_pca,raman_shift,metodo,0,0)
            elif metodo_grafico == 2:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                #espectro_acotado(df_media_pca, 0,2)
            elif metodo_grafico == 3:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                #grafico_tipo(df_media_pca,raman_shift,metodo,0,0)
            # elif metodo_grafico == 4:
            #     #grafico_tipo_acotado()
            else:
                df_media_pca.to_csv('output_media.csv', index=False, header=True)
                
            print("Procesando los datos")
            print("Por favor espere un momento...")
        elif opcion == '3':
            metodo = 3
            print("Como deseas ver el espectro")
            print("1-Grafico completo")
            print("2-Grafico acotado")
            print("3-Grafico por tipo")
            print("4-Grafico acotado por tipo")
            print("5- Descargar .csv normalizado por Area")
            metodo_grafico = int(input("Opcion: "))
            if metodo_grafico == 1:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                mostrar_espectros(df_concatenado_cabecera_nueva_area,raman_shift,metodo,0,0)
            elif metodo_grafico == 2:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                #espectro_acotado(df_concatenado_cabecera_nueva_area, 0,3)
            elif metodo_grafico == 3:
                print("Procesando los datos")
                print("Por favor espere un momento...")
                #grafico_tipo(df_concatenado_cabecera_nueva_area,raman_shift,metodo,0,0)
            # elif metodo_grafico == 4:
            #     #grafico_tipo_acotado()
            else:
                df_concatenado_cabecera_nueva_area.to_csv('output_area.csv', index=False, header=True)
            print("Procesando los datos")
            print("Por favor espere un momento...")           
        else:
            print("Opción no válida. Inténtalo de nuevo.")




#### ver ya como carajo extructurar todo.




if __name__ == "__main__":
     main()



























































def espectro_acotado(datos, pca_op,nor_op):
    if pca_op == 0:
        df_aux = df.iloc[1:,1:].to_numpy()
        #print("df_aux")
        #print(df_aux)
    else:
        #print("entro en el else")
        df_aux = datos.to_numpy()
        
    cabecera_np = df.iloc[0,:].to_numpy()   # la primera fila contiene los encabezados
    cabecera_np = cabecera_np[1:]
    intensidades_np = df_aux[: , :] # apartamos las intensidades
    raman =  df.iloc[:, 0].to_numpy().astype(float)  # Primera columna (Raman Shift)
    raman = raman[1:]
    intensidades =  intensidades_np[:, 1:].astype(float)  # Columnas restantes (intensidades)
    min_rango = int(input("Rango minimo: "))  # Cambia según lo que necesites
    max_rango = int(input("Rango maximo: "))  # Cambia según lo que necesites
    indices_acotados = (raman >= min_rango) & (raman <= max_rango)  #retorna false o true para los que estan en el rango
    raman_acotado = raman[indices_acotados]
    intensidades_acotadas = intensidades[indices_acotados,:]

    # Crear un DataFrame a partir de las dos variables
    df_acotado = pd.DataFrame(
    data=np.column_stack([raman_acotado, intensidades_acotadas]),
    columns=["Raman Shift"] + list(cabecera_np[1:])  # Encabezados para el DataFrame
    )

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
                                plt.plot(raman_acotado , df_acotado[col], color=color_actual, alpha=0.6, linewidth = 0.1) 
                                leyendas.add(tipo) 
                      pos_y+=1 
           
    
        titulo_plot_acotado(nor_op,min_rango,max_rango)
        
    else: 
        return df_acotado , raman_acotado # creo que no hace falta retornarn nada ya que si una funcion le llama seria solamente para graficarla y retorna tiene quw retornar tambien su raman_shift acotado







    intensity = df.iloc[1:, 1:] 
    cabecera = df.iloc[[0]].copy() # EXTRAEMOS LA PRIMERA FILA 
    scaler = StandardScaler() 
    cal_nor = scaler.fit_transform(intensity) #calcula la media y desviación estándar
    dato_normalizado = pd.DataFrame(cal_nor, columns=intensity.columns) # lo convertimos de vuelta en un DataFrame
    df_concatenado = pd.concat([cabecera,dato_normalizado], axis=0, ignore_index=True)
    df_concatenado.columns = df_concatenado.iloc[0]  # Asigna la primera fila como nombres de columna
    df_concatenado_cabecera_nueva = df_concatenado[1:].reset_index(drop=True)
    df_media_pca= pd.DataFrame(df_concatenado_cabecera_nueva.iloc[:,1:])
    return  df_media_pca















