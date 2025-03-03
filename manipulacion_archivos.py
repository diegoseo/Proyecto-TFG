#   ACA IRA TODO LO RELACIONADO A LA LECTURA DE LOS ARCHIVOS

import time # Para el sleep de archivo_existe
import os
import pandas as pd
import re # PARA LA EXPRECION REGULAR DE LOS SUFIJOS
import csv # PARA ENCONTRAR EL TIPO DE DELIMITADOR DEL ARCHIVO .CSV


pila = []
pila_df = [] # SE UTILIZARA PARA ALMACENAR LOS DF FINAL


def nombre_archivo(archivo_nombre):
    return archivo_nombre

def cargar_archivo(nombre_archivo):
        existe = False # VERIFICAMOS QUE EL ARCHIVO EXISTA
        while existe == False:   
            if archivo_existe(nombre_archivo):  
                print("Encontrado!.")
                time.sleep(1)
                bd_name = nombre_archivo#Este archivo contiene los datos espectroscópicos que serán leídos
                delimitador = identificar_delimitador(bd_name)
                print("EL DELIMITADOR ES: ", delimitador)
                df = pd.read_csv(bd_name, delimiter = delimitador , header=None)
                pila.append(df.copy())
                existe = True
                if detectar_labels(df) == "columna" :
                    print("SE HIZO LA TRASPUESTA") # si se hizo la transpuesta es por que si o si ha al menos 2 label por lo que eliminaremos uno, el codigo fallara si hay mas de dos label
                    df = df.T
                    #  Eliminar la fila con índice 0
                    df = df.drop(index=0)  
                    #  Reiniciar los índices después de la eliminación
                    df = df.reset_index(drop=True)
                    print("eliminacion de label")
                    print(df)
                else:
                    print("NO SE HIZO LA TRANSPUESTA")
            else:
                print("El archivo no existe.")
                nombre_archivo = input("Ingrese el nombre del archivo: ")
                
        df = igualar_dimensiones_filas(df)
        print("DF DESPUES DEL CORTE")
        print(df)
        df = del_sufijos(df)
            
        #pila_df.append(df.copy())  # ACA ES DONDE ALMACENAMOS TODOS LOS DF DE CADA ARCHIVO LEIDO

        return df # ASI FUNCIONA BIEN CUANDO ES SOLO UN ARCHIVO PERO CUANDO SEAN VARIOS ARCHIVOS NO DEBE DE RETORNAR NADA AUN







def archivo_existe(ruta_archivo):
    # print("Buscando archivo.")
    # time.sleep(3)
    # print("Buscando archivo..")
    # time.sleep(2)
    print("Buscando archivo...")
    time.sleep(1)
    return os.path.isfile(ruta_archivo) # RETORNA TRUE O FALSE





# Función para detectar el delimitador automáticamente por que los archivos pueden estar ceparados por , o ; etc
def identificar_delimitador(archivo):
    """Detecta el delimitador en un archivo de texto automáticamente."""
    with open(archivo, 'r', encoding="utf-8", newline='') as file:
        muestra_csv = file.read(4096)  # Leer una muestra del archivo

        # Si el archivo está vacío, retornar None
        if not muestra_csv:
            return None

        # Intentar detectar delimitador con csv.Sniffer
        try:
            caracter = csv.Sniffer()
            delimitador = caracter.sniff(muestra_csv).delimiter
            return delimitador
        except csv.Error:
            # Si Sniffer falla, probar manualmente con delimitadores comunes
            delimitadores_comunes = [",", ";", "\t", "|", " "]
            for delim in delimitadores_comunes:
                if delim in muestra_csv:
                    return delim  # Retorna el primer delimitador encontrado
            return None  # Si no detecta ningún delimitador

    delim = archivo
    
    if delim:
        print(f"El delimitador detectado es: '{repr(delim)}'")  # repr() muestra caracteres invisibles como \t
    else:
        print("No se pudo detectar un delimitador válido.")
    
    




def detectar_labels(df): #Detecta si los labels están en la fila o en la columna para ver si hacemos la transpuesta  o no

    # Verificar la primera fila (si contiene strings)
    if df.iloc[0].apply(lambda x: isinstance(x, str)).all():
        return "fila" #si los labels están en la primera fila
    
    # Verificar la primera columna (si contiene strings)
    elif df.iloc[:, 0].apply(lambda x: isinstance(x, str)).all():
        return "columna" #si los labels están en la primera columna
    
    # Si no hay etiquetas detectadas
    return "ninguno" #si no se detectan labels.




def columna_con_menor_filas(df):
    
    # Calcular el número de valores no nulos en cada columna
    valores_no_nulos = df.notna().sum()
    
    # Encontrar la columna con la menor cantidad de valores no nulos
    columna_menor = valores_no_nulos.idxmin()
    cantidad_menor = valores_no_nulos.min()
    
    return columna_menor, cantidad_menor

def igualar_dimensiones_filas(df): 
    #print("Tamaño de la pila inicial:", len(pila))
    col,fil = columna_con_menor_filas(df)
    if len(df) == fil:
        print("EL DATASET TIENE LA MISMA CANTIDAD DE FILAS EN CADA COLUMNA")
        return df
    else:
        while True:
            print("SALIO DEL WHILE")
            if len(df) == fil:
                print("EL DATASET TIENE LA MISMA CANTIDAD DE FILAS EN CADA COLUMNA")
               #return df
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
                    
                   
                    pila.append(df.copy())
                    
                    menor_cant_filas = df.dropna().shape[0] # Buscamos la columna con menor cantidad de intensidades
                    # print("menor cantidad de filas:", menor_cant_filas)
            
                    df_truncado = df.iloc[:menor_cant_filas] # Hacemos los cortes para igualar las columnas
            
                    df = df_truncado
                    
                    #print("Tamaño de la pila:", len(pila))
                    # print(df.shape)
                    col,fil = columna_con_menor_filas(df)
                    break
                elif opcion == 2:
                    
                    pila.append(df.copy())
                    # print(df.shape)
                    df.drop(columns=[col], inplace=True)
                    # print(df.shape
                    #print("Tamaño de la pila:", len(pila))
                    col,fil = columna_con_menor_filas(df)
                    break
                elif opcion == 3:
                    print(df)
                elif opcion == 4:
                      #print("Tamaño de la pila final inicial2:", len(pila))
                      if len(pila) > 1 :
                          # Recuperar el último estado del DataFrame
                          df = pila.pop()
                          print("Se ha revertido al estado anterior.")
                      else:
                          print("No hay acciones para deshacer.")
                      #print("Tamaño de la pila final:", len(pila))
                      col,fil = columna_con_menor_filas(df)
                      break            
                elif opcion == 5:
                    nombre_archivo = input("Ingrese nombre del archivo:")
                    df.to_csv(nombre_archivo, index=False, header=0)
                else:
                    print("Saliendo")
                    return df
            
    
# NO ELIMINE LA CELDA 0.0 CAPAZ CREE PROBLEMAS MAS ADELANTE

# renombramos la celda [0,0]

# print("Cambiar a cero: ",df.iloc[0,0])

#df.iloc[0,0] = float(0)

# print("Cambiar a cero: ",df.iloc[0,0])

#print(df)



# HACEMOS LA ELIMINACION DE LOS SUFIJOS EN CASO DE TENER

def del_sufijos(df):
    for col in df.columns:
        valor = re.sub(r'[_\.]\d+$', '', str(df.at[0, col]).strip())  # Eliminar sufijos con _ o .
        try:
            df.at[0, col] = float(valor)  # Convertir de nuevo a float si es posible
        except ValueError:
            df.at[0, col] = valor  # Mantener como string si no es convertible
    
    
    # print("Luego de eliminar los sufijos")
    print(df)
    return df


