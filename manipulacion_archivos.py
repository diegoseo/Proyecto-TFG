import pandas as pd
import csv
import os
import re


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
    
    
def cargar_archivo(ruta_archivo):
    print("ENTRO 3")
    if not os.path.isfile(ruta_archivo):
        raise FileNotFoundError("Archivo no encontrado")
    
    delimitador = identificar_delimitador(ruta_archivo)
    print("El deliminatador es: ",delimitador)
    df = pd.read_csv(ruta_archivo, delimiter=delimitador, header=None)
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
        
    print("DF DENTRO DE CARGSR ARCHIVO")
    print(df)
    df = del_sufijos(df)
    return df

def del_sufijos(df):
    print("entro del del_sufijos")
    for col in df.columns:
        valor = re.sub(r'[_\.]\d+$', '', str(df.at[0, col]).strip())  # Eliminar sufijos con _ o .
        try:
            df.at[0, col] = float(valor)  # Convertir de nuevo a float si es posible
        except ValueError:
            df.at[0, col] = valor  # Mantener como string si no es convertible
    return df

def detectar_labels(df): #Detecta si los labels están en la fila o en la columna para ver si hacemos la transpuesta  o no
        # Verificar la primera fila (si contiene strings)
        if df.iloc[0].apply(lambda x: isinstance(x, str)).all():
            return "fila" #si los labels están en la primera fila
        
        # Verificar la primera columna (si contiene strings)
        elif df.iloc[:, 0].apply(lambda x: isinstance(x, str)).all():
            return "columna" #si los labels están en la primera columna
        
        # Si no hay etiquetas detectadas
        return "ninguno" #si no se detectan labels.
    