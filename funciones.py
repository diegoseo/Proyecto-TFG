#   ACA PONDRE TODO LO RELACIONADO A LAS FUNCIONES QUE RETORNARAN VALORES

#import main
import titulo
import matplotlib.pyplot as plt
import webbrowser
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go#permite mover la gráfica, hacer zoom y visualizar mejor las relaciones entre los componentes
import scipy.cluster.hierarchy as sch
import matplotlib.ticker as ticker # PARA QUE EL EJE X MUESTRE EN ALGUN INTERVALO Y NO SALGA ENCIMADO (SE USO EN LOS GRAFICOS DE LOADINGS)
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler # PARA LA NORMALIZACION POR LA MEDIA 
from scipy.signal import savgol_filter # Para suavizado de Savitzky Golay
from scipy.ndimage import gaussian_filter # PARA EL FILTRO GAUSSIANO
from sklearn.decomposition import PCA
from scipy.stats import chi2 # PARA GRAFICAR LOS ELIPSOIDES
from scipy.spatial.distance import pdist, squareform


diccionario_archivos = []
diccionario_nombre = []
lista_pc = []

# MOSTRAMOS LA LEYENDA PARA CADA TIPO
def mostrar_leyendas(df,diccionario,cant_tipos):
    
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


def guardar_archivo(archivo_nombre,df):
    diccionario_nombre.append(archivo_nombre)
    diccionario_archivos.append(df)
    print("VER TODOS LOS ARCHIVOS")
    print(diccionario_archivos)
    
    
    
    
# FUNCION PARA SABER SI LOS RAMAN_SHIFT SON IGUALES O NO
def comparar_long_ondas(diccionario_archivos,diccionario_nombre, cant_archivos):
    
    print("\nANALIZANDO CONTENIDO LOGITUD DE ONDA\n")
    
    for num in range(len(diccionario_archivos) - 1):  # Iteramos sobre cada DataFrame
        long_ondas = diccionario_archivos[num].iloc[1:, 0].values  # Raman_Shift del DF actual (ignorando primera fila)

        for i in range(num + 1, len(diccionario_archivos)):  # Compara con los siguientes DataFrames
            raman_actual = diccionario_archivos[i].iloc[1:, 0].values  # Raman_Shift del otro DF
            
            if long_ondas.shape != raman_actual.shape:
                print(f"Diferente cantidad de longitudes de onda entre {diccionario_nombre[num]} ({long_ondas.shape[0]}) y {diccionario_nombre[i]} ({raman_actual.shape[0]})")
                continue  # Salta esta comparación si los tamaños son diferentes

            if np.array_equal(long_ondas, raman_actual):  # Comparación correcta
                print(f"Longitudes de onda iguales en {diccionario_nombre[num]} y {diccionario_nombre[i]}")
            else:
                print(f"Diferencias en las longitudes de onda entre {diccionario_nombre[num]} ({long_ondas.shape[0]}) y {diccionario_nombre[i]} ({raman_actual.shape[0]})")

    



def comparar_cant_filas(diccionario_archivos,diccionario_nombre, cant_archivos):
   
    print("\nANALIZANDO CANTIDAD DE FILAS\n") 
   
    for num in range(len(diccionario_archivos) - 1):
       long_filas = diccionario_archivos[num].shape[0] 

       for i in range(num + 1, len(diccionario_archivos)):  # Compara con los siguientes DataFrames
           fila_actual = diccionario_archivos[i].shape[0] # Raman_Shift del otro DF
           
           if long_filas != fila_actual:
                print(f"Diferente cantidad de filas entre {diccionario_nombre[num]} ({long_filas}) y {diccionario_nombre[i]} ({fila_actual})")
           else:
                print(f"Cantidad de filas iguales en {diccionario_nombre[num]} y {diccionario_nombre[i]}")


def comparar_cant_col(diccionario_archivos,diccionario_nombre, cant_archivos):
       
    print("\nANALIZANDO CANTIDAD DE COLUMNAS\n") 
   
    for num in range(len(diccionario_archivos) - 1):
       long_col = diccionario_archivos[num].shape[1] 

       for i in range(num + 1, len(diccionario_archivos)):  # Compara con los siguientes DataFrames
           col_actual = diccionario_archivos[i].shape[1] # Raman_Shift del otro DF
           
           if long_col != col_actual:
                print(f"Diferente cantidad de columnas entre {diccionario_nombre[num]} ({long_col}) y {diccionario_nombre[i]} ({col_actual})")
           else:
                print(f"Cantidad de columnas iguales en {diccionario_nombre[num]} y {diccionario_nombre[i]}")




def datos_sin_normalizar(df): #SIRVE PARA ELIMINAR LA COLUMNA DE RAMAN_SHIFT
    
    df2 = df.copy()
    df2.columns = df2.iloc[0]
    #print(df2)
    df2 = df2.drop(0).reset_index(drop=True) #eliminamos la primera fila
    df2 = df2.drop(df2.columns[0], axis=1) #eliminamos la primera columna el del rama_shift
    #print(df2) # aca ya tenemos la tabla de la manera que necesitamos, fila cero es la cabecera con los nombres de los tipos anteriormente eran indice numericos consecutivos
    df2 = df2.apply(pd.to_numeric, errors='coerce') #CONVERTIMOS A NUMERICO
    #print("EL DATAFRAME DEL ESPECTRO SIN NORMALIZAR ES")
    #print(df2) #ESTA VARIABLE SE USA PARA EL PCA TAMBIEN
    #print(df2.shape)
    print("DF22222")
    print(df2)
    return df2





def descargar_csv(df,normalizado,dato,raman_shift_actual):
    print("DATO dentro de descargar csv")
    print(dato)
    df_aux = dato.copy() # obtenemos el dataframe              
    raman_shift_aux = raman_shift_actual[:len(df_aux)] #nos aseguramos de que el tengan la misa longitud         
    df_aux.insert(0, 'Raman_shift', raman_shift_aux)    # Insertamos la columna raman_shift en la posición 0
    #df_aux.columns = df_aux.iloc[0]  # Asigna la primera fila como nombres de columnas
    #df_aux = df_aux[1:].reset_index(drop=True)  # Elimina la primera fila y reindexa
    
    if normalizado == 1:
        df.to_csv('output_sin_normalizar.csv', index=False, header=False)
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_sin_normalizar.csv")
    elif normalizado == 2:
        df_aux.to_csv('output_media.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_media.csv")
    elif normalizado == 3:
        df_aux.to_csv('output_area.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_area.csv")
    elif normalizado == 4:
        df_aux.to_csv('output_suavizado_saviztky_golay.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_suavizado_saviztky_golay.csv")
    elif normalizado == 5:
        df_aux.to_csv('output_suavizado_filtro_gausiano.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_suavizado_filtro_gausiano.csv")
    elif normalizado == 6:
        df_aux.to_csv('output_suavizado_media_movil.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_suavizado_media_movil.csv")
    elif normalizado == 7:
        df_aux.to_csv('output_primera_derivada.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_primera_derivada.csv")
    elif normalizado == 8:
        df_aux.to_csv('output_segunda_derivada.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_segunda_derivada.csv")
    elif normalizado == 9:
        df_aux.to_csv('output_correcionBase.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_correcionBase.csv")
    elif normalizado == 10:
        df_aux.to_csv('output_correcionShirley.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_correcionShirley.csv")




def descargar_csv_acotado(df,datos, opcion,raman_shift_actual):   # ESTA PARTE SE PUEDE OPTIMIZAR YA QUE 2-Grafico acotado  Y 6- Descargar .csv acotado por la media HACE LA MISMA COSA, SOLO QUE UNO GENERA UN .CSV Y EL OTRO LO GRAFICA
    df_aux = datos.to_numpy()
    print("PRINT")
    print(df_aux)
    cabecera_np = df.iloc[0, 1:].to_numpy()  # La primera fila contiene los encabezados
    #print("CABECERA_NP")
    #print(cabecera_np)
    intensidades_np = df_aux[:, :]  # Excluir la primera fila y primera columna
    #print("INTENSIDADES_NP")
    #print(intensidades_np)
    #raman = df.iloc[1:, 0].to_numpy().astype(float)  # Primera columna (Raman Shift) este es el ORIGINAL
    raman = raman_shift_actual.to_numpy().astype(float)
    print("RAMAN")
    print(raman)
    intensidades = intensidades_np.astype(float)  # Columnas restantes (intensidades)
    #print("INTENSIDADES")
    #print(intensidades)

    min_rango = int(input("Rango minimo: "))  
    max_rango = int(input("Rango maximo: "))  

    indices_acotados = (raman >= min_rango) & (raman <= max_rango)  
    #print("INDICES_ACOTADOS")
    #print(indices_acotados)
    #print(indices_acotados.shape)
    raman_acotado = raman[indices_acotados]
    #print("RAMAN_ACOTADO")
    #print(raman_acotado)
    intensidades_acotadas = intensidades[indices_acotados, :]
    #print("INTENSIDADES_ACOTADAS")
    #print(intensidades_acotadas)
    
    
    # Crear DataFrame filtrado
    df_acotado = pd.DataFrame(
        data=np.column_stack([raman_acotado, intensidades_acotadas]),
        columns=["Raman Shift"] + list(cabecera_np[:])  # Encabezados para el DataFrame
    )

    if opcion == 1:
        df_acotado.to_csv('output_acotado_sinNormalizar.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_sinNormalizar.csv")
    elif opcion == 2:
        df_acotado.to_csv('output_acotado_media.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_media.csv")
    elif opcion == 3:
        df_acotado.to_csv('output_acotado_area.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_area.csv")
    elif opcion == 4:
        df_acotado.to_csv('output_acotado_suavizado_saviztky_golay.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_suavizado_saviztky_golay.csv")
    elif opcion == 5:
        df_acotado.to_csv('output_acotado_filtro_gausiano.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_filtro_gausiano.csv")
    elif opcion == 6:
        df_acotado.to_csv('output_acotado_media_movil.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_media_movil.csv")
    elif opcion == 7:
        df_acotado.to_csv('output_acotado_primera_derivada.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_primera_derivada.csv")
    elif opcion == 8:
        df_acotado.to_csv('output_acotado_segunda_derivada.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_segunda_derivada.csv")
    elif opcion == 9:
        df_acotado.to_csv('output_acotado_corregido.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_corregido.csv")
    elif opcion == 10:
        df_acotado.to_csv('output_acotado_Shirley.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_Shirley.csv")
                                    
        

def descargar_csv_tipo(datos,opcion,raman_shift_actual):
    
    mostrar_tipo = input("Ingrese el tipo que deseas visualizar: ")

    columnas_eliminar = [] # GUARDAMOS EN ESTA LISTA TODO LO QUE SE VAS A ELIMINAR

    for col in datos.columns:
       
        if col != mostrar_tipo: # SI ESA COLUMNA NO CONINCIDE CON EL TIPO DESEADO SE AGREGAR EN columnas_eliminar
            columnas_eliminar.append(col)
    

    datos_filtrados = datos.drop(columns=columnas_eliminar) # CREAMOS UN DATAFRAME ELIMINANDO TODO LO QUE ESTE DENTRO DE columnas_eliminar
    
    datos_filtrados.insert(0, "raman_shift",raman_shift_actual)  # Insertamos en la primera posición los valores de raman_shift
    #print("Datos filtrados con 'raman_shift' agregado:")
    #print(datos_filtrados)
        
    
    
    if opcion == 1:
        datos_filtrados.to_csv('output_tipo_sinNormalizar.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_tipo_sinNormalizar.csv")
    elif opcion == 2:
        datos_filtrados.to_csv('output_tipo_media.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_tipo_media.csv")
    elif opcion == 3:
        datos_filtrados.to_csv('output_tipo_area.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_tipo_area.csv")
    elif opcion == 4:
        datos_filtrados.to_csv('output_tipo_suavizado_saviztky_golay.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_tipo_suavizado_saviztky_golay.csv")
    elif opcion == 5:
        datos_filtrados.to_csv('output_tipo_suavizado_filtro_gausiano.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_tipo_suavizado_filtro_gausiano.csv")
    elif opcion == 6:
        datos_filtrados.to_csv('output_tipo_suavizado_media_movil.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_tipo_media_movil.csv")
    elif opcion == 7:
        datos_filtrados.to_csv('output_tipo_primera_derivada.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_tipo_primera_derivada.csv")
    elif opcion == 8:
        datos_filtrados.to_csv('output_tipo_segunda_derivada.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_tipo_segunda_derivada.csv")
    elif opcion == 9:
        datos_filtrados.to_csv('output_tipo_corregido.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_tipo_corregido.csv")
    elif opcion == 10:
        datos_filtrados.to_csv('output_tipo_shirley.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_tipo_shirley.csv")




def descargar_csv_acotado_tipo(datos,opcion,raman_shift_actual):
   
    mostrar_tipo = input("Ingrese el tipo que deseas visualizar: ")


    columnas_eliminar = [] # GUARDAMOS EN ESTA LISTA TODO LO QUE SE VAS A ELIMINAR

    for col in datos.columns:
       
        if col != mostrar_tipo: # SI ESA COLUMNA NO CONINCIDE CON EL TIPO DESEADO SE AGREGAR EN columnas_eliminar
            columnas_eliminar.append(col)
    

    datos_filtrados = datos.drop(columns=columnas_eliminar) # CREAMOS UN DATAFRAME ELIMINANDO TODO LO QUE ESTE DENTRO DE columnas_eliminar
    
    datos_filtrados.insert(0, "raman_shift",raman_shift_actual)  # Insertamos en la primera posición los valores de raman_shift
    #print("Datos filtrados con 'raman_shift' agregado:")
    #print(datos_filtrados)
    datos_filtrados = datos_filtrados.astype(object)  # Convierte todo el DataFrame a tipo object       
    df_aux = datos_filtrados.iloc[:,1:].to_numpy()
    #print("PRINT")
    #print(df_aux)
    datos_filtrados.iloc[0, 1:] = mostrar_tipo
    cabecera_np = datos_filtrados.iloc[0, 1:].to_numpy()  # La primera fila contiene los encabezados
    #print("CABECERA_NP")
    #print(cabecera_np)
    intensidades_np = df_aux[:, :]
    #print("INTENSIDADES_NP")
    #print(intensidades_np)
    raman = raman_shift_actual.to_numpy().astype(float)  # Primera columna (Raman Shift)
    #print("RAMAN")
    #print(raman)
    intensidades = intensidades_np.astype(float)  # Columnas restantes (intensidades)
    # print("INTENSIDADES")
    # print(intensidades)
    
    min_rango = int(input("Rango minimo: "))  
    max_rango = int(input("Rango maximo: "))  
    
    indices_acotados = (raman >= min_rango) & (raman <= max_rango)  
    # print("INDICES_ACOTADOS")
    # print(indices_acotados)
    # print(indices_acotados.shape)
    raman_acotado = raman[indices_acotados]
    # print("RAMAN_ACOTADO")
    # print(raman_acotado)
    intensidades_acotadas = intensidades[indices_acotados, :]
    # print("INTENSIDADES_ACOTADAS")
    # print(intensidades_acotadas)
    
    # print("murio aca")
    
    # Crear DataFrame filtrado
    datos_acotado_tipo = pd.DataFrame(
        data=np.column_stack([raman_acotado, intensidades_acotadas]),
        columns=["Raman Shift"] + list(cabecera_np[:]) # Encabezados para el DataFrame
    )
    # print("datos_acotado_tipo ")
    # print(datos_acotado_tipo )
    
    if opcion == 1:
        datos_acotado_tipo.to_csv('output_acotado_tipo_sinNormalizar.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_tipo_sinNormalizar.csv")
    elif opcion == 2:
        datos_acotado_tipo.to_csv('output_acotado_tipo_media.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_tipo_media.csv")
    elif opcion == 3:
        datos_acotado_tipo.to_csv('output_acotado_tipo_area.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_tipo_area.csv")
    elif opcion == 4:
        datos_acotado_tipo.to_csv('output_acotado_tipo_suavizado_saviztky_golay.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_tipo_suavizado_saviztky_golay.csv")
    elif opcion == 5:
        datos_acotado_tipo.to_csv('output_acotado_tipo_suavizado_filtro_gaussiano.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_tipo_suavizado_filtro_gaussiano.csv")
    elif opcion == 6:
        datos_acotado_tipo.to_csv('output_acotado_tipo_suavizado_media_movil.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_tipo_suavizado_media_movil.csv")
    elif opcion == 7:
        datos_acotado_tipo.to_csv('output_acotado_tipo_primera_derivada.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_tipo_primera_derivada.csv")
    elif opcion == 8:
        datos_acotado_tipo.to_csv('output_acotado_tipo_segunda_derivada.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_tipo_segunda_derivada.csv")
    elif opcion == 9:
        datos_acotado_tipo.to_csv('output_acotado_tipo_corregido.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_tipo_corregido.csv")
    elif opcion == 10:
        datos_acotado_tipo.to_csv('output_acotado_tipo_corregidoShirley.csv', index=False, header=True)# Guardarmos el DataFrame en un archivo CSV
        print("SE HA DESCARGADO EL ARCHIVO CON EL NOMBRE: output_acotado_tipo_corregidoShirley.csv")


##ESTE CODIGO FUNCIONA PARA LAS OPCIONES 4,5,6 DE LOS SUAVIZADOS
def suavizado_menu(df,raman_shift):
    print("NORMALIZAR POR:")
    print("1-Media")
    print("2-Area")
    print("3-Sin normalizar")
    print("4- Volver")
    opcion = int(input("Selecciona una opción: "))
    
    if opcion == 1 :
        suavizar = normalizado_media(df)
    elif opcion == 2 :
        suavizar = normalizado_area(df,raman_shift)
    elif opcion == 3 :
        suavizar = datos_sin_normalizar(df)
    elif opcion == 4 :
        return None , None

    return suavizar , opcion



def mostrar_espectros(archivo_nombre,datos,raman_shift,asignacion_colores,metodo,nor_op,op_der,derivada): # LA VARIABLE DERIVADA ES  SOLO PARA SABER SI ES LA 1RA O LA 2DA

        
    print("ENTRO EN MOSTRAR ESPECTROS")
    print(datos)

    # Graficar los espectros
    if nor_op != 0:
        print("Procesando los datos")
        print("Por favor espere un momento...")
    
    plt.figure(figsize=(10, 6))
 
  
    leyendas_tipos = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
    pos_y=0
    for col in datos.columns :
        #print('col =', col)
        #col agarrar el nombre de los tipos de cada columna
        for tipo in asignacion_colores:
            #print("tipo: ",tipo)
            if tipo == col :
                #print("tipo==color")
                color_actual= asignacion_colores[tipo] #ACA YA ENCONTRAMOS EL COLOR CORRESPONDIENTE A ESE TIPO   
                if isinstance(col, str):  #Verifica que el nombre de la columna sea un string
                    if tipo in leyendas_tipos:
                        #print("primer plot")
                        plt.plot(raman_shift , datos[col], color=color_actual, alpha=0.5, linewidth = 0.1,label=col)   
                        #plt.xticks(np.arange(min(raman_shift[1:].astype(float)), max(raman_shift[1:].astype(float)), step=300))
                        break
                    else:
                        #print("segundo plot")
                        plt.plot(raman_shift , datos[col], color=color_actual, alpha=0.5, linewidth = 0.1) 
                        #plt.xticks(np.arange(min(raman_shift[1:].astype(float)), max(raman_shift[1:].astype(float)), step=300))         
                        leyendas_tipos.add(tipo) 
                pos_y+=1 
    print(leyendas_tipos)
    if op_der == 0:
        titulo.titulo_plot_mostrar(archivo_nombre,metodo,nor_op)

    if derivada == 1:
         titulo.titulo_plot_primera_derivada(archivo_nombre,nor_op,op_der)
    elif derivada == 2:
         titulo.titulo_plot_segunda_derivada(archivo_nombre,nor_op,op_der)
    elif derivada == 3:  # PARA QUE VAYA AL PLOT DE CORRECION DE LINEA BASE OPCION 9
         titulo.titulo_plot_correcion_base(archivo_nombre,nor_op,op_der)
    # elif derivada == 4:
    #     titulo_plot_correcion_shirley(nor_op,op_der)



##### ver como solucionar el tema de los datos que tiran todo iguales sus graficos, 
def espectro_acotado(archivo_nombre,asignacion_colores,df,datos,raman_shift_corregido ,pca_op,nor_op,op_der,derivada): # raman_shift_corregido = es por que la correcion de linea base modifica el raman_shift original
      
    df_aux = datos.to_numpy()
    #print("PRINT")
    #print(df_aux)
    cabecera_np = df.iloc[0, 1:].to_numpy()  # La primera fila contiene los encabezados
    #print("CABECERA_NP")
    #print(cabecera_np)
    intensidades_np = df_aux[:, :]  # Excluir la primera fila y primera columna
    #print("INTENSIDADES_NP")
    #print(intensidades_np)
    
    #TODO ESTE CODIGO DE ABAJO ES POR QUE RAMAN_SHIFT_CORREGIDO NO ES UN TIPO DE DATO NUMERO
    if isinstance(raman_shift_corregido, (int, float)) and raman_shift_corregido == 0:
        print("ENTRO EN EL IF POR QUE NO VIENE DE NINGUN METODO DE CORRECCION")
        raman = df.iloc[1:, 0].to_numpy().astype(float)  # Primera columna (Raman Shift)
    elif isinstance(raman_shift_corregido, pd.Series) and not raman_shift_corregido.empty:
        print("Entro en el else con un Pandas Series no vacío")
        raman = raman_shift_corregido.to_numpy().astype(float)
    elif isinstance(raman_shift_corregido, np.ndarray) and raman_shift_corregido.size > 0:
        print("Entro en el else con un NumPy array no vacío")
        raman = raman_shift_corregido.astype(float)
    else:
        print("Entro en el else para cualquier otro caso")
        raman = df.iloc[1:, 0].to_numpy().astype(float)  # Manejo por defecto
        
        
        
    #print("RAMAN")
    #print(raman)
    intensidades = intensidades_np.astype(float)  # Columnas restantes (intensidades)
    #print("INTENSIDADES")
    #print(intensidades)
    # Solicitar el rango
    min_rango = int(input("Rango minimo: "))  # Cambia según lo que necesites
    max_rango = int(input("Rango maximo: "))  # Cambia según lo que necesites
    
    if pca_op == 0:
        print("Procesando los datos")
        print("Por favor espere un momento...")
    
    # Filtrar los datos en el rango
    indices_acotados = (raman >= min_rango) & (raman <= max_rango)  # Filtra los índices
    #print("INDICES_ACOTADOS")
    #print(indices_acotados)
    #print(indices_acotados.shape)
    raman_acotado = raman[indices_acotados]
    #print("RAMAN_ACOTADO")
    #print(raman_acotado)
    intensidades_acotadas = intensidades[indices_acotados, :]
    #print("INTENSIDADES_ACOTADAS")
    #print(intensidades_acotadas)
    
    
    # Crear DataFrame filtrado
    df_acotado = pd.DataFrame(
        data=np.column_stack([raman_acotado, intensidades_acotadas]),
        columns=["Raman Shift"] + list(cabecera_np[:])  # Encabezados para el DataFrame
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
                                #print("error 2")
                                #print(raman)
                                #print(df_acotado[col])
                                plt.plot(raman_acotado , df_acotado[col], color=color_actual, alpha=0.6, linewidth = 0.1) 
                                leyendas.add(tipo) 
                      pos_y+=1 
           
        if op_der == 0:
            titulo.titulo_plot_acotado(archivo_nombre,nor_op,min_rango,max_rango)


        if derivada == 1:
            titulo.titulo_plot_primera_derivada(archivo_nombre,nor_op,op_der)
        elif derivada == 2:
            titulo.titulo_plot_segunda_derivada(archivo_nombre,nor_op,op_der)
        elif derivada == 3:  
            titulo.titulo_plot_correcion_base(archivo_nombre,nor_op,op_der)


    
        
    # else: 
    #     return df_acotado , raman_acotado # creo que no hace falta retornarn nada ya que si una funcion le llama seria solamente para graficarla y retorna tiene quw retornar tambien su raman_shift acotado


def grafico_tipo(archivo_nombre,asignacion_colores,datos,raman_shift,nor_op,metodo,op_der,derivada):

    
    #print("ENTRO EN MOSTRAR ESPECTROS")
    #print(datos)
    
    mostrar_tipo = input("Ingrese el tipo que deseas visualizar: ")
    
    print("Procesando los datos")
    print("Por favor espere un momento...")
    
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
    if op_der == 0:
        titulo.titulo_plot_tipo(archivo_nombre,metodo,mostrar_tipo,nor_op,metodo)

    if derivada == 1:
        titulo.titulo_plot_primera_derivada(archivo_nombre,metodo,op_der)
    elif derivada == 2:
        titulo.titulo_plot_segunda_derivada(archivo_nombre,metodo,op_der)
    elif derivada == 3:
         titulo.titulo_plot_correcion_base(archivo_nombre,nor_op,op_der)
    # elif derivada == 4:
    #     titulo_plot_correcion_shirley(nor_op,op_der)





def grafico_acotado_tipo(archivo_nombre,asignacion_colores,df,datos,raman_shift_corregido,metodo,opcion,op_der,derivada):
    
    mostrar_tipo = input("Ingrese el tipo que deseas visualizar: ")


    #print("ENTRO EN EL ESPECTRO ACOTADO")
    #print(datos)

    df_aux = datos.to_numpy()
    
    # print("RAMAN SHIFT CORREGIDO")
    # print(raman_shift_corregido)
        
    # print("ENTRO EN EL ESPECTRO ACOTADO222")
    # print(df_aux)
    # print(df_aux.shape)
    
    cabecera_np = df.iloc[0,:].to_numpy()   # la primera fila contiene los encabezados 
    cabecera_np = cabecera_np[1:]
    #print("la cabeceras son:")
    #print(cabecera_np)
    #print(cabecera_np.shape)
    
    
    intensidades_np = df_aux[: , :] # apartamos las intensidades
    # print("intensidades_np son:")
    # print(intensidades_np)
    # #print(intensidades_np.shape)
    
    #raman = raman_shift.to_numpy().astype(float)
    raman =  raman_shift_corregido.to_numpy().astype(float)  # Primera columna (Raman Shift) ESTE ES DEL ORIGINAL
    #raman = raman[1:]
    intensidades =  intensidades_np.astype(float)  # Columnas restantes (intensidades)
    # print("RAMAN:")
    # print(raman)
    # # print(raman.shape)
    # print("INTENSIDADES:")
    # print(intensidades)
    # print(intensidades.shape)
    # Filtrado del rango de las intensidades
    min_rango = int(input("Rango minimo: "))  # Cambia según lo que necesites
    max_rango = int(input("Rango maximo: "))  # Cambia según lo que necesites
    
    
    print("Procesando los datos")
    print("Por favor espere un momento...")
    
    indices_acotados = (raman >= min_rango) & (raman <= max_rango) #retorna false o true para los que estan en el rango
    # print("Indices acotados")
    # print(indices_acotados)
    # print(indices_acotados.shape)
    
    raman_acotado = raman[indices_acotados]
    intensidades_acotadas = intensidades[indices_acotados,:]

    
        
    # # # Imprimir resultados
    # print("Raman Shift Acotado:")
    # print(raman_acotado)
    # print("\nIntensidades Acotadas:")
    # print(intensidades_acotadas)
        
    
    # Crear un DataFrame a partir de las dos variables
    df_acotado = pd.DataFrame(
    data=np.column_stack([raman_acotado, intensidades_acotadas]),
    columns=["Raman Shift"] + list(cabecera_np[:])  # Encabezados para el DataFrame
    )

    # # Mostrar el DataFrame resultante
    # print("df_acotado")
    # # df_acotado = pd.DataFrame(df_acotado)
    # print(df_acotado)


   
    # Graficar los espectros
    plt.figure(figsize=(10, 6))
    
    #print("entro en el graficador")
    #DESCOMENTAR EL CODIGO DE ABAJO ESE ES MIO, EL DE ARRIBA ES CHATGPT  
         
    leyendas = set()  # almacenamos los tipos que enocntramos y la funcion set() nos ayuda a quer no se repitan
    pos_y=0
    for col in df_acotado.columns :
        if col == mostrar_tipo:
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
           
    print("llego a graficar pero falta el plot")
    if op_der == 0:
       titulo.titulo_plot_tipo_acotado(archivo_nombre,metodo,mostrar_tipo,min_rango,max_rango,opcion) #el 0 es por el m_suavi que no esta implementado aun
    if derivada == 1:
        titulo.titulo_plot_primera_derivada(archivo_nombre,opcion,op_der)
    elif derivada == 2:
        titulo.titulo_plot_segunda_derivada(archivo_nombre,opcion,op_der)
    elif derivada == 3:  
        titulo.titulo_plot_correcion_base(archivo_nombre,opcion,op_der)
    # elif derivada == 4:
    #     titulo_plot_correcion_shirley(opcion,op_der)




"""
VARIABLES DE NORMALIZAR POR LA MEDIA    tratar de hacer por la forma del ejemplo y no por z-core para ver si se soluciona lo de la raya
"""
def normalizado_media(df):
        
    intensity = df.iloc[1:, 1:] # EXTRAEMOS TODAS DEMAS COLUMNAS EXCEPTO LA PRIMERA FILA Y PRIMERA COLUMNA
    #print(intensity)     
    cabecera = df.iloc[[0]].copy() # EXTRAEMOS LA PRIMERA FILA 
    #print(cabecera)
    #print(cabecera.shape)
    scaler = StandardScaler(with_mean=True) 
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
    print('normalizacion media')
    print(df_media_pca)
    #df_media_pca.to_csv("nor_media_df.csv", index=False)
    return  df_media_pca





"""
VARIABLES DE NORMALIZAR POR AREA    tratar de hacer otro sin np.trap para ver si se soluciona lo de la raya
"""
def normalizado_area(df,raman_shift):
    
    intensity = df.iloc[1:, 1:] # EXTRAEMOS TODAS DEMAS COLUMNAS EXCEPTO LA PRIMERA FILA Y PRIMERA COLUMNA
    #print(intensity)  
    
    cabecera = df.iloc[[0]].copy() # EXTRAEMOS LA PRIMERA FILA 
    #print(cabecera)
    
    df3 = pd.DataFrame(intensity)
    #print("DataFrame de Intensidades:")
    #print(df3)
    df3 = df3.apply(pd.to_numeric, errors='coerce')  # Convierte a numérico, colocando NaN donde haya problemas
    #print(df3)
    np_array = raman_shift.astype(float).to_numpy() #CONVERTIMOS INTENSITY AL TIPO NUMPY POR QUE POR QUE NP.TRAPZ UTILIZA ESE TIPO DE DATOS
    #print("valor de np_array: ")
    #print(np_array)
    
    df3_normalizado = df3.copy()
    #print("EL VALOR DE DF3 ES :")
    #print(df3)
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
    return df_concatenado_cabecera_nueva_area




# SUAVIZADO POR SAVIZTKY-GOLAY 
#no genera valores NaN por si mismo

def suavizado_saviztky_golay(dato_suavizar):  #acordarse que se puede suavizar por la media, area y directo

    ventana = int(input("INGRESE EL VALOR DE LA VENTANA: "))
    orden = int(input("INGRESE EL VALOR DEL ORDEN: "))
            
    dato = dato_suavizar.to_numpy() #PASAMOS LOS DATOS A NUMPY POR QUE SAVGOL_FILTER USA SOLO NUMPY COMO PARAMETRO (PIERDE LA CABECERA DE TIPOS AL HACER ESTO)
    suavizado = savgol_filter(dato, window_length=ventana, polyorder=orden)
    suavizado_pd = pd.DataFrame(suavizado) # PASAMOS SUAVIZADO A PANDAS Y GUARDAMOS EN SUAVIZADO_PD
    suavizado_pd.columns = dato_suavizar.columns # AGREGAMOS LA CABECERA DE TIPOS
        
  
    print("Saviztky", suavizado_pd.shape)
    print("Saviztky", suavizado_pd)
    return suavizado_pd



# SUAVIZADO POR FILTRO GAUSIANO
# tampoco genera valores NaN
def suavizado_filtroGausiano(df,dato_suavizar):  #acordarse que se puede suavizar por la media, area y directo
   
    sigma = int(input("INGRESE EL VALOR DE SIGMA: ")) #Un valor mayor de sigma produce un suavizado más fuerte

    cabecera = df.iloc[[0]].copy() # EXTRAEMOS LA PRIMERA FILA 
    #print("avanzo")
    #print(pca_op)
    #print(type(normalizado))  
    #print(normalizado)
    dato = dato_suavizar.to_numpy() #PASAMOS LOS DATOS A NUMPY (PIERDE LA CABECERA DE TIPOS AL HACER ESTO)
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
    
    print("FILTRO GAUSIANO = ", suavizado_gaussiano_pd.shape)
    print("FILTRO GAUSIANO = ", suavizado_gaussiano_pd)
    
    return suavizado_gaussiano_pd




# GENERA VALORES NAM POR LA FORMA EN LA QUE SE CALCULA EL MEDIA MOVIL
# si puede generar valores NaN
def suavizado_mediamovil(dato_suavizar):

    ventana = int(input("INGRESE EL VALOR DE LA VENTANA: "))
    normalizado = dato_suavizar

      
    suavizado_media_movil = pd.DataFrame()
    
    
    suavizado_media_movil = normalizado.rolling(window=ventana, center=True).mean() # mean() es para hallar el promedio
    

    print("SUAVIZADO MEDIAMOVIL",suavizado_media_movil.shape)
    print("SUAVIZADO MEDIAMOVIL",suavizado_media_movil)

    
    return suavizado_media_movil
    
    
   # print(suavizado_media_movil)


# GENERA VALORES NAM POR LA FORMA EN LA QUE SE CALCULA EL MEDIA MOVIL
# si puede generar valores NaN
def suavizado_mediamovil_paraPCA(dato_suavizar , raman_shift):

    ventana = int(input("INGRESE EL VALOR DE LA VENTANA: "))
    normalizado = dato_suavizar

      
    suavizado_media_movil = pd.DataFrame()
    
    
    suavizado_media_movil = normalizado.rolling(window=ventana, center=True).mean() # mean() es para hallar el promedio
    
    # Eliminar las filas donde hay NaN en el suavizado
    mask = ~suavizado_media_movil.isna().any(axis=1)  # Crea una máscara para las filas sin NaN
    suavizado_media_movil = suavizado_media_movil[mask]  # Filtrar los datos suavizados
    raman_shift_sinNaN = raman_shift[mask]  # Aplicar el mismo filtro a raman_shift

    print("SUAVIZADO MEDIAMOVIL",suavizado_media_movil.shape)
    print("SUAVIZADO MEDIAMOVIL",suavizado_media_movil)
    print("Raman shift sin NAN", raman_shift_sinNaN)
    print(raman_shift_sinNaN)
    
    return suavizado_media_movil , raman_shift_sinNaN
    
    
   # print(suavizado_media_movil)




def primera_derivada(datos,op,raman): # SI op es 0 es por que no viene del PCA, si es diferente a 0 es por que viene del PCA y debe de retornar los valores sin NaN
            
    if op == 0 :
        df_derivada = datos.apply(pd.to_numeric, errors='coerce') # PASAMOS A NUMERICO SI ES NECESARIO
        print("xXXXXXXXxxXXXX")
        print(df_derivada)
        
        # Calcular la primera derivada
        df_derivada_diff = df_derivada.diff()
        
        # Imprimir la primera derivada
        #print("Primera Derivada:")
        #print(df_derivada_diff)
        
        
        print("PRIMERA DERIVADA = ", df_derivada_diff.shape)
        print("PRIMERA DERIVADA = ", df_derivada_diff)
        return df_derivada_diff
    
    else:
        df_derivada = datos.apply(pd.to_numeric, errors='coerce')  # Convertir a numérico
        df_derivada_diff = df_derivada.diff()  # Calcular la primera derivada

        # Crear una máscara para identificar las filas sin NaN
        mask = ~df_derivada_diff.isna().any(axis=1)
    
        # Aplicar la máscara para eliminar filas con NaN en la derivada
        df_derivada_diff = df_derivada_diff[mask]  
        raman_shift_sinNaN = raman[mask]  # Ajustar raman_shift
    
        return df_derivada_diff, raman_shift_sinNaN


    #para la llamada del PCA



def segunda_derivada(datos,op,raman):
    
    if op == 0 :
        df_derivada = datos.apply(pd.to_numeric, errors='coerce') # PASAMOS A NUMERICO SI ES NECESARIO
        df_derivada_diff = df_derivada.diff()
        df_derivada_diff = df_derivada_diff.diff()
        print("SEGUNDA DERIVADA = ", df_derivada_diff.shape)
        print("SEGUNDA DERIVADA = ", df_derivada_diff)
        return df_derivada_diff
    else:
        df_derivada = datos.apply(pd.to_numeric, errors='coerce') # PASAMOS A NUMERICO SI ES NECESARIO
        df_derivada_diff = df_derivada.diff()
        df_derivada_diff = df_derivada_diff.diff()
        # Crear una máscara para identificar las filas sin NaN
        mask = ~df_derivada_diff.isna().any(axis=1)  
        # Aplicar la máscara para eliminar filas con NaN en la derivada
        df_derivada_diff = df_derivada_diff[mask]  
        raman_shift_sinNaN = raman[mask]  # Ajustar raman_shift 
        return df_derivada_diff, raman_shift_sinNaN
              



##ESTE CODIGO FUNCIONA PARA LAS OPCIONES 7.8 PARA LAS DERIVADAS
def suavizado_menu_derivadas(df,raman_shift):
   while True:  # Ciclo para permitir "Volver" sin salir de la función
        print("NORMALIZAR POR:")
        print("1- Media")
        print("2- Area")
        print("3- Sin normalizar")
        print("4- Volver")
        opcion = int(input("Selecciona una opción: "))
        
        if opcion == 1 :
            suavizar = normalizado_media(df)
        elif opcion == 2 :
            suavizar = normalizado_area(df,raman_shift)
        elif opcion == 3 :
            suavizar = datos_sin_normalizar(df)
        elif opcion == 4 :
            return None , None , None
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            continue  # Regresa al inicio del ciclo

        while True:  # Submenú de suavizado
            print("DESEA SUAVIZAR")
            print("1. SI")
            print("2. NO")
            print("3. Volver")
            opcion_s =  int(input("OPCION: "))
            if opcion_s == 1:
                while True:  # Bucle para el submenú de suavizado
                    print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
                    print("1- SUAVIZADO POR SAVIZTKY-GOLAY")
                    print("2- SUAVIZADO POR FILTRO GAUSIANO")
                    print("3- SUAVIZADO POR MEDIA MOVIL")
                    print("4- Volver")
                    metodo_suavizado = int(input("OPCION: "))
                    if metodo_suavizado == 1:
                        suavizar = suavizado_saviztky_golay(suavizar)
                        return suavizar , opcion , metodo_suavizado
                    elif metodo_suavizado == 2:
                        suavizar = suavizado_filtroGausiano(df,suavizar)
                        return suavizar , opcion , metodo_suavizado
                    elif metodo_suavizado == 3:
                        suavizar = suavizado_mediamovil(suavizar)   
                        return suavizar , opcion , metodo_suavizado
                    elif metodo_suavizado == 4:
                        break
                    else:
                        print("Opción no válida. Inténtalo de nuevo.")
            elif opcion_s == 2:
                metodo_suavizado = 5 # DEBO ASIGNAR UN VALOR A METODO_SUAVIZADO POR QUE EN CASO DE QUE NO QUIERA SUAVIZAR
                                     # IGUAL DEBE DE RETORNAR UN VALOR O DARA ERROR, PUSE 5 PARA QUE SABER QUE NO SE SUAVIZAO
                
                return suavizar , opcion , metodo_suavizado
            elif opcion_s == 3:
                break
            else: 
                print("Opción no válida. Inténtalo de nuevo.")
                continue
            


# ##ESTE CODIGO FUNCIONA PARA LAS OPCIONES 9,10 PARA LAS CORRECIONES
def menu_correccion(df,raman_shift):
    print("entrooooooooooooo")
    while True:  # Ciclo principal para manejar "Volver"
        print("NORMALIZAR POR:")
        print("1- Media")
        print("2- Área")
        print("3- Sin normalizar")
        print("4- Volver")
        
 
        opcion = int(input("Selecciona una opción: "))


        if opcion == 1:
            suavizar = normalizado_media(df)
        elif opcion == 2:
            suavizar = normalizado_area(df,raman_shift)
        elif opcion == 3:
            suavizar = datos_sin_normalizar(df)
        elif opcion == 4:
            return None, None, None
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            continue

        opcion_s = 0
        while True:  # Submenú 
             if opcion_s == 3:
                 break
             while True:
                print("DESEA SUAVIZAR")
                print("1. Sí")
                print("2. No")
                print("3. Volver")
    
                opcion_s = int(input("Opción: "))
    
                if opcion_s == 1:
                    while True:  # Submenú de métodos de suavizado
                        print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
                        print("1- Suavizado por Saviztky-Golay")
                        print("2- Suavizado por Filtro Gaussiano")
                        print("3- Suavizado por Media Móvil")
                        print("4- Volver")
                        
    
                        metodo_suavizado = int(input("Opción: "))
                        
    
                        if metodo_suavizado == 1:
                            suavizar = suavizado_saviztky_golay(suavizar)
                            break  # Lleva al menú de derivadas
                        elif metodo_suavizado == 2:
                            suavizar = suavizado_filtroGausiano(df,suavizar)
                            break  # Lleva al menú de derivadas
                        elif metodo_suavizado == 3:
                            suavizar = suavizado_mediamovil(suavizar)
                            break  # Lleva al menú de derivadas
                        elif metodo_suavizado == 4:
                            break  # Regresa al menú 
                        else:
                            print("Opción no válida. Inténtalo de nuevo.")
    
                    # Ir al menú de derivadas después del suavizado
                    while True:  # Submenú para derivadas
                        print("\n--- DESEA DERIVAR ---")
                        print("1- Derivar por Primera Derivada")
                        print("2- Derivar por Segunda Derivada")
                        print("3- No Derivar")
                        print("4- Volver")
    
                        opcion_d = int(input("Opción: "))
    
                        if opcion_d == 1:
                            derivada = primera_derivada(suavizar,0,raman_shift)
                            print("Primera derivada aplicada.")
                            return derivada, opcion, metodo_suavizado
                        elif opcion_d == 2:
                            derivada = segunda_derivada(suavizar,0,raman_shift)
                            print("Segunda derivada aplicada.")
                            return derivada, opcion, metodo_suavizado
                        elif opcion_d == 3:
                            print("No se aplicó derivada.")
                            derivada = suavizar  # Devolver el resultado suavizado sin derivar
                            return derivada, opcion, metodo_suavizado
                        elif opcion_d == 4:
                            print("Volviendo al menú '¿Desea suavizar?'.")
                            break  # Regresa al menú 
                        else:
                            print("Opción no válida. Inténtalo de nuevo.")
    
                elif opcion_s == 2:
                    metodo_suavizado = 5  # Valor para indicar que no se aplicó suavizado
                    
                    # Ir directamente al menú de derivadas
                    while True:
                        print("\n--- DESEA DERIVAR ---")
                        print("1- Derivar por Primera Derivada")
                        print("2- Derivar por Segunda Derivada")
                        print("3- No Derivar")
                        print("4- Volver")
                        
                        opcion_d = int(input("Opción: "))
    
                        if opcion_d == 1:
                            derivada = primera_derivada(suavizar,0,raman_shift)
                            return derivada, opcion, metodo_suavizado
                        elif opcion_d == 2:
                            derivada = segunda_derivada(suavizar,0,raman_shift)
                            return derivada, opcion, metodo_suavizado
                        elif opcion_d == 3:
                            derivada = suavizar  # Devolver el resultado suavizado sin derivar
                            return derivada, opcion, metodo_suavizado
                        elif opcion_d == 4:
                            break  # Regresa al menú 
                        else:
                            print("Opción no válida. Inténtalo de nuevo.")
    
                elif opcion_s == 3:
                    break  # Regresa al menú principal
                else:
                    print("Opción no válida. Inténtalo de nuevo.")
                    continue
                




# ##ESTE CODIGO FUNCIONA PARA LAS OPCIONES 9,10 PARA LAS CORRECIONES
def menu_correccion_pca(df,raman_shift):
    print("entrooooooooooooo")
    while True:  # Ciclo principal para manejar "Volver"
        print("NORMALIZAR POR:")
        print("1- Media")
        print("2- Área")
        print("3- Sin normalizar")
        print("4- Volver")
        
 
        opcion = int(input("Selecciona una opción: "))


        if opcion == 1:
            suavizar = normalizado_media(df)
        elif opcion == 2:
            suavizar = normalizado_area(df,raman_shift)
        elif opcion == 3:
            suavizar = datos_sin_normalizar(df)
        elif opcion == 4:
            return None, None, None
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            continue

        opcion_s = 0
        while True:  # Submenú 
             if opcion_s == 3:
                 break
             while True:
                print("DESEA SUAVIZAR")
                print("1. Sí")
                print("2. No")
                print("3. Volver")
    
                opcion_s = int(input("Opción: "))
    
                if opcion_s == 1:
                    while True:  # Submenú de métodos de suavizado
                        print("\n--- POR CUAL METODO DESEAS SUAVIZAR ---")
                        print("1- Suavizado por Saviztky-Golay")
                        print("2- Suavizado por Filtro Gaussiano")
                        print("3- Suavizado por Media Móvil")
                        print("4- Volver")
                        
    
                        metodo_suavizado = int(input("Opción: "))
                        
    
                        if metodo_suavizado == 1:
                            suavizar = suavizado_saviztky_golay(suavizar)
                            break  # Lleva al menú de derivadas
                        elif metodo_suavizado == 2:
                            suavizar = suavizado_filtroGausiano(df,suavizar)
                            break  # Lleva al menú de derivadas
                        elif metodo_suavizado == 3:
                            suavizar , raman_shift_sinNaN = suavizado_mediamovil_paraPCA(suavizar,raman_shift)
                            break  # Lleva al menú de derivadas
                        elif metodo_suavizado == 4:
                            break  # Regresa al menú 
                        else:
                            print("Opción no válida. Inténtalo de nuevo.")
    
                    # Ir al menú de derivadas después del suavizado
                    while True:  # Submenú para derivadas
                        print("\n--- DESEA DERIVAR ---")
                        print("1- Derivar por Primera Derivada")
                        print("2- Derivar por Segunda Derivada")
                        print("3- No Derivar")
                        print("4- Volver")
    
                        opcion_d = int(input("Opción: "))
    
                        if opcion_d == 1:
                            if metodo_suavizado != 3:
                                derivada , raman_actual = primera_derivada(suavizar,1,raman_shift)
                                return derivada, opcion, metodo_suavizado , raman_shift
                            else:
                                derivada , raman_actual = primera_derivada(suavizar,1,raman_shift_sinNaN)
                                return derivada, opcion, metodo_suavizado , raman_actual
                        elif opcion_d == 2:
                            if metodo_suavizado != 3:
                                derivada , raman_actual = segunda_derivada(suavizar,1,raman_shift)
                                return derivada, opcion, metodo_suavizado , raman_shift
                            else:
                                derivada , raman_actual = segunda_derivada(suavizar,1,raman_shift_sinNaN)
                                return derivada, opcion, metodo_suavizado , raman_actual
                        elif opcion_d == 3:
                            print("No se aplicó derivada.")
                            derivada = suavizar  # Devolver el resultado suavizado sin derivar                      
                            if metodo_suavizado != 3:
                                return derivada, opcion, metodo_suavizado , raman_shift
                            else:
                                return derivada, opcion, metodo_suavizado , raman_shift_sinNaN
                        elif opcion_d == 4:
                            print("Volviendo al menú '¿Desea suavizar?'.")
                            break  # Regresa al menú 
                        else:
                            print("Opción no válida. Inténtalo de nuevo.")
    
                elif opcion_s == 2:
                    metodo_suavizado = 5  # Valor para indicar que no se aplicó suavizado
                    
                    # Ir directamente al menú de derivadas
                    while True:
                        print("\n--- DESEA DERIVAR ---")
                        print("1- Derivar por Primera Derivada")
                        print("2- Derivar por Segunda Derivada")
                        print("3- No Derivar")
                        print("4- Volver")
                        
                        opcion_d = int(input("Opción: "))
    
                        if opcion_d == 1:
                            if metodo_suavizado != 3:
                                derivada , raman_actual = primera_derivada(suavizar,1,raman_shift)
                                return derivada, opcion, metodo_suavizado , raman_shift
                            else:
                                derivada , raman_actual = primera_derivada(suavizar,1,raman_shift_sinNaN)
                                return derivada, opcion, metodo_suavizado , raman_actual
                        elif opcion_d == 2:
                            if metodo_suavizado != 3:
                                derivada , raman_actual = segunda_derivada(suavizar,1,raman_shift)
                                return derivada, opcion, metodo_suavizado , raman_shift
                            else:
                                derivada , raman_actual = segunda_derivada(suavizar,1,raman_shift_sinNaN)
                                return derivada, opcion, metodo_suavizado , raman_actual
                        elif opcion_d == 3:
                            derivada = suavizar  # Devolver el resultado suavizado sin derivar
                            if metodo_suavizado != 3:
                                return derivada, opcion, metodo_suavizado , raman_shift
                            else:
                                return derivada, opcion, metodo_suavizado , raman_shift_sinNaN
                        elif opcion_d == 4:
                            break  # Regresa al menú 
                        else:
                            print("Opción no válida. Inténtalo de nuevo.")
    
                elif opcion_s == 3:
                    break  # Regresa al menú principal
                else:
                    print("Opción no válida. Inténtalo de nuevo.")
                    continue
                




# # # POR EL METODO DE REGRESION LINEAL
def correcion_LineaB(normalizado_f,raman_shift):
    print("NORMALIZADO-F")
    print(normalizado_f)

    # Obtener los índices de las filas válidas
    indices_validos = normalizado_f.dropna().index
    print("INDICES VALIDOS")
    print(indices_validos)
    # Filtrar raman_shift por los índices válidos
    raman_shift_filtrado = raman_shift.loc[indices_validos].reset_index(drop=True)
    print("RAMAN SHIFT FILTRADO")
    print(raman_shift_filtrado)
    # Filtrar normalizado_f por los índices válidos
    normalizado_f_filtrado = normalizado_f.loc[indices_validos].reset_index(drop=True)
    print("NORMALIZADO F FILTRADO")
    print(normalizado_f_filtrado)

 
        #print("NORMALIZADO-F")
        #print(normalizado_f)
    cabecera_aux = normalizado_f_filtrado.columns
        #print("XDDDDDDDDDD")
        #print(cabecera_aux)
    np_corregido = normalizado_f_filtrado.to_numpy() # pasamos a numpy para borrar la cabecera de tipos
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
    print("AHORA ENTRARA EN EL FOR")
    for col in df_corregido.columns:
            intensidades = df_corregido[col]  # Extraer la columna actual
            intensidades = pd.to_numeric(intensidades, errors='coerce') #ASEGURAMOS QUE LAS INTENSIDADES SEAN NUMERICOS           
            raman_shift_filtrado = raman_shift_filtrado.astype(str).str.replace(",", ".")  # Asegurar que usa '.' como separador decimal
            raman_shift_filtrado = pd.to_numeric(raman_shift_filtrado, errors="coerce")  # Convertir a float
            print("Tipos en raman_shift_filtrado:", raman_shift_filtrado.dtype)
            print("Tipos en intensidades:", intensidades.dtype)
            print("INTENSIDADES")
            print(intensidades)
            # print(cont)                        
            # Calcularmos la pendiente 
            coef = np.polyfit(raman_shift_filtrado, intensidades, 1)  # Grado 1 para línea recta , coef = coeficiente de Y=mx+b , coef[0] es la pendiente m, y coef[1] es la intersección b.
            # SE PONE DE GRADO 1 POR QUE QUEREMOS AJUSTAR UN POLINOMIO DE LA FORMA Y=MX+B 
            print("PRIMERA INTERACION CON EL POLYFIT")
            pendiente = coef[0]  # La pendiente (m) está en el índice 0 de los coeficientes
            interseccion = coef[1] # La interseccion (b) esta en el indice 1 
            # Guardamos la pendiente y intersecciones en el diccionario
            pendientes[col] = pendiente
            intersecciones[col] = interseccion
            #print(raman_shift[pos])
            #print("xD")
            y_ajustado = []
            for pos, intensidad in enumerate(intensidades):
                y = pendiente * raman_shift_filtrado[pos] + interseccion  
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
    print("LLEGO HASTA ACA")
    print(df_y_ajustados)

    return df_y_ajustados, raman_shift_filtrado









def pca(dato, raman_shift,archivo_nombre,asignacion_colores,types, datafusion): #datafusion = "opcion del data fusion" , si es 1 es por que viene del mid-level a cargar los componentes seleccionados por el usuario

    print("Datos PCA:")
    print(dato.shape)  
    print(dato)
    num_muestras, num_variables = dato.shape #OBTENEMOS LA CANTIDAD DE  FILAS Y COLUMNAS

    max_pc = min(num_muestras, num_variables) #CANTIDAD MAXIMA DE N ES EL MENOR NUMERO

    aux = 0
    if datafusion != 1:
        print("Cómo deseas visualizar los componentes principales:")
        print("1 - Elegir 2 Componentes para graficar en 2D")
        print("2 - Elegir 3 Componentes para graficar en 3D")
        print("3.  Grafico de Loanding")
        print("4-  Salir")
        visualizar_pca = int(input("Opción = "))
    else: #PARA EL DATAFUSION
        #POR LO QUE ENTENDI EL RESULTADO DE LOS COMPONENTES PRINCIPALES ES EL MISMO YA SEA UN n=2 O UN n=3
        componentes_x = []
        componentes_y = []
        componentes_z = []

        cant_pca = int(input("Cantidad de componentes principales= "))
        
        print(f"Que componentes principales del archivo {archivo_nombre} deseas visualizar:")
        # PREGUNTAR ACA QUE PASA CUANDO ENTRE LOS ARCHIVOS FUSIONADOS TENEMOS MAS DE 3 COMPONENTES PRINCIPALES

        # NO VA A FUNCIONAR AHORA EL MID-LEVEL POR QUE a visualizar_pca no se le carga un valor


    if visualizar_pca == 4:
        return
    elif visualizar_pca == 3:
        aux = 1 # su unica funcion es cambiar el valor de visualizar_pca a 3 cuando esta dentro de las condicionales de visualizar_pca = a y 1 o 2 cuando es llamado por la opcion loanding 
        print("Cuantos Componentes principales deseas visualizar?")
        print("1. Dos PCA")
        print("2. Tres PCA")
        cant_pca = int(input("Cantidad Pca = "))
        if cant_pca == 1:
            visualizar_pca = 1
        elif cant_pca == 2:
            visualizar_pca = 2
        
    componentes_x = []
    componentes_y = []
    componentes_z = []

    while True:
        n_componentes = int(input(f"Ingrese la cantidad de componentes principales (1-{max_pc}): "))
        if 1 < n_componentes <= max_pc:
            break
        print(f"n debe estar entre 2 y {max_pc}.")



    dato = dato.dropna() #eliminamos las filas con valores NAN
    datos_df = dato.transpose() #PASAMOS LA CABECERA DE TIPOS A LA COLUMNA
    
    
    #with_mean=True → Resta la media (centraliza los datos).
    #with_std=True → Divide por la desviación estándar (escala los datos).
    #StandardScaler() y StandardScaler(with_std=True) son lo mismo porque el valor por defecto de with_std es True
    
    #DESCOMENTAR LO DE ABAJO SI QUIERO CENTRALIZAR Y DIVIDIR POR SU DESVACION ESTANDAR
    escalado = StandardScaler()  #Escalar los datos para que cada columna (intensidades en cada longitud de onda) tenga una media de 0 y una desviación estándar de 1.
    dato_escalado = escalado.fit_transform(datos_df)

    print(f"🔹 Número de columnas en dato_escalado: {dato_escalado.shape[1]}")  # Debería ser 234, no 731
    print(f"🔹 Tamaño de raman_shift: {raman_shift.shape[0]}")  # Debe ser 234


    # print("dato escalado dentro de la funcio de pca")
    # print(dato_escalado.shape)


    #SI SOLO QUIERO CENTRALIZAR Y NO DIVIDIR POR LA DESVIACION ESTANDAR (Igual al Orange)
    #centralizador = StandardScaler(with_std=False)  # Solo restamos la media sin escalar por desviación estándar
    #dato_centralizado = centralizador.fit_transform(datos_df)

    # # Ver cuántos valores únicos tiene cada columna
    # print(dato_escalado.nunique(axis=0))
    
    # # Ver cuántas columnas tienen valores NaN
    # print(dato_escalado.isnull().sum())


    pca = PCA(n_components=n_componentes) 
    dato_pca = pca.fit_transform(dato_escalado)  # fit_transform ya hace el calcilo de los eigenvectores y eigenvalores y matriz de covarianza, (cambiar dato_centralizado por dato_escalado si quiero usar el otro metodo)
    # El pca en pca.fit_transform(dato_escalado) no cambia su valor, sino que almacena internamente los parámetros del PCA (eigenvectores, varianza explicada, etc.) 
    #y es por eso que dentro de la funcion de loading puede hacer el calculo de loadings = pca.components_ y de viene que loadings saclos valores 
    # print("Datos_PCA xd:")
    # print(dato_pca[:5]) 
    # print("pca.components_PCAAA")
    # print(pca.components_)

    print(f"🔹 Dimensiones de loadings (pca.components_): {pca.components_.shape}")  # Debería ser (n_componentes, 731)

    # Obtener Eigenvalores (Varianza explicada)
    eigenvalores = pca.explained_variance_
    suma_eigenvalores = sum(eigenvalores) # Hallamos su suma solo para despues ver el % de importancia
    porcentaje_varianza = (eigenvalores / suma_eigenvalores) * 100
    # Obtener Eigenvectores (Componentes principales)
    eigenvectores = pca.components_ * -1 # en realidad multiplique por -1 por que quiero cambiar el orden de los signos, No afecta el resultado final
 
    
     # descomentar para ver los datos de los de abajo
    # # Mostrar Eigenvalores
    # print("Eigenvalores (Varianza explicada):")
    # print(eigenvalores)
   
    # # Mostrar Eigenvectores
    # print("\nEigenvectores (Componentes principales):")
    # print(eigenvectores.T)
    
    # print("Porcentaje de Varianza Explicada por cada componente")
    # print(porcentaje_varianza)




    if visualizar_pca == 1: # lee los datos para el 2D
        componentes_x = [int(input(f"Ingrese el número de PC para X (1-{n_componentes}): ")) - 1]
        componentes_y = [int(input(f"Ingrese el número de PC para Y (1-{n_componentes}): ")) - 1]
        eje_x = dato_pca[:, componentes_x] #CARGAMOS LAS COORDENADAS EN X QUE CARGO EL USUARIO
        eje_y = dato_pca[:, componentes_y] #CARGAMOS LAS COORDENADAS EN Y QUE CARGO EL USUARIO
        dato_pca = np.column_stack((eje_x, eje_y)) #GUARDAMOS DENTRO DE LA VARIABLE LA UNION DE LOS PC SELECCIONADOS

        eje_z = "No Aporta" # SOLO PARA QUE APAREZCA ESE MENSAJE EN EL ARCHIVO
        porcentaje_varianza_z = "No Aporta" # SOLO PARA QUE APAREZCA ESE MENSAJE EN EL ARCHIVO
        componentes_z = "No Aporta" # SOLO PARA QUE APAREZCA ESE MENSAJE EN EL ARCHIVO
        # ESTA PARTE ES SOLO PARA APARTAR SU PORCENTAJE DENTRO DE UNA VARIABLE Y ENVIAR AL GRAFICADOR PARA QUE APAREZCA EN LOS EJES DEL GRAFICO EL %
        porcentaje_varianza_x = porcentaje_varianza[componentes_x]
        #print("PORCENTAJE DE VARIANZA X", porcentaje_varianza_x)
        porcentaje_varianza_y = porcentaje_varianza[componentes_y]
        #print("PORCENTAJE DE VARIANZA Y", porcentaje_varianza_y)
        # print("x= ",componentes_x)
        # print("y= ",componentes_y)
        if aux == 1:
            visualizar_pca = 3 # pongo 3 para que se vaya a la funcion de grafica loanding una vez obtenido los datos
            componente_seleccionado = [componentes_x[0] , componentes_y[0]]
            
            
        print("DATO PCA")
        print(dato_pca)
        print(dato_pca.shape)


    elif visualizar_pca == 2: # lee los datos para el 3D
        componentes_x = [int(input(f"Ingrese el número de PC para X (1-{n_componentes}): ")) - 1]
        componentes_y = [int(input(f"Ingrese el número de PC para Y (1-{n_componentes}): ")) - 1]
        componentes_z = [int(input(f"Ingrese el número de PC para Z (1-{n_componentes}): ")) - 1]
        eje_x = dato_pca[:, componentes_x]
        eje_y = dato_pca[:, componentes_y]
        eje_z = dato_pca[:, componentes_z]
        
        dato_pca = np.column_stack((eje_x, eje_y, eje_z)) #GUARDAMOS DENTRO DE LA VARIABLE LA UNION DE LOS PC SELECCIONADOS
        
        # print("DATO PCA")
        # print(dato_pca)
        if aux == 1:
            visualizar_pca = 3 # pongo 3 para que se vaya a la funcion de grafica loanding una vez obtenido los datos
            componente_seleccionado = [componentes_x[0] , componentes_y[0] , componentes_z[0]]
    

        # ESTA PARTE ES SOLO PARA APARTAR SU PORCENTAJE DENTRO DE UNA VARIABLE Y ENVIAR AL GRAFICADOR PARA QUE APAREZCA EN LOS EJES DEL GRAFICO EL %
        porcentaje_varianza_x = porcentaje_varianza[componentes_x]
        print("PORCENTAJE DE VARIANZA X", porcentaje_varianza_x)
        porcentaje_varianza_y = porcentaje_varianza[componentes_y]
        print("PORCENTAJE DE VARIANZA Y", porcentaje_varianza_y)
        porcentaje_varianza_z = porcentaje_varianza[componentes_z]
        print("PORCENTAJE DE VARIANZA Z", porcentaje_varianza_z)
    #     print("x= ",componentes_x)
    #     print("y= ",componentes_y)
    #     print("z= ",componentes_z)
        
    
    print("Deseas generar un informe del loadings")
    print("1. Generar Informe")
    print("2. No")
    informe = int(input("Opcion: "))
    
    print(f"🔹 Dimensiones antes de PCA: {dato_escalado.shape}")  # Debe ser (233, 731)
    print(f"🔹 Dimensiones después de PCA: {dato_pca.shape}")  # Debe ser (233, n_componentes)
    print(f"🔹 Dimensión de raman_shift: {raman_shift.shape}")  # Debe ser (731,)
    print(f"🔹 Dimensión de loadings: {pca.components_.shape}")  # Debe ser (n_componentes, 731)

    
    
    if informe == 1:
        nombre_archivo = input("Ingrese el nombre del archivo: ")
        with open(nombre_archivo, "w", encoding="utf-8") as file:
            file.write("INFORME SOBRE EL PCA.\n")
            file.write("-------------------------\n")
            file.write(f"Ingrese la cantidad de componentes principales (1-{max_pc}): {n_componentes}\n")
            file.write("Eigenvalores\n")
            file.write(f"{eigenvalores}")
            file.write("\nEigenvectores\n")
            file.write(f"{eigenvectores.T}\n")
            file.write("Porcentaje de Varianza Explicada por cada componente\n")
            file.write(f"{porcentaje_varianza}\n")
            file.write(f"Componente Principal para eje X = {componentes_x[0]+1}\n")
            file.write(f"Componente Principal para eje Y = {componentes_y[0]+1}\n")
            if visualizar_pca == 1:
                file.write("Componente Principal para eje Z = No aporta\n")
            elif visualizar_pca == 2:
                file.write(f"Componente Principal para eje Z = {componentes_z[0]+1}\n")
            # file.write(f"Eje X = \n {eje_x}\n")
            # file.write(f"Eje Y = \n {eje_y}\n")
            # file.write(f"Eje Z = \n {eje_z}\n")
            file.write(f"Porcentajes de Varianza X = {porcentaje_varianza_x[0]}\n")
            file.write(f"Porcentajes de Varianza Y = {porcentaje_varianza_y[0]}\n")
            file.write(f"Porcentajes de Varianza Z = {porcentaje_varianza_z[0]}\n")
            file.write("Eje X \t\t Eje Y \t\t Eje Z\n")
            for x , y , z in zip(eje_x , eje_y , eje_z):
                file.write(f"{x}\t{y}\t{z}\n")
            file.write("RESULTADO PCA\n")
            file.write(f"{dato_pca}")
    else:
        print("Generando Grafico...")

    # LLAMAMOS A LAS FUNCIONES PARA GRAFICA
    # COMO plot_pca_2d(dato_pca) ES EN 2D TIENE QUE LLAMAR A LA FUNCION generar_elipse
    # COMO plot_pca_3d(dato_pca) ES EN 3D TIENE QUE LLAMAR A LA FUNCION generar_elipsoide
    if visualizar_pca == 1:
        plot_pca_2d(archivo_nombre,asignacion_colores,types,dato_pca,porcentaje_varianza_x,porcentaje_varianza_y,componentes_x,componentes_y)
    elif visualizar_pca == 2:
        plot_pca_3d(archivo_nombre,asignacion_colores,types,dato_pca,porcentaje_varianza_x,porcentaje_varianza_y,porcentaje_varianza_z,componentes_x,componentes_y,componentes_z)
    elif visualizar_pca == 3:
        grafico_loading(pca, raman_shift, componente_seleccionado)


def plot_pca_2d(archivo_nombre,asignacion_colores,types,dato_pca,porcentaje_varianza_x,porcentaje_varianza_y,componentes_x,componentes_y):
    """
    Realiza PCA, grafica en 2D y dibuja elipses de confianza para cada tipo.
    """

    
    df_pca = pd.DataFrame(dato_pca, columns=['PC1', 'PC2']) # Convertir a DataFrame
    df_pca['Tipo'] = types  # Agregar tipo de cada punto
    
    
    print("Cantidad de puntos para graficar:", len(df_pca))
    print("Tipos:", df_pca['Tipo'].unique())
    print("DF_PCA")
    print(df_pca)
    
    intervalo_confianza = (float(input("INGRESE EL INTERVALO DE CONFIAZA % = ")))/100

    # Crear la figura en Plotly
    fig = go.Figure()
    

    for tipo in np.unique(types): #OBTENEMOS LOS TIPOS DE LOS DATOS PERO SIN REPETIR COMO COLLAGEN,GLYCOGEN,DNA
        print("TIPO")
        print(tipo)
        indices = df_pca['Tipo'] == tipo #FILTRA DATOS QUE COINCIDE CON TIPO, EJEMPLO: TIPO=LIPIDS Y SI df_pca['Tipo'] NO ES EL MISMO TIPO TIRA FALSO
        print("INDICES")
        print(indices)
        fig.add_trace(go.Scatter(
            x=df_pca.loc[indices, 'PC1'], # Usa los valores de PC1 del tipo actual. Selecciona solo las filas donde indices es True, es decir, solo los puntos de ese tipo
            y=df_pca.loc[indices, 'PC2'], #  Usa los valores de PC2 del tipo actual. Selecciona solo las filas donde indices es True, es decir, solo los puntos de ese tipo
            mode='markers',
            marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
            name=f'Tipo {tipo}' # PARA LA LEYENDA
            
        ))
    
    
        # Calcular el centro y la covarianza del grupo
        datos_tipo = df_pca.loc[indices, ['PC1', 'PC2']].to_numpy() # CONVERTIMOS LOS DATOS DEL TIPO ACTUAL A NUMPY
        print("DATOS_TIPO")
        print(datos_tipo[:5])
        centro = np.mean(datos_tipo, axis=0) # CALCULAMOS LA MEDIA DE PC1 Y PC2 OSEA SERIA EL CENTRO DEL GRUPO 
        print("CENTRO",centro) 
        cov = np.cov(datos_tipo.T) #CALCULA LA MATRIZ DE COVARIANZA PARA TENER LA FORMA Y ORIENTACION DEL ELIPSE
        #print("DATOS_TIPO_T (transpuesta)")
        #print(datos_tipo.T[:5])
        print("COV: ubicar estas coordenadas en el grafico para saber el limite del tamanio de la elipse") # cov es covarianza
        print(cov)
        
        
        if datos_tipo.shape[0] > 2:  # nos aseguramos de que haya suficientes puntos, tiene ser  un minimo de 2, por logica no puede ser 1 ya que como vas a halla el promedio de un solo numero, La media de un solo punto sigue siendo el mismo punto
            elipse = generar_elipse(centro, cov, color=asignacion_colores[tipo], intervalo_confianza = intervalo_confianza)
            fig.add_trace(elipse)
    
    # Configurar los ejes y el título
    fig.update_layout(
        title=f'Análisis de Componentes Principales 2D de {archivo_nombre}',
        xaxis_title= f'PC{componentes_x[0]+1} {porcentaje_varianza_x[0]:.2f}%',
        yaxis_title= f'PC{componentes_y[0]+1} {porcentaje_varianza_y[0]:.2f}%',
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Mostrar la gráfica
    fig.show(renderer="browser")
    fig.show(renderer="svg")


# RECIBE MUCHOS PARAMETROS SOLO POR QUE QUIERO QUE SALGA LINDO EL NOMBRE DE LOS EJES
def plot_pca_3d(archivo_nombre,asignacion_colores,types,dato_pca,porcentaje_varianza_x,porcentaje_varianza_y,porcentaje_varianza_z,componentes_x,componentes_y,componentes_z):

    df_pca = pd.DataFrame(dato_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca['Tipo'] = types

    # print("Cantidad de puntos para graficar:", len(df_pca))
    # print("Tipos:", df_pca['Tipo'].unique())
    
    intervalo_confianza = (float(input("INGRESE EL INTERVALO DE CONFIAZA % = ")))/100

    fig = go.Figure() #Usas Plotly

    for tipo in np.unique(types):
        indices = df_pca['Tipo'] == tipo
        fig.add_trace(go.Scatter3d(
            x=df_pca.loc[indices, 'PC1'], # Usa los valores de PC1 del tipo actual. Selecciona solo las filas donde indices es True, es decir, solo los puntos de ese tipo
            y=df_pca.loc[indices, 'PC2'],
            z=df_pca.loc[indices, 'PC3'],
            mode='markers',
            marker=dict(size=5, color=asignacion_colores[tipo], opacity=0.7),
            name=f'Tipo {tipo}'
        ))

        # Generar elipsoide de confianza
        datos_tipo = df_pca.loc[indices, ['PC1', 'PC2', 'PC3']].to_numpy()
        if datos_tipo.shape[0] > 3:
            centro = np.mean(datos_tipo, axis=0)
            cov = np.cov(datos_tipo.T)
            elipsoide = generar_elipsoide(centro, cov, asignacion_colores[tipo],intervalo_confianza = intervalo_confianza)
            fig.add_trace(elipsoide)


    fig.update_layout(
        legend=dict(
                font=dict(
                size=18  # Aumenta el tamaño de la leyenda (puedes probar con 16, 18, etc.)
                ),
                title=dict(
                            text="Tipos de Muestras",  # Título de la leyenda
                            font=dict(size=16, family="Arial", color="black")  # Configuración del título
                          ),
                itemsizing="constant",  # Mantiene el tamaño de los íconos proporcional
                bordercolor="black",  # Color del borde de la leyenda
                borderwidth=2,  # Grosor del borde
                bgcolor="rgba(255,255,255,0.7)"  # Fondo semitransparente para la leyenda
        ),
        title=dict(
                    text=f'<b><u>Análisis de Componentes Principales 3D de {archivo_nombre}</u></b>',  # Negrita y subrayado
                    x=0.5,  # Centrar el título (0 izquierda, 1 derecha, 0.5 centro)
                    xanchor="center",  # Asegura que esté alineado al centro
                    font=dict(
                    family="Arial",  # Tipo de letra
                    size=20,  # Tamaño del título
                    color="black"  # Color del título
                    )),
        scene=dict(
            xaxis_title= f'PC{componentes_x[0]+1} {porcentaje_varianza_x[0]:.2f}%',# PARA LA ETIQUETAS
            yaxis_title= f'PC{componentes_y[0]+1} {porcentaje_varianza_y[0]:.2f}%',
            zaxis_title= f'PC{componentes_z[0]+1} {porcentaje_varianza_z[0]:.2f}%',
            # PARA QUE EL CUBO SEA DE COLOR GRIS
            #xaxis=dict(backgroundcolor="rgb(240, 240, 240)", gridcolor="gray", showbackground=True),
            #yaxis=dict(backgroundcolor="rgb(240, 240, 240)", gridcolor="gray", showbackground=True),
            #zaxis=dict(backgroundcolor="rgb(240, 240, 240)", gridcolor="gray", showbackground=True)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    #print("LLEGO HASTA ACA")
    fig.show(renderer="browser") 


def generar_elipsoide(centro, cov, color='rgba(150,150,150,0.3)',intervalo_confianza = 0.95):
    
    print("INTERVALO CONFIANZA")
    print(intervalo_confianza)
    
    U, S, _ = np.linalg.svd(cov)
    radii = np.sqrt(chi2.ppf(intervalo_confianza, df=3) * S) # 0.999 para que encierre lo mas que pueda todas las muestras dentro del elipsoide

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = centro + np.dot(U, np.multiply(radii, [x[i, j], y[i, j], z[i, j]]))

    return go.Surface(x=x, y=y, z=z, opacity=0.3, colorscale=[[0, color], [1, color]], showscale=False)




# num_puntos=100: Número de puntos para dibujar la elipse.
# color='rgba(150,150,150,0.3)': Color de la elipse (por defecto, gris semitransparente).
def generar_elipse(centro, cov, num_puntos=100, color='rgba(150,150,150,0.3)',intervalo_confianza = 0.95):
    
    print("INTERVALO CONFIANZA")
    print(intervalo_confianza)
        
    #Descompone en valores singulares (SVD) la matriz de covarianza cov:
    #         U: Matriz de rotación (eigenvectores).
    #         S: Valores singulares (eigenvalores, escalas de la elipse).
    #         _: No se usa la tercera matriz de la descomposición.
    #¿Por qué SVD?
    #     U nos da la orientación de la elipse.
    #     S nos da las escalas (los radios) de la elipse.
    
    # Determina hasta dónde se extiende la elipse en cada dirección para cubrir el 95% (o el porcentaje elegido) de los datos.
    U, S, _ = np.linalg.svd(cov)
    radii = np.sqrt(chi2.ppf(intervalo_confianza, df=2) * S)  # Intervalo de confianza del 95%, ESTE TIENE QUE METER EL USUARIO 95%
    
    theta = np.linspace(0, 2 * np.pi, num_puntos)
    x = np.cos(theta)
    y = np.sin(theta)
    
    #     np.array([x, y]).T → Crea una matriz de (num_puntos, 2) con las coordenadas originales.
    # np.diag(radii) → Escala los puntos en las direcciones correctas.
    # @ U.T → Rota la elipse usando la matriz de eigenvectores.
    # + centro → Mueve la elipse al centro adecuado.
    
    elipse = np.array([x, y]).T @ np.diag(radii) @ U.T + centro
    
    return go.Scatter(
        x=elipse[:, 0], # Coordenadas X de la elipse.
        y=elipse[:, 1], # Coordenadas Y de la elipse.
        mode='lines', #Dibuja líneas en lugar de puntos.
        line=dict(color=color, width=2), # Define el color y grosor de la línea.
        showlegend=False
    )




def grafico_loading(pca, raman_shift, op_pca):
    
    print("ENTRO EN GRAFICO DE LOADING")
    # print("selected_components=",op_pca)
    # print("PCA dentro de la función de loading")
    # print(pca)
    # print("Raman shift =",raman_shift.shape)
    # print(raman_shift)
    plt.figure(figsize=(10, 6))

    # if not hasattr(pca, 'components_'):
    #     raise ValueError("El objeto PCA proporcionado no tiene componentes. Asegúrate de pasar el modelo PCA y no los datos transformados.")




    #pca = PCA(n_components=n_componentes)
    #dato_pca = pca.fit_transform(dato_escalado)  # fit_transform ya hace el calcilo de los eigenvectores y eigenvalores y matriz de covarianza
    ## El pca en pca.fit_transform(dato_escalado) no cambia su valor, sino que almacena internamente los parámetros del PCA (eigenvectores, varianza explicada, etc.) 
    ##y es por eso que dentro de la funcion de loading puede hacer el calculo de loadings = pca.components_ y de viene que loadings saclos valores 
    loadings = pca.components_  # Obtener los loadings aca es donde hace los calculos de los pessos
    print("DESPUES DE CALCULAR LOS PESO DE LOADING")
    # print("pca.components_")
    # print(pca.components_)
    # print("LOADING")
    # print(loadings)
    # print("LOADING.shape")
    # print(loadings.shape)
    n_componentes = loadings.shape[0]  # Número de componentes principales
    # print(" n_components=", n_componentes)
    # print(f"Shape de loadings: {loadings.shape}")  # (n_components, n_features)
    # print(f"Tamaño de raman_shift: {len(raman_shift)}")


    # Asegurar que `selected_components` sea una lista de enteros
    # if isinstance(selected_components, int):  
    #     selected_components = [selected_components]  # Convertir a lista si es un solo número

    # if not isinstance(selected_components, list):
    #     raise ValueError("Los componentes seleccionados deben ser una lista de índices.")


    #PARA LA OPCION DE DESCARGAR UN .CSV
    df_loadings = pd.DataFrame({"Raman_Shift": raman_shift}) #CREAMOS UN DF Y ASIGNAMOS EL RAMANSHIFT COMO PRIMERA COLUMNA

    for i in op_pca:  
        print("ENTRO EN EL FOR DE LOADIN")
        if not isinstance(i, int):  
            raise ValueError(f"El índice de componente debe ser un entero, pero se recibió {type(i)}.")
        if i >= n_componentes:
            print(f"Advertencia: PC {i+1} no existe en los datos de PCA.")
            continue


        # print("Dimensión de raman_shift:", raman_shift.shape)
        # print("Dimensión de PCA components:", loadings.shape)
        # print("ramanshift=", raman_shift.shape, "PCA=", loadings[i, :].shape)
        # #print(loadings[i, :])
        # print(f"Tamaño de raman_shift: {len(raman_shift)}")
        # print(f"Tamaño de loadings[i, :]: {loadings[i, :].shape}")

        #print("Raman shift")
        #print(raman_shift)

        plt.plot(raman_shift, loadings[i, :], label=f'PC {i+1}')
       
        #PARA LA OPCION DE DESCARGAR UN .CSV
        df_loadings[f'PC{i+1}'] = loadings[i, :] #LUEGO ASIGANOMS LOS VALORES DE LOANDING AL df_loadings
        print("iteraccion: ",i)
    
    # para que en el eje x muestre los valores cada 250 por que osino sale muy encimado
    # Ajustar el intervalo del eje X a cada 500 cm⁻¹
    ax = plt.gca()  # Obtener el objeto de los ejes
    ax.xaxis.set_major_locator(ticker.MultipleLocator(200)) # PARA QUE SE ENTIENDA EL EJE X Y SALGA EN INTERVALO

    ay = plt.gca()  # Obtener el objeto de los ejes
    ay.yaxis.set_major_locator(ticker.MultipleLocator(0.1)) # PARA QUE SE ENTIENDA EL EJE X Y SALGA EN INTERVALO


    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Loading')
    plt.title('Loading Plot para PCA y Raman Shift')
    plt.legend()
    plt.grid()
    plt.show()
   
    
    print("Descargar csv de loading:")
    print("1. Si")
    print("2. No")
    des_loand = int(input("Opcion: "))
    if des_loand == 1:
        df_loadings.to_csv("loadings.csv", index=False)
        print("Descargando...")
        time.sleep(1)
    elif des_loand == 2:
        print("Salir..")
        
    #TAREA
    #HACER CODIGO DE SHIRLEY
    #CORREGIR EL DEDROGRAMA
    #CORREGIR TODOS LOS ERRORES QUE TIRA AL MOVER ATRAS
    #CORREGIR EL TEMA DEL AREA QUE ESTA MULTIPLICANDO POR -1
    #CORREGIR EL DEMDOGRAMA POR QUE SALE MUY FEO
    # tira error al querer normalizar el archivo de LucasNo.csv
    #CREO QUE EL GRAFICO MIS ARCHIVOS .CSV QUE GENERO NO FUNCIONA EN ORANGE POR QUE LA CELDA 0,0 NO DICE RAMANSHITF






def hca(dato,raman_shift):
    print("Datos:")
    print(dato)
    print("Raman_shift")
    print(raman_shift)
    datos = dato.dropna()  # Eliminamos filas con NaN ya que el algoritmo de linkage no lee esos tipos de datos
    print("Datos despues de eliminar NAN:")
    print(datos)
    while True:  # Bucle principal del menú
        print("\n--- Menú Principal ---")
        print("Cual método de distancias deseas utilizar:")
        print("1. Euclidiana")
        print("2. Manhattan")
        print("3. Coseno")
        print("4. Chebyshev")
        print("5. Correlación Pearson")
        print("6. Correlación Spearman")
        print("7. Jaccard") #foto que tome de la notebook de Edher
        m_dis = int(input("Opción: "))
        if m_dis == 1:
            nombre_plot= "Euclidiana"
            distancia = pdist(datos.T, metric='euclidean')  # Transponer porque queremos calcular entre columnas
            print("DISTANCIA")
            print(distancia)
            df_distancias = pd.DataFrame(squareform(distancia), index=datos.T.index, columns=datos.T.index)
            # 📌 Guardar la matriz de distancias en un archivo de texto
            ruta_archivo = "matriz_distancias.txt"
            df_distancias.to_csv(ruta_archivo, sep='\t', index=True)
            # # Convertir a matriz cuadrada (opcional)
            # distancia_cuadrada = squareform(distancia_euclidiana)
            
        elif m_dis == 2:
            nombre_plot= "Manhattan"
            distancia = pdist(datos.T, metric='cityblock')  # Transponer porque queremos calcular entre columnas
            print("DISTANCIA")
            print(distancia)
        elif m_dis == 3:
            nombre_plot= "Coseno"
            distancia = pdist(datos.T, metric='cosine')  # Transponer para calcular entre columnas
            print("DISTANCIA")
            print(distancia)
        elif m_dis == 4:
            nombre_plot= "Chebyshev"
            distancia = pdist(datos.T, metric='chebyshev')  # Transponer para calcular entre columnas
            print("DISTANCIA")
            print(distancia)
        elif m_dis == 5:
            nombre_plot= "Pearson"
            correlacion_pearson = datos.corr(method='pearson') # Calcular la matriz de correlación de Pearson
            distancia_pearson = 1 - correlacion_pearson # Convertir correlación en distancia: d = 1 - r
            distancia = squareform(distancia_pearson)  # Convertir la matriz de distancias en formato condensado
            print("DISTANCIA")
            print(distancia)
        elif m_dis == 6:  
            nombre_plot= "Spearman"
            correlacion_spearman = datos.corr(method='spearman')  # Calcular la matriz de correlación de Spearman
            distancia_spearman = 1 - correlacion_spearman # Convertir correlación en distancia: d = 1 - r
            distancia = squareform(distancia_spearman) # Convertir la matriz de distancias en formato condensado
            print("DISTANCIA")
            print(distancia)
        elif m_dis == 7:
            nombre_plot= "Jaccard"
            distancia = pdist(datos.T, metric='jaccard')  # Transponer para calcular entre columnas
            print("DISTANCIA")
            print(distancia)      
            
        while True:  # Submenú para el método de enlace
            if m_dis == 1 or m_dis == 2:
                print("\nCual método de enlace entre Clústeres deseas utilizar:")
                print("1. Ward")
                print("2. Single Linkage")
                print("3. Complete Linkage")
                print("4. Average Linkage")
                print("5. Volver")  # Volver al menú de distancia
                print("6. Salir")
                m_enlace = int(input("Opción: "))
            elif m_dis == 3 or m_dis == 4 or m_dis == 5 or m_dis == 6 or m_dis == 7:
                print("\nCual método de enlace entre Clústeres deseas utilizar:")
                print("1. Single Linkage")
                print("2. Complete Linkage")
                print("3. Average Linkage")
                print("4. Volver")  # Volver al menú de distancia
                print("5. Salir")
                m_enlace = int(input("Opción: ")) + 1  # SUMO 1 PARA QUEN SE VAYA EN EL IF CORRECTO 
        
            if m_enlace == 1:
                nombre_enlace = "ward"
                dendrograma = sch.linkage(distancia, method='ward')
                print("Dendrograma")
                print(dendrograma)
               
                #  Convertirmos la matriz de linkage a un DataFrame
                df_linkage = pd.DataFrame(dendrograma, columns=["Cluster 1", "Cluster 2", "Distancia", "Elementos fusionados"])              
                # Guardamos la matriz de linkage en un archivo de texto
                ruta_archivo = "matriz_linkage.txt"
                df_linkage.to_csv(ruta_archivo, sep='\t', index=False)

            elif m_enlace == 2:
                print("Entro en single")
                nombre_enlace = "single"
                dendrograma = sch.linkage(distancia, method='single') 
                print("Dendrograma")
                print(dendrograma)
            elif m_enlace == 3:
                nombre_enlace = "complete"
                dendrograma = sch.linkage(distancia, method='complete') 
                print("Dendrograma")
                print(dendrograma)
            elif m_enlace == 4:
                nombre_enlace = "average"
                dendrograma = sch.linkage(distancia, method='average')
                print("Dendrograma")
                print(dendrograma)
            elif m_enlace == 5:
                break
            else:
                return


            # Números sin paréntesis (Ejemplo: 361) Representan una muestra individual dentro del dataset.
            # Números con paréntesis (Ejemplo: (36))  Indican un clúster que ya agrupa múltiples muestras, 36 en este caso
            # por ejemplo al sumar todo el eje x de Limpio.csv deberia de dar 731 ,(361) vale solo 1
            # Usar truncate_mode='level' para agrupar clusters pequeños.
            # El eje Y muestra la "distancia" o disimilitud entre los clústeres.
            # El eje X en tu dendrograma representa las muestras o clusters fusionado
            # Los números indican cuántas muestras se han fusionado en cada cluster. ejemplo= (17) significa que en ese punto se han fusionado 17 muestras en un solo cluster.
            #  La altura en el eje Y sigue representando la distancia entre los clusters.
            plt.figure(figsize=(16, 8))
            sch.dendrogram(dendrograma, truncate_mode='level', p=4)  # Solo muestra los 10 niveles superiores
            #dendrogram(dendrograma)
            plt.title('Dendrograma usando {nombre_enlace} linkage con distancia de {nombre_plot} (HCA)'.format(
                nombre_enlace=nombre_enlace,     
                nombre_plot=nombre_plot
                ))
            plt.xlabel('Muestras')
            plt.ylabel('Distancia')
            #   COMENTAR TODO ESTO PARA NO VER EN EL NAVEGADOR
            # 🔹 Guardar la imagen
            ruta_imagen = "dendrograma.png"
            plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
            
            # 🔹 Abrir la imagen en el navegador automáticamente
            webbrowser.open(ruta_imagen)
            
            # También mostrarlo en pantalla si deseas
            plt.show()
               


# ORDENAMOS LAS MUESTRAS PARA QUE EN CADA ARCHIVO LAS MUESTRAS TENGAN EL MISMO ORDEN, A A A B , A B A A RESULTADO A A A B 
def ordenar_muestras(lista_df):
    print("ENTRO EN ORDENAR MUESTRAS")

    tipos_orden = lista_df[0].iloc[0, 1:].tolist()

    for i in range(len(lista_df)):
        df = lista_df[i]
        fila_tipos = 0

        col_0 = df.columns[0]
        tipos_actuales = df.iloc[fila_tipos, 1:].copy() 
        nuevas_cols = [col_0]

        for tipo in tipos_orden:
            coincidencias = tipos_actuales[tipos_actuales == tipo]

            if len(coincidencias) == 0:
                print(f"Tipo '{tipo}' no se encontro en el DataFrame {i}.")
                continue  # saltamos al siguiente tipo

            idx = coincidencias.index[0]
            nuevas_cols.append(idx)
            tipos_actuales.at[idx] = "USADO"

        lista_df[i] = df[nuevas_cols]

    return lista_df



# AHORA QUE TENGO ORDENADO LA MUESTRA NECESITO ENCONTRAR SI TIENE EJE COMUN O NO PARA PODER DETERMINAR CUAL SERA EL PASO DE LA INTERPOLACION
# def val_eje_comun():

    




# # TIENE QUE RECIBIR EL RAMAN SHIFT
# def cal_pasos():
    




def low_level(df):
    df = [df_i.copy() for df_i in df]

    for i in range(len(df)):
        df[i].columns = df[i].iloc[0]                  # Promover fila 0 como encabezado
        df[i] = df[i].iloc[1:].reset_index(drop=True)  # Eliminar la fila de encabezados original

    # Mantener la primera columna solo del primer archivo (Raman Shift)
    df_base = df[0]
    df_restantes = [df_i.iloc[:, 1:] for df_i in df[1:]]  # Quitar Raman Shift de los demás

    # Concatenar todo
    df_fusionado = pd.concat([df_base] + df_restantes, axis=1)

    # Guardar sin índice
    df_fusionado.to_csv("lowfusion.csv", index=False)

    print("✅ Fusionado correctamente con nombres reales en la cabecera.")
    print(df_fusionado)
    # NO RETORNA NADA POR QUE LO QUE HAGO ES GENERAR EL ARCHIVO CONCATENADO Y EN LA FUSION MAIN LLAMAR AL ARCHIVO DE VUELTA, NO ME GUSTA PERO ASI FUNCIONO



def mid_level(df,archivo_nombre):
    
    df_base = df[0]  # Primer DataFrame de la lista
    raman_shift = pd.to_numeric(df_base.iloc[1:, 0], errors="coerce").reset_index(drop=True) # CORROBORAMOS QUE SEA NUMERICO
    tipos = df_base.iloc[0, 1:] # EXTRAEMOS LA FILA DE LOS TIPOS
    
    #print("Valores únicos en tipos: ", tipos.unique())
    
    types = tipos.tolist() # COMVERTIMOS A LISTA 
    
    tipos_nombres = list(set(types))  # Convierte a conjunto y luego a lista para evitar problemas
    
    cmap = plt.cm.Spectral
    colores = [cmap(i) for i in np.linspace(0, 1, len(tipos_nombres))]
    asignacion_colores = {tipo: mcolors.to_hex(colores[i]) for i, tipo in enumerate(tipos_nombres)}
    
    df_numerico_lista = [df_i.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce") for df_i in df] # convertir todos los DataFrames en `df` a formato numérico
    
    #Revisamos si hay valores NaN en algún DataFrame después de la conversión
    for i, df_num in enumerate(df_numerico_lista):
        if df_num.isna().sum().sum() > 0:
            print(f"Advertencia: Se encontraron valores NaN en el DataFrame {i} después de la conversión.")
    
    
    pca(df_numerico_lista[0], raman_shift, archivo_nombre, asignacion_colores, types) # Llamamos a PCA con el primer DataFrame convertido


