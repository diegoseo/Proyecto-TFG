#   ACA PONDRE TODO LO RELACIONADO A LAS FUNCIONES QUE RETORNARAN VALORES

#import main
import titulo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # PARA LA NORMALIZACION POR LA MEDIA 
from scipy.signal import savgol_filter # Para suavizado de Savitzky Golay
from scipy.ndimage import gaussian_filter # PARA EL FILTRO GAUSSIANO
     



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


def guardar_archivo(df):
    print("ACA GUARDAR TODOS LOS DATAFRAME")

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
                        print("primer plot")
                        plt.plot(raman_shift , datos[col], color=color_actual, alpha=0.5, linewidth = 0.1,label=col)   
                        #plt.xticks(np.arange(min(raman_shift[1:].astype(float)), max(raman_shift[1:].astype(float)), step=300))
                        break
                    else:
                        print("segundo plot")
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
    # elif derivada == 3:  # PARA QUE VAYA AL PLOT DE CORRECION DE LINEA BASE OPCION 9
    #     titulo_plot_correcion_base(nor_op,op_der)
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
    # elif derivada == 3:
    #     titulo_plot_correcion_base(nor_op,op_der)
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
    # elif derivada == 3:  
    #     titulo_plot_correcion_base(opcion,op_der)
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

def suavizado_saviztky_golay(dato_suavizar):  #acordarse que se puede suavizar por la media, area y directo

    ventana = int(input("INGRESE EL VALOR DE LA VENTANA: "))
    orden = int(input("INGRESE EL VALOR DEL ORDEN: "))
            
    dato = dato_suavizar.to_numpy() #PASAMOS LOS DATOS A NUMPY POR QUE SAVGOL_FILTER USA SOLO NUMPY COMO PARAMETRO (PIERDE LA CABECERA DE TIPOS AL HACER ESTO)
    suavizado = savgol_filter(dato, window_length=ventana, polyorder=orden)
    suavizado_pd = pd.DataFrame(suavizado) # PASAMOS SUAVIZADO A PANDAS Y GUARDAMOS EN SUAVIZADO_PD
    suavizado_pd.columns = dato_suavizar.columns # AGREGAMOS LA CABECERA DE TIPOS
        
  
    
    return suavizado_pd



# SUAVIZADO POR FILTRO GAUSIANO

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
    
    return suavizado_gaussiano_pd




# GENERA VALORES NAM POR LA FORMA EN LA QUE SE CALCULA EL MEDIA MOVIL
def suavizado_mediamovil(dato_suavizar):

    ventana = int(input("INGRESE EL VALOR DE LA VENTANA: "))
    normalizado = dato_suavizar

      
    suavizado_media_movil = pd.DataFrame()
    
    
    suavizado_media_movil = normalizado.rolling(window=ventana, center=True).mean() # mean() es para hallar el promedio
    
   
    return suavizado_media_movil
    
    
   # print(suavizado_media_movil)





def primera_derivada(datos):
            
    
    df_derivada = datos.apply(pd.to_numeric, errors='coerce') # PASAMOS A NUMERICO SI ES NECESARIO
    #print("xXXXXXXXxxXXXX")
    #print(df_derivada)
    
    # Calcular la primera derivada
    df_derivada_diff = df_derivada.diff()
    
    # Imprimir la primera derivada
    #print("Primera Derivada:")
    #print(df_derivada_diff)
    

    return df_derivada_diff
    #para la llamada del PCA



def segunda_derivada(datos):
    
    df_derivada = datos.apply(pd.to_numeric, errors='coerce') # PASAMOS A NUMERICO SI ES NECESARIO
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
    return df_derivada_diff
                        



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
            

