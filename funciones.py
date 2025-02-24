#   ACA PONDRE TODO LO RELACIONADO A LAS FUNCIONES QUE RETORNARAN VALORES

#import main
import titulo
import matplotlib.pyplot as plt
import pandas as pd



     



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




def datos_sin_normalizar(df):
    
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
    # print("DF22222")
    # print(df2)
    return df2


def mostrar_espectros(datos,raman_shift,asignacion_colores,metodo,nor_op,op_der,derivada): # LA VARIABLE DERIVADA ES  SOLO PARA SABER SI ES LA 1RA O LA 2DA

        
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
        titulo.titulo_plot_mostrar(metodo,nor_op)

    # if derivada == 1:
    #     titulo_plot_primera_derivada(nor_op,op_der)
    # elif derivada == 2:
    #     titulo_plot_segunda_derivada(nor_op,op_der)
    # elif derivada == 3:  # PARA QUE VAYA AL PLOT DE CORRECION DE LINEA BASE OPCION 9
    #     titulo_plot_correcion_base(nor_op,op_der)
    # elif derivada == 4:
    #     titulo_plot_correcion_shirley(nor_op,op_der)


