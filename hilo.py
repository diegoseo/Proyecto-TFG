# Importa QThread y Signal, y la función para cargar archivos.
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PySide6.QtCore import QThread, Signal
from manipulacion_archivos import cargar_archivo
from PySide6.QtWebEngineWidgets import QWebEngineView
import plotly.io as pio
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QMainWindow
)
from funciones import (
    corregir_base_lineal, corregir_shirley, normalizar_por_media, normalizar_por_area, suavizar_sg, suavizar_gaussiano, suavizar_media_movil, primera_derivada, segunda_derivada,
    pca, plot_pca_2d, plot_pca_3d, tsne, plot_tsne_2d, plot_tsne_3d, tsne_pca, generar_informe, calculo_hca, grafico_loading
)

#Define la clase de hilo, con una señal que envía una lista.
class HiloCargarArchivo(QThread):
    print("ENTRO 2")
    archivo_cargado = Signal(pd.DataFrame) # emitirá un  DataFrame

    # Constructor: recibe una lista de rutas.
    def __init__(self, rutas_archivos, parent=None):
        super().__init__(parent)
        self.rutas_archivos = rutas_archivos  # <== ESTA LÍNEA ES FUNDAMENTAL

    #Método que se ejecuta al iniciar el hilo:
    #Carga cada archivo con cargar_archivo.
    #Si hay error, lo imprime.
    #Cuando termina, emite la señal con la lista de DataFrames.
    def run(self):
        for ruta in self.rutas_archivos:
            try:
                df = cargar_archivo(ruta)
                self.archivo_cargado.emit(df)  # emitir de a uno
            except Exception as e:
                print(f"Error al cargar {ruta}: {e}")
                



class HiloGraficarEspectros(QThread):
    print("ENTRO EN HILO QUE GRAFICA")
    graficar_signal = Signal(object, object, object)  # Emitimos: datos, raman_shift, asignacion_colores

    def __init__(self, datos, raman_shift, asignacion_colores):
        super().__init__()
        self.datos = datos
        self.raman_shift = raman_shift
        self.asignacion_colores = asignacion_colores

    def run(self):
        self.graficar_signal.emit(self.datos, self.raman_shift, self.asignacion_colores)
        
        
        
class HiloMetodosTransformaciones(QThread):
    data_frame_resultado = Signal(object)
    
    def __init__(self, df_original, opciones):
        super().__init__()
        self.df = df_original.copy()
        self.opciones = opciones
        
    # CORRECIONES -> NORMALIZACION -> SUAVIZADO -> DERIVADAS : NO IMPORTA EL ORDEN EN QUE SE ESTE MARCANDO EL BOX, SIEMPRE HARA EN ESTE ORDEN
    def run(self):
        df = self.df
        
        ############## ARREGLAMOS EL DF PARA PARA ENVIAR A LAS FUNCIONES################
        # Separamos la primera fila
        cabecera = df.iloc[0]  # Primera fila
        df_sin_cabecera = df[1:].copy()  # Quitamos la primera fila

        # Eliminamos la primera columna (Raman shift)
        raman_shift = df_sin_cabecera.iloc[:, 0].astype(float)  # Guardamos Raman shift
        df_solo_intensidad = df_sin_cabecera.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')  # Aseguramos que sean números

        
        #  SE USA SOLO IF Y NO ELIF POR QUE VA A IMPEDIR QUE SE APLIQUEN MULTIPLES TRANSFORMACIONES SI SE SELECCIONA MAS DE UNA CASILLA
        if self.opciones.get("correccion_lineal"):
            df = corregir_base_lineal(df_solo_intensidad,raman_shift)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
            print("CORRECCION BASE LINEAL")
            print(df)
        if self.opciones.get("correccion_shirley"):
            df = corregir_shirley(df_solo_intensidad,raman_shift)  
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
            print("CORRECCION SHIRLEY")
            print(df)  
        if self.opciones.get("normalizar_media", {}).get("activar"):
            metodo = self.opciones["normalizar_media"]["metodo"]
            df = normalizar_por_media(df_solo_intensidad, metodo) 
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
            print("NORMALIZACION POR MEDIA")
            print(df)
        if self.opciones.get("normalizar_area"):
            df = normalizar_por_area(df_solo_intensidad,raman_shift)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
            print("Normalizacion por Area")
            print(df)
        if self.opciones.get("suavizar_sg"):
            ventana = self.opciones["suavizar_sg"]["ventana"]
            orden = self.opciones["suavizar_sg"]["orden"]
            df = suavizar_sg(df_solo_intensidad, ventana, orden)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
            print("Suavizado Savizky Golay")
            print(df)
        if self.opciones.get("suavizar_fg"):
            sigma = self.opciones["suavizar_fg"]["sigma"]
            df = suavizar_gaussiano(df_solo_intensidad, sigma)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
            print("Suavizado FILTRO GAUSIANO")
            print(df)
        if self.opciones.get("suavizar_mm"):
            ventana = self.opciones["suavizar_mm"]["ventana"]
            df = suavizar_media_movil(df, ventana)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
            print("Suavizado MEDIA MOVIL")
            print(df)
        if self.opciones.get("derivada_1"):
            df = primera_derivada(df_solo_intensidad,raman_shift)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
            print("PRIMERA DERIVADA")
            print(df)
        if self.opciones.get("derivada_2"):
            df = segunda_derivada(df_solo_intensidad,raman_shift)
            df_solo_intensidad = df # ACTUALIZAMOS LOS CAMBIOS DEL DF
            print("SEGUNDA DERIVADA")
            print(df)

        ########### YA QUE NUESTRO DF ES SOLO DE INTENSIDADES PURAS TENEMOS QUE VOLVER A CONCATENAR LA COLUMNA RAMAN_SHIFT Y LA FILA DE LOS TIPOS

        # Agregamos nuevamente el Raman Shift como primera columna
        df_final = pd.concat([raman_shift.reset_index(drop=True), df_solo_intensidad.reset_index(drop=True)], axis=1)

        # Restauramos la fila de tipos como primera fila simulada
        df_final.columns = ['Raman Shift'] + list(cabecera.iloc[1:])  # cabecera original sin la primera celda
        df_final = pd.concat([pd.DataFrame([df_final.columns], columns=df_final.columns), df_final], ignore_index=True)

        # No modificamos la primera fila, solo renombramos la cabecera (fila no enumerada)
        n_cols = df_final.shape[1]
        df_final.columns = [0] + list(range(1, n_cols))

        self.data_frame_resultado.emit(df_final)
        
        ############ VERIFICAR LOS RESULTADOS QUE RETORNAN ESTOS METODOS #################
        
    
    
        
class HiloMetodosReduccion(QThread):
    signal_figura_pca_2d = Signal(object)
    signal_figura_pca_3d = Signal(object)
    signal_figura_tsne_2d =Signal(object)
    signal_figura_tsne_3d =Signal(object)
    signal_figura_loading =Signal(object)
    
    def __init__(self, df_original, opciones,componentes,intervalo,nombre_informe,componentes_seleccionados,cp_pca,cp_tsne,componentes_selec_loading,cant_componentes_loading):
        super().__init__()
        self.df = df_original.copy()
        self.opciones = opciones
        self.componentes = componentes
        self.intervalo = intervalo
        self.nombre_informe = nombre_informe
        self.cp_pca = cp_pca
        self.cp_tsne = cp_tsne
        self.componentes_seleccionados = componentes_seleccionados
        self.componentes_selec_loading = componentes_selec_loading
        self.cant_componentes_loading = cant_componentes_loading
        self.tipos = self.df.iloc[0, 1:]    # Fila 0 tiene los tipos (collagen, DNA, etc.)
        print("tipos:")
        print(self.tipos)
        tipos_nombres = self.tipos.unique()
        cmap = plt.cm.Spectral
        colores = [cmap(i) for i in np.linspace(0, 1, len(tipos_nombres))]
        self.asignacion_colores = {tipo: mcolors.to_hex(colores[i]) for i, tipo in enumerate(tipos_nombres)}
        self.raman_shift = self.df.iloc[1:, 0].reset_index(drop=True)
        self.varianza_porcentaje = None
        # Inicializamos los resultados como None por defecto para que no cree conflicto a generar el informe
        self.pca_resultado = None
        self.tsne_resultado = None
        self.tsne_pca_resultado = None
        
        print("ENTRO EN HILOMETODOSREDUCCION")
        print(self.df)
        print("Compornentes:",componentes)
        print("Intervalo:",intervalo)
        print("Nombre Informe:",nombre_informe)
            
    def run(self):
        #  SE USA SOLO IF Y NO ELIF POR QUE VA A IMPEDIR QUE SE APLIQUEN MULTIPLES TRANSFORMACIONES SI SE SELECCIONA MAS DE UNA CASILLA
        #  El orden en que se ejecutan las transformaciones no depende del orden de selección del usuario, sino del orden en el código.
        #  VALIDAR QUE SI YA SE DIO CHECK EN 2D QUE NO PERMITA DAR CHECK EN 3D , VER LA FORMA DE QUE SE PUEDA HACER TSNE Y SOBRE EL RESULTADO PCA
        #   t-SNE(PCA(X)) SI SE PUEDE PERO PCA(t-SNE(X)) NO SE PUEDE, TECNICAMENTE SI PERO NO TIENE SENTIDO
        if self.opciones.get("PCA"):
            # ACA LLAMAMOS A LA FUNCION PCA
            self.df = self.df.iloc[1:,1:].T
            self.pca_resultado,self.varianza_porcentaje = pca(self.df,self.componentes) #  pca(dato, raman_shift,archivo_nombre,asignacion_colores,types, datafusion)
            #print(self.pca_resultado)
            #print("PORCENTAJE VARIANZA")
            #print(self.varianza_porcentaje)
            if self.opciones.get("GRAFICO 2D"):
                pc_x, pc_y = self.componentes_seleccionados
                #print("PCX:",pc_x,"PCY:",pc_y)
                print("ENTRO GRAFICO 2D")
                fig = plot_pca_2d(self.pca_resultado, self.varianza_porcentaje,self.asignacion_colores,self.tipos,pc_x,pc_y,self.intervalo)
                self.signal_figura_pca_2d.emit(fig) 
                # Mostrar en ventana embebida
                #self.ventana_pca = VentanaGraficoPCA2D(fig)
                #self.ventana_pca.show()
            
            if self.opciones.get("GRAFICO 3D"):
                pc_x, pc_y,pc_z = self.componentes_seleccionados
                fig = plot_pca_3d(self.pca_resultado, self.varianza_porcentaje,self.asignacion_colores,self.tipos,pc_x,pc_y,pc_z,self.intervalo)
                self.signal_figura_pca_3d.emit(fig)
                #print("PCX:",pc_x,"PCY:",pc_y,"PCZ:",pc_z)
                print("ENTRO EN GRAFICO 3D")
            
            
        if self.opciones.get("TSNE"): # SOLO ACEPTA COMPONENTES PRINCIPALES DE 2 Y 3, INVESTIGAR POR QUE NO ACEPTA MAS
            # perplexity (float) = controla el número aproximado de vecinos más cercanos que t-SNE considera al posicionar los puntos.
            # learning_rate (float) = tasa de aprendizaje para el descenso de gradiente. Afecta qué tan rápido se ajustan las posiciones de los puntos.
            # random_state (int) = fija la semilla aleatoria. Útil para que el resultado sea reproducible.
            print("ENTRO TSNE")
            df_intensidades = self.df.iloc[1:, 1:].T  # HACEMOS LA TRANSPUESTA POR QUE TSNE TIENE QUE TENER LAS MUESTRA EN CADA FILA
            #df_intensidades = df_intensidades.apply(pd.to_numeric, errors='coerce') # AL HACER LA COVERSION GENERA VALORES NAN
            #df_intensidades = df_intensidades.dropna() # ELIMINAMOS LOS VALORES NAN
            self.tsne_resultado = tsne(df_intensidades,int(self.componentes))
            
            if self.opciones.get("GRAFICO 2D"):
                fig = plot_tsne_2d(self.tsne_resultado, self.tipos, self.asignacion_colores, int(self.intervalo)/100)
                self.signal_figura_tsne_2d.emit(fig)
            if self.opciones.get("GRAFICO 3D"):
                fig = plot_tsne_3d(self.tsne_resultado, self.tipos, self.asignacion_colores, int(self.intervalo)/100)
                self.signal_figura_tsne_3d.emit(fig)
                print("Prueba")
            
        if self.opciones.get("t-SNE(PCA(X))"):
            print("ENTRO EN OTROS")
            
            df_intensidades = self.df.iloc[1:, 1:].T  # HACEMOS LA TRANSPUESTA POR QUE TSNE TIENE QUE TENER LAS MUESTRA EN CADA FILA
            #df_intensidades = df_intensidades.apply(pd.to_numeric, errors='coerce') # AL HACER LA COVERSION GENERA VALORES NAN
            #df_intensidades = df_intensidades.dropna() # ELIMINAMOS LOS VALORES NAN
            self.tsne_pca_resultado = tsne_pca(df_intensidades, self.cp_pca, self.cp_tsne)
            
            if self.cp_tsne == 2:
                fig = plot_tsne_2d(self.tsne_pca_resultado, self.tipos, self.asignacion_colores, int(self.intervalo)/100)
                self.signal_figura_tsne_2d.emit(fig)
            if self.cp_tsne == 3:
                fig = plot_tsne_3d(self.tsne_pca_resultado, self.tipos, self.asignacion_colores, int(self.intervalo)/100)
                self.signal_figura_tsne_3d.emit(fig)
            
        if self.opciones.get("Grafico Loading (PCA)"): # Hay que pasarle los resultados de los PCA, Raman_shift y componentes seleccionados
            print("ENTRO EN GRAFICO DE LOADING")
            # OBTENEMOS EL RESULTADO DEL PCA
            self.df = self.df.iloc[1:,1:].T
            #self.pca_resultado,self.varianza_porcentaje = pca(self.df,self.cant_componentes_loading) #  pca(dato, raman_shift,archivo_nombre,asignacion_colores,types, datafusion)
            #print(self.pca_resultado)
            fig = grafico_loading(self.df,self.raman_shift,self.componentes_selec_loading)
            self.signal_figura_loading.emit(fig)

        if self.opciones.get("GENERAR INFORME"):
            print("ENTRO EN GENERAR INFORME")
            generar_informe(self.nombre_informe ,self.opciones,self.componentes,self.intervalo,self.cp_pca,self.cp_tsne,self.componentes_seleccionados,self.asignacion_colores,self.pca_resultado,self.varianza_porcentaje,self.tsne_resultado,self.tsne_pca_resultado)



class HiloHca(QThread):
    signal_figura_hca = Signal(object)

    def __init__(self, df_original, opciones):
        super().__init__()
        self.df = df_original.copy()
        self.opciones = opciones  # Guardamos las opciones recibidas
        print("RAMAN SHIFT DENTRO DE HILOHCA")
        self.raman_shift = self.df.iloc[1:, 0].reset_index(drop=True)
        print(self.raman_shift)
    def run(self):
        print("HILO INICIADO CON OPCIONES:")
        print(self.opciones)
        
        fig = calculo_hca(self.df, self.raman_shift,self.opciones)

        # Emitimos la señal cuando termine
        self.signal_figura_hca.emit(fig)

        


########### REVISAR HCA #############






# Esta clase se encarga de recibir el fig de Plotly y mostrarlo como una ventana GUI
# class VentanaGraficoPCA2D(QMainWindow):
#     def __init__(self, fig_plotly, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Gráfico PCA 2D")

#         # Convertir fig Plotly a HTML
#         html = pio.to_html(fig_plotly, full_html=False)

#         # Crear contenedor y vista web
#         central = QWidget()
#         layout = QVBoxLayout(central)
#         webview = QWebEngineView()
#         webview.setHtml(html)

#         layout.addWidget(webview)
#         self.setCentralWidget(central)






# def run(self):
#     pca_df = None

#     if self.opciones.get("PCA"):
#         print("ENTRO PCA")
#         pca_df = self.realizar_pca(self.df)
#         self.pca_resultado_df = pca_df  # guardamos el resultado

#     if self.opciones.get("TSNE"):
#         print("ENTRO TSNE")
#         tsne_df = self.realizar_tsne(self.df)
#         # si querés usar el tsne en algo posterior, también guardás: self.tsne_resultado_df = tsne_df

#     if self.opciones.get("GRAFICO 2D"):
#         print("ENTRO GRAFICO 2D")
#         if pca_df is not None:
#             self.graficar_2d(pca_df)
#         else:
#             QMessageBox.warning(None, "Error", "Primero debe ejecutarse PCA para poder graficar en 2D.")
