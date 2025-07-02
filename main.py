import sys,os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import tempfile
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem, QInputDialog, QLabel, QDialog, QLineEdit, QCheckBox, QHBoxLayout, QGroupBox, QComboBox,
    QSpinBox
)
from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
from hilo import HiloCargarArchivo , HiloGraficarEspectros, HiloMetodosTransformaciones, HiloMetodosReduccion, HiloHca # CLASE PERSONALIZADA
from funciones import columna_con_menor_filas
from graficado import GraficarEspectros, GraficarEspectrosAcotados, GraficarEspectrosTipos, GraficarEspectrosAcotadoTipos
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # PARA EL HCA

# #from funciones import App  # App manejará la lógica de carga
# pila = []
# pila_df = [] # SE UTILIZARA PARA ALMACENAR LOS DF FINAL

class MenuPrincipal(QWidget): # Clase principal que representa la ventana con el menú principal.
    # Inicializa la ventana principal y le da título y tamaño.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Menú Principal - Análisis de Espectros")
        self.resize(400, 600)
        #Crea un layout vertical y lo alinea arriba.
        # layout = QVBoxLayout()
        # layout.setAlignment(Qt.AlignTop)
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)
        self.dataframes = [] # lista de df cargados
        self.nombres_archivos = []  # lista de nombres de Los archivo
        self.df_final = None  # Inicializamos el df que usaremos siempre
        # OPCIONES MENU PRINCIPAL
        opciones = [
            "1. CARGAR ARCHIVO",
            "2. VER DATAFRAME",
            "3. MOSTRAR ESPECTROS",
            "4. NORMALIZAR POR MEDIA",
            "5. NORMALIZAR POR AREA",
            "6. SUAVIZADO POR SAVIZTKY-GOLAY",
            "7. SUAVIZADO POR FILTRO GAUSIANO",
            "8. SUAVIZADO POR MEDIA MOVIL",
            "9. PRIMERA DERIVADA",
            "10. SEGUNDA DERIVADA",
            "11. CORRECCION BASE LINEAL",
            "12. CORRECION SHIRLEY",
            "13. REDUCIR DIMENSIONALIDAD",
            "14. GRAFICO HCA",
            "15. CAMBIAR ARCHIVO",
            "16. DATA FUSION",
            "17. Salir"
        ]

        # Crea el botón que, al hacer clic, ejecuta la función abrir_dialogo_archivos.
        self.boton_cargar = QPushButton("1. CARGAR ARCHIVO")# Botón para cargar archivo
        self.boton_cargar.clicked.connect(self.abrir_dialogo_archivos)
        self.layout.addWidget(self.boton_cargar)
        #Aplica el layout a la ventana.
        self.setLayout(self.layout)

        self.boton_verdf = QPushButton("2. VER DATAFRAME")# Botón para cargar archivo
        self.boton_verdf.clicked.connect(self.ver_dataframe)
        self.layout.addWidget(self.boton_verdf)
        self.setLayout(self.layout)       #Aplica el layout a la ventana.

        self.boton_ver_espectro = QPushButton("3. MOSTRAR ESPECTROS")# Botón para cargar archivo
        self.boton_ver_espectro.clicked.connect(self.ver_espectros)
        self.layout.addWidget(self.boton_ver_espectro)
        self.setLayout(self.layout)


        ####################
        #PENSAR EN UNA MANERA DE REDUCIR TODAS LAS OPCIONES DE NORMALIZACION, DERIVADAS,CORRECIONES Y SUAVIZADOS SIN CREAR TANTOS SUBMENUS Y DESPUES DE TODOO ESO RECIEN LLAMAR A ver_espectro para despegar el submenu
        # TAMBIEN TIENE QUE TENER LA POSIBILIDAD DE VER EL DATAFRAME YA TODOO MODIFICADO
        ##################


        self.boton_ver_espectro = QPushButton("4. ARREGLAR DATOS")# Botón para cargar archivo
        self.boton_ver_espectro.clicked.connect(self.arreglar_datos)
        self.layout.addWidget(self.boton_ver_espectro)
        self.setLayout(self.layout)


        self.boton_ver_espectro = QPushButton("5. METODO DE REDUCCION DE DIMENSIONALIDAD")# Botón para cargar archivo
        self.boton_ver_espectro.clicked.connect(self.abrir_dialogo_dimensionalidad)
        self.layout.addWidget(self.boton_ver_espectro)
        self.setLayout(self.layout)

        # self.boton_ver_espectro = QPushButton("6. GRAFICO DE LOADING (creo que movere dentro de la opcion 5)")# Botón para cargar archivo
        # self.boton_ver_espectro.clicked.connect(self.abrir_dialogo_loading)
        # self.layout.addWidget(self.boton_ver_espectro)
        # self.setLayout(self.layout)

        self.boton_ver_espectro = QPushButton("6. Análisis de Conglomerados Jerárquico (HCA)")# Botón para cargar archivo
        self.boton_ver_espectro.clicked.connect(self.abrir_dialogo_hca)
        self.layout.addWidget(self.boton_ver_espectro)
        self.setLayout(self.layout)

        self.boton_ver_espectro = QPushButton("7. DATA FUSION")# Botón para cargar archivo
        self.boton_ver_espectro.clicked.connect(self.abrir_dialogo_datafusion)
        self.layout.addWidget(self.boton_ver_espectro)
        self.setLayout(self.layout)

        # self.boton_ver_espectro = QPushButton("3. MOSTRAR ESPECTROS")# Botón para cargar archivo
        # self.boton_ver_espectro.clicked.connect(self.sub_menu)
        # self.layout.addWidget(self.boton_ver_espectro)
        # self.setLayout(self.layout)

        # self.submenu_espectros = QGroupBox("Opciones de espectros")
        # self.submenu_layout = QVBoxLayout()
        # self.submenu_espectros.setLayout(self.submenu_layout)
        # self.submenu_espectros.setVisible(False)  # ⬅️ al principio está oculto

    def abrir_dialogo_dimensionalidad(self):
        self.ventana_opciones_dim = VentanaReduccionDim(self.dataframes, self.nombres_archivos,self)
        self.ventana_opciones_dim.show()

    def arreglar_datos(self):
        self.ventana_prueba = VentanaTransformaciones(self.dataframes, self.nombres_archivos,self)
        self.ventana_prueba.show()

    # Abre un diálogo para seleccionar uno o más archivos CSV
    def abrir_dialogo_archivos(self):
        print("ENTRO 1")
        rutas, _ = QFileDialog.getOpenFileNames(self, "Seleccionar archivos CSV", "", "CSV Files (*.csv)")
        # Si se seleccionaron rutas, se lanza un hilo con HiloCargarArchivo, se conecta la señal a procesar_archivos y se inicia el hilo.
        if rutas:
            self.nombres_archivos.extend(rutas)# ACA GUARDAMOS EL NOMBRE DE LOS ARCHIVOS
            self.hilo = HiloCargarArchivo(rutas)
            self.hilo.archivo_cargado.connect(self.procesar_archivos)
            self.hilo.start()
        else: # Si no se seleccionaron archivos, muestra advertencia.
            QMessageBox.warning(self, "Sin selección", "No se seleccionaron archivos.")

    def abrir_dialogo_hca(self):
        self.ventana_opciones_hca = VentanaHca(self.dataframes, self.nombres_archivos,self)
        self.ventana_opciones_hca.show()
        
    def abrir_dialogo_datafusion(self):
        self.ventana_opciones_datafusion = VentanaDataFusion(self.dataframes, self.nombres_archivos,self)
        self.ventana_opciones_datafusion.show()


    # def abrir_dialogo_loading(self):
    #     self.ventana_opciones_loading = VentanaLoading(self.dataframes, self.nombres_archivos,self)
    #     self.ventana_opciones_loading.show()

    # Esta función se ejecuta cuando termina el hilo. Guarda los DataFrames y muestra un mensaje de éxito.
    def procesar_archivos(self,df):
        self.df_original = df.copy()
        self.df = df
        self.df_final = df.copy()  # por defecto, este es el df final si no hay corrección
        self.dataframe = self.df_final # seria el puntero
        self.dataframes.append(df)  # podés guardar en lista si querés
        self.index_actual = len(self.dataframes) - 1
        print("ENTRO 4")
        print(df)

        col,fil = columna_con_menor_filas(df)

        print("Col=" ,col, "fil=",fil)

        if len(df) != fil:
            self.eliminar_filas = ArreglarDf(df.copy())  # le pasamos el df
            self.eliminar_filas.df_modificado.connect(self.recibir_df_modificado)
            self.eliminar_filas.show()

        else:
            print("no hay que arreglar nada, directo graficar los espectros")

        print("lista de dataframe")
        print(self.dataframes)

    def recibir_df_modificado(self, df_nuevo):
        self.df = df_nuevo
        self.df_final = df_nuevo
        self.dataframe = df_nuevo

        # Actualizamos el DataFrame corregido dentro de la lista
        if hasattr(self, "index_actual") and self.index_actual is not None:
            self.dataframes[self.index_actual] = df_nuevo

        print("DF corregido recibido en ventana principal")

    def ver_dataframe(self, df=None):
        if not self.dataframes:
            QMessageBox.warning(self, "Sin datos", "Todavía no se ha cargado ningún archivo.")
            return

        opciones = [os.path.basename(nombre) for nombre in self.nombres_archivos] # LISTA LOS NOMBRES DE LOS ARCHIVOS

        seleccion, ok = QInputDialog.getItem(
            self,
            "Seleccionar DataFrame",
            "Elegí un archivo para visualizar:",
            opciones,
            0,
            False
        )

        if ok and seleccion:
            index = opciones.index(seleccion)
            df_a_mostrar = self.dataframes[index]
            self.ventana_tabla = VerDf(df_a_mostrar)
            self.ventana_tabla.show()


    def ver_espectros(self, df=None):
        print("Df dentro de ver espectro 3")
        print(df)
        if isinstance(df, pd.DataFrame):
            # Si te pasan un DataFrame, lo usamos directamente
            self.df_completo = df.copy()
            self.df_original = df.copy()
            self.raman_shift = self.df_completo.iloc[1:, 0].reset_index(drop=True)
            tipos = self.df_completo.iloc[0, 1:]
            tipos_nombres = tipos.unique()
        else:
            if not self.dataframes:
                QMessageBox.warning(self, "Sin datos", "No hay DataFrames cargados para graficar.")
                return

            opciones = [os.path.basename(nombre) for nombre in self.nombres_archivos] # LISTA LOS NOMBRES DE LOS ARCHIVOS

            seleccion, ok = QInputDialog.getItem(
                self,
                "Seleccionar DataFrame",
                "Elegí un DataFrame para graficar:",
                opciones,
                0,
                False
            )
            if ok and seleccion:
                index = opciones.index(seleccion)
                self.df_completo = self.dataframes[index].copy()
                self.df_original = self.dataframes[index].copy()
                # Extraer Raman Shift y tipos ANTES de modificar df
                self.raman_shift = self.df_completo.iloc[1:, 0].reset_index(drop=True)
                tipos = self.df_completo.iloc[0, 1:]
                tipos_nombres = tipos.unique()

        cmap = plt.cm.Spectral
        colores = [cmap(i) for i in np.linspace(0, 1, len(tipos_nombres))]
        self.asignacion_colores = {tipo: mcolors.to_hex(colores[i]) for i, tipo in enumerate(tipos_nombres)}



        opciones_sub = [
            "1. Gráfico completo",
            "2. Gráfico acotado",
            "3. Gráfico por tipo",
            "4. Gráfico acotado por tipo",
            "5. Descargar .csv",
            "6. Descargar .csv acotado",
            "7. Descargar .csv por tipo",
            "8. Descargar .csv acotado por tipo"
        ]

        seleccion_sub, ok_sub = QInputDialog.getItem(
            self,
            "Opciones de gráfico",
            "Selecciona una opción:",
            opciones_sub,
            0,
            False
        )

        if ok_sub and seleccion_sub:
            self.procesar_opcion_grafico(seleccion_sub)



    # PARA EL SUB_MENU
    def procesar_opcion_grafico(self, opcion):
        print(f"Seleccionaste: {opcion}")

        if opcion.startswith("1"):
            # df_a_graficar debe incluir la fila 0 (tipos) y todas las columnas
            df_a_graficar = self.df_completo.reset_index(drop=True)

            # Iniciar hilo
            self.hilo_graficar = HiloGraficarEspectros(df_a_graficar, self.raman_shift, self.asignacion_colores)
            self.hilo_graficar.graficar_signal.connect(self.mostrar_grafico)
            self.hilo_graficar.start()

        elif opcion.startswith("2"):
            dialogo = DialogoRangoRaman()
            if dialogo.exec():
                self.min_val = dialogo.valor_min
                self.max_val = dialogo.valor_max

            # df_a_graficar debe incluir la fila 0 (tipos) y todas las columnas
            df_a_graficar = self.df_completo.reset_index(drop=True)

            # Iniciar hilo
            self.hilo_graficar = HiloGraficarEspectros(df_a_graficar, self.raman_shift, self.asignacion_colores)
            self.hilo_graficar.graficar_signal.connect(self.mostrar_grafico_acotado)
            self.hilo_graficar.start()

        elif opcion.startswith("3"):
            dialogo = DialogoRangoRamanTipo()
            if dialogo.exec():
                self.tipo_graficar = dialogo.tipo_graficar

            # df_a_graficar debe incluir la fila 0 (tipos) y todas las columnas
            df_a_graficar = self.df_completo.reset_index(drop=True)

            # Iniciar hilo
            self.hilo_graficar = HiloGraficarEspectros(df_a_graficar, self.raman_shift, self.asignacion_colores)
            self.hilo_graficar.graficar_signal.connect(self.mostrar_grafico_tipo)
            self.hilo_graficar.start()

        elif opcion.startswith("4"):
            dialogo = DialogoRangoRamanTipoAcotado()
            if dialogo.exec():
                self.tipo_graficar = dialogo.tipo_graficar
                self.min_val = dialogo.valor_min
                self.max_val = dialogo.valor_max

            # df_a_graficar debe incluir la fila 0 (tipos) y todas las columnas
            df_a_graficar = self.df_completo.reset_index(drop=True)

            # Iniciar hilo
            self.hilo_graficar = HiloGraficarEspectros(df_a_graficar, self.raman_shift, self.asignacion_colores)
            self.hilo_graficar.graficar_signal.connect(self.mostrar_grafico_tipo_acotado)
            self.hilo_graficar.start()
        elif opcion.startswith("5"):
            # df_a_graficar debe incluir la fila 0 (tipos) y todas las columnas
            #df_a_graficar = self.df_completo.reset_index(drop=True)
            self.arreglar_df = ArreglarDf(self.df_original)
            self.arreglar_df.gen_csv()
        elif opcion.startswith("6"):
            dialogo = DialogoRangoRaman()
            if dialogo.exec():
                self.min_val = dialogo.valor_min
                self.max_val = dialogo.valor_max

            tipos = self.df_completo.iloc[0, :]
            print(tipos)
            self.raman = self.df_completo.iloc[:, 0].reset_index(drop=True)
            print("SELF DF COMPLETO")
            print(self.df_completo)
            df_acotado = self.descargar_csv_acotado(self.df_completo,self.raman,self.min_val,self.max_val,self.df_final)
            print(df_acotado)
            self.arreglar_df = GenerarCsv(df_acotado)
            self.arreglar_df.generar_csv()
        elif opcion.startswith("7"):
            dialogo = DialogoRangoRamanTipo()
            if dialogo.exec():
                self.tipo_graficar = dialogo.tipo_graficar

            tipos = self.df_completo.iloc[0, :]
            print(tipos)
            self.raman = self.df_completo.iloc[:, 0].reset_index(drop=True)
            print("SELF DF COMPLETO")
            print(self.df_completo)
            df_acotado = self.descargar_csv_tipo(self.df_completo,self.raman,self.df_final,self.tipo_graficar)
            print(df_acotado)
            self.arreglar_df = GenerarCsv(df_acotado)
            self.arreglar_df.generar_csv()
        elif opcion.startswith("8"):
            dialogo = DialogoRangoRamanTipoAcotado() #ACA ES DONDE LLAMO A LA INTERFAZ PARA QUE EL USUARIO CARGUE LOS VALORES
            if dialogo.exec():
                # Y ACA ES DONDE ALMACENO LO QUE RETORNA DE LA INTERFAZ
                self.tipo_graficar = dialogo.tipo_graficar
                self.min_val = dialogo.valor_min
                self.max_val = dialogo.valor_max

            tipos = self.df_completo.iloc[0, :]
            print(tipos)
            self.raman = self.df_completo.iloc[:, 0].reset_index(drop=True)
            print("SELF DF COMPLETO")
            print(self.df_completo)
            df_acotado = self.descargar_csv_tipo_acotado(self.df_completo,self.raman,self.df_final,self.tipo_graficar,self.min_val,self.max_val)
            print(df_acotado)
            self.arreglar_df = GenerarCsv(df_acotado)
            self.arreglar_df.generar_csv()



    def mostrar_grafico(self, datos, raman_shift, asignacion_colores):
        self.grafico_pg = GraficarEspectros(datos, raman_shift, asignacion_colores)
        self.grafico_pg.show()

    def mostrar_grafico_acotado(self, datos, raman_shift, asignacion_colores):
        self.grafico_pg = GraficarEspectrosAcotados(datos, raman_shift, asignacion_colores, self.min_val, self.max_val)
        self.grafico_pg.show()

    def mostrar_grafico_tipo(self, datos, raman_shift, asignacion_colores):
        self.grafico_pg = GraficarEspectrosTipos(datos, raman_shift, asignacion_colores, self.tipo_graficar)
        self.grafico_pg.show()

    def mostrar_grafico_tipo_acotado(self, datos, raman_shift, asignacion_colores):
        self.grafico_pg = GraficarEspectrosAcotadoTipos(datos, raman_shift, asignacion_colores, self.tipo_graficar,self.min_val, self.max_val)
        self.grafico_pg.show()

    def descargar_csv_acotado(self,datos,raman,val_min,val_max,df_final):
        nueva_cabecera = datos.iloc[0]             # Fila 0 tiene los tipos (nombres deseados de columnas)
        datos = datos[1:]                          # Eliminar esa fila del DataFrame
        datos.columns = nueva_cabecera             # Asignar como cabecera
        datos.reset_index(drop=True, inplace=True) # Resetear el índice
        datos = datos.iloc[:,1:]
        print("DATOS")
        print(datos)

        df_aux = datos.to_numpy()
        cabecera_np = df_final.iloc[0, 1:].to_numpy()
        print("CABECERA_NP")
        print(cabecera_np)

        intensidades_np = df_aux[:, :]

        raman = raman[1:].to_numpy().astype(float)
        print("RAMAN")
        print(raman)
        intensidades = intensidades_np.astype(float)

        indices_acotados = (raman >= val_min) & (raman <= val_max)
        raman_acotado = raman[indices_acotados]
        intensidades_acotadas = intensidades[indices_acotados, :]

        df_acotado = pd.DataFrame(
            data=np.column_stack([raman_acotado, intensidades_acotadas]),
            columns=["Raman Shift"] + list(cabecera_np)
        )

        print("DF ACOTADO")
        print(df_acotado)

        return df_acotado


        # x_total = np.array(raman, dtype=float)  # Eje X completo
        # mascara = (x_total >= val_min) & (x_total <= val_max)
        # print("Longitud de datos:", len(datos))
        # print("Longitud de máscara:", len(mascara))
        # x_filtrado = x_total[mascara]



        # print("DF ACOTADOOO")
        # print(df_acotado)

        # self.df_filtrado = df_acotado
        # return self.df_filtrado
    def descargar_csv_tipo(self,datos,raman,df_final,tipo_graficar):
        nueva_cabecera = datos.iloc[0]             # Fila 0 tiene los tipos (nombres deseados de columnas)
        datos = datos[1:]                          # Eliminar esa fila del DataFrame
        datos.columns = nueva_cabecera             # Asignar como cabecera
        datos.reset_index(drop=True, inplace=True) # Resetear el índice
        datos = datos.iloc[:,1:]
        print("DATOS")
        print(datos)
        columnas_eliminar = [] # GUARDAMOS EN ESTA LISTA TODO LO QUE SE VAS A ELIMINAR
        raman = raman[1:].to_numpy().astype(float)
        for col in datos.columns:

            if col != tipo_graficar: # SI ESA COLUMNA NO CONINCIDE CON EL TIPO DESEADO SE AGREGAR EN columnas_eliminar
                columnas_eliminar.append(col)


        datos_filtrados = datos.drop(columns=columnas_eliminar) # CREAMOS UN DATAFRAME ELIMINANDO TODO LO QUE ESTE DENTRO DE columnas_eliminar

        datos_filtrados.insert(0, "raman_shift",raman)  # Insertamos en la primera posición los valores de raman_shift
        #print("Datos filtrados con 'raman_shift' agregado:")
        #print(datos_filtrados)

        return datos_filtrados


    def descargar_csv_tipo_acotado(self,datos,raman,df_final,tipo_graficar,min_val,max_val):
        nueva_cabecera = datos.iloc[0]             # Fila 0 tiene los tipos (nombres deseados de columnas)
        datos = datos[1:]                          # Eliminar esa fila del DataFrame
        datos.columns = nueva_cabecera             # Asignar como cabecera
        datos.reset_index(drop=True, inplace=True) # Resetear el índice
        datos = datos.iloc[:,1:]
        columnas_eliminar = [] # GUARDAMOS EN ESTA LISTA TODOO LO QUE SE VAS A ELIMINAR
        raman = raman[1:].to_numpy().astype(float)
        for col in datos.columns:

            if col != tipo_graficar: # SI ESA COLUMNA NO CONINCIDE CON EL TIPO DESEADO SE AGREGAR EN columnas_eliminar
                columnas_eliminar.append(col)


        datos_filtrados = datos.drop(columns=columnas_eliminar) # CREAMOS UN DATAFRAME ELIMINANDO TODOO LO QUE ESTE DENTRO DE columnas_eliminar

        datos_filtrados.insert(0, "raman_shift",raman)  # Insertamos en la primera posición los valores de raman_shift
        #print("Datos filtrados con 'raman_shift' agregado:")
        #print(datos_filtrados)
        datos_filtrados = datos_filtrados.astype(object)  # Convierte todo el DataFrame a tipo object
        df_aux = datos_filtrados.iloc[:,1:].to_numpy()
        #print("PRINT")
        #print(df_aux)
        datos_filtrados.iloc[0, 1:] = tipo_graficar
        cabecera_np = datos_filtrados.iloc[0, 1:].to_numpy()  # La primera fila contiene los encabezados
        #print("CABECERA_NP")
        #print(cabecera_np)
        intensidades_np = df_aux[:, :]
        #print("INTENSIDADES_NP")
        #print(intensidades_np)
        ###raman = raman.to_numpy().astype(float)  # Primera columna (Raman Shift)
        #print("RAMAN")
        #print(raman)
        intensidades = intensidades_np.astype(float)  # Columnas restantes (intensidades)
        # print("INTENSIDADES")
        # print(intensidades)

        indices_acotados = (raman >= min_val) & (raman <= max_val)
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

        return datos_acotado_tipo



    # Método auxiliar para futuras opciones del menú.
    def ejecutar_opcion(self, texto):
        if texto == "17. Salir":
            self.close()
        else:
            QMessageBox.information(self, "Opción seleccionada", f"Elegiste: {texto}")



class ArreglarDf(QWidget):
    df_modificado = Signal(object)  # emitirá el DataFrame corregido
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Arreglar DataFrame")
        self.resize(300, 150)
        self.df = df  # Guardamos el DataFrame original
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)

        #PARA LA OPCION 4 DE VOLVER AL ESTADO ANTERIOR
        self.pila = []
        self.pila.append(self.df.copy())

        self.col,self.fil = columna_con_menor_filas(df)

        self.tabla = QTableWidget()
        self.layout.addWidget(self.tabla)

        # Botón que solo se muestra si el DataFrame es inconsistente
        self.boton_fila = QPushButton("1. ELIMINAR TODAS LAS FILAS HASTA IGUALAR A LA MENOR")
        self.boton_fila.clicked.connect(self.del_filas)
        self.layout.addWidget(self.boton_fila)

        # Botón que solo se muestra si el DataFrame es inconsistente
        self.boton_col = QPushButton("2- ELIMINAR LA COLUMNA CON MENOR NUMERO DE FILAS")
        self.boton_col.clicked.connect(self.del_col)
        self.layout.addWidget(self.boton_col)

        # Botón que solo se muestra si el DataFrame es inconsistente
        self.boton_ver = QPushButton("3- VER DATAFRAME ACTUAL")
        self.boton_ver.clicked.connect(self.ver_df)
        self.layout.addWidget(self.boton_ver)

        # Botón que solo se muestra si el DataFrame es inconsistente
        self.boton_volver = QPushButton("4- VOLVER ESTADO ANTERIOR")
        self.boton_volver.clicked.connect(self.volver_estado)
        self.layout.addWidget(self.boton_volver)

        # Botón que solo se muestra si el DataFrame es inconsistente
        self.boton = QPushButton("5- GENERAR .CSV")
        self.boton.clicked.connect(self.gen_csv)
        self.layout.addWidget(self.boton)

        # Botón que solo se muestra si el DataFrame es inconsistente
        self.boton = QPushButton("6- SALIR")
        self.boton.clicked.connect(self.salir)
        self.layout.addWidget(self.boton)

    def del_filas(self):
        self.pila.append(self.df.copy())
        menor_cant_filas = self.df.dropna().shape[0] # Buscamos la columna con menor cantidad de intensidades
        # print("menor cantidad de filas:", menor_cant_filas)
        df_truncado = self.df.iloc[:menor_cant_filas] # Hacemos los cortes para igualar las columnas
        self.df = df_truncado
        #print("Tamaño de la pila:", len(pila))
        # print(df.shape)
        print(self.df)

    def del_col(self):
        self.pila.append(self.df.copy())
        col ,_ = columna_con_menor_filas(self.df) # EL _ ES POR QUE LA FUNCION RETORNA DOS VALORES PERO SOLO NECESITAMOS EL COL
        self.df.drop(columns=[col], inplace=True)
        print(self.df)


    def ver_df(self):
        self.ventana_tabla = VerDf(self.df)
        self.ventana_tabla.show()

    def volver_estado(self):
        if len(self.pila) > 1 :
            # Recuperar el último estado del DataFrame
            self.df = self.pila.pop()
            print("Se ha revertido al estado anterior.")
        else:
            print("No hay acciones para deshacer.")

    def gen_csv(self):
        nombre, ok = QInputDialog.getText(self, "Guardar CSV", "Nombre del archivo:") # ok es booleano , retorna True si da en aceptar o False caso contrario

        if ok and nombre:
            # Asegura que tenga extensión .csv
            if not nombre.endswith(".csv"):
                nombre += ".csv"
            try:
                self.df.to_csv(nombre, index=False, header=0)
                print(f"Archivo guardado como: {nombre}")
            except Exception as e:
                print(f"Error al guardar el archivo: {e}")
        else:
            print("Guardado cancelado por el usuario.")

    def salir(self): # AL SALIR EMITE EL DF NUEVO A PROCESAR ARCHIVO(SERIA COMO EL RETURN)
        self.df_modificado.emit(self.df)
        self.close()

class VerDf(QWidget): # si hago con hilos puedo hacer que se actualice el df sin tener que cerrar para visualizar el actualizado (mejoras para el futuro)
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Vista del DataFrame")
        self.resize(800, 800)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        tabla = QTableWidget()
        tabla.setRowCount(len(df))
        tabla.setColumnCount(len(df.columns))
        tabla.setHorizontalHeaderLabels(df.columns.astype(str))

        for i in range(len(df)):
            for j in range(len(df.columns)):
                valor = str(df.iat[i, j])
                tabla.setItem(i, j, QTableWidgetItem(valor))

        layout.addWidget(tabla)


class DialogoRangoRaman(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rango Raman Shift")
        self.setMinimumWidth(300)

        layout = QVBoxLayout()

        self.label_min = QLabel("Ingrese valor mínimo de Raman Shift:")
        self.input_min = QLineEdit()
        layout.addWidget(self.label_min)
        layout.addWidget(self.input_min)

        self.label_max = QLabel("Ingrese valor máximo de Raman Shift:")
        self.input_max = QLineEdit()
        layout.addWidget(self.label_max)
        layout.addWidget(self.input_max)

        self.boton_aceptar = QPushButton("Aceptar")
        self.boton_aceptar.clicked.connect(self.validar_y_enviar)
        layout.addWidget(self.boton_aceptar)

        self.setLayout(layout)

        self.valor_min = None
        self.valor_max = None

    def validar_y_enviar(self):
        try:
            self.valor_min = float(self.input_min.text())
            self.valor_max = float(self.input_max.text())

            if self.valor_min >= self.valor_max:
                raise ValueError("El mínimo debe ser menor al máximo.")

            self.accept()  # Cierra el diálogo correctamente
        except ValueError as e:
            QMessageBox.warning(self, "Error", f"Entrada inválida: {e}")




class DialogoRangoRamanTipo(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tipos para Graficar")
        self.setMinimumWidth(300)

        layout = QVBoxLayout()

        self.label_min = QLabel("Ingrese el tipo que desea graficar:")
        self.input_min = QLineEdit()
        layout.addWidget(self.label_min)
        layout.addWidget(self.input_min)

        self.boton_aceptar = QPushButton("Aceptar")
        self.boton_aceptar.clicked.connect(self.validar_y_enviar)
        layout.addWidget(self.boton_aceptar)

        self.setLayout(layout)

    def validar_y_enviar(self):
        self.tipo_graficar = self.input_min.text().strip()
        self.accept()



class DialogoRangoRamanTipoAcotado(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tipos para Graficar")
        self.setMinimumWidth(300)

        layout = QVBoxLayout()

        self.label_min = QLabel("Ingrese el tipo que desea graficar:")
        self.input_tipo = QLineEdit()
        layout.addWidget(self.label_min)
        layout.addWidget(self.input_tipo)

        self.label_min = QLabel("Ingrese valor mínimo de Raman Shift:")
        self.input_min = QLineEdit()
        layout.addWidget(self.label_min)
        layout.addWidget(self.input_min)

        self.label_max = QLabel("Ingrese valor máximo de Raman Shift:")
        self.input_max = QLineEdit()
        layout.addWidget(self.label_max)
        layout.addWidget(self.input_max)

        self.boton_aceptar = QPushButton("Aceptar")
        self.boton_aceptar.clicked.connect(self.validar_y_enviar)
        layout.addWidget(self.boton_aceptar)

        self.setLayout(layout)

        self.valor_min = None
        self.valor_max = None

    def validar_y_enviar(self):
        self.tipo_graficar = self.input_tipo.text().strip()
        self.valor_min = float(self.input_min.text())
        self.valor_max = float(self.input_max.text())
        self.accept()
        # try:
        #     self.valor_min = float(self.input_min.text())
        #     self.valor_max = float(self.input_max.text())

        #     if self.valor_min >= self.valor_max:
        #         raise ValueError("El mínimo debe ser menor al máximo.")

        #     self.accept()  # Cierra el diálogo correctamente
        # except ValueError as e:
        #     QMessageBox.warning(self, "Error", f"Entrada inválida: {e}")



class GenerarCsv(QWidget):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle("Arreglar DataFrame")
        self.resize(300, 150)
        self.df = df  # Guardamos el DataFrame original
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)

    def generar_csv(self):
        nombre, ok = QInputDialog.getText(self, "Guardar CSV", "Nombre del archivo:") # ok es booleano , retorna True si da en aceptar o False caso contrario

        if ok and nombre:
            # Asegura que tenga extensión .csv
            if not nombre.endswith(".csv"):
                nombre += ".csv"
            try:
                self.df.to_csv(nombre, index=False, header=True)
                print(f"Archivo guardado como: {nombre}")
            except Exception as e:
                print(f"Error al guardar el archivo: {e}")
        else:
            print("Guardado cancelado por el usuario.")

class VentanaTransformaciones(QWidget):
    def __init__(self, lista_df, nombres_archivos,menu_principal):
        super().__init__()
        self.menu_principal = menu_principal
        self.setWindowTitle("Opciones de Transformación")
        self.resize(400, 200)
        self.lista_df = lista_df.copy() # SI O SI HAY QUE HACER ESTA LINEA POR QUE SI NO SE PONE EL SELF ENTONCES LISTA_DF SOLO SE PODRA USAR EN ESTE METODO Y NO EN OTRO DEF
        self.nombres_archivos = nombres_archivos # SI O SI HAY QUE HACER ESTA LINEA POR QUE SI NO SE PONE EL SELF ENTONCES NOMBRES_ARCHIVOS SOLO SE PODRA USAR EN ESTE METODO Y NO EN OTRO DEF
        self.df = None # recien cuando el usuario seleccione el df deseado se le asignara

        self.selector_df = QComboBox()
        opciones = [os.path.basename(nombre) for nombre in self.nombres_archivos] # PARA QUE APAREZCA EL NOMBRE DE LOS ARCHIVO QUE SE QUIERE TRANSFORMAR
        for nombre in opciones:
            self.selector_df.addItem(nombre)

        self.selector_df.currentIndexChanged.connect(self.seleccionar_df)
        self.seleccionar_df(0)  # Selecciona automáticamente el primer df, no llama a al metodo seleccionar_df cuando solo hay un archivo por que currentIndexChanged solo se dispara cuando el usuario cambia manualmente el índice por lo que hay que asignar manualmente el df cuando solo hay uno
        # layout = QVBoxLayout()
        # layout.addWidget(QLabel("Selecciona un DataFrame para transformar:"))
        # layout.addWidget(self.selector_df)

        # Crear un contenedor para la normalización
        self.grupo_normalizar = QGroupBox("Normalización Media")
        self.grupo_normalizar.setCheckable(True)  # Activa/Desactiva todo el grupo
        self.grupo_normalizar.setChecked(False)   # Inicialmente desactivado


        # ComboBox para elegir método
        self.combo_normalizar = QComboBox()
        self.combo_normalizar.addItems([
            "Standardize u=0, v2=1",
            "Center to u=0",
            "Scale to v2=1",
            "Normalize to interval [-1,1]",
            "Normalize to interval [0,1]"
        ])

        layout_normalizar = QVBoxLayout()
        layout_normalizar.addWidget(self.combo_normalizar)
        self.grupo_normalizar.setLayout(layout_normalizar)

        # Checkboxes tienen nombres diferentes para que no se pierda todas las casillas marcadas
        self.normalizar_a = QCheckBox("Normalizar por area")
        self.derivada_pd = QCheckBox("Primera Derivada")
        self.derivada_sd = QCheckBox("Segunda Derivada")
        self.correccion_cbl = QCheckBox("Correccion Base Lineal")
        self.correccion_cs = QCheckBox("Correccion Shirley")

        #################################################
        # ESTILO PARA EL COMBO BOX DE NORMALIZAR MEDIA
        estilo_grupo_y_combo = """
            QGroupBox {
                color: white;
                font-weight: bold;
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }

            QGroupBox::indicator {
                width: 15px;
                height: 15px;
                border: 1px solid gray;
                background-color: white;
            }

            QGroupBox::indicator:checked {
                background-color: green;
                border: 1px solid black;
            }

            QComboBox {
                background-color: #2c3e50;
                color: white;
                padding: 5px;
                border: 1px solid gray;
                border-radius: 3px;
            }

            QComboBox QAbstractItemView {
                background-color: #2c3e50;
                color: white;
                selection-background-color: #34495e;
                selection-color: white;
            }
        """

        self.grupo_normalizar.setStyleSheet(estilo_grupo_y_combo)
        self.combo_normalizar.setStyleSheet(estilo_grupo_y_combo)
        #################################################################################

        ##############################################################################
        # Grupo Savitzky-Golay ESTILOS CSS
        estilo_checkbox = """
            QGroupBox {
                color: white;
                background-color: #2c3e50;
                border: 1px solid #3e3e3e;
                border-radius: 5px;
                margin-top: 6px;
                font-weight: bold;
                padding-top: 20px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px;
            }

            QCheckBox {
                color: white;
                background-color: #2c3e50;
                padding: 5px;
            }

            QCheckBox::indicator,
            QGroupBox::indicator {
                width: 15px;
                height: 15px;
            }

            QCheckBox::indicator:unchecked,
            QGroupBox::indicator:unchecked {
                border: 1px solid gray;
                background-color: white;
            }

            QCheckBox::indicator:checked,
            QGroupBox::indicator:checked {
                background-color: green;
                border: 1px solid black;
            }

            QLabel {
                color: white;
            }

            QLineEdit {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #555;
                padding: 4px;
                border-radius: 4px;
            }
        """
        # Grupo Savitzky-Golay
        self.grupo_sg = QGroupBox("Suavizado Savitzky-Golay")
        self.grupo_sg.setCheckable(True)
        self.grupo_sg.setChecked(False)

        self.label_ventana_sg = QLabel("Ventana:")
        self.input_ventana_sg = QLineEdit()
        self.input_ventana_sg.setPlaceholderText("Ej: 5")

        self.label_orden_sg = QLabel("Orden:")
        self.input_orden_sg = QLineEdit()
        self.input_orden_sg.setPlaceholderText("Ej: 2")

        layout_sg = QVBoxLayout()
        layout_sg.addWidget(self.label_ventana_sg)
        layout_sg.addWidget(self.input_ventana_sg)
        layout_sg.addWidget(self.label_orden_sg)
        layout_sg.addWidget(self.input_orden_sg)

        self.grupo_sg.setLayout(layout_sg)
        self.grupo_sg.setStyleSheet(estilo_checkbox)

        # Grupo Filtro Grausiano
        self.grupo_fg = QGroupBox("Suavizado Filtro Gausiano")
        self.grupo_fg.setCheckable(True)
        self.grupo_fg.setChecked(False)

        self.label_sigma_fg = QLabel("Sigma:")
        self.input_sigma_fg = QLineEdit()
        self.input_sigma_fg.setPlaceholderText("Ej: 2")

        layout_fg = QVBoxLayout()
        layout_fg.addWidget(self.label_sigma_fg)
        layout_fg.addWidget(self.input_sigma_fg)

        self.grupo_fg.setLayout(layout_fg)
        self.grupo_fg.setStyleSheet(estilo_checkbox)

        # Grupo Media Movil
        self.grupo_mm = QGroupBox("Suavizado Media Movil")
        self.grupo_mm.setCheckable(True)
        self.grupo_mm.setChecked(False)

        self.label_ventana_mm = QLabel("Ventana:")
        self.input_ventana_mm = QLineEdit()
        self.input_ventana_mm.setPlaceholderText("Ej: 2")

        layout_mm = QVBoxLayout()
        layout_mm.addWidget(self.label_ventana_mm)
        layout_mm.addWidget(self.input_ventana_mm)

        self.grupo_mm.setLayout(layout_mm)
        self.grupo_mm.setStyleSheet(estilo_checkbox)


        #################################################################################

        ################################################################################
        # USAMOS ESTILOS CSS PARA CAMBIAR EL COLOR DE LOS CHECKBOX POR QUE NO SE LOGRA DISTINGIR CON EL FONDO OSCURO
        estilo_checkbox = """
            QCheckBox {
                color: white;
                background-color: #2c3e50;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid gray;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: green;
                border: 1px solid black;
            }
        """
        self.normalizar_a.setStyleSheet(estilo_checkbox)
        self.derivada_pd.setStyleSheet(estilo_checkbox)
        self.derivada_sd.setStyleSheet(estilo_checkbox)
        self.correccion_cbl.setStyleSheet(estilo_checkbox)
        self.correccion_cs.setStyleSheet(estilo_checkbox)
        ##########################################################

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Selecciona un DataFrame para transformar:"))
        layout.addWidget(self.selector_df)
        layout.addWidget(self.grupo_normalizar)
        layout.addWidget(self.normalizar_a)
        layout.addWidget(self.grupo_sg)
        layout.addWidget(self.grupo_fg)
        layout.addWidget(self.grupo_mm)
        layout.addWidget(self.derivada_pd)
        layout.addWidget(self.derivada_sd)
        layout.addWidget(self.correccion_cbl)
        layout.addWidget(self.correccion_cs)


        botones_layout = QHBoxLayout()
        btn_aceptar = QPushButton("Aceptar")
        btn_cancelar = QPushButton("Cancelar")
        botones_layout.addWidget(btn_aceptar)
        botones_layout.addWidget(btn_cancelar)



        # Conexiones
        btn_aceptar.clicked.connect(self.aplicar_transformaciones_y_cerrar)
        btn_cancelar.clicked.connect(self.close)

        layout.addLayout(botones_layout)
        self.setLayout(layout)

    def seleccionar_df(self, index):
        print("ENTRO EN SELECCIONAR ARCHIVO")
        self.df = self.lista_df[index].copy()
        nombre_archivo = os.path.basename(self.nombres_archivos[index])
        print(f"DataFrame seleccionado: {nombre_archivo} con forma {self.df.shape}")

    # FUNCION PARA QUE SE CIERRE LA VENTANA AL APRETAR ACEPTAR
    def aplicar_transformaciones_y_cerrar(self):
        self.aplicar_transformaciones()
        self.close()


    ####     LOS CALCULOS DE ESTAS OPCIONES TRATAR DE HACER DENTRO DE FUNCIONES.PY
    ####     SEGUN INFORMACIONES NO ES LO MISMO POR EJEMPLO SUAVIZAR Y LUEGO NORMALIZAR QUE NORMALIZAR Y LUEGO SUAVIZAR
    ####     SEGUN CHATGPT EL ORDEN IMPORTA Y RECOMIENDA QUE SEA: CORRECIONES -> NORMALIZACION -> SUAVIZADO -> DERIVADAS
    def aplicar_transformaciones(self):
        #df = self.df.copy()
        opciones = {} # CREAMOS UN DICCIONARIO PARA LOS CASOS ESPECIALES EN DONDE EL HILO TIENE QUE RECIBIR MAS PAREMETROS(VENTANA,ORDEN,SIGMA..) Y NO SOLO EL DF A MODIFICAR

        # Normalizacion Media
        if self.grupo_normalizar.isChecked():
            metodo = self.combo_normalizar.currentText()
            opciones["normalizar_media"] = {
                "activar": True,
                "metodo": metodo
            }

        # Savitzky-Golay
        if self.grupo_sg.isChecked():
            ventana = int(self.input_ventana_sg.text()) # SE USA LA VARIABLE input_ventana_sg POR QUE ASI SE ESCRIBIO EN LA PARTE QUE SE CREA LA INTERFAZ PARA LA LECTURA DE DATOS
            orden = int(self.input_orden_sg.text())
            opciones["suavizar_sg"] = {"ventana": ventana, "orden": orden}

        # Filtro Gaussiano
        if self.grupo_fg.isChecked():
            sigma = int(self.input_sigma_fg.text())
            opciones["suavizar_fg"] = {"sigma": sigma}

        # Media Móvil
        if self.grupo_mm.isChecked():
            ventana_mm = int(self.input_ventana_mm.text())
            opciones["suavizar_mm"] = {"ventana": ventana_mm}

        # CORRECCIONES
        if self.correccion_cbl.isChecked():
            opciones["correccion_lineal"] = True

        if self.correccion_cs.isChecked():
            opciones["correccion_shirley"] = True

        # NORMALIZACION AREA
        if self.normalizar_a.isChecked():
            opciones["normalizar_area"] = True

        # DERIVADAS
        if self.derivada_pd.isChecked():
            opciones["derivada_1"] = True

        if self.derivada_sd.isChecked():
            opciones["derivada_2"] = True



        self.hilo = HiloMetodosTransformaciones(self.df,opciones) # LLAMAMOS AL HILO Y LE PASAMOS EL DF ORIGINAL Y LA OPCION SELECCIONADA
        self.hilo.data_frame_resultado.connect(self.recibir_df_transformado)
        self.hilo.start()

    # FUNCIONA PERO COMENTE PARA PROBAR OTRA COSA
    # def recibir_df_transformado(self, df_transformado):
    #     print("DataFrame transformado recibido:")
    #     print(df_transformado)

    #     nombre_df, ok = QInputDialog.getText(self, "Guardar DataFrame", "Ingrese un nombre para el DataFrame transformado:")
    #     if ok and nombre_df.strip():
    #         self.menu_principal.dataframes.append(df_transformado)
    #         self.menu_principal.nombres_archivos.append(nombre_df.strip())

    #         self.ventana_opciones = VentanaOpcionesPostTransformacion(self.menu_principal, df_transformado)
    #         self.ventana_opciones.show()

    def recibir_df_transformado(self, df_transformado):
        print("DataFrame transformado recibido:")
        print(df_transformado)

        # Solicita al usuario un nombre para guardar el DataFrame transformado
        nombre_df, ok = QInputDialog.getText(self, "Guardar DataFrame", "Ingrese un nombre para el DataFrame transformado:")
        if ok and nombre_df.strip():
            self.menu_principal.dataframes.append(df_transformado)
            self.menu_principal.nombres_archivos.append(nombre_df.strip())

            QMessageBox.information(self, "Éxito", f"DataFrame transformado guardado como '{nombre_df.strip()}'")

            # (Opcional) Mostrar espectros automáticamente del nuevo DF
            self.menu_principal.ver_espectros(df_transformado)


        # if self.correccion_cbl.isChecked():
        #     self.worker = HiloMetodosTransformaciones(self.df, opciones)
        #     self.worker.progreso.connect(self.mostrar_mensaje)  # Puedes conectar a una barra o consola
        #     self.worker.terminado.connect(self.actualizar_dataframe)  # Qué hacer al terminar
        #     self.worker.start()
        #     print("HACER CORRECION BASE LINEAL")

        # if self.correccion_cs.isChecked():
        #     print("HACER CORRECION SHIRLEY")

        # if self.grupo_normalizar.isChecked():
        #     metodo = self.combo_normalizar.currentText()
        #     print("Normalización activada:", metodo)

        # if self.normalizar_a.isChecked():
        #     print("HACER NORMALIZACION AREA")
        #     #df = df.rolling(window=3, min_periods=1).mean()

        # if self.grupo_sg.isChecked():
        #     ventana = int(self.input_ventana_sg.text())
        #     orden = int(self.input_orden_sg.text())
        #     print(f"Savitzky-Golay activado: ventana={ventana}, orden={orden}")

        # if self.grupo_fg.isChecked():
        #     sigma = int(self.input_sigma_fg.text())
        #     print(f"Filtro Gausiano activado: Sigma={sigma}")

        # if self.grupo_mm.isChecked():
        #     ventana = int(self.input_ventana_mm.text())
        #     print(f"Media Movil activado: Ventana={ventana}")

        # if self.derivada_pd.isChecked():
        #     print("HACER PRIMERA DERIVADA")
        # #     df = df.diff().fillna(0)
        # if self.derivada_sd.isChecked():
        #     print("HACER SEGUNDA DERIVADA")

# UNA VEZ QUE SE GENERA EL NUEVO DF TRANSFORMADO SE HABRE UNA NUEVA VENTANA
# class VentanaOpcionesPostTransformacion(QWidget):
#     def __init__(self, menu_principal, df_transformado):
#         super().__init__()
#         self.menu_principal = menu_principal
#         self.df = df_transformado

#         self.setWindowTitle("Acciones con el DataFrame transformado")

#         layout = QVBoxLayout()
#         layout.addWidget(QLabel("¿Qué desea hacer con el DataFrame transformado?"))

#         btn_ver_df = QPushButton("Ver DataFrame")
#         btn_ver_espectro = QPushButton("Mostrar Espectros")

#         btn_ver_df.clicked.connect(self.ver_df)
#         btn_ver_espectro.clicked.connect(self.ver_espectros)

#         layout.addWidget(btn_ver_df)
#         layout.addWidget(btn_ver_espectro)
#         self.setLayout(layout)

#     def ver_df(self):
#         print("Df transformado 1 self.df")
#         print(self.df)
#         self.menu_principal.ver_dataframe(self.df)
#         self.close()
#     def ver_espectros(self):
#         self.menu_principal.ver_espectros(self.df)
#         print("Df transformado 2 self.df")
#         print(self.df)
#         self.close()


class VentanaReduccionDim(QWidget):
    def __init__(self, lista_df, nombres_archivos, menu_principal):
        super().__init__()
        self.menu_principal = menu_principal
        self.setWindowTitle("Reducción de Dimensionalidad")
        self.resize(400, 300)
        self.lista_df = lista_df.copy()
        self.nombres_archivos = nombres_archivos
        self.df = None
        #self.asignacion_colores = asignacion_colores
        # Selector de DataFrame
        self.selector_df = QComboBox()
        opciones = [os.path.basename(nombre) for nombre in self.nombres_archivos] # PARA QUE APAREZCA EL NOMBRE DE LOS ARCHIVO QUE SE QUIERE TRANSFORMAR
        for nombre in opciones:
            self.selector_df.addItem(nombre)

        self.selector_df.currentIndexChanged.connect(self.seleccionar_df)
        self.seleccionar_df(0)  # Selecciona automáticamente el primer df, no llama a al metodo seleccionar_df cuando solo hay un archivo por que currentIndexChanged solo se dispara cuando el usuario cambia manualmente el índice por lo que hay que asignar manualmente el df cuando solo hay uno

        ##############################################################################
        # Para los estilos de las casillas de intervalo de confianza y numero de componentes principales
        estilo_checkbox_datos = """
            QGroupBox {
                color: white;
                background-color: #2c3e50;
                border: 1px solid #3e3e3e;
                border-radius: 5px;
                margin-top: 6px;
                font-weight: bold;
                padding-top: 20px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 5px;
            }

            QCheckBox {
                color: white;
                background-color: #2c3e50;
                padding: 5px;
            }

            QCheckBox::indicator,
            QGroupBox::indicator {
                width: 15px;
                height: 15px;
            }

            QCheckBox::indicator:unchecked,
            QGroupBox::indicator:unchecked {
                border: 1px solid gray;
                background-color: white;
            }

            QCheckBox::indicator:checked,
            QGroupBox::indicator:checked {
                background-color: green;
                border: 1px solid black;
            }

            QLabel {
                color: white;
            }

            QLineEdit {
                background-color: #1e1e1e;
                color: white;
                border: 1px solid #555;
                padding: 4px;
                border-radius: 4px;
            }
        """
        # # Grupo Savitzky-Golay
        # self.grupo_sg = QGroupBox("Suavizado Savitzky-Golay")
        # self.grupo_sg.setCheckable(True)
        # self.grupo_sg.setChecked(False)

        self.label_reduccion_dim_componentes = QLabel("Numero de Componentes Principales:")
        self.input_reduccion_dim_componentes = QLineEdit()
        self.input_reduccion_dim_componentes.setPlaceholderText("Ej: 2")

        self.label_reduccion_dim_intervalo = QLabel("Intervalo de Confianza:")
        self.input_reduccion_dim_intervalo = QLineEdit()
        self.input_reduccion_dim_intervalo.setPlaceholderText("Ej: 90")

        layout_dim = QVBoxLayout() #ORGANIZA LO WIDGET DE FORMA VERTICAL
        layout_dim.addWidget(self.label_reduccion_dim_componentes)
        layout_dim.addWidget(self.input_reduccion_dim_componentes)
        layout_dim.addWidget(self.label_reduccion_dim_intervalo)
        layout_dim.addWidget(self.input_reduccion_dim_intervalo)

        self.label_reduccion_dim_componentes.setStyleSheet(estilo_checkbox_datos)
        self.input_reduccion_dim_componentes.setStyleSheet(estilo_checkbox_datos)
        self.label_reduccion_dim_intervalo.setStyleSheet(estilo_checkbox_datos)
        self.input_reduccion_dim_intervalo.setStyleSheet(estilo_checkbox_datos)

        ###########################################################
        # Crea un contenedor
        # self.grupo_pca = QGroupBox("Análisis de Componentes Principales (PCA)")
        # self.grupo_pca.setCheckable(True)  # Activa/Desactiva todo el grupo
        # self.grupo_pca.setChecked(False)   # Inicialmente desactivado

        # # ComboBox para elegir método
        # self.combo_pca = QComboBox()
        # self.combo_pca.addItems([
        #     "Grafico en 2D",
        #     "Grafico en 3D",
        # ])

        # layout_pca = QVBoxLayout()
        # layout_pca.addWidget(self.combo_pca)
        # self.grupo_pca.setLayout(layout_pca)

        # if self.combo_pca.currentText() == "Grafico en 2D":
        #     self.grafico = 0
        # elif self.combo_pca.currentText() == "Grafico en 3D":
        #     self.grafico = 1

        # Checkboxes , AGREGAR OPCION DE 2D Y 3D, VER LA MANERA MAS LINDO DE ARREGALR ES
        self.pca = QCheckBox("Análisis de Componentes Principales (PCA)")
        self.tsne = QCheckBox("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
        self.tsne_pca = QCheckBox("t-SNE(PCA(X))")
        self.grafico2d = QCheckBox("Grafico 2D")
        self.grafico3d = QCheckBox("Grafico 3D")
        self.graficoloading = QCheckBox("Grafico Loading (PCA)")
        self.geninforme = QCheckBox("Generar Informe")

        # PARA QUE AL DAR CLICK EN t-SNE(PCA(X)) ME MUESTRE EL CAMPO DE PEDIDO DE  NUMERO DE COMPONENTES PRINCIPALES PARA PCA Y TSNE
        self.tsne_pca.stateChanged.connect(self.toggle_tsne_pca)
        self.input_comp_pca = QLineEdit()
        self.input_comp_pca.setPlaceholderText("Ingrese el número de CP para PCA:")
        self.input_comp_tsne = QLineEdit()
        self.input_comp_tsne.setPlaceholderText("Ingrese el número de CP para TSNE [2,3]:")
        self.contenedor_componentes_tsne_pca = QWidget()
        layout_tsne_pca = QVBoxLayout()
        layout_tsne_pca.addWidget(self.input_comp_pca)
        layout_tsne_pca.addWidget(self.input_comp_tsne)
        self.contenedor_componentes_tsne_pca.setLayout(layout_tsne_pca)
        self.contenedor_componentes_tsne_pca.hide()  # Ocultamos todo el contenedor


        # PARA QUE AL DAR CLICK EN GENERAR INFORME ME MUESTRE UN CAMPO SOLICITANDO EL NOMBRE DEL INFORME
        self.geninforme.stateChanged.connect(self.toggle_nombre_informe)
        self.label_nombre_informe = QLabel("Nombre del archivo del informe:")
        self.input_nombre_informe = QLineEdit()
        self.input_nombre_informe.setPlaceholderText("Ej: informe.txt")
        self.contenedor_nombre_informe = QWidget()
        layout_nombre_informe = QHBoxLayout()
        layout_nombre_informe.addWidget(self.label_nombre_informe)
        layout_nombre_informe.addWidget(self.input_nombre_informe)
        self.contenedor_nombre_informe.setLayout(layout_nombre_informe)
        self.contenedor_nombre_informe.hide()  # Ocultamos todo el contenedor

        # PARA QUE AL DAR CLICK EN GRAFICO 2D ME MUESTRE EL CAMPO DE PEDIDO DE  NUMERO DE COMPONENTES PRINCIPALES PARA GRAFICAR [X,Y]
        self.grafico2d.stateChanged.connect(self.toggle_gen2d)
        self.input_x_2d = QLineEdit()
        self.input_x_2d.setPlaceholderText("Ingrese el número de PC para X:")
        self.input_y_2d = QLineEdit()
        self.input_y_2d.setPlaceholderText("Ingrese el número de PC para Y:")
        self.contenedor_componentes2d = QWidget()
        layout_numero_cmp_2d = QVBoxLayout()
        layout_numero_cmp_2d.addWidget(self.input_x_2d)
        layout_numero_cmp_2d.addWidget(self.input_y_2d)
        self.contenedor_componentes2d.setLayout(layout_numero_cmp_2d)
        self.contenedor_componentes2d.hide()  # Ocultamos todo el contenedor

        # PARA QUE AL DAR CLICK EN GRAFICO 3D ME MUESTRE EL CAMPO DE PEDIDO DE  NUMERO DE COMPONENTES PRINCIPALES PARA GRAFICAR [X,Y,Z]
        self.grafico3d.stateChanged.connect(self.toggle_gen3d)
        self.input_x_3d = QLineEdit()
        self.input_x_3d.setPlaceholderText("Ingrese el número de PC para X:")
        self.input_y_3d = QLineEdit()
        self.input_y_3d.setPlaceholderText("Ingrese el número de PC para Y:")
        self.input_z_3d = QLineEdit()
        self.input_z_3d.setPlaceholderText("Ingrese el número de PC para Z:")
        self.contenedor_componentes3d = QWidget()
        layout_numero_cmp_3d = QVBoxLayout()
        layout_numero_cmp_3d.addWidget(self.input_x_3d)
        layout_numero_cmp_3d.addWidget(self.input_y_3d)
        layout_numero_cmp_3d.addWidget(self.input_z_3d)
        self.contenedor_componentes3d.setLayout(layout_numero_cmp_3d)
        self.contenedor_componentes3d.hide()  # Ocultamos todo el contenedor
        
        # PARA QUE AL DAR CLICK EN LOANDING ME MUESTRE EL CAMPO DE PEDIDO DE  NUMERO DE COMPONENTES PRINCIPALES PARA GRAFICAR [X,Y] O [X,Y,Z]
        self.graficoloading.stateChanged.connect(self.toggle_loading)
        self.input_cant_comp = QLineEdit()
        self.input_cant_comp.setPlaceholderText("Ingrese cantidad de componentes principales")
        self.input_x_loading = QLineEdit()
        self.input_x_loading.setPlaceholderText("Ingrese el número de PC para X:")
        self.input_y_loading = QLineEdit()
        self.input_y_loading.setPlaceholderText("Ingrese el número de PC para Y:")
        self.input_z_loading = QLineEdit()
        self.input_z_loading.setPlaceholderText("Ingrese el número de PC para Z:")
        self.contenedor_loading = QWidget()
        layout_numero_cmp_loading = QVBoxLayout()
        layout_numero_cmp_loading.addWidget(self.input_cant_comp)
        layout_numero_cmp_loading.addWidget(self.input_x_loading)
        layout_numero_cmp_loading.addWidget(self.input_y_loading)
        layout_numero_cmp_loading.addWidget(self.input_z_loading)
        self.contenedor_loading.setLayout(layout_numero_cmp_loading)
        self.contenedor_loading.hide()  # Ocultamos todo el contenedor

        # estilo_grupo_y_combo = """
        #     QGroupBox {
        #         color: white;
        #         font-weight: bold;
        #         border: 1px solid gray;
        #         border-radius: 5px;
        #         margin-top: 10px;
        #         padding: 10px;
        #     }

        #     QGroupBox::title {
        #         subcontrol-origin: margin;
        #         subcontrol-position: top left;
        #         padding: 0 5px;
        #     }

        #     QGroupBox::indicator {
        #         width: 15px;
        #         height: 15px;
        #         border: 1px solid gray;
        #         background-color: white;
        #     }

        #     QGroupBox::indicator:checked {
        #         background-color: green;
        #         border: 1px solid black;
        #     }

        #     QComboBox {
        #         background-color: #2c3e50;
        #         color: white;
        #         padding: 5px;
        #         border: 1px solid gray;
        #         border-radius: 3px;
        #     }

        #     QComboBox QAbstractItemView {
        #         background-color: #2c3e50;
        #         color: white;
        #         selection-background-color: #34495e;
        #         selection-color: white;
        #     }
        # """

        # PARA LOS CHECKBOX DEL PCA
        estilo_checkbox = """
            QCheckBox {
                color: white;
                background-color: #2c3e50;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid gray;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: green;
                border: 1px solid black;
            }
        """

        #self.combo_pca.setStyleSheet(estilo_grupo_y_combo)
        #self.grupo_pca.setStyleSheet(estilo_grupo_y_combo)
        self.pca.setStyleSheet(estilo_checkbox)
        self.tsne.setStyleSheet(estilo_checkbox)
        self.tsne_pca.setStyleSheet(estilo_checkbox)
        self.grafico2d.setStyleSheet(estilo_checkbox)
        self.grafico3d.setStyleSheet(estilo_checkbox)
        self.geninforme.setStyleSheet(estilo_checkbox)
        self.graficoloading.setStyleSheet(estilo_checkbox)
        # Botones
        btn_aceptar = QPushButton("Aceptar")
        btn_cancelar = QPushButton("Cancelar")
        btn_aceptar.clicked.connect(self.aplicar_transformaciones_y_cerrar)
        btn_cancelar.clicked.connect(self.close)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Selecciona un DataFrame y técnicas de reducción de dimensionalidad:"))
        layout.addWidget(self.selector_df)
        #layout.addWidget(self.grupo_pca)
        layout.addWidget(self.pca)
        layout.addWidget(self.tsne)
        layout.addWidget(self.tsne_pca)
        layout.addWidget(self.contenedor_componentes_tsne_pca)
        layout.addWidget(self.label_reduccion_dim_componentes)
        layout.addWidget(self.input_reduccion_dim_componentes)
        layout.addWidget(self.label_reduccion_dim_intervalo)
        layout.addWidget(self.input_reduccion_dim_intervalo)
        layout.addWidget(self.grafico2d)
        layout.addWidget(self.contenedor_componentes2d)
        layout.addWidget(self.grafico3d)
        layout.addWidget(self.contenedor_componentes3d)
        layout.addWidget(self.graficoloading)
        layout.addWidget(self.contenedor_loading)
        layout.addWidget(self.geninforme)
        layout.addWidget(self.contenedor_nombre_informe)



        # self.input_comp_pca.setPlaceholderText("Ingrese el número de PC para PCA:")
        # self.input_comp_tsne.setPlaceholderText("Ingrese el número de PC para TSNE([0,1]):")



        botones_layout = QHBoxLayout()
        botones_layout.addWidget(btn_aceptar)
        botones_layout.addWidget(btn_cancelar)
        layout.addLayout(botones_layout)

        self.setLayout(layout)

        # PARA QUE EL USUARIO NO PUEDA MARCAR GRAFICO 2D Y GRAFICO 3D AL MISMO TIEMPO, AL MARCAR UNO SE DESACTIVA EL OTRO
        # self.grafico2d.stateChanged.connect(self.toggle_checkboxes)
        # self.grafico3d.stateChanged.connect(self.toggle_checkboxes)


    def toggle_nombre_informe(self, state):
        print(f"Estado del checkbox Generar Informe: {state}")
        self.contenedor_nombre_informe.setVisible(bool(state))

    def toggle_gen2d(self, state):
        print(f"Estado del checkbox Grafico 2D: {state}")
        self.contenedor_componentes2d.setVisible(bool(state))

    def toggle_gen3d(self, state):
        print(f"Estado del checkbox Grafico 3D: {state}")
        self.contenedor_componentes3d.setVisible(bool(state))

    def toggle_tsne_pca(self, state):
        print(f"Estado del checkbox TSNE(PCA): {state}")
        self.contenedor_componentes_tsne_pca.setVisible(bool(state))
        
    def toggle_loading(self, state):
        print(f"Estado del checkbox loading: {state}")
        self.contenedor_loading.setVisible(bool(state))

    # LA IDEA ERA QUE EL USUARIO NO PUEDA MARCAR GRAFICO 2D Y GRAFICO 3D AL MISMO TIEMPO, AL MARCAR UNO SE DESACTIVA EL OTRO
    # def toggle_checkboxes(self):
    #     if self.grafico2d.isChecked():
    #         self.grafico3d.setChecked(False)
    #     elif self.grafico3d.isChecked():
    #         self.grafico2d.setChecked(False)

    def seleccionar_df(self, index):
        if 0 <= index < len(self.lista_df):
            self.df = self.lista_df[index].copy()
            print("ENTRO EN SELECCIONAR_DF")
            print(self.df)

    def aplicar_transformaciones_y_cerrar(self):
        componentes = self.input_reduccion_dim_componentes.text().strip() # text() devuelve el texto que el usuario escribió en ese campo y strip() elimina los espacios en blanco
        intervalo = self.input_reduccion_dim_intervalo.text().strip()
        nombre_informe = self.input_nombre_informe.text().strip()
        cant_componentes_loading = self.input_cant_comp.text().strip() # CANTIDAD DE CP PARA LOS GRAFICO DE LOADING (PRIMERO VA A LA FUNCION PCA)
        num_x_loading = self.input_x_loading.text().strip() # PARA LOADING COMPONENTES A GRAFICAR X
        num_y_loading = self.input_y_loading.text().strip() # PARA LOADING COMPONENTES A GRAFICAR Y
        num_z_loading = self.input_z_loading.text().strip() # PARA LOADING COMPONENTES A GRAFICAR Z (PUEDE NO TENER Z), SI NO  SE INGRESO NADA num_z_loading == ""
        
        if num_z_loading == "":
            num_z_loading = 0 

        if self.df is None:
            QMessageBox.warning(self, "Sin selección", "Debe seleccionar un DataFrame.")
            return

        componentes_selec = []
        opciones = {}
        
        # Valores por defecto para el caso de que no se use cp_pca , cp_tsne y igual necesite enviar algun dato para evitar errores
        cp_pca = None
        cp_tsne = None
        
        if self.pca.isChecked():
            opciones["PCA"] = True   # ACA QUIERO QUE HALLE SOLAMENTE EL VALOR DEL PCA
        if self.tsne.isChecked():
            opciones["TSNE"] = True   # ACA QUIERO SOLO EL CALCULO DEL TSNE
        if self.tsne_pca.isChecked():
            opciones["t-SNE(PCA(X))"] = True  # PREPARADO POR LAS DUDAS MAS ADELANTE SE NECESITE
            cp_pca = int(self.input_comp_pca.text())
            cp_tsne = int(self.input_comp_tsne.text())
        if self.grafico2d.isChecked():
            opciones["GRAFICO 2D"] = True # VALIDAR QUE EL USUARIO HALLA ELEGIDO(CHECK) EL PCA O EL TSNE
            self.pc_x = int(self.input_x_2d.text())
            self.pc_y = int(self.input_y_2d.text())
            componentes_selec = [self.pc_x, self.pc_y]
        if self.grafico3d.isChecked():
            opciones["GRAFICO 3D"] = True # VALIDAR QUE EL USUARIO HALLA ELEGIDO(CHECK) EL PCA O EL TSNE
            self.pc_x = int(self.input_x_3d.text())
            self.pc_y = int(self.input_y_3d.text())
            self.pc_z = int(self.input_z_3d.text())
            componentes_selec = [self.pc_x, self.pc_y, self.pc_z]
        
        if self.geninforme.isChecked():
            opciones["GENERAR INFORME"] = True  # VALIDAR QUE EL USUARIO HALLA ELEGIDO(CHECK) EL PCA O EL TSNE PARA GENERAR EL REPORTE
            
        if self.graficoloading.isChecked():
            opciones["Grafico Loading (PCA)"] = True  # SI NO SE INGRESA VALOR DE Z GRAFICARA SOLO X e Y
            num_x_loading = int(num_x_loading)
            num_y_loading = int(num_y_loading)
            num_z_loading = int(num_z_loading)
            componentes_selec_loading = [num_x_loading,num_y_loading,num_z_loading]


        self.hilo = HiloMetodosReduccion(self.df, opciones,componentes,intervalo,nombre_informe,componentes_selec,cp_pca,cp_tsne,componentes_selec_loading,int(cant_componentes_loading))
        # Conectamos la señal emitida desde el hilo (UN HILO PUEDE TENER VARIOS SIGNAL)
        self.hilo.signal_figura_pca_2d.connect(self.mostrar_grafico_pca_2d)
        self.hilo.signal_figura_pca_3d.connect(self.mostrar_grafico_pca_3d)
        self.hilo.signal_figura_tsne_2d.connect(self.mostrar_grafico_tsne_2d)
        self.hilo.signal_figura_tsne_3d.connect(self.mostrar_grafico_tsne_3d)
        self.hilo.signal_figura_loading.connect(self.mostrar_grafico_loading)
        # Iniciar el hilo
        self.hilo.start()


    def mostrar_grafico_pca_2d(self, fig):
        self.ventana_pca = VentanaGraficoPCA2D(fig)
        self.ventana_pca.show()

    def mostrar_grafico_pca_3d(self, fig):
        self.ventana_pca = VentanaGraficoPCA3D(fig)
        self.ventana_pca.show()

    def mostrar_grafico_tsne_2d(self, fig):
        self.ventana_tsne = VentanaGraficoTSNE2D(fig)
        self.ventana_tsne.show()
        
    def mostrar_grafico_tsne_3d(self, fig):
        self.ventana_tsne = VentanaGraficoTSNE3D(fig)
        self.ventana_tsne.show()
    
    def mostrar_grafico_loading(self, fig):
        print("VOLVIO AL MAIN LOADGING")
        self.ventana_tsne = VentanaGraficoLoading(fig)
        self.ventana_tsne.show()
        
# VentanaGraficoPCA2D y VentanaGraficoPCA3D son lo mismo solo que separo de por si quiero hacerle mejoras independiente(mas botones o algun tipo de leyenda especial)
# VER QUE HACE LINEA POR LINEA
class VentanaGraficoPCA2D(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gráfico PCA 2D")

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        # Guardar figura Plotly en un archivo temporal HTML
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.load(f"file://{f.name}")

        # Guardar la ruta para borrar luego si querés
        self.tempfile_path = f.name

    def closeEvent(self, event):
        # Borra el archivo temporal al cerrar la ventana
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()

class VentanaGraficoPCA3D(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gráfico PCA 3D")

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        # Guardar figura Plotly 3D como HTML temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.load(f"file://{f.name}")
            self.tempfile_path = f.name  # Guardar la ruta

    def closeEvent(self, event):
        # Eliminar el archivo temporal al cerrar
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()




class VentanaGraficoTSNE2D(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gráfico t-SNE 2D")

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        # Guardar figura en archivo HTML temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.load(f"file://{f.name}")
            self.tempfile_path = f.name

    def closeEvent(self, event):
        # Eliminar archivo temporal al cerrar
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()



class VentanaGraficoTSNE3D(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gráfico t-SNE 3D")

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        # Guardar figura en archivo HTML temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.load(f"file://{f.name}")
            self.tempfile_path = f.name

    def closeEvent(self, event):
        # Eliminar archivo temporal al cerrar
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()

class VentanaGraficoLoading(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gráfico Loading")

        layout = QVBoxLayout()
        self.browser = QWebEngineView()
        layout.addWidget(self.browser)
        self.setLayout(layout)

        # Guardar figura Plotly en un archivo temporal HTML
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            fig.write_html(f.name)
            self.browser.load(f"file://{f.name}")

        # Guardar la ruta para borrar luego si querés
        self.tempfile_path = f.name

    def closeEvent(self, event):
        # Borra el archivo temporal al cerrar la ventana
        if os.path.exists(self.tempfile_path):
            os.remove(self.tempfile_path)
        event.accept()


############## ACA ES DONDE EXPLICO BIEN COMO FUNCIONA LOS LAYOUT Y SU ORDEN########################
class VentanaHca(QWidget):
    def __init__(self, lista_df, nombres_archivos, menu_principal):
        super().__init__()
        self.menu_principal = menu_principal
        self.setWindowTitle("HCA (Análisis de Conglomerados Jerárquico)")
        self.resize(400, 300)
        self.lista_df = lista_df.copy()
        self.nombres_archivos = nombres_archivos
        self.df = None
        
        estilo_checkbox = """
            QCheckBox {
                color: white;
                background-color: #2c3e50;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid gray;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: green;
                border: 1px solid black;
            }
        """
        
        #self.asignacion_colores = asignacion_colores
        # Selector de DataFrame
        self.selector_df = QComboBox()
        opciones = [os.path.basename(nombre) for nombre in self.nombres_archivos] # PARA QUE APAREZCA EL NOMBRE DE LOS ARCHIVO QUE SE QUIERE TRANSFORMAR
        for nombre in opciones:
            self.selector_df.addItem(nombre)

        self.selector_df.currentIndexChanged.connect(self.seleccionar_df)
        self.seleccionar_df(0)  # Selecciona automáticamente el primer df, no llama a al metodo seleccionar_df cuando solo hay un archivo por que currentIndexChanged solo se dispara cuando el usuario cambia manualmente el índice por lo que hay que asignar manualmente el df cuando solo hay uno
        
        # Botones
        btn_aceptar = QPushButton("Aceptar")
        btn_cancelar = QPushButton("Cancelar")
        btn_aceptar.clicked.connect(self.aplicar_transformaciones_y_cerrar)
        btn_cancelar.clicked.connect(self.close)
        botones_layout = QHBoxLayout()
        botones_layout.addWidget(btn_aceptar)
        botones_layout.addWidget(btn_cancelar)
                
        # PRIMERP CREAMOS LOS CHECKBOX (PASO 1)
        self.label_distancia_metodo = QLabel("¿Qué método de distancias deseas utilizar?")
        self.euclidiana = QCheckBox("Euclidiana")
        self.manhattan = QCheckBox("Manhattan")
        self.coseno = QCheckBox("Coseno")
        self.chebyshev = QCheckBox("Chebyshev")
        self.correlación_pearson = QCheckBox("Correlación Pearson")
        self.correlación_spearman = QCheckBox("Correlación Spearman")
        self.jaccard = QCheckBox("Jaccard")
        
        self.label_cluster_metodo = QLabel("¿Qué método de enlace entre clústeres deseas utilizar?")
        self.ward = QCheckBox("Ward")
        self.single_linkage = QCheckBox("Single Linkage")
        self.complete_linkage = QCheckBox("Complete Linkage")
        self.average_linkage = QCheckBox("Average Linkage")
        
        # ESTE SERIA SOLO PARA QUE CUANDO SE MARQUE OTRA OPCION QUE NO SEA EUCLIDIANA O MANHATTAN QUE SE DESACTIVE EL WARD
        self.euclidiana.stateChanged.connect(self.actualizar_estado_enlaces)
        self.manhattan.stateChanged.connect(self.actualizar_estado_enlaces)
        

        # LUEGO LO AGREGAMOS EN UN LAYOUT QUE ES PARA QUE LAS OPCIONES SE VEAN EN HORIZONTAL (PASO 2)
        distancia_layout = QHBoxLayout() 
        distancia_layout.addWidget(self.euclidiana)
        distancia_layout.addWidget(self.manhattan)
        distancia_layout.addWidget(self.coseno)
        distancia_layout.addWidget(self.chebyshev)
        distancia_layout.addWidget(self.correlación_pearson)
        distancia_layout.addWidget(self.correlación_spearman)
        distancia_layout.addWidget(self.jaccard)
        
        cluster_layout = QHBoxLayout()
        cluster_layout.addWidget(self.ward)
        cluster_layout.addWidget(self.single_linkage)
        cluster_layout.addWidget(self.complete_linkage)
        cluster_layout.addWidget(self.average_linkage)
        
        # ACA AGREGAMOLOS LOS ESTILOS CSS A LOS CHECKBOX PARA QUE SEAN BLANCOS (NO OBLIGATORIO)
        self.euclidiana.setStyleSheet(estilo_checkbox)
        self.manhattan.setStyleSheet(estilo_checkbox)
        self.coseno.setStyleSheet(estilo_checkbox)
        self.chebyshev.setStyleSheet(estilo_checkbox)
        self.correlación_pearson.setStyleSheet(estilo_checkbox)
        self.correlación_spearman.setStyleSheet(estilo_checkbox)
        self.jaccard.setStyleSheet(estilo_checkbox)
        self.ward.setStyleSheet(estilo_checkbox)
        self.single_linkage.setStyleSheet(estilo_checkbox)
        self.complete_linkage.setStyleSheet(estilo_checkbox)
        self.average_linkage.setStyleSheet(estilo_checkbox)
        
        
        # LUEGO CREAMOS EL LAYOUT PRINCIPAL EN DONDE SE AGREGAN TODOS LOS LAYOUT CREADOS, EJEMPLO: LAYOUT PASO 2
        layout = QVBoxLayout()  
        layout.addWidget(QLabel("Selecciona un DataFrame:"))
        layout.addWidget(self.selector_df)
        layout.addWidget(self.label_distancia_metodo)
        layout.addLayout(distancia_layout)
        layout.addWidget(self.label_cluster_metodo)
        layout.addLayout(cluster_layout)
        layout.addLayout(botones_layout)
        

        self.setLayout(layout) # POR ULTIMO SE HACE UN SETLAYOUT DE LAYOUT PRINCIPAL PARA QUE APAREZCAN EN PATALLA

    def seleccionar_df(self, index):
        print("ENTRO EN SELECCIONAR ARCHIVO")
        self.df = self.lista_df[index].copy()
        nombre_archivo = os.path.basename(self.nombres_archivos[index])
        print(f"DataFrame seleccionado: {nombre_archivo} con forma {self.df.shape}")
    
    def aplicar_transformaciones_y_cerrar(self):
        if self.df is None:
            QMessageBox.warning(self, "Sin selección", "Debe seleccionar un DataFrame.")
            return

        opciones = {}
                
        if self.euclidiana.isChecked():
            opciones["Euclidiana"] = True   
        if self.manhattan.isChecked():
            opciones["Manhattan"] = True   
        if self.coseno.isChecked():
            opciones["Coseno"] = True 
        if self.chebyshev.isChecked():
            opciones["Chebyshev"] = True 
        if self.correlación_pearson.isChecked():
            opciones["Correlación Pearson"] = True
        if self.correlación_spearman.isChecked():
            opciones["Correlación Spearman"] = True
        if self.jaccard.isChecked():
            opciones["Jaccard"] = True
        if self.ward.isChecked():
            opciones["Ward"] = True
        if self.single_linkage.isChecked():
            opciones["Single Linkage"] = True
        if self.complete_linkage.isChecked():
            opciones["Complete Linkage"] = True
        if self.average_linkage.isChecked():
            opciones["Average Linkage"] = True

        self.hilo = HiloHca(self.df,opciones)
        # Conectamos la señal emitida desde el hilo (UN HILO PUEDE TENER VARIOS SIGNAL)
        self.hilo.signal_figura_hca.connect(self.generar_hca)
        # Iniciar el hilo
        self.hilo.start()


    # SI NO ESTA MARCADO EUCLIDIANA O MANHATTAN DESABILITA WARD
    def actualizar_estado_enlaces(self):
        if not (self.euclidiana.isChecked() or self.manhattan.isChecked()):
            self.ward.setEnabled(False)
            self.ward.setChecked(False)
        else:
            self.ward.setEnabled(True)
            
    def generar_hca(self, fig):
        print("generar hca")
        self.ventana_hca = VentanaGraficoHCA(fig)
        self.ventana_hca.show()
        
class VentanaGraficoHCA(QWidget):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gráfico HCA")

        layout = QVBoxLayout()
        self.canvas = FigureCanvas(fig)
        layout.addWidget(self.canvas)
        self.setLayout(layout)


# class VentanaLoading(QWidget):
#     def __init__(self, lista_df, nombres_archivos, menu_principal):
#         super().__init__()
#         self.menu_principal = menu_principal
#         self.setWindowTitle("GRAFICO DE LOADING")
#         self.resize(400, 300)
#         self.lista_df = lista_df.copy()
#         self.nombres_archivos = nombres_archivos
#         self.df = None






class VentanaDataFusion(QWidget):
    def __init__(self, lista_df, nombres_archivos, menu_principal):
        super().__init__()
        self.menu_principal = menu_principal
        self.setWindowTitle("Data Fusion")
        self.resize(400, 300)
        self.lista_df = lista_df.copy()
        self.nombres_archivos = nombres_archivos
        self.df = None
        
        estilo_checkbox = """
            QCheckBox {
                color: white;
                background-color: #2c3e50;
                padding: 5px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid gray;
                background-color: white;
            }
            QCheckBox::indicator:checked {
                background-color: green;
                border: 1px solid black;
            }
        """
        
        #self.asignacion_colores = asignacion_colores
        # Selector de DataFrame
        self.selector_df = QComboBox()
        opciones = [os.path.basename(nombre) for nombre in self.nombres_archivos] # PARA QUE APAREZCA EL NOMBRE DE LOS ARCHIVO QUE SE QUIERE TRANSFORMAR
        for nombre in opciones:
            self.selector_df.addItem(nombre)

        self.selector_df.currentIndexChanged.connect(self.seleccionar_df)
        self.seleccionar_df(0)  # Selecciona automáticamente el primer df, no llama a al metodo seleccionar_df cuando solo hay un archivo por que currentIndexChanged solo se dispara cuando el usuario cambia manualmente el índice por lo que hay que asignar manualmente el df cuando solo hay uno
        
    

        # LUEGO CREAMOS EL LAYOUT PRINCIPAL EN DONDE SE AGREGAN TODOS LOS LAYOUT CREADOS, EJEMPLO: LAYOUT PASO 2
        layout = QVBoxLayout()  
        layout.addWidget(QLabel("Selecciona los DataFrame a fusionar:"))
        layout.addWidget(self.selector_df)
        # crear una variable en donde almacene solo los archivos que desea fusionar


        self.setLayout(layout) # POR ULTIMO SE HACE UN SETLAYOUT DE LAYOUT PRINCIPAL PARA QUE APAREZCAN EN PATALLA

    def seleccionar_df(self, index):
        print("ENTRO EN SELECCIONAR ARCHIVO")
        self.df = self.lista_df[index].copy()
        nombre_archivo = os.path.basename(self.nombres_archivos[index])
        print(f"DataFrame seleccionado: {nombre_archivo} con forma {self.df.shape}")
    






# Lanza la aplicación y muestra la ventana.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = MenuPrincipal()
    ventana.show()
    sys.exit(app.exec())







############### PROXIMA ACTIVIDAD
# PROBAR HASTA AHORA COMO VA QUEDANDO TODAS LAS OPCIONES EN BUSCA DE ERRORES O MEJORAS
# COMENZAR CON EL PCA Y TAMBIEN IMPLEMENTAR EL TSNE
# PARA HALLAR EL TSNE ES RECOMENDABLE HACER PRIMERO EL PCA PERO TAMBIEN PUEDE NO SER NECESARIO