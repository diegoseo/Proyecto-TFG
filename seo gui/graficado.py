from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


class GraficarEspectros(QWidget):
    
    def __init__(self, datos, raman_shift, asignacion_colores):
        super().__init__()
        print("ENTRO EN GRAFICAR ESPECTRO 1")
        self.setWindowTitle("Gráfico de Espectros")
        self.resize(800, 600)
        print("ENTRO EN GRAFICAR ESPECTRO 2")
        layout = QVBoxLayout()
        self.setLayout(layout)
        print("ENTRO EN GRAFICAR ESPECTRO 3")
        self.plot_widget = pg.PlotWidget()
        print("ENTRO EN GRAFICAR ESPECTRO 4")
        self.plot_widget.addLegend()
        self.plot_widget.setLabel('left', 'Intensidad')
        self.plot_widget.setLabel('bottom', 'Raman Shift')
        layout.addWidget(self.plot_widget)

        print("ENTRO EN GRAFICAR ESPECTRO 5")
        
        leyendas_tipos = set()  # almacenamos los tipos que enocontramos y la funcion set() nos ayuda a quer no se repitan

        datos = datos.iloc[:,1:] # APARTAMOS LA PRIMERA COLUMNA DE LONGITUDES DE ONDAS

        tipos = datos.iloc[0, :]    # Fila 0 tiene los tipos (collagen, DNA, etc.)
        intensidades = datos.iloc[1:, :].copy()  # Desde la fila 1 en adelante son datos

        intensidades.columns = tipos.values  # Cambiar nombres de columnas a sus tipos

        intensidades = intensidades.astype(float) # Convertimos a valores numéricos

        datos = intensidades

        print("DATOS NUEVO")
        print(datos)


        leyendas_tipos = set() # ACA GUARDAMOS LOS NOMBRES DE LOS TIPOS SIN QUE SE REPITAN
        tipos_unicos = datos.columns.unique()
        x = np.array(raman_shift, dtype=float)  # Convertimos el eje X (Raman shift) a un array de floats

        for tipo in tipos_unicos: #BUSCA TODAS LAS COINCIDENCIAS EN TODAS LAS POSICIONES CON TIPOS_UNICOS
            indices = [i for i, col in enumerate(datos.columns) if col == tipo]

            for idx in indices:
                y_fila = datos.iloc[:, idx] # extraemos todas las intensidades

                if isinstance(y_fila, pd.DataFrame):
                    y_fila = y_fila.iloc[:, 0]

                try:
                    y = np.array(y_fila, dtype=float).flatten()
                    color_actual = asignacion_colores.get(tipo, "#FFFFFF") # ASIGNAMOS UN COLOR POR DEFECTO
                    pen = pg.mkPen(color=color_actual, width=0.3)

                    if tipo in leyendas_tipos:
                        self.plot_widget.plot(x, y, pen=pen) # Graficar sin leyenda
                    else:
                        self.plot_widget.plot(x, y, pen=pen, name=tipo)
                        leyendas_tipos.add(tipo) # Agregar el tipo a la lista de ya graficados con leyenda

                except Exception as e:
                    print(f"Error al graficar columna {idx} ({tipo}): {e}")



class GraficarEspectrosAcotados(QWidget):
    def __init__(self, datos, raman_shift, asignacion_colores, val_min,val_max):
        super().__init__()
        self.setWindowTitle("Gráfico Acotado")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend()
        self.plot_widget.setLabel('left', 'Intensidad')
        self.plot_widget.setLabel('bottom', 'Raman Shift')
        layout.addWidget(self.plot_widget)

        print("min val = ", val_min)
        print("max val = ", val_max)

        # raman = datos.iloc[1:,0] # SELECIONAMOS LA LONG DE ONDAS
        # print("RAMAN")
        # print(raman)

        datos = datos.iloc[:,1:] # APARTAMOS LA PRIMERA COLUMNA DE LONGITUDES DE ONDAS
        print("Datos:")
        print(datos)

        tipos = datos.iloc[0, :]    # Fila 0 tiene los tipos (collagen, DNA, etc.)
        print("TIPOS:")
        print(tipos)
        
        intensidades = datos.iloc[1:, :].copy()  # Desde la fila 1 en adelante son datos

        intensidades.columns = tipos.values  # Cambiar nombres de columnas a sus tipos

        intensidades = intensidades.astype(float) # Convertimos a valores numéricos

        print("INTENSIDADES")
        print(intensidades)

        datos = intensidades

        print("DATOS NUEVO")
        print(datos)
                
        leyendas_tipos = set()  # Guardamos los tipos sin repetir
        tipos_unicos = datos.columns.unique()
        x_total = np.array(raman_shift, dtype=float)  # Eje X completo

        mascara = (x_total >= val_min) & (x_total <= val_max)
        x_filtrado = x_total[mascara]

        for tipo in tipos_unicos:
            indices = [i for i, col in enumerate(datos.columns) if col == tipo] # separa el indice cuando el nombre de columna es igual al tipo actual
            # print("INDICES")
            # print(indices)
            for x in indices:
                y_fila = datos.iloc[:, x]

                if isinstance(y_fila, pd.DataFrame):
                    y_fila = y_fila.iloc[:, 0]

                try:
                    y_total = np.array(y_fila, dtype=float).flatten()

                    # Aplicar el mismo filtro al eje Y
                    y_filtrado = y_total[mascara]

                    color_actual = asignacion_colores.get(tipo, "#FFFFFF")
                    pen = pg.mkPen(color=color_actual, width=0.3)

                    if tipo in leyendas_tipos:
                        self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen)
                    else:
                        self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen, name=tipo)
                        leyendas_tipos.add(tipo)

                except Exception as e:
                    print(f"Error al graficar columna {x} ({tipo}): {e}")


                

class GraficarEspectrosTipos(QWidget):
    def __init__(self, datos, raman_shift, asignacion_colores, tipo_deseado):
        super().__init__()
        self.setWindowTitle("Gráfico de Espectros")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend()
        self.plot_widget.setLabel('left', 'Intensidad')
        self.plot_widget.setLabel('bottom', 'Raman Shift')
        layout.addWidget(self.plot_widget)

        leyendas_tipos = set()  # almacenamos los tipos que enocontramos y la funcion set() nos ayuda a quer no se repitan

        datos = datos.iloc[:,1:] # APARTAMOS LA PRIMERA COLUMNA DE LONGITUDES DE ONDAS

        tipos = datos.iloc[0, :]    # Fila 0 tiene los tipos (collagen, DNA, etc.)
        intensidades = datos.iloc[1:, :].copy()  # Desde la fila 1 en adelante son datos

        intensidades.columns = tipos.values  # Cambiar nombres de columnas a sus tipos

        intensidades = intensidades.astype(float) # Convertimos a valores numéricos

        datos = intensidades

        print("DATOS NUEVO")
        print(datos)


        leyendas_tipos = set() # ACA GUARDAMOS LOS NOMBRES DE LOS TIPOS SIN QUE SE REPITAN
        tipos_unicos = datos.columns.unique()
        x = np.array(raman_shift, dtype=float)  # Convertimos el eje X (Raman shift) a un array de floats

        indices = [i for i, col in enumerate(datos.columns) if col == tipo_deseado] # LINEA IMPORTANTE
        for index in indices:
                y_fila = datos.iloc[:, index] # extraemos todas las intensidades

                if isinstance(y_fila, pd.DataFrame):
                    y_fila = y_fila.iloc[:, 0]

                try:
                    y = np.array(y_fila, dtype=float).flatten()
                    color_actual = asignacion_colores.get(tipo_deseado, "#FFFFFF") # ASIGNAMOS UN COLOR POR DEFECTO
                    pen = pg.mkPen(color=color_actual, width=0.3)

                    if tipo_deseado in leyendas_tipos:
                        self.plot_widget.plot(x, y, pen=pen) # Graficar sin leyenda
                    else:
                        self.plot_widget.plot(x, y, pen=pen, name=tipo_deseado)
                        leyendas_tipos.add(tipo_deseado) # Agregar el tipo a la lista de ya graficados con leyenda

                except Exception as e:
                    print(f"Error al graficar columna {index} ({tipo_deseado}): {e}")



class GraficarEspectrosAcotadoTipos(QWidget):
    def __init__(self, datos, raman_shift, asignacion_colores,tipo_deseado,val_min,val_max):
        super().__init__()
        self.setWindowTitle("Gráfico Acotado")
        self.resize(800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.addLegend()
        self.plot_widget.setLabel('left', 'Intensidad')
        self.plot_widget.setLabel('bottom', 'Raman Shift')
        layout.addWidget(self.plot_widget)

        print("min val = ", val_min)
        print("max val = ", val_max)

        # raman = datos.iloc[1:,0] # SELECIONAMOS LA LONG DE ONDAS
        # print("RAMAN")
        # print(raman)

        datos = datos.iloc[:,1:] # APARTAMOS LA PRIMERA COLUMNA DE LONGITUDES DE ONDAS
        print("Datos:")
        print(datos)

        tipos = datos.iloc[0, :]    # Fila 0 tiene los tipos (collagen, DNA, etc.)
        print("TIPOS:")
        print(tipos)
        
        intensidades = datos.iloc[1:, :].copy()  # Desde la fila 1 en adelante son datos

        intensidades.columns = tipos.values  # Cambiar nombres de columnas a sus tipos

        intensidades = intensidades.astype(float) # Convertimos a valores numéricos

        print("INTENSIDADES")
        print(intensidades)

        datos = intensidades

        print("DATOS NUEVO")
        print(datos)
                
        leyendas_tipos = set()  # Guardamos los tipos sin repetir
        tipos_unicos = datos.columns.unique()
        x_total = np.array(raman_shift, dtype=float)  # Eje X completo

        mascara = (x_total >= val_min) & (x_total <= val_max)
        x_filtrado = x_total[mascara]

        indices = [i for i, col in enumerate(datos.columns) if col == tipo_deseado] # separa el indice cuando el nombre de columna es igual al tipo actual
            # print("INDICES")
            # print(indices)
        for x in indices:
            y_fila = datos.iloc[:, x]

            if isinstance(y_fila, pd.DataFrame):
                y_fila = y_fila.iloc[:, 0]

            try:
                y_total = np.array(y_fila, dtype=float).flatten()

                # Aplicar el mismo filtro al eje Y
                y_filtrado = y_total[mascara]

                color_actual = asignacion_colores.get(tipo_deseado, "#FFFFFF")
                pen = pg.mkPen(color=color_actual, width=0.3)

                if tipo_deseado in leyendas_tipos:
                    self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen)
                else:
                    self.plot_widget.plot(x_filtrado, y_filtrado, pen=pen, name=tipo_deseado)
                    leyendas_tipos.add(tipo_deseado)

            except Exception as e:
                print(f"Error al graficar columna {x} ({tipo_deseado}): {e}")



    # def aplicar_transformaciones(self):
    #     df = self.df.copy()

    #     if self.chk_normalizar.isChecked():
    #         df = (df - df.mean()) / df.std()

    #     if self.chk_suavizar.isChecked():
    #         df = df.rolling(window=3, min_periods=1).mean()

    #     if self.chk_derivada.isChecked():
    #         df = df.diff().fillna(0)

    #     # Guardar CSV con opciones aplicadas
    #     nombre_archivo, _ = QFileDialog.getSaveFileName(self, "Guardar archivo CSV", "", "CSV (*.csv)")
    #     if nombre_archivo:
    #         try:
    #             df.to_csv(nombre_archivo, index=False, header=True)
    #             QMessageBox.information(self, "Éxito", f"Archivo guardado como: {nombre_archivo}")
    #         except Exception as e:
    #             QMessageBox.critical(self, "Error", f"No se pudo guardar el archivo: {e}")





"""
Metodo = K-Nearest Neighbors (KNN)
accuracy = porcentaje de aciertos
- Mide cuantas predicciones hace bien un modelo respecto al total
Ejemplo = Si clasificas 10 muestras y acertas 9, el accuracy = 90%   

FUNCIONA BUSCANDO LOS VECINOS MAS CERCANO A UN PUNTO, POR EJEMPLO SI TENEMOS UN CONJUNTO DE PUNTOS ROJOS Y AZULES
Y LUEGO AGREGAMOS OTRO PUNTO Y QUEREMOS SABER DE QUE COLOR SERA ESE PUNTO APLICAMOS KNN, POR EJEMPLO K = 5 QUE SELECCIONARA
LOS 5 PUNTOS MAS CERCANOS A ESE PUNTO Y CONTARA CUANTOS ROJOS Y AZULES HAY PARA PODER ELEGIR EL COLOR DEL PUNTO NUEVO QUE SERA 
EL MAYOR 
YOUTUBE: https://www.youtube.com/watch?v=gs9E7E0qOIc
"""

# def calcular_accuracy(dataframe_pca, etiquetas):

#     #Recibe el DataFrame PCA reducido y las etiquetas de clase.
#     #Devuelve el porcentaje de aciertos usando KNN.
    
#     # Asegura que solo se usen columnas numéricas para el modelo
#     columnas_numericas = [col for col in dataframe_pca.columns if dataframe_pca[col].dtype in [np.float64, np.float32, np.int64]]
    
#     X = dataframe_pca[columnas_numericas].values
#     y = etiquetas.values

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#     clf = KNeighborsClassifier(n_neighbors=3) # BUSCAMOS LOS 3 VECINOS MAS CERCANOS
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     return round(accuracy * 100, 2)

def calcular_accuracy(dataframe_pca, etiquetas):
    #Recibe el DataFrame PCA reducido y las etiquetas de clase.Devuelve el porcentaje de aciertos usando KNN.
    print("dataframe_pca")
    print(dataframe_pca)
    columnas_numericas = [col for col in dataframe_pca.columns if dataframe_pca[col].dtype in [np.float64, np.float32, np.int64]] # Asegura que solo se usen columnas numéricas para el modelo
    
    X = dataframe_pca[columnas_numericas]
    y = etiquetas

    # Combinamos X e y en un solo DataFrame para eliminar filas con NaN
    df_completo = X.copy()
    df_completo["__etiqueta__"] = y

    df_completo = df_completo.dropna() # Eliminamos cualquier fila que tenga NaN

    # Separamos nuevamente X e y
    X_clean = df_completo.drop(columns=["__etiqueta__"]).values
    y_clean = df_completo["__etiqueta__"].values

    # Escaladomos (ver si hay que volver a escalar esta parte)
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X_clean)
    X_scaled = X_clean  # Ya están escalados
    
    # División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clean, test_size=0.3, random_state=42)

    # Clasificador KNN
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    # Predicción y accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return round(accuracy * 100, 2)

