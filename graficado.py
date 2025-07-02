from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
import pandas as pd

class GraficarEspectros(QWidget):
    def __init__(self, datos, raman_shift, asignacion_colores):
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
