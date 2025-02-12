#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:09:36 2025

@author: diego
"""

print("EJECUTABLE 2")
import plotly.express as px

# 📊 Crear gráfico 3D con Plotly
fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3',
                     color='Tipo',  # Asigna colores según el tipo
                     title="PCA en 3D con Plotly",
                     labels={"PC1": "Componente 1", "PC2": "Componente 2", "PC3": "Componente 3"})

fig.show()









      label_x = "".join([f"PC{c+1}" for c in componentes_x])  
      label_y = "".join([f"PC{c+1}" for c in componentes_y]) 
      label_z = "".join([f"PC{c+1}" for c in componentes_z]) 
      
     
      colores_pca_original = [asignacion_colores.get(type_, 'black') for type_ in types]

      
      # Crear la figura en Plotly
      fig = go.Figure()
      
      # Agregar el gráfico de dispersión 3D
      fig.add_trace(go.Scatter3d(
          x=dato_pca[:, 0],  # PC1
          y=dato_pca[:, 1],  # PC2
          z=dato_pca[:, 2],  # PC3
          mode='markers',
          marker=dict(
              size=5,
              color=colores_pca_original,  # Colores asignados a cada punto
              opacity=0.7
          )
      ))
      
      # Configurar los ejes
      fig.update_layout(
          title=f'Análisis de Componentes Principales 3D de {archivo_nombre}',
          scene=dict(
              xaxis_title=label_x,
              yaxis_title=label_y,
              zaxis_title=label_z
          ),
          margin=dict(l=0, r=0, b=0, t=40)  # Ajustar los márgenes
      )
      
      # Mostrar la gráfica interactiva
      fig.show()
      

