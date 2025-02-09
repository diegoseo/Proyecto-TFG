# Procesamiento de Espectros Raman

Este documento describe los conceptos y métodos clave para el procesamiento de espectros Raman, incluyendo **normalización**, **suavizado** y **corrección de Shirley**.

---

## 1. Normalización

La **normalización** ajusta los valores de un espectro para que estén en una escala común, facilitando la comparación entre espectros.

### **Propósito:**
- Comparar espectros que tienen diferentes escalas de intensidad global.
- Eliminar efectos de variaciones absolutas en la intensidad, como diferencias en la concentración o potencia del láser.

### **Métodos de Normalización:**

| **Método**           | **Cómo funciona**                                                                                      | **Aplicación principal**                                                                   |
|-----------------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **Escalado Min-Max**  | Escala los valores al rango \([0,1]\), dividiendo entre el rango \((\text{máximo} - \text{mínimo})\). | Comparar espectros con diferentes rangos de intensidad.                                   |
| **Z-Score**           | Escala los valores para que tengan media 0 y desviación estándar 1.                                  | Análisis estadístico (e.g., PCA o clustering).                                            |
| **Por Área**          | Divide los valores por el área total bajo la curva.                                                  | Comparar la forma general de los espectros eliminando diferencias absolutas de intensidad.|
| **Por Máximo**        | Divide los valores por el valor máximo del espectro.                                                 | Resaltar las intensidades relativas de los picos más altos.                               |

### **Ejemplo de Aplicación:**
- Comparar la forma de los espectros independientemente de la intensidad absoluta.
- Preparar los datos para análisis como PCA o ajuste de picos.

---

## 2. Suavizado

El **suavizado** reduce el ruido en los espectros manteniendo las características principales, como los picos.

### **Propósito:**
- Eliminar fluctuaciones rápidas causadas por el ruido.
- Mejorar la claridad visual de los espectros sin distorsionar los picos.

### **Métodos de Suavizado:**

| **Método**            | **Cómo funciona**                                                                                   | **Aplicación principal**                                                                  |
|------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Savitzky-Golay**     | Ajusta un polinomio local en ventanas móviles y calcula un valor suavizado para cada punto.         | Ideal para espectros con picos estrechos, preserva bordes y detalles.                    |
| **Filtro Gaussiano**   | Aplica una convolución con una función gaussiana que asigna mayor peso a los puntos cercanos.        | Adecuado para espectros con ruido aleatorio más disperso.                                |

### **Ejemplo de Aplicación:**
- Mejorar la claridad de los espectros Raman eliminando ruido instrumental.
- Preparar datos para análisis de picos o comparación visual.

---

## 3. Corrección de Shirley

La **corrección de Shirley** elimina el fondo no lineal que aparece en los espectros, mejorando la visibilidad de los picos.

### **Propósito:**
- Eliminar el fondo generado por efectos secundarios, como fluorescencia o emisiones inelásticas.
- Mejorar la precisión del análisis de picos (posición, altura, ancho).

### **Cómo funciona:**
- Ajusta iterativamente el fondo para que el área debajo de la curva ajustada sea igual al área encima.
- Resta este fondo al espectro original, dejando solo los picos.

### **Ejemplo de Aplicación:**
- Identificar picos Raman eliminando fondos de fluorescencia.
- Análisis en espectros XPS para determinar estados electrónicos precisos.

---

## Diferencias Clave

| **Aspecto**               | **Normalización**                                         | **Suavizado**                                      | **Corrección de Shirley**                          |
|----------------------------|----------------------------------------------------------|---------------------------------------------------|---------------------------------------------------|
| **Propósito**              | Ajustar los valores a una escala común para comparar espectros. | Reducir ruido en los datos preservando los picos importantes. | Eliminar el fondo no lineal que afecta la visibilidad de los picos. |
| **Métodos**                | Min-Max, Z-Score, Área, Máximo.                          | Savitzky-Golay, Filtro Gaussiano.                 | Iterativo, basado en el área acumulativa.         |
| **Afecta Intensidades Absolutas** | Sí, ajusta los valores pero mantiene relaciones proporcionales. | No, solo suaviza las fluctuaciones.              | Sí, resta el fondo y puede alterar las intensidades absolutas. |
| **Preserva la Forma Relativa** | Depende del método (por área sí, Min-Max y Z-Score pueden alterar proporciones). | Sí, preserva picos y características generales.   | Sí, pero elimina el fondo superpuesto.           |
| **Cuándo Usar**            | Comparar formas de espectros o preparar datos para análisis multivariado. | Reducir ruido antes de análisis de picos.         | Eliminar fondos para un análisis más preciso de los picos. |

---

## ¿Cuál usar y cuándo?

1. **Normalización:**
   - Cuando necesitas comparar espectros en diferentes escalas o preparar datos para análisis estadístico.

2. **Suavizado:**
   - Cuando los espectros tienen ruido significativo y necesitas mejorar la claridad.

3. **Corrección de Shirley:**
   - Cuando hay un fondo superpuesto que afecta la interpretación de los picos.
