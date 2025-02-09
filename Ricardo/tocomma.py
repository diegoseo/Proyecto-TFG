import csv
import os

def convertir_csv():
    try:
        # Solicitar el nombre del archivo de entrada
        input_file = input("Ingresa el nombre del archivo CSV separado por ';' (incluye la extensión .csv): ").strip()

        # Verificar si el archivo existe
        if not os.path.exists(input_file):
            print(f"El archivo {input_file} no existe. Por favor verifica el nombre e inténtalo de nuevo.")
            return

        # Generar el nombre del archivo de salida
        base_name, _ = os.path.splitext(input_file)
        output_file = f"{base_name}_COMMA.csv"

        # Leer y convertir el archivo CSV, ignorando caracteres nulos
        with open(input_file, mode='r', encoding='latin-1', errors='ignore') as infile:
            reader = csv.reader((line.replace('\0', '') for line in infile), delimiter=';')
            rows = list(reader)

        with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            writer.writerows(rows)

        print(f"Archivo convertido con éxito. Guardado como: {output_file}")
    except Exception as e:
        print(f"Se produjo un error: {e}")

# Ejecutar la función
convertir_csv()
