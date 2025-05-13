# counting_clean_group_proposals.py
import pandas as pd
import sys

def main(city, year):
    # Definir las rutas de los archivos
    input_file_title = f"./data/results3/{city}/{year}/prop_gpt.csv"
    input_file_title_description = f"./data/results3/{city}/{year}/prop_gpt_desc.csv"

    try:
        # Cargar las bases de datos desde los archivos CSV separados por "|"
        db_title = pd.read_csv(input_file_title, sep="|", decimal=',')
        db_title_description = pd.read_csv(input_file_title_description, sep="|" , decimal=',')

        # Ordenar las bases de datos por 'score' en orden descendente
        db_sorted_title = db_title.sort_values(by='score', ascending=False)
        db_sorted_title_description = db_title_description.sort_values(by='score', ascending=False)

        # Eliminar duplicados basados en 'id', manteniendo la entrada con mayor ranking
        db_cleaned_title = db_sorted_title.drop_duplicates(subset='id', keep='first')
        db_cleaned_title_description = db_sorted_title_description.drop_duplicates(subset='id', keep='first')

        db_cleaned_title = db_cleaned_title.copy()
        db_cleaned_title_description = db_cleaned_title_description.copy()
        # Guardar las bases de datos limpias
        for col in db_cleaned_title.select_dtypes(include=['float', 'int']).columns:
            db_cleaned_title[col] = db_cleaned_title[col].astype(object).apply(str)
            db_cleaned_title[col] = db_cleaned_title[col].apply(lambda x: x.replace('.', ','))
        db_cleaned_title.to_csv(f"./data/results3/{city}/{year}/cl_gpt.csv", sep="|", index=False)
        
        # Corregir el SettingWithCopyWarning para columnas con decimales
        for col in db_cleaned_title_description.select_dtypes(include=['float', 'int']).columns:
            db_cleaned_title_description[col] = db_cleaned_title_description[col].astype(object).apply(str)
            db_cleaned_title_description[col] = db_cleaned_title_description[col].apply(lambda x: x.replace('.', ','))

        db_cleaned_title_description.to_csv(f"./data/results3/{city}/{year}/cl_gpt_desc.csv", sep="|", index=False)

        # Agrupar por 'group' y contar las ocurrencias
        db_title_group = db_cleaned_title.groupby(["type", "group"]).size().reset_index(name="count")
        db_title_description_group = db_cleaned_title_description.groupby(["type", "group"]).size().reset_index(name="count")

        # Guardar los resultados agrupados
        db_title_group.to_csv(f"./data/results3/{city}/{year}/cl_count_gpt.csv", sep="|", index=False)
        db_title_description_group.to_csv(f"./data/results3/{city}/{year}/cl_count_gpt_desc.csv", sep="|", index=False)

        print(f"Procesamiento completado para {city} en {year}. Guardado cl_count_gpt.csv y cl_count_gpt_desc.csv")
    except Exception as e:
        print(f"Error procesando los archivos para {city} en {year}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python counting_clean_group_proposals.py <ciudad> <año>")
    else:
        city = sys.argv[1]
        year = sys.argv[2]
        main(city, year)
