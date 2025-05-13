# counting_group_proposals.py
import pandas as pd
import sys

def main(city, year):
    # Definir rutas de los archivos
    input_file_title_gpt = f"./data/results3/{city}/{year}/prop_gpt.csv"
    input_file_title_gpt_description = f"./data/results3/{city}/{year}/prop_gpt_desc.csv"

    try:
        # Cargar las bases de datos desde los archivos CSV separados por "|"
        db_title_gpt = pd.read_csv(input_file_title_gpt, sep="|", decimal=',')
        db_title_gpt_description = pd.read_csv(input_file_title_gpt_description, sep="|", decimal=',')

        # Contar el número de propuestas por grupo
        db_title_gpt_group = db_title_gpt.groupby(["type", "group"]).size().reset_index(name="count")
        db_title_gpt_description_group = db_title_gpt_description.groupby(["type", "group"]).size().reset_index(name="count")

        db_title_gpt_group = db_title_gpt_group.copy()
        db_title_gpt_description_group = db_title_gpt_description_group.copy()
        # Guardar los resultados en nuevos archivos
        for col in db_title_gpt_group.select_dtypes(include=['float', 'int']).columns:
            db_title_gpt_group[col] = db_title_gpt_group[col].astype(object).apply(str)
            db_title_gpt_group[col] = db_title_gpt_group[col].apply(lambda x: x.replace('.', ','))
        
        for col in db_title_gpt_description_group.select_dtypes(include=['float', 'int']).columns:
            db_title_gpt_description_group[col] = db_title_gpt_description_group[col].astype(object).apply(str)
            db_title_gpt_description_group[col] = db_title_gpt_description_group[col].apply(lambda x: x.replace('.', ','))
        db_title_gpt_group.to_csv(f"./data/results3/{city}/{year}/ncl_count_gpt.csv", sep="|", index=False)
        db_title_gpt_description_group.to_csv(f"./data/results3/{city}/{year}/ncl_count_gpt_desc.csv", sep="|", index=False)

        print(f"Procesamiento completado para {city} en {year}.")
    except Exception as e:
        print(f"Error procesando los archivos para {city} en {year}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python counting_group_proposals.py <ciudad> <año>")
    else:
        city = sys.argv[1]
        year = sys.argv[2]
        main(city, year)
