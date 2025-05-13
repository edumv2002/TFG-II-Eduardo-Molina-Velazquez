# Definir las ciudades y los años
$ciudades = @("cambridge", "miami", "newyork")
$anios = @("2014", "2015", "2016", "2017")

# Ruta al script de Python
$pythonScript = "./data/treat_data/counting_group_proposals.py"

# Iterar sobre las combinaciones de ciudad y año
foreach ($ciudad in $ciudades) {
    foreach ($anio in $anios) {
        Write-Host "Ejecutando para Ciudad: $ciudad, Año: $anio"
        python $pythonScript $ciudad $anio

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error al procesar Ciudad: $ciudad, Año: $anio" -ForegroundColor Red
        } else {
            Write-Host "Procesamiento completado para Ciudad: $ciudad, Año: $anio" -ForegroundColor Green
        }
    }
}
