# Definir las ciudades, años y datasets
$ciudades = @("cambridge", "miami", "newyork")
$anios = @("2014", "2015", "2016", "2017")
$datasets = @(
    "Cambridge-2014", "Cambridge-2015", "Cambridge-2016", "Cambridge-2017",
    "Miami-2014", "Miami-2015", "Miami-2016", "Miami-2017",
    "NewYorkCity-2014", "NewYorkCity-2015", "NewYorkCity-2016", "NewYorkCity-2017"
)

# Ruta base para la base de datos y los índices
$indexBasePath = "./data/indexes2"

# Ruta al código fuente y binarios
$libPath = "lib\lucene-9.5.0\modules\*;lib\*;."
$binPath = "bin"
$srcPath = @("src\db\*.java", "src\entities\*.java", "src\fairness\DUSAProposalAnnotator.java")

# Expandir las rutas de los archivos fuente
$sourceFiles = foreach ($path in $srcPath) { Get-ChildItem -Path $path | ForEach-Object { '"' + $_.FullName + '"' } }

# Convertir las rutas a un string separado por espacios
$sourceFilesString = $sourceFiles -join " "

# Compilar el código
Write-Host "Compilando el código..."
javac -cp "lib\lucene-9.5.0\modules\*;lib\*;." -d bin src\db\*.java src\entities\*.java src\fairness\DUSAProposalAnnotator.java

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error durante la compilación." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Compilación completada con éxito." -ForegroundColor Green

# Ejecutar el programa para cada combinación de ciudad, año y dataset
foreach ($ciudad in $ciudades) {
    foreach ($anio in $anios) {
        # Buscar el dataset correspondiente
        $dataset = $datasets | Where-Object { $_ -match "$ciudad" -and $_ -match "$anio" }

        # Verificar si se encontró el dataset
        if (-not $dataset) {
            Write-Host "No se encontró dataset para Ciudad: $ciudad, Año: $anio" -ForegroundColor Yellow
            continue
        }

        # Generar rutas dinámicas
        $indexPath = "$indexBasePath/index$($ciudad.Substring(0,1).ToUpper() + $ciudad.Substring(1))$anio/"

        Write-Host "Ejecutando para Ciudad: $ciudad, Año: $anio, Dataset: $dataset"
        java -cp "lib\lucene-9.5.0\modules\*;lib\*;$binPath" fairness.DUSAProposalAnnotator $ciudad $anio $dataset $indexPath

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error al ejecutar para Ciudad: $ciudad, Año: $anio" -ForegroundColor Red
        } else {
            Write-Host "Ejecución completada para Ciudad: $ciudad, Año: $anio" -ForegroundColor Green
        }
    }
}
