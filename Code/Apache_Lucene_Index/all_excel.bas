Sub GenerateExcelForCitiesAndYears()
    Dim Cities As Variant
    Dim Years As Variant
    Dim FolderPath As String
    Dim City As Variant ' Cambiado a Variant
    Dim Year As Variant ' Cambiado a Variant
    Dim WB As Workbook
    Dim CSVFolder As String
    Dim OutputPath As String
    Dim FileName As String
    Dim CSVFullPath As String
    Dim WS As Worksheet
    Dim SheetName As String
    Dim TableName As String
    Dim MaxFiles As Long

    ' Configuración inicial
    Cities = Array("cambridge", "miami", "newyork")
    Years = Array("2014", "2015", "2016", "2017")
    FolderPath = "C:\Users\molin\Documents\TFG_Info\Code\Apache_Lucene_Index\data\results\"
    OutputPath = "C:\Users\molin\Documents\TFG_Info\Code\GeneratedExcels\" ' Ruta donde se guardarán los libros de Excel

    ' Crear carpeta de salida si no existe
    If Dir(OutputPath, vbDirectory) = "" Then
        MkDir OutputPath
    End If

    ' Iterar sobre ciudades y años
    For Each City In Cities
        For Each Year In Years
            ' Crear un nuevo libro
            Set WB = Application.Workbooks.Add

            ' Definir la carpeta de los CSV
            CSVFolder = FolderPath & City & "\" & Year & "\"
            
            ' Verificar si la carpeta existe
            If Dir(CSVFolder, vbDirectory) = "" Then
                MsgBox "La carpeta " & CSVFolder & " no existe. Se omite esta combinación.", vbExclamation
                GoTo NextCombination
            End If

            ' Establece un límite de archivos para evitar bloqueos
            MaxFiles = 50 ' Ajusta según el rendimiento esperado
            
            ' Obtener el primer archivo CSV del directorio
            FileName = Dir(CSVFolder & "*.csv")
            
            ' Procesar los archivos CSV
            Do While FileName <> "" And MaxFiles > 0
                ' Generar la ruta completa del archivo
                CSVFullPath = CSVFolder & FileName
                
                ' Generar el nombre de la hoja desde el nombre del archivo (sin extensión)
                SheetName = Left(FileName, InStrRev(FileName, ".") - 1)
                
                ' Crear una nueva hoja
                On Error Resume Next
                Set WS = WB.Sheets.Add
                WS.Name = SheetName
                On Error GoTo 0
                
                ' Importar el CSV al rango A1 de la nueva hoja
                With WS.QueryTables.Add(Connection:="TEXT;" & CSVFullPath, Destination:=WS.Range("A1"))
                    .TextFileParseType = xlDelimited
                    .TextFileOtherDelimiter = "|" ' Separador de columnas
                    .TextFileConsecutiveDelimiter = False
                    .TextFileColumnDataTypes = Array(1) ' Formato general para todas las columnas
                    .Refresh BackgroundQuery:=False
                End With
                
                ' Eliminar la conexión activa de la consulta
                On Error Resume Next
                WS.QueryTables(1).Delete
                On Error GoTo 0
                
                ' Crear una tabla estructurada a partir de los datos
                Dim LastRow As Long, LastCol As Long
                LastRow = WS.Cells(WS.Rows.Count, 1).End(xlUp).Row
                LastCol = WS.Cells(1, WS.Columns.Count).End(xlToLeft).Column
                
                ' Definir el rango de la tabla
                Dim TableRange As Range
                Set TableRange = WS.Range(WS.Cells(1, 1), WS.Cells(LastRow, LastCol))
                
                ' Asignar un nombre único a la tabla
                TableName = SheetName & "_Table"
                
                ' Crear la tabla
                WS.ListObjects.Add(xlSrcRange, TableRange, , xlYes).Name = TableName
                
                ' Reducir el contador de archivos
                MaxFiles = MaxFiles - 1
                
                ' Obtener el siguiente archivo
                FileName = Dir
            Loop
            
            ' Guardar el libro de Excel con el nombre adecuado
            WB.SaveAs Filename:=OutputPath & City & "_" & Year & ".xlsx", FileFormat:=xlOpenXMLWorkbook
            WB.Close SaveChanges:=False
            
NextCombination:
        Next Year
    Next City
    
    MsgBox "Proceso completado. Los archivos de Excel se han generado en " & OutputPath, vbInformation
End Sub
