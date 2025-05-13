Sub ImportCSVsToTablesWithSheetNames()
    Dim FolderPath As String
    Dim FileName As String
    Dim WS As Worksheet
    Dim WB As Workbook
    Dim CSVFullPath As String
    Dim SheetName As String
    Dim MaxFiles As Long ' Número máximo de archivos a procesar
    Dim TableName As String ' Nombre de la tabla en cada hoja

    ' Define la ruta de la carpeta que contiene los archivos CSV
    FolderPath = "C:\Users\molin\Documents\TFG_Info\Code\Apache_Lucene_Index\data\results2\cambridge\2014\"

    ' Verifica si la carpeta existe
    If Dir(FolderPath, vbDirectory) = "" Then
        MsgBox "La carpeta especificada no existe. Verifica la ruta.", vbCritical
        Exit Sub
    End If

    ' Establece un límite de archivos para evitar bloqueos
    MaxFiles = 50 ' Ajusta según el rendimiento esperado

    ' Obtener el primer archivo CSV del directorio
    FileName = Dir(FolderPath & "*.csv")

    ' Crear un nuevo libro de trabajo
    Set WB = ThisWorkbook

    ' Iterar sobre los archivos CSV en la carpeta
    Do While FileName <> "" And MaxFiles > 0
        ' Generar la ruta completa del archivo
        CSVFullPath = FolderPath & FileName

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

    MsgBox "Importación completada con formato tabla. Archivos procesados con nombres de hojas: " & 50 - MaxFiles, vbInformation
End Sub

