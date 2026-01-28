import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def describe_df(df):
    """
    Genera un DataFrame resumen con información descriptiva de cada variable.

    El DataFrame resultante contiene una columna por cada variable del DataFrame
    original y las siguientes filas: tipo de dato, porcentaje de valores nulos,
    número de valores únicos y porcentaje de cardinalidad.

    Argumentos:
    df (pandas.DataFrame): DataFrame de entrada.

    Retorna:
    pandas.DataFrame: DataFrame resumen con el formato especificado.
    """
      # Comprobación de que la entrada es un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Debes introducir un pandas DataFrame válido.")
        return None

    n_filas = len(df)   # Número de filas del DataFrame

 # Listas para almacenar la información de cada columna
    data_type = []
    missings = []
    unique_values = []
    cardin = []

    for col in df.columns:
        data_type.append(df[col].dtype)                         # Tipo de dato
        missings.append(round(df[col].isna().mean() * 100, 2))  # % de nulos
        unique = df[col].nunique()                              # Valores únicos
        unique_values.append(unique)
        cardin.append(round((unique / n_filas) * 100, 2))       # % cardinalidad

 # Construcción del DataFrame resumen
    resumen = pd.DataFrame(
        [
            data_type,
            missings,
            unique_values,
            cardin
        ],
        index=["DATA_TYPE", "MISSINGS (%)", "UNIQUE_VALUES", "CARDIN (%)"],
        columns=df.columns
    )

    return resumen

def tipifica_variables(df, umbral_categoria=None, umbral_continua=None):
    """
    Clasifica cada variable de un DataFrame según su tipo sugerido 
    (Binaria, Categórica, Numérica Discreta o Numérica Continua),
    utilizando reglas basadas en cardinalidad y porcentaje de cardinalidad.

    Argumentos:
    df (pd.DataFrame): DataFrame cuyas columnas se desean tipificar
    umbral_categoria (int): Número mínimo de valores únicos para considerar 
    que una variable deja de ser categórica
    umbral_continua (float): Porcentaje mínimo de cardinalidad (cardinalidad/n_filas)
    para considerar que una variable numérica es continua

    Retorna:
    pd.DataFrame: DataFrame con dos columnas:
    - "nombre_variable": nombre de cada columna del DF original.
    - "tipo_sugerido": tipo asignado según las reglas definidas.
    """
    #--------------------------inicio de las validaciones que se han añadido ahora
    #validar que df es df
    if not isinstance(df, pd.DataFrame):
        raise TypeError("El argumento 'df' debe ser un DF")
        return None
    #vlidar umbral_categoria (si se proporciona)
    if umbral_categoria is not None:
        #intentar convertir a int (captura letras, floats raros, etc)
        try:
            umbral_categoria = int(umbral_categoria) #mas que nada por si el usuario pone "10", la intencion es buena y lo transforma.
        except (ValueError, TypeError):                #se podría quitar
            raise ValueError("'umbral_categoria' debe ser un entero válido.")
            return None
        if umbral_categoria <= 0:
            raise ValueError("'umbral_categoria' debe ser mayor que 0.")
            return None

    #validar umbral_continua (si se proporciona)
    if umbral_continua is not None:
        #lo mismo de antes
        try:
            umbral_continua = float(umbral_continua)
        except (ValueError, TypeError):
            raise ValueError("'umbral_continua' debe ser un número (float) válido")
            return None
        if umbral_continua <= 0:
            raise ValueError("'umbral_continua' debe ser mayor que 0")
            return None


    #Obtenemos la descripción del DataFrame
    desc = describe_df(df) #Con la función de Lucía podemos integrarlo (ver con equipo)

    #esto lo he añadido para que sea opcional ponerlo.
    #CÁLCULO AUTOMÁTICO DE UMBRALES SOLO SI NO SE PROPORCIONAN
    if umbral_categoria is None or umbral_continua is None:

        #Cardinalidad de cada variable
        cardinalidades = desc.loc["UNIQUE_VALUES"] #como en la de lucia

        #Porcentaje de cardinalidad (convertido a %)
        porcentajes = desc.loc["CARDIN (%)"] / 100 #como en la de lucia

        #Percentil 75 como punto de corte natural
        if umbral_categoria is None:
            umbral_categoria = int(cardinalidades.quantile(0.75))  #debe ser int

        if umbral_continua is None:
            umbral_continua = float(porcentajes.quantile(0.75)) #debe ser float

    #Lista de amlacenamiento
    resultados = []

    #Recorremos cada col del df original
    for col in df.columns:

        cardinalidad = desc.loc["UNIQUE_VALUES", col] #como en la de lucia
        porcentaje = desc.loc["CARDIN (%)", col] / 100 #como en la de lucia

        #REGLAS

        if cardinalidad == 2:
            tipo= "Binaria"

        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"

        else:
            if porcentaje >= umbral_continua:
                tipo= "Numerica Continua"
            else:
                tipo= "Numerica Discreta"

        resultados.append({
            "nombre_variable": col,
            "tipo_sugerido": tipo
        })

    return pd.DataFrame(resultados)
    

def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Devuelve columnas numéricas con correlación alta respecto a un target para regresión.

    Selecciona las columnas numéricas del DataFrame cuya correlación (Pearson) con `target_col`
    sea superior en valor absoluto a `umbral_corr`. Si `pvalue` no es None, además exige que
    la correlación sea estadísticamente significativa (p-valor <= pvalue).

    Argumentos:
    df (pandas.DataFrame): DataFrame de entrada.
    target_col (str): Nombre de la columna objetivo (debe ser numérica y con alta cardinalidad).
    umbral_corr (float): Umbral de correlación absoluta, entre 0 y 1.
    pvalue (float | None): Umbral de p-valor (entre 0 y 1). Si es None, no se filtra por significancia.

    Retorna:
    list | None: Lista de columnas numéricas que cumplen el criterio o None si hay entradas inválidas.
    """


    # Check básico: df debe ser DataFrame
    if not isinstance(df, pd.DataFrame):
        print("df debe ser un pandas.DataFrame.")
        return None

    # Check básico: target_col debe existir
    if not isinstance(target_col, str) or target_col not in df.columns:
        print("target_col debe ser el nombre (str) de una columna existente del DataFrame.")
        return None

    # Check básico: umbral_corr entre 0 y 1
    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("umbral_corr debe ser un número entre 0 y 1.")
        return None

    # Check básico: pvalue None o entre 0 y 1
    if pvalue is not None and (not isinstance(pvalue, (int, float)) or not (0 <= pvalue <= 1)):
        print("pvalue debe ser None o un número entre 0 y 1.")
        return None

    # Check: el target debe ser numérico
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("target_col debe referenciar una columna numérica.")
        return None

    # Check: el target debe tener alta cardinalidad (criterio simple)
    if df[target_col].nunique(dropna=True) <= 10:
        print("target_col no parece una variable de regresión (baja cardinalidad).")
        return None

    # Seleccionar columnas numéricas y quitar el target de candidatas
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    # Si no hay candidatas numéricas
    if len(num_cols) == 0:
        return []

    # Caso sin pvalue: filtrar solo por correlación
    if pvalue is None:
        corr = df[num_cols].corrwith(df[target_col]).dropna()
        return corr[corr.abs() > umbral_corr].index.tolist()

    # Caso con pvalue: filtrar por correlación y significancia
    try:
        from scipy.stats import pearsonr
    except Exception:
        print("No se pudo importar scipy para calcular p-valores. Usa pvalue=None o instala scipy.")
        return None

    selected = []
    for col in num_cols:
        tmp = df[[target_col, col]].dropna()

        # Pearson necesita variabilidad y suficientes datos
        if len(tmp) < 3:
            continue
        if tmp[target_col].nunique() < 2 or tmp[col].nunique() < 2:
            continue

        r, p = pearsonr(tmp[target_col], tmp[col])

        # Filtrar por umbral de correlación y p-valor
        if abs(r) > umbral_corr and p <= pvalue:
            selected.append(col)

    return selected


def plot_features_num_regression(
    df,
    target_col="",
    columns=None,
    umbral_corr=0,
    pvalue=None):
    """
    Selecciona variables numéricas significativas según correlación y p-value,
    y genera pairplots en grupos de máximo 5 columnas (incluyendo target_col).

    Si 'columns' NO está vacía:
        → solo evalúa esas columnas
        → solo plotea las que cumplan correlación y pvalue
        → devuelve solo esas columnas

    Si 'columns' está vacía:
        → usa todas las numéricas excepto target_col
        → aplica los mismos criterios
        → devuelve solo las que cumplan

    Si ninguna cumple:
        → devuelve []
        → no plotea nada
    """

    #  VALIDACIONES 

    # df debe ser DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un DataFrame.")
        return None

    # target_col debe ser string no vacío
    if not isinstance(target_col, str) or target_col == "":
        print("Error: 'target_col' debe ser un string no vacío.")
        return None

    # target_col debe existir
    if target_col not in df.columns:
        print(f"Error: la columna '{target_col}' no existe en el DataFrame.")
        return None

    # target_col debe ser numérica continua
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: 'target_col' debe ser una variable numérica continua.")
        return None

    # Validación de columns
    if columns is not None:
        if not isinstance(columns, list):
            print("Error: 'columns' debe ser una lista de strings o None.")
            return None
        for col in columns:
            if col not in df.columns:
                print(f"Error: la columna '{col}' indicada en 'columns' no existe en el DataFrame.")
                return None
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"Error: la columna '{col}' no es numérica y no puede evaluarse.")
                return None

    # Validación de umbral_corr
    try:
        umbral_corr = float(umbral_corr)
    except:
        print("Error: 'umbral_corr' debe ser un número válido.")
        return None
   
    if umbral_corr < 0:
        print("Error: 'umbral_corr' no puede ser negativo.")
        return None

    # Validación de pvalue
    if pvalue is not None:
        try:
            pvalue = float(pvalue)
        except:
            print("Error: 'pvalue' debe ser un número válido o None.")
            return None
        if pvalue <= 0 or pvalue >= 1:
            print("Error: 'pvalue' debe estar entre 0 y 1.")
            return None

    #  DETERMINAR COLUMNAS A EVALUAR 

    if columns is None or len(columns) == 0:
        # usar todas las numéricas excepto target
        columns = df.select_dtypes(include="number").columns.tolist()
        if target_col in columns:
            columns.remove(target_col)

    #  SELECCIÓN SEGÚN CRITERIOS 

    # obtener todas las columnas significativas según correlación/pvalue
    selected_all = get_features_num_regression(
        df=df,
        target_col=target_col,
        umbral_corr=umbral_corr,
        pvalue=pvalue
    )

    if selected_all is None:
        print("Error: get_features_num_regression detectó valores no válidos.")
        return None

    # quedarnos solo con las columnas de 'columns' que cumplen criterios
    selected = [c for c in columns if c in selected_all]

    if len(selected) == 0:
        print("Ninguna de las columnas indicadas cumple los criterios.")
        return []

    # ---------------- GENERAR PAIRPLOTS ----------------

    cols_grafica = [target_col] + selected
    max_cols = 5

    for i in range(0, len(cols_grafica), max_cols - 1):
        subset = [target_col] + cols_grafica[i+1:i + (max_cols - 1)]
        if len(subset) > 1:
            sns.pairplot(df[subset], diag_kind="hist")
            plt.show()

    return selected      


### Funcion: get_features_cat_regression

# Esta función recibe como argumentos un dataframe, el nombre de una de las columnas del mismo (argumento 'target_col'), que debería ser el target de un hipotético modelo de regresión, es decir debe ser una variable numérica continua o discreta pero con alta cardinalidad y una variable float "pvalue" cuyo valor por defecto será 0.05.

# La función debe devolver una lista con las columnas categóricas del dataframe cuyo test de relación con la columna designada por 'target_col' supere en confianza estadística el test de relación que sea necesario hacer (es decir la función debe poder escoger cuál de los dos test que hemos aprendido tiene que hacer).

# La función debe hacer todas las comprobaciones necesarias para no dar error como consecuecia de los valores de entrada. Es decir hará un check de los valores asignados a los argumentos de entrada y si estos no son adecuados debe retornar None y printar por pantalla la razón de este comportamiento. Ojo entre las comprobaciones debe estar que "target_col" hace referencia a una variable numérica continua del dataframe.

def get_features_cat_regression(df,target_col = '',umbral_categoria=10, umbral_continua=0.5, pvalue=0.05):
    """
    La funcion te permite realizar el test de las variables categóricas y elige las variables mas optimas en funcion del pvalue.

    Parámetros:
    df (pandas.DataFrame): DataFrame de entrada.
    target_col: El usuario debe de poner el nombre de la columna que considere como la target.
    umbral_categoria (int): Número mínimo de valores únicos para considerar que una variable deja de ser categórica
    umbral_continua (float): Porcentaje mínimo de cardinalidad (cardinalidad/n_filas) para considerar que una variable numérica es continua
    pvalue (float): Umbral de p-valor (entre 0 y 1). Está predeterminado como 0.5, es decir, IC del 95%

        """
    # Librerias:
    from pandas.api.types import (is_numeric_dtype,is_bool_dtype,is_datetime64_any_dtype)
    from scipy.stats import (ttest_ind, f_oneway)
    # Llamada de otra funcion:
    tipos = tipifica_variables(df,umbral_categoria, umbral_continua)
    
    
    columns_cat = tipos.loc[tipos['tipo_sugerido'] == "Categórica","nombre_variable"].to_list()
    
    # Validacion de inputs:
    no_col = []

    if isinstance(df, pd.DataFrame): # 1
        if target_col in df.columns: # 2
            if is_numeric_dtype(df[target_col]) and not is_bool_dtype(df[target_col]) and not is_datetime64_any_dtype(df[target_col]): # 3
                for col in columns_cat:
                    if col in df.columns:
                        continue
                    elif col not in df.columns: # 4.2
                        no_col.append(col)
                        print(f'ERROR #4.2: Las columnas: {no_col}, no se encuentra en tu DataFrame.')
                        return None
                    else: # 4.1
                        print(f'ERROR #4.1: Columnas no comprobadas correctamente.')
                        return None
                       
            else:
                print('ERROR #3: Su target_col no es numerica. Por favor vuelva a revisarlo.')
                return None
        else:
            print('ERROR #2: Su target no se encuentra entre las columnas del dataframe.')
        
    else:
        print('ERROR #1: El grupo de datos que debemos de analizar no es considerado un DataFrame, por favor introduzca un DataFrame.')
        return None
    

    if type(pvalue) != str: # 5
        if pvalue >= 0 and pvalue <= 1: # 6
            print('Validacion de inputs terminada, podemos seguir...')
        else:
            print('ERROR #6: El pvalue introducido no esta entre los valores marcados (0 - 1). Porfavor vuelva a repetirlo.')
            return None
    else:
        print('ERROR #5: El pvalue introducido es erroneo, por favor, introduzca un numero.')
        return None


    # Seleccion de features categoricas para regresion.
    
    X_validas = []
    for col in columns_cat:
        if df[col].nunique() < 2:
            print('No consideramos como categorica.')
            continue
        elif df[col].nunique() == 2:
                catego = df[col].dropna().unique() 
                grupo_1 = df[df[col] == catego[0]][target_col]
                grupo_2 = df[df[col] == catego[1]][target_col]
                stat_ttest,pvalue_ttest = ttest_ind(grupo_1,grupo_2)
                print(f'Para la columna {col}, su rtdo es: Stat: {stat_ttest}p-value: {pvalue_ttest} con un IC del {1-pvalue}\n')
                if pvalue_ttest < pvalue:
                    X_validas.append(col)
            
        else:
            grupos = []
            for categoria in df[col].dropna().unique():
                grupos.append(df[df[col] == categoria][target_col])
            stat_anova,pvalue_anova = f_oneway(*grupos) # El * es para desempaquetar la lista formadas por mas listas.
            print(f'Para la columna {col}, su rtdo es: Stat: {stat_anova}p-value: {pvalue_anova} con un IC del {1-pvalue}\n')
            if pvalue_anova < pvalue:
                X_validas.append(col)
        print(f'Nuestras features válidas son: {X_validas}')
        
    return X_validas


# ### Funcion: plot_features_cat_regression
# Esta función recibe un dataframe, una argumento "target_col" con valor por defecto "", una lista de strings ("columns") cuyo valor por defecto es la lista vacía, un argumento ("pvalue") con valor 0.05 por defecto y un argumento "with_individual_plot" a False.

# Si la lista no está vacía, la función pintará los histogramas agrupados de la variable "target_col" para cada uno de los valores de las variables categóricas incluidas en columns que cumplan que su test de relación con "target_col" es significatio para el nivel 1-pvalue de significación estadística. La función devolverá los valores de "columns" que cumplan con las condiciones anteriores. 

# Si la lista está vacía, entonces la función igualará "columns" a las variables numéricas del dataframe y se comportará como se describe en el párrafo anterior.

# De igual manera que en la función descrita anteriormente deberá hacer un check de los valores de entrada y comportarse como se describe en el último párrafo de la función `get_features_cat_regression`.

def plot_features_cat_regression(df,columns=[],umbral_categoria=10, umbral_continua=0.5,target_col = '',pvalue = 0.05, with_individual_plot=False):

    """
    El objetivo de la funcion es sacarme una representacion grafica de las X_validas categoricas que nos introduzca el usuario, en el caso que no introduzca, 
    las columnas que se utiliza son las numericas haciendo un test para seleccionar las columnas optimas para su representación. 
    
    Descripción de los argumentos:
    df: DataFrame que contiene los datos a analizar.
    columns: lista de variables categóricas a evaluar. Si está vacía, la función analiza automáticamente las variables numéricas del DataFrame.
    umbral_categoria: valor utilizado para clasificar una variable como categórica en función de su cardinalidad.
    umbral_continua: porcentaje de cardinalidad a partir del cual una variable numérica se considera continua.
    target_col: nombre de la variable objetivo, que debe ser numérica.
    pvalue: nivel de significación estadística utilizado en los tests (por defecto 0.05).
    with_individual_plot: si es True, se generan visualizaciones individuales para cada variable seleccionada.     
    
    """

    # Librerias
    import seaborn as sns
    from pandas.api.types import (is_numeric_dtype,is_bool_dtype,is_datetime64_any_dtype)
    from scipy.stats import (ttest_ind, f_oneway, pearsonr)


    # Validacion de inputs:
    if isinstance(df, pd.DataFrame): # 1
        if target_col in df.columns: # 2
            if is_numeric_dtype(df[target_col]) and not is_bool_dtype(df[target_col]) and not is_datetime64_any_dtype(df[target_col]): # 3 (entiendo que aqui es tambien numerica)
                print('OK, Dataframe y target.')         
            else:
                print('ERROR #3: Su target_col no es numerica. Por favor vuelva a revisarlo.')
                return None
        else:
            print('ERROR #2: Su target no se encuentra entre las columnas del dataframe.')
        
    else:
        print('ERROR #1: El grupo de datos que debemos de analizar no es considerado un DataFrame, por favor introduzca un DataFrame.')
        return None
    
    if type(pvalue) != str: # 5
        if pvalue >= 0 and pvalue <= 1: # 6
            print('pvalue es OK')
        else:
            print('ERROR #6: El pvalue introducido no esta entre los valores marcados (0 - 1). Porfavor vuelva a repetirlo.')
            return None
    else:
        print('ERROR #5: El pvalue introducido es erroneo, por favor, introduzca un numero.')
        return None
    
    # Seleccion de variables categóricas:

    
    # Seleccion de variables numéricas:
    tipos = tipifica_variables(df,umbral_categoria, umbral_continua)
    
    selected_num = tipos.loc[(tipos['tipo_sugerido'] != "Categórica") & (tipos['tipo_sugerido'] != "Binaria"),"nombre_variable"].to_list() 

    if target_col in selected_num:
        selected_num.remove(target_col)
 

    # CASO A: Existen columnas categoricas:
    col_validas = []
    no_col = [] 
    X_validasA = []
    if type(columns) == int:
        print('ERROR #7: La columna introducida no es tipo string')  
        return None  
    if columns != []:            
        for col in columns:               
            if col in df.columns:
                col_validas.append(col)
            else: # 4
                no_col.append(col)
                print(f'ERROR #4: Las columnas: {no_col}, no se encuentra en tu DataFrame.')
                return None
        print(f'columnas validas que se encuentran en el Dataframe: {col_validas}')

        for col in col_validas:
            if df[col].nunique() < 2:
                print('No consideramos como categorica.')
                continue
            elif df[col].nunique() == 2:
                    catego = df[col].dropna().unique() 
                    grupo_1 = df[df[col] == catego[0]][target_col].dropna()
                    grupo_2 = df[df[col] == catego[1]][target_col].dropna()
                    stat_ttest,pvalue_ttest = ttest_ind(grupo_1,grupo_2)
                    print(f'Para la columna {col}, su rtdo es: Stat: {stat_ttest}p-value: {pvalue_ttest} con un IC del {1-pvalue}\n')
                    if pvalue_ttest < pvalue:
                        X_validasA.append(col)
                
            else:
                grupos = []

                categorias = df[col].dropna().unique()
                for categoria in categorias:
                    grupo = df[df[col] == categoria][target_col].dropna()
                    grupos.append(grupo)

                stat_anova,pvalue_anova = f_oneway(*grupos) # El * es para desempaquetar la lista formadas por mas listas.
                print(f'Para la columna {col}, su rtdo es: Stat: {stat_anova}p-value: {pvalue_anova} con un IC del {1-pvalue}\n')
                if pvalue_anova < pvalue:
                    X_validasA.append(col)
        print(f'Nuestras features válidas son: {X_validasA}')

        if with_individual_plot is True:
            for col in X_validasA:
                print(f'Relacion de {target_col} con {col}:\n')
                sns.pairplot(df[X_validasA + [target_col]], diag_kind="hist")  
                plt.xlabel(col)
                plt.ylabel(target_col)
                plt.show()
        else:
            print(f'Relacion de {target_col} con {X_validasA}:\n')
            sns.pairplot(df[X_validasA + [target_col]], diag_kind="hist")
            plt.xlabel(X_validasA)
            plt.ylabel(target_col)
            plt.show()

        return X_validasA

    else:
        selected_cat = get_features_cat_regression(df,target_col,umbral_categoria, umbral_continua, pvalue)
        if selected_cat != []:
            for col in selected_cat:
                if col in df.columns:
                    col_validas.append(col)
                else: # 4
                    no_col.append(col)
                    print(f'ERROR #4: Las columnas: {no_col}, no se encuentra en tu DataFrame.')
                    return None
            print(f'columnas validas que se encuentran en el Dataframe: {col_validas}')

        # Seleccion de features:

            X_validasA = []
            for col in col_validas:
                if df[col].nunique() < 2:
                    print('No consideramos como categorica.')
                    continue
                elif df[col].nunique() == 2:
                        catego = df[col].dropna().unique() 
                        grupo_1 = df[df[col] == catego[0]][target_col].dropna()
                        grupo_2 = df[df[col] == catego[1]][target_col].dropna()
                        stat_ttest,pvalue_ttest = ttest_ind(grupo_1,grupo_2)
                        print(f'Para la columna {col}, su rtdo es: Stat: {stat_ttest}p-value: {pvalue_ttest} con un IC del {1-pvalue}\n')
                        if pvalue_ttest < pvalue:
                            X_validasA.append(col)
                
                else:
                    grupos = []

                    categorias = df[col].dropna().unique()
                    for categoria in categorias:
                        grupo = df[df[col] == categoria][target_col].dropna()
                        grupos.append(grupo)

                    stat_anova,pvalue_anova = f_oneway(*grupos) # El * es para desempaquetar la lista formadas por mas listas.
                    print(f'Para la columna {col}, su rtdo es: Stat: {stat_anova}p-value: {pvalue_anova} con un IC del {1-pvalue}\n')
                    if pvalue_anova < pvalue:
                        X_validasA.append(col)
            print(f'Nuestras features válidas son: {X_validasA}')
            
            
            # PLOTS PARA VARIABLES CATEGÓRICAS
            if with_individual_plot is True:
                for col in X_validasA:
                    print(f'Relacion de {target_col} con {col}:\n')
                    sns.pairplot(df[X_validasA + [target_col]], diag_kind="hist")  
                    plt.xlabel(col)
                    plt.ylabel(target_col)
                    plt.show()
            else:
                print(f'Relacion de {target_col} con {X_validasA}:\n')
                sns.pairplot(df[X_validasA + [target_col]], diag_kind="hist")
                plt.xlabel(X_validasA)
                plt.ylabel(target_col)
                plt.show()

            return X_validasA




    # CASO B: No existen columnas categoricas validas.
        else:
            columns_val = []
            for col in selected_num:
                if is_numeric_dtype(df[col]) and not is_bool_dtype(df[col]) and not is_datetime64_any_dtype(df[col]): # 8
                    columns_val.append(col)
                    continue
                else:
                    continue
            print(f'Como al llamar la funcion no has introducido una lista de columns, se ha añadido las variables numericas que son: {columns_val}')
            # Seleccion de features:    

            X_validasB = []

            for col in columns_val:
                corr, pvalue_col = pearsonr(df[target_col], df[col])
                print(f'Para la columna {col}, tiene una correlacion del {corr} y un p-value del {pvalue_col} en un IC del {1-pvalue}')
                if pvalue_col < pvalue:
                    X_validasB.append(col)

            print(f'Las features elegidas son: {X_validasB}')

            # PLOTS PARA VARIABLES NUMÉRICAS
            if with_individual_plot is True:
                for col in X_validasB:
                    print(f'Relacion de {target_col} con {col}:\n')
                    sns.pairplot(df[X_validasB + [target_col]], diag_kind="hist")                
                    plt.xlabel(col)
                    plt.ylabel(target_col)
                    plt.show()
            else:
                print(f'Relacion de {target_col} con {X_validasB}:\n')
                sns.pairplot(df[X_validasB + [target_col]], diag_kind="hist")
                plt.show()
                    
            return X_validasB
