import pandas as pd
import sqlite3
from sqlalchemy import create_engine, inspect
import tempfile
import os
import openpyxl

def generate_excel_columns(n):
    """
    Generate Excel-style column names (A, B, ..., Z, AA, AB, ...).
    
    Args:
        n (int): Number of column names to generate.
    
    Returns:
        list: List of column names.
    """
    columns = []
    for i in range(n):
        name = ""
        while True:
            name = chr(65 + (i % 26)) + name
            i //= 26
            if i == 0:
                break
        columns.append(name)
    return columns

def get_excel_sheets(file):
    """
    Get a list of sheet names from an Excel file.
    
    Args:
        file (UploadedFile): Streamlit UploadedFile object (Excel .xlsx or .xls file).
    
    Returns:
        list: List of sheet names.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    wb = openpyxl.load_workbook(tmp_file_path, read_only=True)
    sheet_names = wb.sheetnames
    wb.close()
    os.unlink(tmp_file_path)

    if not sheet_names:
        raise Exception("No sheets found in the Excel file")
    return sheet_names

def load_file_data(file, file_type, has_headers=True, excel_sheet=None):
    """
    Load data from a file into a pandas DataFrame.
    
    Args:
        file (UploadedFile): Streamlit UploadedFile object.
        file_type (str): Type of file ('csv', 'json', 'excel').
        has_headers (bool): Whether the file contains column headers.
    
    Returns:
        pandas DataFrame: Loaded data.
    """

    if file_type == "csv":
        if has_headers:
            df = pd.read_csv(file)
        else:
            df = pd.read_csv(file, header=None)
            df.columns = generate_excel_columns(len(df.columns))
    elif file_type == "json":
        df = pd.read_json(file)
        if not has_headers:
            df.columns = generate_excel_columns(len(df.columns))
    elif file_type == "excel":
        file.seek(0)
        preview = pd.read_excel(
            file,
            sheet_name=excel_sheet,
            nrows=5,
            engine="openpyxl"
        )
        valid_columns = [
            col for col in preview.columns
            if not col.startswith("Unnamed") and not preview[col].isna().all()
        ]
        
        file.seek(0)
        if has_headers:
            df = pd.read_excel(
                file,
                sheet_name=excel_sheet,
                usecols=valid_columns,
                engine="openpyxl"
            )
        else:
            df = pd.read_excel(
                file,
                sheet_name=excel_sheet,
                header=None,
                usecols=range(len(valid_columns)),
                engine="openpyxl"
            )
            df.columns = generate_excel_columns(len(df.columns))
        
        df = df.dropna(axis=1, how="all")
        df = df.loc[:, ~df.columns.str.contains("^Unnamed", na=False)]
    elif file_type == "text":
        df = pd.read_csv(file, delimiter="\t")
    
    return df

    
def get_sqlite_tables(file):
    """
    Get a list of table names from a SQLite database file.
    
    Args:
        file (UploadedFile): Streamlit UploadedFile object (SQLite .db file).
    
    Returns:
        list: List of table names, or empty list if none found or error occurs.
    """
    try:
        # Create a temporary file to store the uploaded SQLite database
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
            # Write the uploaded file's contents to the temporary file
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        # Connect to the SQLite database using the temporary file path
        conn = sqlite3.connect(tmp_file_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        conn.close()

        # Delete the temporary file
        os.unlink(tmp_file_path)

        if tables.empty:
            raise Exception("No tables found in the database")
        return tables["name"].tolist()

    except Exception as e:
        raise Exception(f"Error accessing SQLite database: {e}")

def load_sqlite_data(file_type, file, sqlite_table):
    """
    Load data from a specified SQLite table into a pandas DataFrame.
    
    Args:
        file_type (str): Type of file (e.g., 'sqlite').
        file (UploadedFile): Streamlit UploadedFile object (SQLite .db file).
        sqlite_table (str): Name of the table to query.
    
    Returns:
        pandas DataFrame: Data from the specified table.
    """
    if file_type != "sqlite":
        raise ValueError("File type must be 'sqlite' for this function")
    if not sqlite_table:
        raise ValueError("SQLite table name must be provided")

    try:
        # Create a temporary file to store the uploaded SQLite database
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp_file:
            # Write the uploaded file's contents to the temporary file
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name

        # Connect to the SQLite database using the temporary file path
        conn = sqlite3.connect(tmp_file_path)
        df = pd.read_sql(f"SELECT * FROM {sqlite_table}", conn)
        conn.close()

        # Delete the temporary file
        os.unlink(tmp_file_path)

        return df

    except Exception as e:
        raise Exception(f"Error loading data from table '{sqlite_table}': {e}")

def get_db_tables(db_type, user, password, host, port, database):
    """
    Get a list of table names from an external database.
    
    Args:
        db_type (str): Database type ('postgresql', 'mysql', 'sqlserver').
        user (str): Database username.
        password (str): Database password.
        host (str): Database host.
        port (str): Database port.
        database (str): Database name.
    
    Returns:
        list: List of table names.
    """
    if db_type == "postgresql":
        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    elif db_type == "mysql":
        conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    elif db_type == "sqlserver":
        conn_str = f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    else:
        raise ValueError("Unsupported database type.")
    
    engine = create_engine(conn_str)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    engine.dispose()
    if not tables:
        raise Exception("No tables found in the database")
    return tables

def load_db_data(db_type, user, password, host, port, database, table_name=None):
    """
    Load data from a database into a pandas DataFrame.
    
    Args:
        db_type (str): Database type ('postgresql', 'mysql', 'sqlserver').
        user (str): Database username.
        password (str): Database password.
        host (str): Database host.
        port (str): Database port.
        database (str): Database name.
        table_name (str, optional): Table name to load (generates SELECT * FROM table_name).
    
    Returns:
        tuple: (DataFrame, None)
    """
    
    if db_type == "postgresql":
        conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    elif db_type == "mysql":
        conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    elif db_type == "sqlserver":
        conn_str = f"mssql+pyodbc://{user}:{password}@{host}:{port}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    else:
        raise ValueError("Unsupported database type.")
    
    engine = create_engine(conn_str)
    if table_name:
        query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df