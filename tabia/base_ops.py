import duckdb
from .table import Table


def select(table, condition):
    """
    Select rows from a table based on a condition.

    Args:
        table (duckdb.Table): The input table.
        condition (str): The SQL condition to filter rows.

    Returns:
        duckdb.Table: A new table with the selected rows.
    """
    data = table.data
    return Table(duckdb.sql(f"SELECT * FROM data WHERE {condition}"))


def _normalise_columns(rel):
    """
    Rename the columns. Should always triples of in, jn, vn, where n is some integer,
    or can be left out if n=1.
    for t1 * t2, n should be 1 and 2, while for a matching with already renamed columns m (with 1 and 2),
    t1 * m should result in 1 2 and 3. (i.e. 1 and 2 renamed to 2 and 3) and 1 inserted.
    """
    ncols = len(rel.columns)
    if ncols % 3 != 0:
        raise ValueError("Relation must have a column count that is a multiple of 3.")
    if ncols == 3:
        return duckdb.sql(f'select "{rel.columns[0]}" as i, "{rel.columns[1]}" as j, "{rel.columns[2]}" as v from rel')
    renamed_columns = [f"{attribute}{tablenr}" for tablenr in range(1, ncols//3 + 1) for attribute in ["i", "j", "v"]]
    aliases = ','.join(f'"{col}" as {newcol}' for col, newcol in zip(rel.columns, renamed_columns))
    return duckdb.sql(f'select {aliases} from rel')


def product(table1, table2):
    """
    Compute the Cartesian product of two tables.

    Args:
        table1 (duckdb.Table): The first input table.
        table2 (duckdb.Table): The second input table.

    Returns:
        duckdb.Table: A new table that is the Cartesian product of the two input tables.
    """
    data1 = table1.data
    data2 = table2.data
    rel = duckdb.sql(f"SELECT COLUMNS(d1.*) AS 'l_\\0', COLUMNS(d2.*) AS 'r_\\0' FROM data1 d1 CROSS JOIN data2 d2")
    rel = _normalise_columns(rel)
    return Table(rel)


def union(table1, table2):
    """
    Union two tables.

    Args:
        table1 (duckdb.Table): The first input table.
        table2 (duckdb.Table): The second input table.

    Returns:
        duckdb.Table: A new table that is the union of the two input tables.
    """
    data1 = table1.data
    data2 = table2.data
    rel = duckdb.sql(f"SELECT * FROM data1 UNION ALL SELECT * FROM data2")
    return Table(rel)


def intersect(table1, table2):
    """
    Intersect two tables.

    Args:
        table1 (duckdb.Table): The first input table.
        table2 (duckdb.Table): The second input table.

    Returns:
        duckdb.Table: A new table that is the intersection of the two input tables.
    """
    rel = duckdb.sql(f"SELECT * FROM {table1.data} INTERSECT SELECT * FROM {table2.data}")
    return Table(rel)


def difference(table1, table2):
    """
    Compute the difference between two tables.

    Args:
        table1 (duckdb.Table): The first input table.
        table2 (duckdb.Table): The second input table.

    Returns:
        duckdb.Table: A new table that contains rows from the first table that are not in the second.
    """
    data1 = table1.data
    data2 = table2.data
    rel = duckdb.sql(f"SELECT * FROM data1 EXCEPT SELECT * FROM data2")
    return Table(rel)


def project(table, columns):
    """
    Project specific columns from a table.

    Args:
        table (duckdb.Table): The input table.
        columns (list): List of column names to project.

    Returns:
        duckdb.Table: A new table with only the specified columns.
    """
    data = table.data
    rel = duckdb.sql(f"SELECT distinct {', '.join(columns)} FROM data")
    rel = _normalise_columns(rel)
    return Table(rel)

