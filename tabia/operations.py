import duckdb
from .base_ops import select, project, union, difference, product, intersect


def transpose(table):
    """
    Transpose a table, swapping rows and columns.
    """
    return project(table, ['j', 'i', 'v'])


def column(table, onval=None, index=None):
    """
    Select a specific column from the table based on a value in the 'v' column.
    """
    if onval is not None:
        return project(select(table * table, f"v1 = '{onval}' and j2 = j1"), ['i2', 'j2', 'v2'])
    return select(table, f"j = {index}")


def row(table, onval=None, index=None):
    """
    Select a specific row from the table based on a value in the 'i' column.
    """
    if onval is not None:
        return project(select(table * table, f"v1 = '{onval}' and i2 = i1"), ['i2', 'j2', 'v2'])
    return select(table, f"i = {index}")


def below(table, onval=None, index=None):
    """
    Select rows below a specific value in the 'v' column.
    """
    if onval is not None:
        return project(select(table * table, f"v1 = '{onval}' and i2 > i1"), ['i2', 'j2', 'v2'])
    return select(table, f"i > {index}")


def above(table, onval=None, index=None):
    """
    Select rows above a specific value in the 'v' column.
    """
    if onval is not None:
        return project(select(table * table, f"v1 = '{onval}' and i2 < i1"), ['i2', 'j2', 'v2'])
    return select(table, f"i < {index}")


def left(table, onval=None, index=None):
    """
    Select rows to the left of a specific value in the 'i' column.
    """
    if onval is not None:
        return project(select(table * table, f"v1 = '{onval}' and j2 < j1"), ['i2', 'j2', 'v2'])
    if index is not None:
        return select(table, f"j < {index}")
    # Left-most
    return table / project(select(table * table, 'j1 < j2'), ['i2', 'j2', 'v2'])


def right(table, onval=None, index=None):
    """
    Select rows to the right of a specific value in the 'i' column.
    """
    if onval is not None:
        return project(select(table * table, f"v1 = '{onval}' and j2 > j1"), ['i2', 'j2', 'v2'])
    if index is not None:
        return select(table, f"j > {index}")
    # Right-most
    return table / project(select(table * table, 'j1 > j2'), ['i2', 'j2', 'v2'])


def top(table):
    """
    Select the top-most row of the table.
    """
    return table / project(select(table * table, 'i1 < i2'), ['i2', 'j2', 'v2'])


def bottom(table):
    """
    Select the bottom-most row of the table.
    """
    return table / project(select(table * table, 'i1 > i2'), ['i2', 'j2', 'v2'])


def move(table, di, dj):
    """
    Move a table by a specified number of rows and columns.
    """
    return project(table, [f"i + {di}", f"j + {dj}", "v"])


def match(table1, table2):
    """
    Match two tables based on their values.
    """
    return select(table1 * table2, "v1 = v2")


def concat_vertically(table1, table2):
    """
    Concatenate two tables vertically.
    """
    c = bottom(table1) * top(table2) * table2
    return table1 + project(c, ['i3 + (i1 - i2) + 1', 'j3', 'v3'])


def concat_horizontally(table1, table2):
    """
    Concatenate two tables horizontally.
    """
    return transpose(transpose(table1) - transpose(table2))


def insert_column(table, column, index):
    """
    Insert a column into the table at a specified index.
    """
    return left(table, index=index) | column | right(table, index=index-1)


def insert_row(table, row, index):
    """
    Insert a row into the table at a specified index.
    """
    return above(table, index=index) - row - below(table, index=index-1)


def delete_column(table, index):
    """
    Delete a column from the table at a specified index.
    """
    return left(table, index=index) | right(table, index=index)


def delete_row(table, index):
    """
    Delete a row from the table at a specified index.
    """
    return above(table, index=index) - below(table, index=index)


def align_tops(table1, table2):
    """
    Align the tops of two tables.
    """
    return project(top(table1) * top(table2) * table2, ['i1 + (i2 - i3)', 'j3', 'v3'])


def concat_align_tops(table1, table2):
    """
    Concatenate two tables aligned at the top.
    """
    return table1 | align_tops(table1, table2)


def duplicate_column(table, index):
    """
    Duplicate a column in the table at a specified index.
    """
    return insert_column(table, column(table, index=index), index=index)


def height(table):
    """
    Calculate the height of the table.
    """
    #  HEIGHT(A) = πi2−i1,j2,v2 TOPM(A) × BOTTOMM(A)
    return project(
        top(table) * bottom(table) * table,
        ['i2 - i1 + 1', 'j3', 'v3']
    )


def width(table):
    """
    Calculate the width of the table.
    """
    return project(
        left(table) * right(table) * table,
        ['j2 - j1 + 1', 'i3', 'v3']
    )


def origin(table):
    """
    Move the table so that its upper-left corner is at (0, 0).
    """
    return project(table * top(table) * left(table), ['i1 - i2', 'j1 - j3', 'v1'])


def fold(indices, columns, values):
    """
    Fold a table with indices, columns, and values such that columns becomes rows.
    """
    ni = project(origin(indices) * origin(columns) * width(columns), ['i1 * i3 + j2', '0', 'v1'])
    nc = project(origin(indices) * origin(columns) * width(columns), ['i1 * i3 + j2', '0', 'v2'])
    nv = project(origin(values) * width(values), ['i1 * i2 + j1', '0', 'v1'])
    return ni | nc | nv



def align_rows(matching, table):
    """
    Align two tables vertically.
    """
    return project(select(matching * table, "i2 = i3"), ['i1', 'j3', 'v3'])


def join_rows(matching, table1, table2):
    """
    Join two tables on rows.
    """
    return table1 | align_rows(matching, table2)
