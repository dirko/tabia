import pandas as pd
import duckdb


def list_to_ijv(table, start=0, skip_none=False):
    rows = [
        (i + start, j + start, v)
        for i, row in enumerate(table)
        for j, v in enumerate(row)
        if not (skip_none and (v is None or v == ""))
    ]
    return pd.DataFrame(rows, columns=["i", "j", "v"])


def df_to_ijv(df, start=0, skip_na=False):
    a = df.to_numpy()
    rows = [(i + start, j + start, a[i, j])
            for i in range(a.shape[0]) for j in range(a.shape[1])
            if not (skip_na and pd.isna(a[i, j]))]
    return pd.DataFrame(rows, columns=["i", "j", "v"])


def ijv_to_df(ijv: pd.DataFrame, fill_value=None) -> pd.DataFrame:
    i0, j0 = ijv["i"].min(), ijv["j"].min()
    M = (ijv.assign(i=ijv["i"] - i0, j=ijv["j"] - j0)
         .pivot_table(index="i", columns="j", values="v", aggfunc="last")
         .sort_index().sort_index(axis=1))
    if fill_value is not None:
        M = M.fillna(fill_value)
    M.index = range(M.shape[0])
    M.columns = range(M.shape[1])
    return M


def ijv_to_lists(ijv: pd.DataFrame, fill_value=None, start=None):
    if ijv.empty:
        return []
    if start is None:
        i0, j0 = ijv["i"].min(), ijv["j"].min()
    else:
        i0, j0 = start, start
    nrows = ijv["i"].max() - i0 + 1
    ncols = ijv["j"].max() - j0 + 1
    grid = [[fill_value for _ in range(ncols)] for _ in range(nrows)]
    for i, j, v in ijv[["i", "j", "v"]].itertuples(index=False):
        grid[i - i0][j - j0] = v
    return grid


class Table:
    """
    A table represented as (i,j,v) triples.
    """

    def __init__(self, data):
        if isinstance(data, list):
            self.data = list_to_ijv(data, skip_none=True)
        elif isinstance(data, pd.DataFrame):
            if set(data.columns) == {'i', 'j', 'v'}:
                self.data = data
            else:
                self.data = df_to_ijv(data, skip_na=True)
        elif isinstance(data, duckdb.DuckDBPyRelation):
            self.data = data
        data = self.data
        self.data = duckdb.sql('select * from data')  # Internally always keep as rel

    def __add__(self, other):
        """
        Add two tables together.
        Args:
            other (Table): Another Table instance to add.
        Returns:
            Table: A new Table instance containing the combined data.
        """
        if isinstance(other, Table):
            from .base_ops import union
            return union(self, other)
        else:
            raise TypeError("Can only add another Table instance.")

    def __truediv__(self, other):
        """
        Subtract another table from this one.
        Args:
            other (Table): Another Table instance to subtract.
        Returns:
            Table: A new Table instance containing the difference.
        """
        if isinstance(other, Table):
            from .base_ops import difference
            return difference(self, other)
        else:
            raise TypeError("Can only subtract another Table instance.")

    def __mul__(self, other):
        """
        Multiply this table with another table.
        Args:
            other (Table): Another Table instance to multiply with.
        Returns:
            Table: A new Table instance containing the product.
        """
        if isinstance(other, Table):
            from .base_ops import product
            return product(self, other)
        else:
            raise TypeError("Can only multiply with another Table instance.")

    def __and__(self, other):
        """
        Intersect this table with another table.
        Args:
            other (Table): Another Table instance to intersect with.
        Returns:
            Table: A new Table instance containing the intersection.
        """
        if isinstance(other, Table):
            from .base_ops import intersect
            return intersect(self, other)
        else:
            raise TypeError("Can only intersect with another Table instance.")

    def __or__(self, other):
        """
        Horizontal concatenation of two tables.
        """
        if isinstance(other, Table):
            from .operations import concat_horizontally
            return concat_horizontally(self, other)
        else:
            raise TypeError("Can only concatenate with another Table instance.")

    def __sub__(self, other):
        """
        Horizontal concatenation of two tables.
        """
        if isinstance(other, Table):
            from .operations import concat_vertically
            return concat_vertically(self, other)
        else:
            raise TypeError("Can only concatenate with another Table instance.")

    def __mod__(self, other):
        """
        Match two tables based on their values.
        """
        if isinstance(other, Table):
            from .operations import match
            return match(self, other)
        else:
            raise TypeError("Can only match with another Table instance.")

    def to_list(self):
        """
        Convert the table to a list of lists.
        Returns:
            list: A list of lists representing the table.
        """
        return ijv_to_lists(self.data.to_df(), start=0)

    def to_df(self, fill_value=None):
        """
        Convert the table to a pandas DataFrame.
        Args:
            fill_value: Value to fill in missing entries.
        Returns:
            pd.DataFrame: A DataFrame representation of the table.
        """
        return ijv_to_df(self.data.to_df(), fill_value)

    def to_tuples(self):
        """
        Convert the table to a list of tuples.
        Returns:
            list: A list of tuples representing the table.
        """
        return self.data.fetchall()

    def __repr__(self):
        return repr(self.to_df())

    # For jupyter repr
    def _repr_html_(self):
        return self.to_df()._repr_html_()

