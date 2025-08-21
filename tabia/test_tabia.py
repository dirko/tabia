"""Tests for the tabia module."""

from .table import Table
from .base_ops import select, project, union, difference, product, intersect
from .operations import transpose, column, row, below, left, right, top, bottom, move, insert_column, delete_column, align_tops, \
    concat_align_tops, duplicate_column, height, fold, join_rows, fill1


def test_selection():
    a = Table([['desc', 'a'], ['school', 'b'], ['1', '2'], ['3', '4']])
    b = select(a, "v = 'school'")
    assert b.to_tuples() == [(1, 0, 'school')]


def test_product_project():
    a = Table([['desc', 'a'], ['school', 'b'], ['1', '2'], ['3', '4']])
    b = project(select(a * a, 'i1 = i2 and j1 < j2'), ['i1', 'j2', 'v1'])
    assert b.to_tuples() == [(0, 1, 'desc'), (1, 1, 'school'), (2, 1, '1'), (3, 1, '3')]


def test_transpose():
    a = Table([['desc', 'a'], ['school', 'b'], ['1', '2'], ['3', '4']])
    b = transpose(a)
    assert b.to_list() == [['desc', 'school', '1', '3'], ['a', 'b', '2', '4']]


def test_column():
    a = Table([['desc', 'a'], ['school', 'b'], ['1', '2'], ['3', '4']])
    b = column(a, onval='school')
    assert b.to_list() == [['desc'], ['school'], ['1'], ['3']]

    c = column(a, index=1)
    assert c.to_list() == [['a'], ['b'], ['2'], ['4']]


def test_below():
    a = Table([['desc', 'a'], ['school', 'b'], ['1', '2'], ['3', '4']])
    b = below(a, onval='school')
    assert b.to_list() == [['1', '2'], ['3', '4']]

    c = below(a, index=1)
    assert c.to_list() == [['1', '2'], ['3', '4']]


def test_left():
    a = Table([['desc', 'a'], ['school', 'b'], ['1', '2'], ['3', '4']])
    b = left(a, onval='school')
    assert b.to_list() == []

    c = left(a, index=1)
    assert c.to_list() == [['desc'], ['school'], ['1'], ['3']]

    d = left(a)
    assert d.to_list() == [['desc'], ['school'], ['1'], ['3']]


def test_top():
    a = Table([['desc', 'a'], ['school', 'b'], ['1', '2'], ['3', '4']])
    b = top(a)
    assert b.to_list() == [['desc', 'a']]


def test_bottom():
    a = Table([['desc', 'a'], ['school', 'b'], ['1', '2'], ['3', '4']])
    b = bottom(a)
    assert b.to_list() == [[None, None], [None, None], [None, None], ['3', '4']]


def test_move():
    a = Table([['desc', 'a'], ['school', 'b']])
    b = move(a, di=1, dj=1)
    assert b.to_tuples() == [(1, 1, 'desc'), (1, 2, 'a'), (2, 1, 'school'), (2, 2, 'b')]
    assert b.to_list() == [[None, None, None], [None, 'desc', 'a'], [None, 'school', 'b']]


def test_match():
    a = Table([['desc', 'a'], ['school', 'b'], ['1', '2'], ['3', '4']])
    b = Table([['school', 'b']])
    c = a % b
    assert c.to_tuples() == [(1, 0, 'school', 0, 0, 'school'), (1, 1, 'b', 0, 1, 'b')]


def test_concat_vertically():
    a = Table([['desc', 'a'], ['school', 'b']])
    b = Table([['1', '2'], ['3', '4']])
    c = a - b
    assert c.to_list() == [['desc', 'a'], ['school', 'b'], ['1', '2'], ['3', '4']]


def test_concat_horizontally():
    a = Table([['desc', 'a'], ['school', 'b']])
    b = Table([['1', '2'], ['3', '4']])
    c = a | b
    assert c.to_list() == [['desc', 'a', '1', '2'], ['school', 'b', '3', '4']]


def test_insert_column():
    a = Table([['desc', 'a'], ['school', 'b']])
    b = Table([['new_col1'], ['new_col2']])
    c = insert_column(a, b, index=1)
    assert c.to_list() == [['desc', 'new_col1', 'a'], ['school', 'new_col2', 'b']]


def test_delete_column():
    a = Table([['desc', 'a'], ['school', 'b']])
    c = delete_column(a, index=1)
    assert c.to_list() == [['desc'], ['school']]


def test_align_tops():
    a = Table([['desc', 'a'], ['school', 'b']])
    b = Table([['1', '2'], ['3', '4']])
    c = row(b, index=1)
    c = align_tops(a, c)
    assert c.to_list() == [['3', '4']]


def test_concat_align_tops():
    a = Table([['desc', 'a'], ['school', 'b']])
    b = row(a, index=1)
    c = concat_align_tops(a, b)
    assert c.to_list() == [['desc', 'a', 'school', 'b'], ['school', 'b', None, None]]


def test_duplicate_column():
    a = Table([['desc', 'a'], ['school', 'b']])
    b = duplicate_column(a, index=1)
    assert b.to_list() == [['desc', 'a', 'a'], ['school', 'b', 'b']]


def test_height():
    a = Table([['desc', 'a'], ['school', 'b']])
    h = height(a)
    assert sorted(h.to_tuples()) == [(2, 0, 'desc'), (2, 0, 'school'), (2, 1, 'a'), (2, 1, 'b')]


def test_fold():
    a = Table([[None, 'a', 'b'], ['A', '1', '2'], ['B', '3', '4']])
    i = below(column(a, index=0), index=0)
    c = right(row(a, index=0), index=0)
    v = right(below(a, index=0), index=0)
    d = fold(i, c, v)
    assert d.to_list() == [['A', 'a', '1'], ['A', 'b', '2'], ['B', 'a', '3'], ['B', 'b', '4']]


def test_join_rows():
    a = Table([['a', '1'], ['b', '2']])
    b = Table([['b', '3'], ['c', '4']])
    c = join_rows(a % b, a, b)
    assert c.to_list() == [['a', '1', None, None], ['b', '2', 'b', '3']]


def test_fill1():
    a = Table([['desc', 'a'], [None, 'b']])
    b = fill1(a)
    assert b.to_list() == [['desc', 'a'], ['desc', 'b'], [None, 'b']]