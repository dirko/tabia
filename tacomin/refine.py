import time

import numpy as np
from scipy.sparse import load_npz
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import duckdb


def load_data(x_file, top_file, k):
    # Load from disk
    print('Loading')
    x = load_npz(x_file)
    topo = np.load(top_file.format(k=k))
    if 'arr_0' in topo:
        top = topo['arr_0']
    else:
        top = topo['top']
    print('Loaded', x.shape, top.shape)

    x_lil = x.tolil()
    x_set = [
        set(row)
        for row in x_lil.rows
    ]
    return top, x_set


def subset(top, x_set, k=20):
    topa = top
    scores = []

    for i, xs in enumerate(tqdm.tqdm(x_set)):
        for j in range(k):
            j_ind = topa[i, j]
            sa = x_set[i] & x_set[j_ind]
            scores.append((i, j_ind, len(sa)))

    # Sorting
    scores = [(i, j, sa) for i ,j, sa in scores]

    scores = sorted(scores, key=lambda x: x[2], reverse=True)
    return scores

def moves(scores, parquet_tall_dir, n_contains=5, n_move_size=5):
    # Now find the moves by joining on value
    res = duckdb.sql(f"""
    with A as (
        select * from '{parquet_tall_dir}/df_*.parquet'
        where fs in ({','.join([str(s[0]) for s in scores if s[2] >= n_contains])})
    ),
    diff as (
        select a.fs as afs, b.fs as bfs, a.v as v, cast(a.i as int) - b.i as di, cast(a.j as int) - b.j as dj,
            a.i as i1, a.j as j1, b.i as i2, b.j as j2
        from A as a join A as b
        on a.v = b.v and a.fs < b.fs
    ),
    moves as (
        select afs, bfs, di, dj, count(*) as move_size,
            min(i1) as mini1, min(j1) as minj1, max(i1) as maxi1, max(j1) as maxj1,
            min(i2) as mini2, min(j2) as minj2, max(i2) as maxi2, max(j2) as max2j
        from diff
        group by afs, bfs, di, dj
        having move_size > {n_move_size}
    )
    select * from moves
    """)
    df_moves = res.to_df()
    return df_moves

def concat(df_moves):
    # Now find triples where the same sheet is involved with two moves
    res = duckdb.sql(f"""
    with move_rels as (
        select distinct afs, bfs from df_moves
    ),
    move_counts as (
        select afs as fs, count(*) as cnt from move_rels group by afs
        union all
        select bfs as fs, count(*) as cnt from move_rels group by bfs
    )
    select fs, sum(cnt) as cnt
    from move_counts
    group by fs
    having cnt >= 3
    order by cnt desc
    """)
    df_triple_candidates = res.to_df()
    #print(df_triple_candidates.shape)

    # Now find the moves where the max of one is the min of the other
    moves = []
    for fs in tqdm.tqdm(df_triple_candidates.fs):
        fso = df_moves.query('afs == @fs')
        #print(fs, fso.shape)
        if len(fso) < 2:
            #print(f"Skipping {fs} as it has only {len(fso)} move")
            continue
        target_mini = []
        target_minj = []
        target_maxi = []
        target_maxj = []
        source_fs = []
        for i, row in fso.iterrows():
            if row['afs'] == fs:  # target <- source
                source_fs.append(row['bfs'])
                target_mini.append(row['mini1'])
                target_minj.append(row['minj1'])
                target_maxi.append(row['maxi1'])
                target_maxj.append(row['maxj1'])
            else:  # target <- source
                source_fs.append(row['afs'])
                target_mini.append(row['mini2'])
                target_minj.append(row['minj2'])
                target_maxi.append(row['maxi2'])
                target_maxj.append(row['maxj2'])
        # Now find the moves where the max of one is the min of the other
        for s1, mini in enumerate(target_mini):
            for s2, maxi in enumerate(target_maxi):
                if maxi + 1 == mini:
                    #print(f"Found concat: {source_fs[s1]} -> {fs} <- {source_fs[s2]}  {maxi}:{mini}")
                    moves.append((fs, source_fs[s1], source_fs[s2], maxi, mini))
    df_concat = pd.DataFrame(moves, columns=['fs', 'source1', 'source2', 'maxi', 'mini'])
    return df_concat


def main(x_file, top_file, parquet_tall_dir, top_k=20, n_contains=5, n_move_size=5):
    top, x_set = load_data(x_file, top_file, top_k)
    t0 = time.time()
    scores = subset(top, x_set)
    filtered_scores = [(i, j, sa) for i ,j, sa in scores if sa >= n_contains]
    t1 = time.time()
    print('Scores: ', len(filtered_scores))
    print('Score time: ', t1 - t0)
    df_moves = moves(scores, parquet_tall_dir, n_contains=n_contains, n_move_size=n_move_size)
    t2 = time.time()
    print('Moves: ', df_moves.shape)
    print('Move time', t2 - t1)
    df_concat = concat(df_moves)
    t3 = time.time()
    print('Concat: ', df_concat.shape)
    print('Concat time', t3 - t2)
