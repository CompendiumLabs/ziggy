# generate similarity metrics

import torch
import pandas as pd
from pathlib import Path
from glob import glob
from ziggy import TextDatabase, TorchVectorIndex
from ziggy.quant import Half
from ziggy.utils import batch_indices

# merge multiple text databases (assumes same qspec, ignores groups)
def merge_databases(paths, output, model, qspec=Half, size=1024):
    if isinstance(paths, str):
        paths = sorted(glob(paths))
    db = TextDatabase(embed=model, device='cpu', qspec=qspec, size=size)
    for path in paths:
        print(path)
        db1 = TextDatabase.load(path, embed=model, device='cpu')
        labels, texts = list(db1.text.keys()), list(db1.text.values())
        vectors = db1.index.values.data[:len(db1),:]
        db.index_text(labels, texts)
        db.index_vecs(labels, vectors)
        del db1
    db.save(output)

def merge_indices(paths, output, dims, size=1024, qspec=Half):
    if isinstance(paths, str):
        paths = sorted(glob(paths))
    index = TorchVectorIndex(dims=dims, size=size, device='cpu', qspec=qspec)
    for path in paths:
        print(path)
        index1 = TorchVectorIndex.load(path, device='cpu')
        vectors = index1.values.data[:len(index1),:]
        index.add(index1.labels, vectors)
        del index1
    index.save(output)

def extract_index(path, output, trim=True):
    db = TextDatabase.load(path, device='cpu')
    if trim:
        db.index.values.data = db.index.values.data[:len(db.index),:].clone()
    db.index.save(output)

# demean and renormalize vectors
def demean_inplace(x):
    x -= x.mean(dim=0)[None,:]
    x /= x.square().sum(dim=1)[:,None]

# load ziggy TorchVectorIndex directly or from TextDatabase
def load_database(path, device='cuda'):
    data = torch.load(path, map_location=device, weights_only=True)
    if 'index' in data:
        data = data['index']
    return TorchVectorIndex.load(data, device=device)

# merge patent metadata with ziggy database
def merge_patents(
    path_vecs, # ziggy database
    path_meta, # metadata csv
    path_pats, # output csv
    id_col='appnum', date_col='appdate'
):
    # load vector index
    print('Loading vector index')
    index = load_database(path_vecs, device='cpu')

    # load metadata csv
    print('Loading patent metadata')
    meta = pd.read_csv(path_meta, usecols=[id_col, date_col], dtype={date_col: 'str'})
    meta = meta.drop_duplicates(id_col).set_index(id_col)
    meta[date_col] = pd.to_datetime(meta[date_col], errors='coerce')

    # merge with index (due to dups)
    pats = pd.DataFrame({id_col: index.labels})
    pats = pats.join(meta, on=id_col)

    # save ordered patent data
    pats.to_csv(path_pats, index=False)

def similarity_topk(
    path_vecs, # ziggy database
    path_pats, # patent metadata csv (for comparison!)
    path_sims, # output torch file
    path_vecs1=None, # comparison ziggy database
    topk=100, batch_size=256, max_rows=None, demean=False
):
    # load vector index
    print('Loading vector index')
    index = load_database(path_vecs)
    n_pats = len(index)

    # load comparison vector index
    if path_vecs1 is not None:
        print('Loading comparison vector index')
        index1 = load_database(path_vecs1)

    # demean vectors is requested
    if demean:
        demean_inplace(index.values.data)
        if index1 is not index:
            demean_inplace(index1.values.data)

    # limit rows if requested
    if max_rows is not None:
        n_pats = min(n_pats, max_rows)

    # load merged patent data
    pats = pd.read_csv(path_pats, nrows=max_rows)

    # convert date to days since unix epoch
    epoch = pd.to_datetime('1970-01-01')
    dates = pats.set_index('patnum')['appdate']
    days = torch.tensor((dates-epoch).dt.days.to_numpy(), device='cuda')

    # create output tensors
    idxt = torch.zeros((n_pats, topk), dtype=torch.int32, device='cuda')
    simt = torch.zeros((n_pats, topk), dtype=torch.float16, device='cuda')

    # generate similarity metrics
    for i1, i2 in batch_indices(n_pats, batch_size):
        print(f'{i1} → {i2}')

        # compute similarities for batch
        vecs = index.values.data[i1:i2] # [B, D]
        sims = index.similarity(vecs) # [B, N1]

        # compute top sims for before
        before = days[None, :] < days[i1:i2, None]
        simb = torch.where(before, sims, -torch.inf)
        topb = simb.topk(topk, dim=1)

        # store in output tensors
        idxt[i1:i2] = topb.indices
        simt[i1:i2] = topb.values

    # save to disk
    torch.save({
        'top_idx': idxt,
        'top_sim': simt,
    }, path_sims)

def similarity_mean(
    path_vecs, # ziggy database
    path_pats, # patent metadata csv (for comparison!)
    path_sims=None, # output torch file
    path_vecs1=None, # comparison ziggy database
    batch_size=8, max_rows=None, demean=False,
    device='cuda', device1='cuda', min_year=1970,
):
    # load vector index
    print('Loading base vector index')
    index = load_database(path_vecs, device=device)
    n_pats = len(index)
    print(f'Loaded {n_pats} vectors')

    # load comparison vector index
    if path_vecs1 is not None:
        print('Loading comparison vector index')
        index1 = load_database(path_vecs1, device=device1)
    else:
        index1 = index
    n_pats1 = len(index1)

    # demean vectors is requested
    if demean:
        demean_inplace(index.values.data)
        if index1 is not index:
            demean_inplace(index1.values.data)

    # limit rows if requested
    if max_rows is not None:
        n_pats = min(n_pats, max_rows)

    # load merged patent data
    print('Loading patent metadata')
    pats = pd.read_csv(path_pats)
    min_date = pd.Timestamp(f'{min_year}-01-01')
    pats['appdate'] = pd.to_datetime(pats['appdate']).fillna(min_date)
    print(f'Loaded {len(pats)} metadata')

    # get application year for patents
    app_year = torch.tensor(pats['appdate'].dt.year, dtype=torch.int32, device=device1)
    year_min, year_max = min_year, app_year.max().item()
    year_idx = app_year - year_min

    # get application year statistics
    n_years = year_max - year_min + 1
    c_years = torch.bincount(year_idx, minlength=n_years)
    y_years = torch.arange(year_min, year_max + 1, device=device1)

    # create output tensors
    avgt = torch.zeros((n_pats, n_years), dtype=torch.float16, device=device1)

    # generate similarity metrics
    for i1, i2 in batch_indices(n_pats, batch_size):
        print(f'{i1} → {i2}')
        n_batch = i2 - i1

        # compute similarities for batch
        vecs = index.values.data[i1:i2].to(device=device1) # [B, D]
        # sims = index1.similarity(vecs).T # [N1, B] - this is slow
        sims = index1.values.data[:n_pats1,:] @ vecs.T # [N1, B]

        # generate offsets
        batch_vec = torch.arange(n_batch, device=device1)
        offsets = year_idx[:,None] * n_batch + batch_vec[None,:]

        # group sum by application year
        sums = torch.bincount(offsets.view(-1), weights=sims.view(-1), minlength=n_years*n_batch)
        avgt[i1:i2] = sums.reshape(n_years, n_batch).T / c_years[None,:]

    # return results
    res = {
        'year_sims': avgt   ,
        'year_nums': c_years,
        'year_inds': y_years,
    }
    if path_sims is not None:
        torch.save(res, path_sims)
    else:
        return res

# compute patent novelty metric
def patent_novelty(sim_data, meta_data, delta=5, device='cuda', year_cut=torch.inf):
    # load if needed
    if isinstance(sim_data, (str, Path)):
        sim_data = torch.load(sim_data, weights_only=True)
    if isinstance(meta_data, (str, Path)):
        meta_data = pd.read_csv(meta_data)

    # load similarity data
    year_sim = sim_data['year_sim'].to(dtype=torch.float, device=device)
    year_count = sim_data['year_count'].to(dtype=torch.float, device=device)
    year_min = sim_data['year_min']
    year_max = sim_data['year_max']

    # get data size
    n_pats, n_years = year_sim.shape

    # load patent metadata
    min_str = f'{year_min}-01-01'
    pat_date = pd.to_datetime(meta_data['appdate'].fillna(min_str))
    app_year = torch.tensor(pat_date.dt.year, dtype=torch.int64, device=device)
    year_idx = app_year - year_min

    # get pre counts
    count_pre = torch.zeros(n_pats, device=device)
    simil_pre = torch.zeros(n_pats, device=device)
    for i in range(delta):
        idx = year_idx - i
        sel = (idx >= 0) & (idx < n_years) & (app_year <= year_cut)

        idx1 = torch.where(sel, idx, 0)
        cnt1 = torch.where(sel, year_count[idx1], 0)
        sim1 = torch.where(sel, year_sim.gather(1, idx1.unsqueeze(1)).squeeze(1), 0)

        count_pre += cnt1
        simil_pre += cnt1 * sim1
    simil_pre /= count_pre

    # get pos counts
    count_pos = torch.zeros(n_pats, device=device)
    simil_pos = torch.zeros(n_pats, device=device)
    for i in range(1, delta + 1):
        idx = year_idx + i
        sel = (idx >= 0) & (idx < n_years) & (app_year <= year_cut)

        idx1 = torch.where(sel, idx, 0)
        cnt1 = torch.where(sel, year_count[idx1], 0)
        sim1 = torch.where(sel, year_sim.gather(1, idx1.unsqueeze(1)).squeeze(1), 0)

        count_pos += cnt1
        simil_pos += cnt1 * sim1
    simil_pos /= count_pos

    # return similarities
    return simil_pre, simil_pos
