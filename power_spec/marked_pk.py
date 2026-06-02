import argparse, json, os, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from pypower import CatalogFFTPower

DEFAULT_DATA_ROOT = '/pscratch/sd/v/vtorresg/hod-astra-box500'
ENVIRONMENTS = ('void', 'sheet', 'filament', 'knot')
PROB_COLS = {'void': 'PVOID',
             'sheet': 'PSHEET',
             'filament': 'PFILAMENT',
             'knot': 'PKNOT'}
PROB_COL_ORDER = [PROB_COLS[name] for name in ENVIRONMENTS]
RAW_NAME_RE = re.compile(r'^zone_(?P<cosmo>c\d{3})_ph(?P<phase>\d+)_seed(?P<seed>\d+)_hod(?P<hod>\d+)\.fits(?:\.gz)?$')


@dataclass
class CatalogJob:
    raw_path: Path
    probability_path: Optional[Path]
    tag: str
    cosmo: str
    phase: str
    seed: str
    hod: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--input', nargs='+', default=None)
    parser.add_argument('--prob-file', type=str, default='')
    parser.add_argument('--cosmo', nargs='+', default=['all'])
    parser.add_argument('--hod', nargs='+', default=['all'])
    parser.add_argument('--field', type=str, default='matter', choices=['matter', 'all'] + list(ENVIRONMENTS))
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--boxsize', type=float, default=500.0)
    parser.add_argument('--boxcenter', nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument('--grid', type=int, default=256)
    parser.add_argument('--mas', type=str, default='CIC', choices=['NGP', 'CIC', 'TSC', 'PCS'])
    parser.add_argument('--interlacing', type=int, default=2)
    parser.add_argument('--engine', type=str, default='pypower')
    parser.add_argument('--position-cols', nargs=3, default=['XCART', 'YCART', 'ZCART'])
    parser.add_argument('--weight-col', type=str, default='', help='If omitted, all selected galaxies get weight 1.')
    parser.add_argument('--data-randiter', type=int, default=-1)
    parser.add_argument('--use-all-rows', action='store_true')
    parser.add_argument('--subtract-shotnoise', action='store_true',
                        help='Write pk_used = pk_raw - shotnoise.')
    parser.add_argument('--k-bin-width', type=float, default=0.0, help='k-bin width. Default is the fundamental mode 2*pi/boxsize.')
    parser.add_argument('--kmax', type=float, default=0.0, help='Maximum k edge. Default is the mesh Nyquist frequency.')
    parser.add_argument('--unmatched-policy', type=str, default='error', choices=['error', 'drop'])
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def strip_fits_suffix(path):
    name = Path(path).name
    for suffix in ('.fits.gz', '.fits'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(path).stem


def normalize_cosmo(value):
    value = str(value).strip().lower()
    if value in ('all', '*'):
        return 'all'
    if value.startswith('c'):
        return f'c{int(value[1:]):03d}'
    return f'c{int(value):03d}'


def normalize_hod(value):
    value = str(value).strip().lower()
    if value in ('all', '*'):
        return 'all'
    if value.startswith('hod'):
        value = value[3:]
    return f'hod{int(value):03d}'


def parse_raw_metadata(raw_path):
    match = RAW_NAME_RE.match(Path(raw_path).name)
    tag = strip_fits_suffix(raw_path)
    if not match:
        return {'tag': tag, 'cosmo': 'unknown', 'phase': 'unknown',
                'seed': 'unknown', 'hod': 'unknown'}
    info = match.groupdict()
    info['tag'] = tag
    info['hod'] = f"hod{int(info['hod']):03d}"
    return info


def derive_probability_path(raw_path):
    raw_path = Path(raw_path)
    tag = strip_fits_suffix(raw_path)
    candidates = []
    if raw_path.parent.name == 'raw':
        cosmo_dir = raw_path.parent.parent
        candidates.append(cosmo_dir / 'release' / 'probabilities'
                          / f'{tag}_probability_iterdata.fits.gz')
        candidates.append(cosmo_dir / 'release' / 'probabilities'
                          / f'{tag}_probability_iterdata.fits')
    candidates.append(raw_path.with_name(f'{tag}_probability_iterdata.fits.gz'))
    candidates.append(raw_path.with_name(f'{tag}_probability_iterdata.fits'))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def expand_requested_fields(field):
    if field == 'all':
        return ['matter'] + list(ENVIRONMENTS)
    return [field]


def discover_jobs(args):
    if args.input:
        raw_paths = [Path(path).expanduser().resolve() for path in args.input]
    else:
        root = Path(args.data_root).expanduser()
        cosmo_filter = {normalize_cosmo(value) for value in args.cosmo}
        hod_filter = {normalize_hod(value) for value in args.hod}
        raw_paths = sorted(root.glob('c*/raw/zone_*.fits'))
        if 'all' not in cosmo_filter:
            raw_paths = [path for path in raw_paths
                         if parse_raw_metadata(path)['cosmo'] in cosmo_filter]
        if 'all' not in hod_filter:
            raw_paths = [path for path in raw_paths
                         if parse_raw_metadata(path)['hod'] in hod_filter]

    jobs = []
    for raw_path in raw_paths:
        if not raw_path.is_file():
            raise FileNotFoundError(f'Raw FITS catalog not found: {raw_path}')
        meta = parse_raw_metadata(raw_path)
        prob_path = Path(args.prob_file).expanduser().resolve() if args.prob_file else derive_probability_path(raw_path)
        jobs.append(CatalogJob(raw_path=raw_path,
                               probability_path=prob_path,
                               tag=meta['tag'],
                               cosmo=meta['cosmo'],
                               phase=meta['phase'],
                               seed=meta['seed'],
                               hod=meta['hod']))

    if args.limit > 0:
        jobs = jobs[:args.limit]
    return jobs


def resolve_outdir(args):
    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = Path(args.data_root).expanduser() / 'power_spectra'
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def read_fits_columns(path, columns):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        if data is None:
            raise RuntimeError(f'No table data found in FITS file: {path}')
        names = set(data.names or [])
        return {col: np.asarray(data[col]).copy() for col in columns}


def read_raw_catalog(path, position_cols, weight_col, data_randiter, use_all_rows):
    needed = list(position_cols)
    if weight_col:
        needed.append(weight_col)
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        if data is None:
            raise RuntimeError(f'No table data found in FITS file: {path}')
        names = set(data.names or [])

        n_total = len(data)
        if use_all_rows or 'RANDITER' not in names:
            mask = np.ones(n_total, dtype=bool)
            selection_desc = 'all rows'
        else:
            mask = np.asarray(data['RANDITER']) == data_randiter
            selection_desc = f'RANDITER == {data_randiter}'

        positions = np.column_stack([np.asarray(data[col][mask], dtype=np.float64) for col in position_cols])
        if weight_col:
            weights = np.asarray(data[weight_col][mask], dtype=np.float64)
        else:
            weights = np.ones(positions.shape[0], dtype=np.float64)

        targetid = None
        if 'TARGETID' in names:
            targetid = np.asarray(data['TARGETID'][mask], dtype=np.int64)

    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.all(np.isfinite(positions), axis=1) & np.isfinite(weights) & (weights >= 0.0)
    if not np.all(valid):
        positions = positions[valid]
        weights = weights[valid]
        if targetid is not None:
            targetid = targetid[valid]

    return {'positions': positions,
            'weights': weights,
            'targetid': targetid,
            'n_total': int(n_total),
            'n_selected_before_finite': int(np.sum(mask)),
            'n_selected': int(positions.shape[0]),
            'selection': selection_desc,
            'n_removed_nonfinite': int(np.sum(~valid))}


def load_environment_classes(targetid_data, probability_path, unmatched_policy):
    cols = ['TARGETID'] + PROB_COL_ORDER
    prob_data = read_fits_columns(probability_path, cols)
    tid_prob = np.asarray(prob_data['TARGETID'], dtype=np.int64)
    probs = np.column_stack([np.asarray(prob_data[col], dtype=np.float64) for col in PROB_COL_ORDER])
    probs = np.nan_to_num(probs, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

    class_prob = np.argmax(probs, axis=1).astype(np.int16)
    max_prob = np.max(probs, axis=1).astype(np.float64)
    class_prob[~np.isfinite(max_prob)] = -1

    order = np.argsort(tid_prob)
    tid_sorted = tid_prob[order]
    class_sorted = class_prob[order]
    max_prob_sorted = max_prob[order]

    tid_data = np.asarray(targetid_data, dtype=np.int64)
    idx = np.searchsorted(tid_sorted, tid_data, side='left')
    in_bounds = idx < tid_sorted.size
    matched = np.zeros(tid_data.size, dtype=bool)
    matched[in_bounds] = tid_sorted[idx[in_bounds]] == tid_data[in_bounds]

    if np.any(~matched) and unmatched_policy == 'error':
        n_missing = int(np.sum(~matched))
        raise RuntimeError(f'{n_missing} selected galaxies are missing from {probability_path}')

    class_data = np.full(tid_data.size, -1, dtype=np.int16)
    max_prob_data = np.full(tid_data.size, np.nan, dtype=np.float64)
    if np.any(matched):
        class_data[matched] = class_sorted[idx[matched]]
        max_prob_data[matched] = max_prob_sorted[idx[matched]]

    counts = {env: int(np.sum(class_data == i)) for i, env in enumerate(ENVIRONMENTS)}
    info = {'probability_path': str(probability_path),
            'n_probability_rows': int(tid_prob.size),
            'n_data': int(tid_data.size),
            'n_matched': int(np.sum(matched)),
            'matched_fraction': float(np.mean(matched)) if tid_data.size else 0.0,
            'unmatched_policy': unmatched_policy,
            'class_rule': 'argmax(PVOID, PSHEET, PFILAMENT, PKNOT)',
            'class_counts': counts,
            'mean_max_probability': float(np.nanmean(max_prob_data)) if np.any(matched) else np.nan}
    return class_data, max_prob_data, info


def mas_to_resampler(mas):
    mapping = {'NGP': 'ngp', 'CIC': 'cic', 'TSC': 'tsc', 'PCS': 'pcs'}
    return mapping[mas.upper()]


def make_k_edges(boxsize, nmesh, k_bin_width, kmax):
    dk = 2.0 * np.pi / float(boxsize) if k_bin_width <= 0.0 else float(k_bin_width)
    k_nyquist = np.pi * float(nmesh) / float(boxsize)
    k_stop = k_nyquist if kmax <= 0.0 else min(float(kmax), k_nyquist)
    edges = np.arange(0.0, k_stop + 0.5 * dk, dk, dtype=np.float64)
    if edges.size < 2 or edges[-1] < k_stop:
        edges = np.append(edges, k_stop)
    return edges


def to_scalar_float(value, default=np.nan):
    if value is None:
        return float(default)
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(np.real(arr.ravel()[0]))


def extract_monopole(poles):
    try:
        pk0 = poles(ell=0, complex=False, remove_shotnoise=False)
        remove_shotnoise_supported = True
    except TypeError:
        pk0 = poles(ell=0, complex=False)
        remove_shotnoise_supported = False

    nmodes = getattr(poles, 'nmodes', None)
    if nmodes is None:
        nmodes = np.full_like(np.asarray(poles.k, dtype=np.float64), np.nan, dtype=np.float64)
    else:
        nmodes = np.asarray(nmodes, dtype=np.float64)

    return {'k': np.asarray(poles.k, dtype=np.float64),
            'pk_raw': np.asarray(pk0, dtype=np.float64),
            'nmodes': nmodes,
            'shotnoise_pypower': to_scalar_float(getattr(poles, 'shotnoise', np.nan)),
            'remove_shotnoise_supported': remove_shotnoise_supported}


def compute_pk_pypower(positions, weights, edges, args):
    result = CatalogFFTPower(data_positions1=positions,
                             data_weights1=weights,
                             edges=edges,
                             ells=(0,),
                             position_type='pos',
                             boxsize=args.boxsize,
                             boxcenter=np.asarray(args.boxcenter, dtype=np.float64),
                             nmesh=args.grid,
                             resampler=mas_to_resampler(args.mas),
                             interlacing=args.interlacing)
    pk = extract_monopole(result.poles)
    pk['engine'] = 'pypower.CatalogFFTPower'
    return pk


def paint_ngp(mesh, grid_pos, weights):
    nmesh = mesh.shape[0]
    idx = np.floor(grid_pos).astype(np.int64) % nmesh
    np.add.at(mesh, (idx[:, 0], idx[:, 1], idx[:, 2]), weights)


def paint_cic(mesh, grid_pos, weights):
    nmesh = mesh.shape[0]
    base = np.floor(grid_pos).astype(np.int64)
    frac = grid_pos - base
    ix0 = base[:, 0] % nmesh
    iy0 = base[:, 1] % nmesh
    iz0 = base[:, 2] % nmesh
    ix1 = (ix0 + 1) % nmesh
    iy1 = (iy0 + 1) % nmesh
    iz1 = (iz0 + 1) % nmesh
    wx = (1.0 - frac[:, 0], frac[:, 0])
    wy = (1.0 - frac[:, 1], frac[:, 1])
    wz = (1.0 - frac[:, 2], frac[:, 2])
    xs = (ix0, ix1)
    ys = (iy0, iy1)
    zs = (iz0, iz1)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = weights * wx[dx] * wy[dy] * wz[dz]
                np.add.at(mesh, (xs[dx], ys[dy], zs[dz]), w)


def assignment_window(kx, ky, kz, cell_size, mas):
    mas = mas.upper()
    if mas == 'NGP':
        power = 1
    elif mas == 'CIC':
        power = 2
    else:
        raise ValueError(f'The numpy FFT engine only supports NGP and CIC, not {mas}.')
    wx = np.sinc(kx * cell_size / (2.0 * np.pi)) ** power
    wy = np.sinc(ky * cell_size / (2.0 * np.pi)) ** power
    wz = np.sinc(kz * cell_size / (2.0 * np.pi)) ** power
    return wx[:, None, None] * wy[None, :, None] * wz[None, None, :]


def rfft_mode_weights(nmesh):
    weights = np.ones(nmesh // 2 + 1, dtype=np.float64)
    if nmesh % 2 == 0:
        weights[1:-1] = 2.0
    else:
        weights[1:] = 2.0
    return weights


def compute_pk(positions, weights, edges, args):
    if positions.shape[0] == 0:
        raise RuntimeError('Cannot compute P(k): selected field has zero galaxies.')
    result = compute_pk_pypower(positions, weights, edges, args)

    volume = float(args.boxsize) ** 3
    sw = float(np.sum(weights, dtype=np.float64))
    sw2 = float(np.sum(weights.astype(np.float64) ** 2, dtype=np.float64))
    shotnoise_analytic = volume * sw2 / (sw * sw)
    shotnoise = result['shotnoise_pypower']
    if not np.isfinite(shotnoise):
        shotnoise = shotnoise_analytic

    result['shotnoise'] = float(shotnoise)
    result['shotnoise_analytic'] = float(shotnoise_analytic)
    result['sum_weights'] = sw
    result['sum_weights2'] = sw2
    result['n_objects'] = int(positions.shape[0])
    if args.subtract_shotnoise:
        result['pk_used'] = result['pk_raw'] - shotnoise
    else:
        result['pk_used'] = result['pk_raw'].copy()
    return result


def align_to_k(k_target, k_src, values_src):
    k_target = np.asarray(k_target, dtype=np.float64)
    k_src = np.asarray(k_src, dtype=np.float64)
    values_src = np.asarray(values_src, dtype=np.float64)
    if k_target.shape == k_src.shape and np.allclose(k_target, k_src, rtol=1e-8, atol=1e-12):
        return values_src.copy()
    order = np.argsort(k_src)
    return np.interp(k_target, k_src[order], values_src[order], left=np.nan, right=np.nan)


def write_csv(path, results):
    fields = list(results)
    first = results[fields[0]]
    k = first['k']
    cols = [k, first['nmodes']]
    headers = ['k_h_mpc', 'nmodes']
    for field in fields:
        result = results[field]
        cols.append(align_to_k(k, result['k'], result['pk_raw']))
        cols.append(align_to_k(k, result['k'], result['pk_used']))
        headers.append(f'pk_raw_{field}')
        headers.append(f'pk_used_{field}')
    np.savetxt(path, np.column_stack(cols), delimiter=',',
               header=','.join(headers), comments='')


def load_pyplot():
    if 'MPLCONFIGDIR' not in os.environ:
        mplconfig = Path(os.environ.get('TMPDIR', '/tmp')) / 'matplotlib-cache'
        mplconfig.mkdir(parents=True, exist_ok=True)
        os.environ['MPLCONFIGDIR'] = str(mplconfig)
    try:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['text.usetex'] = True
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
    except Exception as exc:
        print(f'---> plot skipped: matplotlib is not available ({exc})')
        return None
    return plt


def write_plot(path, results, title):
    plt = load_pyplot()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for field, result in results.items():
        k = result['k']
        pk = result['pk_used']
        mask = (k > 0.0) & np.isfinite(pk) & (pk > 0.0)
        if np.any(mask):
            ax.loglog(k[mask], pk[mask], lw=1.5, label=field)
    ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$P(k)\ [(\mathrm{Mpc}/h)^3]$')
    ax.grid(alpha=0.3, which='both')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return True


def result_metadata(result):
    return {'engine': result['engine'],
            'n_objects': result['n_objects'],
            'sum_weights': result['sum_weights'],
            'sum_weights2': result['sum_weights2'],
            'shotnoise': result['shotnoise'],
            'shotnoise_analytic': result['shotnoise_analytic'],
            'shotnoise_pypower': result['shotnoise_pypower'],
            'remove_shotnoise_supported': bool(result['remove_shotnoise_supported'])}


def process_job(job, args, outdir):
    requested_fields = expand_requested_fields(args.field)
    csv_path = outdir / f'pk_{args.field}_{job.tag}.csv'
    meta_path = outdir / 'meta' / f'run_metadata_pk_{args.field}_{job.tag}.json'
    plot_path = outdir / 'plots' / f'pk_{args.field}_{job.tag}.png'

    if args.skip_existing and csv_path.is_file():
        print(f'---> skipping existing: {csv_path}')
        return {'skipped': True, 'csv': str(csv_path)}

    raw = read_raw_catalog(job.raw_path, args.position_cols, args.weight_col,
                           args.data_randiter, args.use_all_rows)
    print(f"---> {job.tag}: selected {raw['n_selected']} / {raw['n_total']} rows ({raw['selection']})")

    class_data = None
    max_prob = None
    class_info = None
    needs_environment = any(field != 'matter' for field in requested_fields)
    if needs_environment:
        class_data, max_prob, class_info = load_environment_classes(
            raw['targetid'], job.probability_path, args.unmatched_policy)
        print(f"---> {job.tag}: ASTRA class counts {class_info['class_counts']}")

    edges = make_k_edges(args.boxsize, args.grid, args.k_bin_width, args.kmax)
    positions = raw['positions']
    weights = raw['weights']
    results = {}
    field_info = {}

    for field in requested_fields:
        if field == 'matter':
            field_mask = np.ones(positions.shape[0], dtype=bool)
            mark_desc = 'unmarked matter field'
        else:
            env_index = ENVIRONMENTS.index(field)
            field_mask = class_data == env_index
            mark_desc = f'binary ASTRA mark: 1 if argmax probability is {field}, else 0'

        n_field = int(np.sum(field_mask))
        if n_field == 0:
            raise RuntimeError(f'{job.tag}: field {field} has zero selected galaxies.')

        print(f'---> {job.tag}: computing {field} P(k) with {n_field} galaxies')
        results[field] = compute_pk(positions[field_mask], weights[field_mask], edges, args)
        field_info[field] = {'mark': mark_desc,
                             'n_selected': n_field,
                             'selected_fraction': float(n_field / positions.shape[0])}
        if field != 'matter' and max_prob is not None:
            field_info[field]['mean_max_probability_selected'] = float(np.nanmean(max_prob[field_mask]))

    write_csv(csv_path, results)
    plot_written = False
    if args.plot:
        plot_written = write_plot(plot_path, results, f'{job.tag}  field={args.field}')

    metadata = {'raw_path': str(job.raw_path),
                'probability_path': str(job.probability_path) if needs_environment else None,
                'tag': job.tag,
                'cosmo': job.cosmo,
                'phase': job.phase,
                'seed': job.seed,
                'hod': job.hod,
                'field_requested': args.field,
                'fields_computed': requested_fields,
                'raw_selection': raw,
                'astra_classification': class_info,
                'field_info': field_info,
                'pk': {field: result_metadata(result) for field, result in results.items()},
                'boxsize_mpc_h': args.boxsize,
                'boxcenter_mpc_h': list(args.boxcenter),
                'grid': args.grid,
                'mas': args.mas,
                'interlacing': args.interlacing,
                'engine_requested': args.engine,
                'subtract_shotnoise': args.subtract_shotnoise,
                'k_bin_width': args.k_bin_width,
                'kmax': args.kmax,
                'outputs': {'csv': str(csv_path),
                            'metadata': str(meta_path),
                            'plot': str(plot_path) if plot_written else None}}

    metadata['raw_selection']['positions'] = None
    metadata['raw_selection']['weights'] = None
    metadata['raw_selection']['targetid'] = None
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f'---> wrote: {csv_path}')
    if plot_written:
        print(f'---> wrote: {plot_path}')
    print(f'---> wrote: {meta_path}')
    return {'skipped': False, 'csv': str(csv_path), 'metadata': str(meta_path),
            'plot': str(plot_path) if plot_written else None}


def main():
    args = parse_args()
    t0 = time.time()
    outdir = resolve_outdir(args)
    jobs = discover_jobs(args)
    print(f'---> output directory: {outdir}')
    print(f'---> catalogs to process: {len(jobs)}')

    outputs = []
    for i, job in enumerate(jobs, start=1):
        print(f'---> [{i}/{len(jobs)}] {job.raw_path}')
        outputs.append(process_job(job, args, outdir))

    elapsed = time.time() - t0
    run_summary = {'elapsed_sec': elapsed,
                   'n_catalogs': len(jobs),
                   'outputs': outputs}
    summary_path = outdir / 'meta' / f'run_summary_pk_{args.field}_{int(time.time())}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(run_summary, f, indent=2)
    print(f'---> wrote: {summary_path}')
    print(f'---> elapsed: {elapsed:.2f} s')


if __name__ == '__main__':
    main()