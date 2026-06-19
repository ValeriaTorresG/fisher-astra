import argparse, csv, json, os, time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from calc_deriv_cosmo import (DEFAULT_PK_ROOT, DERIVATIVE_KINDS,
                              FIELDS, OBSERVABLE_KINDS,
                              add_bool_argument, align_pk_to_reference,
                              average_pk, check_result_k_range,
                              derivative_axis_label,
                              normalize_cosmo, normalize_fields,
                              normalize_hod, observable_description,
                              safe_log,
                              write_combined_environment_matrix,
                              write_derivative_matrix)


DEFAULT_PARAMS_FILE = 'params/params_hods_c000_ph000_seed0.csv'
DEFAULT_HOD_HEADER_DIR = '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/c000_ph000/seed0'
HOD_PARAMS = ('LOGM_CUT', 'LOGM1', 'SIGMA', 'ALPHA', 'KAPPA')
HOD_HEADER_COLUMNS = ('GAL_TYPE', 'Q_PAR', 'Q_PERP', 'LOGM_CUT', 'LOGM1',
                      'SIGMA', 'ALPHA', 'KAPPA', 'ALPHA_C', 'ALPHA_S',
                      'S', 'S_V', 'S_P', 'S_R', 'ACENT', 'ASAT',
                      'BCENT', 'BSAT', 'IC')
FINITE_DIFFERENCE_SCHEMES = ('central', 'forward', 'backward')
FITS_BLOCK_SIZE = 2880
FITS_CARD_SIZE = 80

DerivativeResult = SimpleNamespace
LocalNeighbor = SimpleNamespace
LocalFit = SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pk-root', type=str, default=DEFAULT_PK_ROOT)
    parser.add_argument('--params-file', type=str, default=DEFAULT_PARAMS_FILE)
    parser.add_argument('--hod-header-dir', type=str, default=DEFAULT_HOD_HEADER_DIR)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--fields', nargs='+', default=['all'], choices=['all'] + list(FIELDS))
    parser.add_argument('--params', nargs='+', default=list(HOD_PARAMS), choices=list(HOD_PARAMS))
    parser.add_argument('--hod', nargs='+', default=['all'])
    parser.add_argument('--cosmo', nargs='+', default=['c000'])
    parser.add_argument('--pk-kind', type=str, default='pk_used', choices=['pk_used', 'pk_raw'])
    parser.add_argument('--observable', type=str, default='f2pk', choices=OBSERVABLE_KINDS)
    parser.add_argument('--derivative-kind', type=str, default='linear', choices=DERIVATIVE_KINDS)
    parser.add_argument('--fiducial-hod', type=str, default='center',)
    parser.add_argument('--n-neighbors', type=int, default=100)
    add_bool_argument(parser, '--fit-intercept', False, '')
    parser.add_argument('--finite-difference', type=str, default=None, choices=FINITE_DIFFERENCE_SCHEMES)
    parser.add_argument('--pk-file-field', type=str, default='all')
    parser.add_argument('--neighbor-target-weight', type=float, default=0.25)
    parser.add_argument('--max-other-distance', type=float, default=0.0)
    add_bool_argument(parser, '--strict-k-range', True, '')
    add_bool_argument(parser, '--plot', True, '')
    parser.add_argument('--skip-missing', action='store_true')
    return parser.parse_args()


def normalize_params(values):
    params = []
    aliases = {name.lower(): name for name in HOD_PARAMS}
    for value in values:
        key = str(value).strip().upper()
        if key not in HOD_PARAMS:
            key = aliases.get(str(value).strip().lower(), key)
        if key not in HOD_PARAMS:
            raise ValueError(f'Unknown HOD parameter: {value}')
        if key not in params:
            params.append(key)
    return params


def resolve_single_cosmo(values):
    tokens = [str(value).strip().lower() for value in values]
    if any(token in ('all', '*') for token in tokens):
        raise RuntimeError('HOD derivatives must be evaluated at one fiducial cosmology; use --cosmo c000.')
    cosmos = [normalize_cosmo(value) for value in values]
    unique = []
    for cosmo in cosmos:
        if cosmo not in unique:
            unique.append(cosmo)
    if len(unique) != 1:
        raise RuntimeError('HOD derivatives must be evaluated at one fiducial cosmology; '
                           f'got {", ".join(unique)}.')
    return unique[0]


def parse_float(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return np.nan


def fits_stem(path):
    name = Path(path).name
    for suffix in ('.fits.gz', '.fits'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(path).stem


def split_fits_value_comment(text):
    in_quote = False
    pieces = []
    i = 0
    while i < len(text):
        char = text[i]
        if char == "'":
            pieces.append(char)
            if in_quote and i + 1 < len(text) and text[i + 1] == "'":
                pieces.append(text[i + 1])
                i += 2
                continue
            in_quote = not in_quote
        elif char == '/' and not in_quote:
            break
        else:
            pieces.append(char)
        i += 1
    return ''.join(pieces).strip()


def parse_fits_value(text):
    value = split_fits_value_comment(text)
    if not value:
        return ''
    if value.startswith("'"):
        chars = []
        i = 1
        while i < len(value):
            char = value[i]
            if char == "'":
                if i + 1 < len(value) and value[i + 1] == "'":
                    chars.append("'")
                    i += 2
                    continue
                break
            chars.append(char)
            i += 1
        return ''.join(chars).strip()
    if value in ('T', 'F'):
        return value == 'T'
    numeric = value.replace('D', 'E').replace('d', 'e')
    try:
        if any(char in numeric for char in ('.', 'E', 'e')):
            return float(numeric)
        return int(numeric)
    except ValueError:
        return value


def parse_fits_card(card):
    key = card[:8].strip()
    if not key or key == 'END':
        return key, None
    if len(card) >= 10 and card[8] == '=':
        return key, parse_fits_value(card[10:])
    return key, None


def read_fits_header_block(f):
    header = {}
    read_any = False
    while True:
        block = f.read(FITS_BLOCK_SIZE)
        if not block:
            return header if read_any else None
        read_any = True
        if len(block) != FITS_BLOCK_SIZE:
            raise RuntimeError('Unexpected end of FITS header block.')
        for i in range(0, FITS_BLOCK_SIZE, FITS_CARD_SIZE):
            card = block[i:i + FITS_CARD_SIZE].decode('ascii', errors='replace')
            key, value = parse_fits_card(card)
            if key == 'END':
                return header
            if value is not None:
                header[key] = value


def padded_fits_data_size(header):
    bitpix = int(header.get('BITPIX', 0) or 0)
    naxis = int(header.get('NAXIS', 0) or 0)
    if bitpix == 0 or naxis == 0:
        data_size = 0
    else:
        data_size = abs(bitpix) // 8
        for axis in range(1, naxis + 1):
            data_size *= int(header.get(f'NAXIS{axis}', 0) or 0)
    data_size += int(header.get('PCOUNT', 0) or 0)
    data_size *= int(header.get('GCOUNT', 1) or 1)
    if data_size == 0:
        return 0
    return ((data_size + FITS_BLOCK_SIZE - 1) // FITS_BLOCK_SIZE) * FITS_BLOCK_SIZE


def iter_fits_headers(path):
    with open(path, 'rb') as f:
        while True:
            header = read_fits_header_block(f)
            if header is None:
                return
            yield header
            skip_bytes = padded_fits_data_size(header)
            if skip_bytes:
                f.seek(skip_bytes, os.SEEK_CUR)


def read_hod_fits_header(path):
    for header in iter_fits_headers(path):
        if all(parameter in header for parameter in HOD_PARAMS):
            return header
    raise RuntimeError(f'No HOD parameter header found in {path}')


def discover_hod_fits_files(hod_header_dir):
    hod_header_dir = Path(hod_header_dir).expanduser().resolve()
    by_hod = {}
    for path in hod_header_dir.glob('hod*.fits'):
        by_hod[normalize_hod(fits_stem(path))] = path
    for path in hod_header_dir.glob('hod*.fits.gz'):
        by_hod.setdefault(normalize_hod(fits_stem(path)), path)
    return [by_hod[hod] for hod in sorted(by_hod, key=lambda value: int(value[3:]))]


def load_hod_params_from_fits(hod_header_dir, cosmo, columns):
    paths = discover_hod_fits_files(hod_header_dir)
    if not paths:
        raise RuntimeError(f'No HOD FITS files found in {hod_header_dir}')

    params = {}
    rows = []
    for path in paths:
        if path.suffix == '.gz':
            raise RuntimeError(f'Compressed FITS headers are not supported without astropy/fitsio: {path}')
        hod = normalize_hod(fits_stem(path))
        header = read_hod_fits_header(path)
        row = {'cosmo': cosmo, 'hod': hod, 'filename': path.name}
        for column in columns:
            row[column] = header.get(column, '')
        rows.append(row)
        params[hod] = {parameter: parse_float(header.get(parameter, ''))
                       for parameter in HOD_PARAMS}
    return params, rows


def load_hod_params(path, columns):
    params = {}
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {key.strip(): str(value).strip() if value is not None else ''
                   for key, value in raw_row.items() if key is not None}
            hod_value = row.get('hod') or row.get('filename', '').replace('.fits', '')
            if not hod_value:
                continue
            hod = normalize_hod(hod_value)
            params[hod] = {column: parse_float(row.get(column, '')) for column in columns}
            row['hod'] = hod
            rows.append(row)
    return params, rows


def load_hod_param_table(params_file, hod_header_dir, cosmo):
    header_dir = Path(hod_header_dir).expanduser().resolve()
    if header_dir.is_dir():
        params, rows = load_hod_params_from_fits(header_dir, cosmo, HOD_HEADER_COLUMNS)
        return params, rows, 'fits_headers', str(header_dir)

    params_path = Path(params_file).expanduser().resolve()
    params, rows = load_hod_params(params_path, HOD_PARAMS)
    table_rows = []
    for row in rows:
        table_row = {'cosmo': row.get('cosmo') or cosmo,
                     'hod': row['hod'],
                     'filename': row.get('filename') or f"{row['hod']}.fits"}
        for column in HOD_HEADER_COLUMNS:
            table_row[column] = row.get(column, '')
        table_rows.append(table_row)
    return params, table_rows, 'csv_fallback', str(params_path)


def write_hod_parameter_table(path, rows):
    headers = ['cosmo', 'hod', 'filename'] + list(HOD_HEADER_COLUMNS)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(normalize_hod(item['hod'])[3:])):
            writer.writerow(row)


def discover_pk_files(pk_root, requested_fields, pk_kind, pk_file_field,
                      hod_filter, cosmo_filter):
    from calc_deriv_cosmo import parse_pk_file

    candidates = []
    for path in sorted(Path(pk_root).expanduser().resolve().rglob('pk_*.csv')):
        pk_file = parse_pk_file(path, requested_fields, pk_kind)
        if pk_file is None:
            continue
        if pk_file.coverage_score < len(requested_fields):
            continue
        if pk_file_field and pk_file.field_request != pk_file_field:
            continue
        if hod_filter and pk_file.hod not in hod_filter:
            continue
        if cosmo_filter and pk_file.cosmo not in cosmo_filter:
            continue
        candidates.append(pk_file)

    best_by_realization = {}
    for pk_file in candidates:
        key = (pk_file.hod, pk_file.cosmo, pk_file.phase, pk_file.seed)
        current = best_by_realization.get(key)
        score = (pk_file.coverage_score,
                 pk_file.field_request == 'all',
                 len(pk_file.available_fields),
                 str(pk_file.path))
        if current is None or score > current[0]:
            best_by_realization[key] = (score, pk_file)

    files_by_hod_cosmo = defaultdict(list)
    for _, pk_file in best_by_realization.values():
        files_by_hod_cosmo[(pk_file.hod, pk_file.cosmo)].append(pk_file)

    for key in list(files_by_hod_cosmo):
        files_by_hod_cosmo[key] = sorted(files_by_hod_cosmo[key],
                                         key=lambda item: (int(item.phase), int(item.seed), str(item.path)))
    return files_by_hod_cosmo


def parameter_scales(hod_params, hods, parameters):
    scales = {}
    for parameter in parameters:
        values = np.asarray([hod_params[hod][parameter] for hod in hods
                             if hod in hod_params and np.isfinite(hod_params[hod][parameter])],
                            dtype=np.float64)
        if values.size < 2:
            raise RuntimeError(f'Need at least two finite HOD values for {parameter}.')
        scale = float(np.nanstd(values, ddof=0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(f'HOD parameter {parameter} has zero standard deviation.')
        scales[parameter] = scale
    return scales


def parameter_vector(hod_params, hod, parameters):
    return np.asarray([hod_params[hod][parameter] for parameter in parameters],
                      dtype=np.float64)


def scale_vector(scales, parameters):
    return np.asarray([scales[parameter] for parameter in parameters],
                      dtype=np.float64)


def finite_parameter_hods(hod_params, hods, parameters):
    finite_hods = []
    for hod in hods:
        if hod not in hod_params:
            continue
        values = parameter_vector(hod_params, hod, parameters)
        if np.all(np.isfinite(values)):
            finite_hods.append(hod)
    return finite_hods


def resolve_fiducial_hod(hod_params, hods, parameters, scales, requested,
                         center_hods=None):
    finite_hods = finite_parameter_hods(hod_params, hods, parameters)
    if not finite_hods:
        raise RuntimeError('No HODs have finite values for all fit parameters.')

    mode = str(requested).strip().lower()
    center_modes = {'center', 'centre', 'nearest-center', 'nearest_center',
                    'lhc-center', 'lhc_center'}
    if mode in center_modes:
        center_pool = center_hods if center_hods is not None else hods
        finite_center_hods = finite_parameter_hods(hod_params, center_pool,
                                                   parameters)
        if not finite_center_hods:
            raise RuntimeError('No HODs are available to define the LHC center.')
        values = np.vstack([parameter_vector(hod_params, hod, parameters)
                            for hod in finite_center_hods])
        center = np.nanmean(values, axis=0)
        scales_array = scale_vector(scales, parameters)
        best = None
        for hod in finite_hods:
            row = parameter_vector(hod_params, hod, parameters)
            offsets = (row - center) / scales_array
            distance = float(np.sqrt(np.sum(offsets ** 2)))
            candidate = (distance, int(hod[3:]), hod)
            if best is None or candidate < best:
                best = candidate
        distance, _, hod = best
        center_values = {parameter: float(value)
                         for parameter, value in zip(parameters, center)}
        return hod, 'nearest_lhc_center', center_values, distance

    hod = normalize_hod(requested)
    if hod not in hod_params:
        raise RuntimeError(f'Fiducial HOD {hod} is missing from the HOD parameter table.')
    if hod not in set(hods):
        raise RuntimeError(f'Fiducial HOD {hod} is not in the candidate P(k) pool.')
    if hod not in finite_hods:
        raise RuntimeError(f'Fiducial HOD {hod} has non-finite fit parameters.')
    return hod, 'requested', {parameter: float(hod_params[hod][parameter])
                              for parameter in parameters}, 0.0


def build_pk_means(files_by_hod_cosmo, hods, cosmos, fields, pk_kind,
                   observable, strict_k_range):
    metadata_cache = {}
    pk_means = {}
    reference = None

    for cosmo in cosmos:
        for hod in hods:
            files = files_by_hod_cosmo.get((hod, cosmo), [])
            if not files:
                continue
            pk = average_pk(files, fields, pk_kind, strict_k_range,
                            observable, metadata_cache)
            if reference is None:
                reference = pk
            else:
                pk = align_pk_to_reference(pk, reference, fields, strict_k_range)
            pk_means[(hod, cosmo)] = pk

    if reference is None:
        raise RuntimeError('No P(k) files could be loaded for HOD derivatives.')
    return pk_means, reference


def local_fit_formula(derivative_kind, fit_intercept):
    left = 'O_h(k) - O_fid(k)'
    derivative = 'dO/dtheta_i'
    if derivative_kind == 'log':
        left = 'ln O_h(k) - ln O_fid(k)'
        derivative = 'd ln O/dtheta_i'
    intercept = '1, ' if fit_intercept else ''
    return (f'{left} = [{intercept}x_1(h), ..., x_p(h)] beta(k), '
            'with x_i=(theta_i(h)-theta_i(fid))/sigma_i and '
            f'{derivative} = beta_i/sigma_i')


def normalized_offsets(hod_params, hod, fiducial_hod, parameters, scales):
    delta = parameter_vector(hod_params, hod, parameters) - parameter_vector(
        hod_params, fiducial_hod, parameters)
    return delta / scale_vector(scales, parameters)


def select_local_neighbors(hod_params, available_hods, fiducial_hod,
                           parameters, scales, n_neighbors):
    neighbors = []
    for hod in available_hods:
        if hod == fiducial_hod or hod not in hod_params:
            continue
        offsets = normalized_offsets(hod_params, hod, fiducial_hod,
                                     parameters, scales)
        if not np.all(np.isfinite(offsets)):
            continue
        distance = float(np.sqrt(np.sum(offsets ** 2)))
        if not np.isfinite(distance) or distance == 0.0:
            continue
        theta = {parameter: float(hod_params[hod][parameter])
                 for parameter in parameters}
        offset_map = {parameter: float(value)
                      for parameter, value in zip(parameters, offsets)}
        neighbors.append(LocalNeighbor(hod=hod,
                                       distance=distance,
                                       offsets=offsets,
                                       theta=theta,
                                       normalized_offsets=offset_map))

    neighbors = sorted(neighbors, key=lambda item: (item.distance, int(item.hod[3:])))
    if n_neighbors > 0:
        neighbors = neighbors[:int(n_neighbors)]
    return neighbors


def build_design_matrix(neighbors, fit_intercept):
    design = np.vstack([neighbor.offsets for neighbor in neighbors]).astype(np.float64)
    if fit_intercept:
        design = np.column_stack([np.ones(design.shape[0], dtype=np.float64), design])
    return design


def observable_difference(values, fiducial_values, derivative_kind):
    if derivative_kind == 'linear':
        return np.asarray(values, dtype=np.float64) - np.asarray(fiducial_values, dtype=np.float64)
    if derivative_kind == 'log':
        return safe_log(values) - safe_log(fiducial_values)
    raise ValueError(f'Unknown derivative kind: {derivative_kind}')


def fit_local_plane(design, response, scales_array, fit_intercept):
    response = np.asarray(response, dtype=np.float64)
    n_params = int(scales_array.size)
    n_values = int(response.shape[1])
    required_rank = n_params + (1 if fit_intercept else 0)
    finite_design = np.all(np.isfinite(design), axis=1)

    slopes = np.full((n_params, n_values), np.nan, dtype=np.float64)
    intercepts = np.full(n_values, np.nan, dtype=np.float64)
    residual_rms = np.full(n_values, np.nan, dtype=np.float64)
    n_used = np.zeros(n_values, dtype=np.int64)
    ranks = np.zeros(n_values, dtype=np.int64)

    for i in range(n_values):
        mask = finite_design & np.isfinite(response[:, i])
        n_used[i] = int(np.sum(mask))
        if n_used[i] < required_rank:
            continue

        x = design[mask]
        y = response[mask, i]
        rank = int(np.linalg.matrix_rank(x))
        ranks[i] = rank
        if rank < required_rank:
            continue

        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ beta
        residual_rms[i] = float(np.sqrt(np.mean((y - fitted) ** 2)))
        if fit_intercept:
            intercepts[i] = beta[0]
            beta = beta[1:]
        slopes[:, i] = beta / scales_array

    diagnostics = {'required_rank': int(required_rank),
                   'min_n_used': int(np.min(n_used)) if n_used.size else 0,
                   'max_n_used': int(np.max(n_used)) if n_used.size else 0,
                   'min_rank': int(np.min(ranks)) if ranks.size else 0,
                   'max_rank': int(np.max(ranks)) if ranks.size else 0,
                   'mean_residual_rms': float(np.nanmean(residual_rms))
                   if np.any(np.isfinite(residual_rms)) else np.nan}
    return slopes, intercepts, residual_rms, diagnostics


def fit_local_derivatives_for_cosmo(cosmo, hod_params, pk_means, available_hods,
                                    fields, derivative_kind, fit_parameters,
                                    scales, fiducial_hod, n_neighbors,
                                    fit_intercept):
    fiducial_pk = pk_means.get((fiducial_hod, cosmo))
    if fiducial_pk is None:
        raise RuntimeError(f'Fiducial HOD {fiducial_hod} has no P(k) for {cosmo}.')

    neighbors = select_local_neighbors(hod_params, available_hods, fiducial_hod,
                                       fit_parameters, scales, n_neighbors)
    required_rank = len(fit_parameters) + (1 if fit_intercept else 0)
    if len(neighbors) < required_rank:
        raise RuntimeError(f'Need at least {required_rank} neighboring HODs for {cosmo}; '
                           f'found {len(neighbors)}.')

    design = build_design_matrix(neighbors, fit_intercept)
    design_rank = int(np.linalg.matrix_rank(design))
    if design_rank < required_rank:
        raise RuntimeError(f'Local HOD design matrix is rank deficient for {cosmo}: '
                           f'rank={design_rank}, required={required_rank}.')

    condition_number = float(np.linalg.cond(design))
    scales_array = scale_vector(scales, fit_parameters)
    derivatives = {}
    intercepts = {}
    residual_rms = {}
    field_diagnostics = {}

    for field in fields:
        response_rows = []
        for neighbor in neighbors:
            pk = pk_means[(neighbor.hod, cosmo)]
            pk = align_pk_to_reference(pk, fiducial_pk, fields, True)
            response_rows.append(
                observable_difference(pk['values'][field],
                                      fiducial_pk['values'][field],
                                      derivative_kind))
        response = np.vstack(response_rows)
        slopes, field_intercepts, field_residual_rms, diagnostics = fit_local_plane(
            design, response, scales_array, fit_intercept)
        derivatives[field] = slopes
        intercepts[field] = field_intercepts
        residual_rms[field] = field_residual_rms
        field_diagnostics[field] = diagnostics

    max_distance = max(neighbor.distance for neighbor in neighbors) if neighbors else np.nan
    return LocalFit(cosmo=cosmo,
                    fiducial_hod=fiducial_hod,
                    n_neighbors=len(neighbors),
                    neighbors=neighbors,
                    design_rank=design_rank,
                    required_rank=required_rank,
                    condition_number=condition_number,
                    max_distance=float(max_distance),
                    derivatives=derivatives,
                    intercepts=intercepts,
                    residual_rms=residual_rms,
                    field_diagnostics=field_diagnostics)


def compute_local_derivatives(hod_params, pk_means, hods_by_cosmo, fields,
                              observable, derivative_kind, output_parameters,
                              fit_parameters, scales, fiducial_hod,
                              n_neighbors, fit_intercept):
    parameter_indices = {parameter: fit_parameters.index(parameter)
                         for parameter in output_parameters}

    if len(hods_by_cosmo) != 1:
        raise RuntimeError('Expected exactly one fiducial cosmology for HOD derivatives; '
                           f'got {", ".join(sorted(hods_by_cosmo))}.')
    cosmo, available_hods = next(iter(hods_by_cosmo.items()))
    if fiducial_hod not in available_hods:
        raise RuntimeError(f'No cosmology has P(k) for fiducial HOD {fiducial_hod}.')

    fit = fit_local_derivatives_for_cosmo(
        cosmo, hod_params, pk_means, available_hods, fields,
        derivative_kind, fit_parameters, scales, fiducial_hod,
        n_neighbors, fit_intercept)
    local_fits = {cosmo: fit}
    first_pk = pk_means[(fiducial_hod, cosmo)]
    used_hods = sorted({fiducial_hod}
                       | {neighbor.hod for neighbor in fit.neighbors},
                       key=lambda value: int(value[3:]))
    results = {}

    for parameter in output_parameters:
        derivatives = {}
        derivative_std = {}
        n_samples = {}
        for field in fields:
            derivative = fit.derivatives[field][parameter_indices[parameter]]
            derivatives[field] = derivative
            derivative_std[field] = np.zeros_like(derivative)
            n_samples[field] = 1
            if not np.any(np.isfinite(derivatives[field])):
                raise RuntimeError(f'No finite local HOD derivative values for {parameter}/{field}.')

        results[parameter] = DerivativeResult(
            parameter=parameter,
            param_column=parameter,
            observable=observable,
            derivative_kind=derivative_kind,
            fit_method='local_linear_regression',
            fiducial_hod=fiducial_hod,
            fit_parameters=fit_parameters,
            scales=scales,
            used_hods=used_hods,
            n_hods=len(used_hods),
            n_samples=n_samples,
            local_fits_by_cosmo=local_fits,
            k=first_pk['k'],
            k_min=first_pk['k_min'],
            k_max=first_pk['k_max'],
            derivatives=derivatives,
            derivative_std=derivative_std)
    return results, local_fits


def latex_parameter_label(parameter):
    labels = {'LOGM_CUT': r'\log M_{\rm cut}',
              'LOGM1': r'\log M_1',
              'SIGMA': r'\sigma',
              'ALPHA': r'\alpha',
              'KAPPA': r'\kappa'}
    return labels.get(parameter, parameter.replace('_', r'\_'))


def plot_field_style(field):
    colors = {'matter': '#ffffff',
              'void': '#8dd3c7',
              'sheet': '#ffff99',
              'filament': '#bebada',
              'knot': '#fb8072'}
    labels = {'matter': 'pmatter',
              'void': 'pvoid',
              'sheet': 'psheet',
              'filament': 'pfilament',
              'knot': 'pknot'}
    return colors.get(field, 'tab:cyan'), labels.get(field, field)


def load_pyplot():
    if 'MPLCONFIGDIR' not in os.environ:
        mplconfig = Path(os.environ.get('TMPDIR', '/tmp')) / 'matplotlib-cache'
        mplconfig.mkdir(parents=True, exist_ok=True)
        os.environ['MPLCONFIGDIR'] = str(mplconfig)
    try:
        import matplotlib
        matplotlib.use('Agg')
        matplotlib.rcParams['text.usetex'] = False
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
    except Exception as exc:
        print(f'---> plot skipped: matplotlib is not available ({exc})')
        return None
    return plt


def write_derivative_plot(path, title, results, fields, parameters):
    plt = load_pyplot()
    if plt is None:
        return False
    first = check_result_k_range(results)
    ylabel = derivative_axis_label(first)
    n_cols = len(parameters)
    fig_width = max(8.0, 4.4 * n_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, 5.2),
                             squeeze=False, sharex=True)
    axes = axes[0]

    handles = []
    labels = []
    ordered_fields = [field for field in FIELDS if field in fields]
    for col, parameter in enumerate(parameters):
        ax = axes[col]
        for field in ordered_fields:
            color, label = plot_field_style(field)
            deriv = np.asarray(results[parameter].derivatives[field], dtype=np.float64)
            deriv_std = np.asarray(results[parameter].derivative_std.get(field, []),
                                   dtype=np.float64)
            mask = np.isfinite(first.k) & np.isfinite(deriv)
            if np.any(mask):
                line, = ax.plot(first.k[mask], deriv[mask], marker='o',
                                lw=1.8, color=color, label=label)
                if col == 0:
                    handles.append(line)
                    labels.append(label)
            if deriv_std.shape == deriv.shape:
                band_mask = mask & np.isfinite(deriv_std)
                if np.any(band_mask):
                    lower = deriv[band_mask] - deriv_std[band_mask]
                    upper = deriv[band_mask] + deriv_std[band_mask]
                    ax.fill_between(first.k[band_mask], lower, upper, color=color,
                                    alpha=0.16, linewidth=0.0)

        ax.axhline(0.0, color='0.7', lw=0.9, ls='--', alpha=0.85)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xscale('log')
        ax.set_title(latex_parameter_label(parameter), fontsize=15)
        ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
        if col == 0:
            ax.set_ylabel(ylabel)

    fig.suptitle(title, fontsize=13)
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(handles),
                   bbox_to_anchor=(0.5, 0.94), frameon=True)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return True


def neighbor_to_metadata(neighbor):
    return {'hod': neighbor.hod,
            'distance': neighbor.distance,
            'theta': neighbor.theta,
            'normalized_offsets': neighbor.normalized_offsets}


def local_fit_to_metadata(fit):
    return {'cosmo': fit.cosmo,
            'fiducial_hod': fit.fiducial_hod,
            'n_neighbors': fit.n_neighbors,
            'design_rank': fit.design_rank,
            'required_rank': fit.required_rank,
            'condition_number': fit.condition_number,
            'max_distance': fit.max_distance,
            'field_diagnostics': fit.field_diagnostics,
            'neighbors': [neighbor_to_metadata(neighbor)
                          for neighbor in fit.neighbors]}


def write_metadata(path, args, candidate_hods, candidate_cosmos, results, fields,
                   parameters, outputs, scales, fiducial_hod,
                   fiducial_selection, fiducial_target, fiducial_parameters,
                   fiducial_center_distance, local_fits, hod_param_source,
                   hod_param_source_path, fiducial_cosmo):
    first = next(iter(results.values()))
    metadata = {'pk_root': str(Path(args.pk_root).expanduser().resolve()),
                'params_file': str(Path(args.params_file).expanduser().resolve()),
                'hod_header_dir': str(Path(args.hod_header_dir).expanduser().resolve()),
                'hod_parameter_source': hod_param_source,
                'hod_parameter_source_path': hod_param_source_path,
                'pk_kind': args.pk_kind,
                'observable': args.observable,
                'observable_description': observable_description(args.observable),
                'derivative_kind': args.derivative_kind,
                'fields': fields,
                'parameters': parameters,
                'fit_parameters': list(first.fit_parameters),
                'fiducial_cosmo': fiducial_cosmo,
                'candidate_hods': candidate_hods,
                'candidate_cosmos': candidate_cosmos,
                'fit_method': 'local_linear_regression',
                'fit_formula': local_fit_formula(args.derivative_kind,
                                                 args.fit_intercept),
                'fiducial_hod_requested': args.fiducial_hod,
                'fiducial_hod': fiducial_hod,
                'fiducial_selection': fiducial_selection,
                'fiducial_target': fiducial_target,
                'fiducial_center_distance': fiducial_center_distance,
                'fiducial_parameters': fiducial_parameters,
                'parameter_scales': {parameter: float(scales[parameter])
                                     for parameter in first.fit_parameters},
                'parameter_scale_definition': (
                    'population standard deviation over the HOD parameter table'),
                'n_neighbors_requested': args.n_neighbors,
                'fit_intercept': args.fit_intercept,
                'neighbor_rule': (
                    'nearest HODs by Euclidean distance in normalized 5D HOD-parameter space'),
                'strict_k_range': args.strict_k_range,
                'plot': args.plot,
                'local_fits_by_cosmo': {
                    cosmo: local_fit_to_metadata(fit)
                    for cosmo, fit in local_fits.items()},
                'parameter_fits': {},
                'outputs': outputs}
    for parameter in parameters:
        result = results[parameter]
        metadata['parameter_fits'][parameter] = {
            'param_column': result.param_column,
            'fit_method': result.fit_method,
            'fiducial_hod': result.fiducial_hod,
            'used_hods': result.used_hods,
            'n_hods': result.n_hods,
            'n_samples_by_field': result.n_samples}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def main():
    args = parse_args()
    t0 = time.time()

    fields = normalize_fields(args.fields)
    parameters = normalize_params(args.params)

    hod_values = {normalize_hod(value) for value in args.hod}
    hod_filter = None if 'all' in hod_values else hod_values
    fiducial_cosmo = resolve_single_cosmo(args.cosmo)
    cosmo_filter = {fiducial_cosmo}

    pk_root = Path(args.pk_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else pk_root / 'hod_derivatives'
    outdir.mkdir(parents=True, exist_ok=True)

    hod_params, hod_param_rows, hod_param_source, hod_param_source_path = (
        load_hod_param_table(args.params_file, args.hod_header_dir, fiducial_cosmo))
    files_by_hod_cosmo = discover_pk_files(pk_root, fields, args.pk_kind,
                                           args.pk_file_field, hod_filter,
                                           cosmo_filter)
    discovered_hods = sorted({hod for hod, _ in files_by_hod_cosmo},
                             key=lambda value: int(value[3:]))
    hods = sorted([hod for hod in discovered_hods if hod in hod_params],
                  key=lambda value: int(value[3:]))
    cosmos = sorted({cosmo for _, cosmo in files_by_hod_cosmo})
    if not hods:
        raise RuntimeError(f'No P(k) files with matching HOD parameters found in {pk_root}.')
    if cosmos != [fiducial_cosmo]:
        raise RuntimeError(f'No P(k) files found for fiducial cosmology {fiducial_cosmo} in {pk_root}.')
    missing_param_hods = [hod for hod in discovered_hods if hod not in hod_params]
    if missing_param_hods:
        print('---> warning: skipping HODs missing from parameter table: '
              f"{' '.join(missing_param_hods)}")

    fit_parameters = list(HOD_PARAMS)
    scale_hods = sorted(hod_params, key=lambda value: int(value[3:]))
    scales = parameter_scales(hod_params, scale_hods, fit_parameters)
    fiducial_hod, fiducial_selection, fiducial_target, fiducial_center_distance = (
        resolve_fiducial_hod(hod_params, hods, fit_parameters, scales,
                             args.fiducial_hod, center_hods=scale_hods))
    fiducial_parameters = {parameter: float(hod_params[fiducial_hod][parameter])
                           for parameter in fit_parameters}
    hods_by_cosmo = {}
    for cosmo in cosmos:
        hods_by_cosmo[cosmo] = sorted(
            [hod for hod in hods if (hod, cosmo) in files_by_hod_cosmo],
            key=lambda value: int(value[3:]))

    pk_means, _ = build_pk_means(files_by_hod_cosmo, hods, cosmos, fields,
                                 args.pk_kind, args.observable, args.strict_k_range)

    print(f'---> pk root: {pk_root}')
    print(f'---> output directory: {outdir}')
    print(f'---> HOD parameter source: {hod_param_source} ({hod_param_source_path})')
    print(f'---> fiducial cosmology for HOD derivatives: {fiducial_cosmo}')
    print(f'---> HODs in candidate pool: {len(hods)}')
    print(f'---> cosmologies in candidate pool: {len(cosmos)}')
    print(f'---> fields: {fields}')
    print(f'---> HOD parameters: {parameters}')
    print(f'---> observable: {args.observable} ({observable_description(args.observable)})')
    print(f'---> derivative kind: {args.derivative_kind}')
    print(f'---> fit method: local linear regression')
    print(f'---> fiducial HOD: {fiducial_hod} ({fiducial_selection})')
    print(f'---> nearest HODs requested: {args.n_neighbors if args.n_neighbors > 0 else "all"}')

    summary = {'elapsed_sec': None,
               'pk_root': str(pk_root),
               'outdir': str(outdir),
               'fields': fields,
               'parameters': parameters,
               'fiducial_cosmo': fiducial_cosmo,
               'cosmos': cosmos,
               'observable': args.observable,
               'derivative_kind': args.derivative_kind,
               'fit_method': 'local_linear_regression',
               'hod_parameter_source': hod_param_source,
               'hod_parameter_source_path': hod_param_source_path,
               'fiducial_hod': fiducial_hod,
               'n_neighbors_requested': args.n_neighbors,
               'candidate_hods': hods,
               'outputs': None}

    results, local_fits = compute_local_derivatives(
        hod_params, pk_means, hods_by_cosmo, fields, args.observable,
        args.derivative_kind, parameters, fit_parameters, scales,
        fiducial_hod, args.n_neighbors, args.fit_intercept)

    for cosmo, fit in local_fits.items():
        print(f'---> {cosmo}: neighbors={fit.n_neighbors}, '
              f'rank={fit.design_rank}/{fit.required_rank}, '
              f'max_distance={fit.max_distance:.6g}')

    for parameter in parameters:
        result = results[parameter]
        print(f'---> {parameter}: HODs={result.n_hods}, '
              f'fit samples={result.n_samples}')

    active_parameters = [parameter for parameter in parameters if parameter in results]
    if not active_parameters:
        raise RuntimeError('No HOD derivative results were produced.')

    plot_outdir = outdir / 'plots'
    plot_outdir.mkdir(parents=True, exist_ok=True)
    matrix_path = outdir / 'deriv_hod.csv'
    combined_path = outdir / 'deriv_hod_combined_env.csv'
    meta_path = outdir / 'deriv_hod_metadata.json'
    hod_table_path = outdir / 'hod_parameter_table.csv'
    plot_path = plot_outdir / 'deriv_hod.png'

    write_hod_parameter_table(hod_table_path, hod_param_rows)
    write_derivative_matrix(matrix_path, results, fields, active_parameters)
    combined_written = write_combined_environment_matrix(combined_path, results, fields, active_parameters)
    plot_written = False
    if args.plot:
        plot_written = write_derivative_plot(
            plot_path, 'HOD local-linear derivatives',
            results, fields, active_parameters)
    outputs = {'matrix': str(matrix_path),
               'combined_environment_matrix': str(combined_path) if combined_written else None,
               'hod_parameter_table': str(hod_table_path),
               'plot': str(plot_path) if plot_written else None,
               'metadata': str(meta_path)}
    write_metadata(meta_path, args, hods, cosmos, results, fields,
                   active_parameters, outputs, scales, fiducial_hod,
                   fiducial_selection, fiducial_target, fiducial_parameters,
                   fiducial_center_distance, local_fits, hod_param_source,
                   hod_param_source_path, fiducial_cosmo)
    summary['outputs'] = outputs

    print(f'---> wrote: {hod_table_path}')
    print(f'---> wrote: {matrix_path}')
    if combined_written:
        print(f'---> wrote: {combined_path}')
    if plot_written:
        print(f'---> wrote: {plot_path}')
    print(f'---> wrote: {meta_path}')

    elapsed = time.time() - t0
    summary['elapsed_sec'] = elapsed
    summary_path = outdir / f'deriv_hod_summary_{int(time.time())}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'---> wrote: {summary_path}')
    print(f'---> elapsed: {elapsed:.2f} s')


if __name__ == '__main__':
    main()