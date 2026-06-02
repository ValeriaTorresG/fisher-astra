import argparse, csv, json, os, re, time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
plt.style.use('dark_background')


DEFAULT_PK_ROOT = '/pscratch/sd/v/vtorresg/hod-astra-box500/power_spectra'
DEFAULT_PARAMS_FILE = 'params/params_cosmo.csv'
FIELDS = ('matter', 'void', 'sheet', 'filament', 'knot')
ENVIRONMENT_FIELDS = ('void', 'sheet', 'filament', 'knot')
PARAM_SPECS = {'Omega_b': {'column': 'omega_b', 'baseline': 'c000', 'plus': 'c100', 'minus': 'c101'},
               'omega_cdm': {'column': 'omega_cdm', 'baseline': 'c000', 'plus': 'c102', 'minus': 'c103'},
               'n_s': {'column': 'n_s', 'baseline': 'c000', 'plus': 'c104', 'minus': 'c105'},
               'sigma_8m': {'column': 'sigma8_m', 'baseline': 'c000', 'plus': 'c112', 'minus': 'c113'}}

PARAM_ALIASES = {'Omega_b': 'Omega_b',
                 'omega_b': 'Omega_b',
                 'omega_cdm': 'omega_cdm',
                 'n_s': 'n_s',
                 'sigma_8m': 'sigma_8m',
                 'sigma8_m': 'sigma_8m'}
FINITE_DIFFERENCE_SCHEMES = ('central', 'forward', 'backward')
PK_FILE_RE = re.compile(r'^pk_(?P<field_request>.+?)_zone_(?P<cosmo>c\d{3})_ph(?P<phase>\d+)'
                        r'_seed(?P<seed>\d+)_hod(?P<hod>\d+)\.csv$')


PkFile = SimpleNamespace
DerivativeResult = SimpleNamespace


def add_bool_argument(parser, name, default, help_text):
    dest = name.lstrip('-').replace('-', '_')
    group = parser.add_mutually_exclusive_group()
    group.add_argument(name, dest=dest, action='store_true', help=help_text)
    group.add_argument(f'--no-{name[2:]}', dest=dest, action='store_false',
                       help=f'Disable: {help_text}')
    parser.set_defaults(**{dest: default})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pk-root', type=str, default=DEFAULT_PK_ROOT)
    parser.add_argument('--params-file', type=str, default=DEFAULT_PARAMS_FILE)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--fields', nargs='+', default=['all'], choices=['all'] + list(FIELDS))
    parser.add_argument('--params', nargs='+', default=list(PARAM_SPECS), choices=list(PARAM_ALIASES))
    parser.add_argument('--hod', nargs='+', default=['all'])
    parser.add_argument('--pk-kind', type=str, default='pk_used', choices=['pk_used', 'pk_raw'])
    parser.add_argument('--finite-difference', type=str, default='central', choices=FINITE_DIFFERENCE_SCHEMES)
    parser.add_argument('--pk-file-field', type=str, default='')
    add_bool_argument(parser, '--match-realizations', True)
    add_bool_argument(parser, '--strict-bins', True)
    add_bool_argument(parser, '--plot', True)
    parser.add_argument('--skip-missing', action='store_true')
    return parser.parse_args()


def normalize_cosmo(value):
    value = str(value).strip().lower()
    if value.startswith('abacus_cosm'):
        value = value.replace('abacus_cosm', 'c')
    if value.startswith('cosm'):
        value = value.replace('cosm', 'c', 1)
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


def normalize_fields(values):
    if any(value == 'all' for value in values):
        return list(FIELDS)
    fields = []
    for value in values:
        if value not in fields:
            fields.append(value)
    return fields


def normalize_params(values):
    params = []
    for value in values:
        name = PARAM_ALIASES[value]
        if name not in params:
            params.append(name)
    return params


def parse_float(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return np.nan


def load_cosmo_params(path):
    params = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {key.strip(): str(value).strip() if value is not None else ''
                   for key, value in raw_row.items() if key is not None}
            root = row.get('root', '')
            if not root:
                continue
            cosmo = normalize_cosmo(root)
            params[cosmo] = {'root': root}
            for spec in PARAM_SPECS.values():
                column = spec['column']
                params[cosmo][column] = parse_float(row.get(column, ''))
    return params


def read_header(path):
    with open(path, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
    return [col.strip() for col in line.split(',') if col.strip()]


def available_fields_from_header(header, pk_kind):
    prefix = f'{pk_kind}_'
    return tuple(col[len(prefix):] for col in header
                 if col.startswith(prefix) and col[len(prefix):] in FIELDS)


def parse_pk_file(path, requested_fields, pk_kind):
    match = PK_FILE_RE.match(path.name)
    if not match:
        return None
    info = match.groupdict()
    header = read_header(path)
    available_fields = available_fields_from_header(header, pk_kind)
    coverage_score = sum(field in available_fields for field in requested_fields)
    return PkFile(path=path,
                  field_request=info['field_request'],
                  cosmo=info['cosmo'],
                  phase=info['phase'],
                  seed=info['seed'],
                  hod=f"hod{int(info['hod']):03d}",
                  available_fields=available_fields,
                  coverage_score=coverage_score)


def discover_pk_files(pk_root, requested_fields, pk_kind, pk_file_field, hod_filter):
    candidates = []
    for path in sorted(Path(pk_root).expanduser().resolve().rglob('pk_*.csv')):
        pk_file = parse_pk_file(path, requested_fields, pk_kind)
        if pk_file is None:
            continue
        if pk_file.coverage_score == 0:
            continue
        if pk_file_field and pk_file.field_request != pk_file_field:
            continue
        if hod_filter and pk_file.hod not in hod_filter:
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
        if current is None:
            best_by_realization[key] = (score, pk_file)
            continue
        if score > current[0]:
            best_by_realization[key] = (score, pk_file)

    files_by_hod_cosmo = defaultdict(list)
    for _, pk_file in best_by_realization.values():
        files_by_hod_cosmo[(pk_file.hod, pk_file.cosmo)].append(pk_file)

    for key in list(files_by_hod_cosmo):
        files_by_hod_cosmo[key] = sorted(files_by_hod_cosmo[key],
                                         key=lambda item: (int(item.phase), int(item.seed), str(item.path)))
    return files_by_hod_cosmo


def require_fields(pk_file, fields, pk_kind):
    missing = [field for field in fields if field not in pk_file.available_fields]
    if missing:
        columns = ', '.join(f'{pk_kind}_{field}' for field in missing)
        raise RuntimeError(f'Missing columns in {pk_file.path}: {columns}')


def load_pk_csv(path, fields, pk_kind):
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding=None)
    data = np.atleast_1d(data)
    names = set(data.dtype.names or [])
    required = ['k_h_mpc'] + [f'{pk_kind}_{field}' for field in fields]
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f'Missing columns in {path}: {", ".join(missing)}')

    k = np.asarray(data['k_h_mpc'], dtype=np.float64)
    k_min = np.asarray(data['k_min_h_mpc'], dtype=np.float64) if 'k_min_h_mpc' in names else np.full_like(k, np.nan)
    k_max = np.asarray(data['k_max_h_mpc'], dtype=np.float64) if 'k_max_h_mpc' in names else np.full_like(k, np.nan)
    values = {field: np.asarray(data[f'{pk_kind}_{field}'], dtype=np.float64)
              for field in fields}
    return {'k': k, 'k_min': k_min, 'k_max': k_max, 'values': values}


def same_bins(a, b):
    return (a['k'].shape == b['k'].shape
            and np.allclose(a['k'], b['k'], rtol=1.0e-8, atol=1.0e-12)
            and np.allclose(a['k_min'], b['k_min'], rtol=1.0e-8, atol=1.0e-12, equal_nan=True)
            and np.allclose(a['k_max'], b['k_max'], rtol=1.0e-8, atol=1.0e-12, equal_nan=True))


def align_pk_to_reference(pk, reference, fields, strict_bins):
    if same_bins(pk, reference):
        return pk
    if strict_bins:
        raise RuntimeError('P(k) files do not use identical k bins.')

    aligned = {'k': reference['k'].copy(),
               'k_min': reference['k_min'].copy(),
               'k_max': reference['k_max'].copy(),
               'values': {}}
    order = np.argsort(pk['k'])
    for field in fields:
        aligned['values'][field] = np.interp(reference['k'], pk['k'][order],
                                             pk['values'][field][order],
                                             left=np.nan, right=np.nan)
    return aligned


def average_pk(files, fields, pk_kind, strict_bins):
    reference = None
    sums = {field: None for field in fields}
    for pk_file in files:
        require_fields(pk_file, fields, pk_kind)
        pk = load_pk_csv(pk_file.path, fields, pk_kind)
        if reference is None:
            reference = pk
            for field in fields:
                sums[field] = pk['values'][field].astype(np.float64).copy()
            continue

        pk = align_pk_to_reference(pk, reference, fields, strict_bins)
        for field in fields:
            sums[field] += pk['values'][field]

    n_files = float(len(files))
    means = {field: sums[field] / n_files for field in fields}
    return {'k': reference['k'],
            'k_min': reference['k_min'],
            'k_max': reference['k_max'],
            'values': means}


def files_for_global_finite_difference(files_by_hod_cosmo, hods, numerator_cosmo,
                                       denominator_cosmo, match_realizations):
    numerator_files = []
    denominator_files = []
    realization_keys = []
    used_hods = set()

    for hod in hods:
        numerator_hod_files = files_by_hod_cosmo.get((hod, numerator_cosmo), [])
        denominator_hod_files = files_by_hod_cosmo.get((hod, denominator_cosmo), [])
        if not numerator_hod_files or not denominator_hod_files:
            continue

        if not match_realizations:
            numerator_files.extend(numerator_hod_files)
            denominator_files.extend(denominator_hod_files)
            used_hods.add(hod)
            continue

        numerator_by_key = {(item.phase, item.seed): item for item in numerator_hod_files}
        denominator_by_key = {(item.phase, item.seed): item for item in denominator_hod_files}
        keys = sorted(set(numerator_by_key) & set(denominator_by_key),
                      key=lambda item: (int(item[0]), int(item[1])))
        if not keys:
            continue

        used_hods.add(hod)
        for phase, seed in keys:
            numerator_files.append(numerator_by_key[(phase, seed)])
            denominator_files.append(denominator_by_key[(phase, seed)])
            realization_keys.append((hod, phase, seed))

    return (numerator_files, denominator_files, sorted(used_hods),
            realization_keys)


def files_for_hod_finite_difference(files_by_hod_cosmo, hod, numerator_cosmo,
                                    denominator_cosmo, match_realizations):
    numerator_files = files_by_hod_cosmo.get((hod, numerator_cosmo), [])
    denominator_files = files_by_hod_cosmo.get((hod, denominator_cosmo), [])
    if not match_realizations:
        return numerator_files, denominator_files, []

    numerator_by_key = {(item.phase, item.seed): item for item in numerator_files}
    denominator_by_key = {(item.phase, item.seed): item for item in denominator_files}
    keys = sorted(set(numerator_by_key) & set(denominator_by_key),
                  key=lambda item: (int(item[0]), int(item[1])))
    return [numerator_by_key[key] for key in keys], [denominator_by_key[key] for key in keys], keys


def finite_difference_cosmos(spec, scheme):
    if scheme == 'central':
        return spec['plus'], spec['minus']
    if scheme == 'forward':
        return spec['plus'], spec['baseline']
    if scheme == 'backward':
        return spec['baseline'], spec['minus']
    raise ValueError(f'Unknown finite-difference scheme: {scheme}')


def compute_derivative(parameter, params, files_by_hod_cosmo, hods, fields, pk_kind,
                       finite_difference, match_realizations, strict_bins):
    spec = PARAM_SPECS[parameter]
    numerator_cosmo, denominator_cosmo = finite_difference_cosmos(spec, finite_difference)
    param_column = spec['column']

    if numerator_cosmo not in params or denominator_cosmo not in params:
        raise RuntimeError(f'Parameter table is missing {numerator_cosmo} or {denominator_cosmo}.')

    theta_numerator = float(params[numerator_cosmo][param_column])
    theta_denominator = float(params[denominator_cosmo][param_column])
    delta_theta = theta_numerator - theta_denominator
    if not np.isfinite(delta_theta) or delta_theta == 0.0:
        raise RuntimeError(f'Invalid finite-difference step for {parameter}: {delta_theta}')

    numerator_files, denominator_files, used_hods, realization_keys = (
        files_for_global_finite_difference(
            files_by_hod_cosmo, hods, numerator_cosmo, denominator_cosmo,
            match_realizations))
    if not numerator_files or not denominator_files:
        raise RuntimeError(f'No matched P(k) files for {parameter}: '
                           f'{numerator_cosmo} has {len(numerator_files)}, '
                           f'{denominator_cosmo} has {len(denominator_files)}.')

    numerator_mean = average_pk(numerator_files, fields, pk_kind, strict_bins)
    denominator_mean = average_pk(denominator_files, fields, pk_kind, strict_bins)
    denominator_mean = align_pk_to_reference(denominator_mean, numerator_mean, fields, strict_bins)

    derivatives = {}
    for field in fields:
        derivatives[field] = (
            numerator_mean['values'][field] - denominator_mean['values'][field]) / delta_theta

    hod_derivative_samples = {field: [] for field in fields}
    hod_sample_keys = []
    for hod in used_hods:
        hod_numerator_files, hod_denominator_files, hod_keys = files_for_hod_finite_difference(
            files_by_hod_cosmo, hod, numerator_cosmo, denominator_cosmo,
            match_realizations)
        if not hod_numerator_files or not hod_denominator_files:
            continue

        hod_numerator_mean = average_pk(hod_numerator_files, fields, pk_kind, strict_bins)
        hod_denominator_mean = average_pk(hod_denominator_files, fields, pk_kind, strict_bins)
        hod_numerator_mean = align_pk_to_reference(
            hod_numerator_mean, numerator_mean, fields, strict_bins)
        hod_denominator_mean = align_pk_to_reference(
            hod_denominator_mean, numerator_mean, fields, strict_bins)

        hod_sample_keys.append((hod, hod_keys))
        for field in fields:
            hod_derivative_samples[field].append(
                (hod_numerator_mean['values'][field]
                 - hod_denominator_mean['values'][field]) / delta_theta)

    derivative_std = {}
    n_hod_samples = {}
    for field in fields:
        if hod_derivative_samples[field]:
            sample_array = np.vstack(hod_derivative_samples[field])
            derivative_std[field] = np.nanstd(sample_array, axis=0, ddof=0)
            n_hod_samples[field] = int(sample_array.shape[0])
        else:
            derivative_std[field] = np.full_like(derivatives[field], np.nan, dtype=np.float64)
            n_hod_samples[field] = 0

    return DerivativeResult(parameter=parameter,
                            param_column=param_column,
                            finite_difference=finite_difference,
                            numerator_cosmo=numerator_cosmo,
                            denominator_cosmo=denominator_cosmo,
                            theta_numerator=theta_numerator,
                            theta_denominator=theta_denominator,
                            delta_theta=delta_theta,
                            used_hods=used_hods,
                            n_hods=len(used_hods),
                            n_numerator=len(numerator_files),
                            n_denominator=len(denominator_files),
                            realization_keys=realization_keys,
                            hod_sample_keys=hod_sample_keys,
                            n_hod_samples=n_hod_samples,
                            k=numerator_mean['k'],
                            k_min=numerator_mean['k_min'],
                            k_max=numerator_mean['k_max'],
                            derivatives=derivatives,
                            derivative_std=derivative_std)


def check_result_bins(results):
    first = next(iter(results.values()))
    for parameter, result in results.items():
        if (result.k.shape != first.k.shape
                or not np.allclose(result.k, first.k, rtol=1.0e-8, atol=1.0e-12)
                or not np.allclose(result.k_min, first.k_min, rtol=1.0e-8, atol=1.0e-12, equal_nan=True)
                or not np.allclose(result.k_max, first.k_max, rtol=1.0e-8, atol=1.0e-12, equal_nan=True)):
            raise RuntimeError(f'Derivative bins for {parameter} do not match the first parameter.')
    return first


def write_derivative_matrix(path, results, fields, parameters):
    first = check_result_bins(results)
    headers = ['k_h_mpc', 'k_min_h_mpc', 'k_max_h_mpc']
    columns = [first.k, first.k_min, first.k_max]
    for field in fields:
        for parameter in parameters:
            headers.append(f'dpk_d{parameter}_{field}')
            columns.append(results[parameter].derivatives[field])
    data = np.column_stack(columns)
    np.savetxt(path, data, delimiter=',', header=','.join(headers), comments='')


def write_combined_environment_matrix(path, results, fields, parameters):
    env_fields = [field for field in ENVIRONMENT_FIELDS if field in fields]
    if not env_fields:
        return False

    first = check_result_bins(results)
    headers = ['component_index', 'field', 'k_h_mpc', 'k_min_h_mpc', 'k_max_h_mpc']
    headers += [f'dpk_d{parameter}' for parameter in parameters]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        component_index = 0
        for field in env_fields:
            for i in range(first.k.size):
                row = [component_index, field, first.k[i], first.k_min[i], first.k_max[i]]
                row.extend(results[parameter].derivatives[field][i] for parameter in parameters)
                writer.writerow(row)
                component_index += 1
    return True


def latex_parameter_label(parameter):
    labels = {'Omega_b': r'\omega_b',
              'omega_cdm': r'\omega_{\rm cdm}',
              'n_s': r'n_s',
              'sigma_8m': r'\sigma_8'}
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


def write_derivative_plot(path, title, results, fields, parameters):
    first = check_result_bins(results)
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
                line, = ax.plot(first.k[mask], deriv[mask], lw=1.8, color=color,
                                label=label)
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
        ax.set_title(rf'${latex_parameter_label(parameter)}$', fontsize=15)
        ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
        if col == 0:
            ax.set_ylabel(r'$\partial O / \partial \theta$')

    fig.suptitle(title, fontsize=13)
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(handles),
                   bbox_to_anchor=(0.5, 0.94), frameon=True)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_metadata(path, args, candidate_hods, results, fields, parameters, outputs):
    metadata = {'pk_root': str(Path(args.pk_root).expanduser().resolve()),
                'params_file': str(Path(args.params_file).expanduser().resolve()),
                'pk_kind': args.pk_kind,
                'fields': fields,
                'parameters': parameters,
                'candidate_hods': candidate_hods,
                'finite_difference': args.finite_difference,
                'finite_difference_formula': ('(mean(P_numerator) - mean(P_denominator)) / '
                                              '(theta_numerator - theta_denominator)'),
                'averaging': 'mean over all matched HOD/phase/seed datavectors before finite differencing',
                'match_realizations': args.match_realizations,
                'strict_bins': args.strict_bins,
                'plot': args.plot,
                'parameter_pairs': {},
                'outputs': outputs}
    for parameter in parameters:
        result = results[parameter]
        metadata['parameter_pairs'][parameter] = {'param_column': result.param_column,
                                                  'finite_difference': result.finite_difference,
                                                  'numerator_cosmo': result.numerator_cosmo,
                                                  'denominator_cosmo': result.denominator_cosmo,
                                                  'theta_numerator': result.theta_numerator,
                                                  'theta_denominator': result.theta_denominator,
                                                  'delta_theta': result.delta_theta,
                                                  'used_hods': result.used_hods,
                                                  'n_hods': result.n_hods,
                                                  'n_numerator': result.n_numerator,
                                                  'n_denominator': result.n_denominator,
                                                  'n_hod_samples_by_field': result.n_hod_samples,
                                                  'realizations': [f'{hod}_ph{phase}_seed{seed}'
                             for hod, phase, seed in result.realization_keys]}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def main():
    args = parse_args()
    t0 = time.time()

    fields = normalize_fields(args.fields)
    parameters = normalize_params(args.params)
    hod_values = {normalize_hod(value) for value in args.hod}
    hod_filter = None if 'all' in hod_values else hod_values

    pk_root = Path(args.pk_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else pk_root / 'cosmo_derivatives'
    outdir.mkdir(parents=True, exist_ok=True)

    params = load_cosmo_params(Path(args.params_file).expanduser().resolve())
    files_by_hod_cosmo = discover_pk_files(pk_root, fields, args.pk_kind,
                                           args.pk_file_field, hod_filter)
    hods = sorted({hod for hod, _ in files_by_hod_cosmo})
    if not hods:
        raise RuntimeError(f'No P(k) files found in {pk_root}.')

    print(f'---> pk root: {pk_root}')
    print(f'---> output directory: {outdir}')
    print(f'---> HODs in candidate pool: {len(hods)}')
    print(f'---> fields: {fields}')
    print(f'---> parameters: {parameters}')

    summary = {'elapsed_sec': None,
               'pk_root': str(pk_root),
               'outdir': str(outdir),
               'fields': fields,
               'parameters': parameters,
               'finite_difference': args.finite_difference,
               'candidate_hods': hods,
               'outputs': None}

    results = {}
    for parameter in parameters:
        try:
            results[parameter] = compute_derivative(
                parameter, params, files_by_hod_cosmo, hods, fields, args.pk_kind,
                args.finite_difference, args.match_realizations, args.strict_bins)
        except RuntimeError as exc:
            if args.skip_missing:
                print(f'---> skipped {parameter}: {exc}')
                continue
            raise

        result = results[parameter]
        print(f"---> {parameter}: finite-difference={result.finite_difference}, "
              f"{result.numerator_cosmo}-{result.denominator_cosmo}, "
              f"HODs={result.n_hods}, N={result.n_numerator}, "
              f"delta={result.delta_theta:.8g}")

    active_parameters = parameters

    plot_outdir = outdir / 'plots'
    plot_outdir.mkdir(parents=True, exist_ok=True)
    matrix_path = outdir / 'deriv_cosmo.csv'
    combined_path = outdir / 'deriv_cosmo_combined_env.csv'
    meta_path = outdir / 'deriv_cosmo_metadata.json'
    plot_path = plot_outdir / 'deriv_cosmo.png'

    write_derivative_matrix(matrix_path, results, fields, active_parameters)
    combined_written = write_combined_environment_matrix(combined_path, results, fields, active_parameters)
    if args.plot:
        write_derivative_plot(plot_path, 'Derivatives from mean observables over HODs', results,
                              fields, active_parameters)
    outputs = {'matrix': str(matrix_path),
               'combined_environment_matrix': str(combined_path) if combined_written else None,
               'plot': str(plot_path),
               'metadata': str(meta_path)}
    write_metadata(meta_path, args, hods, results, fields, active_parameters, outputs)
    summary['outputs'] = outputs

    print(f'---> wrote: {matrix_path}')
    if combined_written:
        print(f'---> wrote: {combined_path}')
    print(f'---> wrote: {plot_path}')
    print(f'---> wrote: {meta_path}')

    elapsed = time.time() - t0
    summary['elapsed_sec'] = elapsed
    summary_path = outdir / f'deriv_cosmo_summary_{int(time.time())}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'---> wrote: {summary_path}')
    print(f'---> elapsed: {elapsed:.2f} s')


if __name__ == '__main__':
    main()