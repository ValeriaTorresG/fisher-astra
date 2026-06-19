import argparse, csv, json, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from calc_deriv_cosmo import (DEFAULT_PK_ROOT, ENVIRONMENT_FIELDS, FIELDS,
                              PARAM_ALIASES, PARAM_SPECS)
from calc_deriv_hod import HOD_PARAMS


DATA_FIELDS = tuple(FIELDS)
CASE_CHOICES = DATA_FIELDS + ('combined', 'all')
DERIVATIVE_PREFIXES = ('dO_d', 'dpk_d', 'dlnO_d', 'dlnpk_d')
LOG_PREFIXES = ('dlnO_d', 'dlnpk_d')
DERIVATIVE_FILENAMES = {
    'cosmo': ('deriv_cosmo.csv', 'deriv_cosmo_combined_env.csv'),
    'hod': ('deriv_hod.csv', 'deriv_hod_combined_env.csv')}
PARAMETER_GROUPS = {
    'cosmo': tuple(PARAM_SPECS),
    'hod': tuple(HOD_PARAMS)}
DerivativeRecord = SimpleNamespace
Component = SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser()
    default_cosmo_deriv_dir = Path(DEFAULT_PK_ROOT) / 'cosmo_derivatives'
    default_hod_deriv_dir = Path(DEFAULT_PK_ROOT) / 'hod_derivatives'
    default_cov_dir = Path(DEFAULT_PK_ROOT) / 'covariances' / 'c000'
    default_outdir = Path(DEFAULT_PK_ROOT) / 'fisher_cosmo_hod'

    parser.add_argument('--deriv-dir', type=str, default='')
    parser.add_argument('--cosmo-deriv-dir', type=str, default=str(default_cosmo_deriv_dir))
    parser.add_argument('--derivatives-file', type=str, default='')
    parser.add_argument('--combined-derivatives-file', type=str, default='')
    parser.add_argument('--hod-deriv-dir', type=str, default=str(default_hod_deriv_dir))
    parser.add_argument('--hod-derivatives-file', type=str, default='')
    parser.add_argument('--hod-combined-derivatives-file', type=str, default='')
    parser.add_argument('--cov-dir', type=str, default=str(default_cov_dir))
    parser.add_argument('--outdir', type=str, default=str(default_outdir))
    parser.add_argument('--cases', nargs='+', default=['all'], choices=CASE_CHOICES)
    parser.add_argument('--params', nargs='+', default=['all'])
    parser.add_argument('--inverse-method', type=str, default='auto', choices=['auto', 'inv', 'pinv'])
    parser.add_argument('--rcond', type=float, default=1.0e-12)
    parser.add_argument('--hartlap-correction', action='store_true')
    parser.add_argument('--allow-log-derivatives', action='store_true')
    parser.add_argument('--no-hod-derivatives', dest='include_hod_derivatives', action='store_false',)
    parser.set_defaults(include_hod_derivatives=True)
    return parser.parse_args()


def normalize_cases(values):
    if 'all' in values:
        return list(DATA_FIELDS) + ['combined']
    cases = []
    for value in values:
        if value not in cases:
            cases.append(value)
    return cases


def case_name(case):
    return case


def normalize_params(values, available):
    if any(value == 'all' for value in values):
        return ordered_available_parameters(available)

    params = []
    for value in values:
        token = str(value).strip()
        token_lower = token.lower()
        if token_lower in ('cosmo', 'cosmology', 'cosmological'):
            expanded = [parameter for parameter in PARAMETER_GROUPS['cosmo']
                        if parameter in available]
        elif token_lower == 'hod':
            expanded = [parameter for parameter in PARAMETER_GROUPS['hod']
                        if parameter in available]
        else:
            parameter = normalize_parameter_name(token)
            if parameter not in available:
                raise RuntimeError(f'Parameter {value} is not available in the derivative files. '
                                   f'Available: {", ".join(available)}')
            expanded = [parameter]
        for parameter in expanded:
            if parameter not in params:
                params.append(parameter)
    return params


def normalize_parameter_name(value):
    if value in PARAM_ALIASES:
        return PARAM_ALIASES[value]
    hod_aliases = {parameter.lower(): parameter for parameter in HOD_PARAMS}
    return hod_aliases.get(str(value).strip().lower(), value)


def ordered_available_parameters(available):
    ordered = [parameter for parameter in PARAMETER_GROUPS['cosmo']
               if parameter in available]
    ordered += [parameter for parameter in PARAMETER_GROUPS['hod']
                if parameter in available and parameter not in ordered]
    ordered += [parameter for parameter in available if parameter not in ordered]
    return ordered


def derivative_column_info(column):
    for prefix in DERIVATIVE_PREFIXES:
        if column.startswith(prefix):
            derivative_kind = 'log' if prefix in LOG_PREFIXES else 'linear'
            return prefix, derivative_kind, column[len(prefix):]
    return None


def split_field_derivative_column(column):
    for field in DATA_FIELDS:
        suffix = f'_{field}'
        if not column.endswith(suffix):
            continue
        base = column[:-len(suffix)]
        info = derivative_column_info(base)
        if info is None:
            continue
        prefix, derivative_kind, parameter = info
        return prefix, derivative_kind, parameter, field
    return None


def read_csv_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise RuntimeError(f'No rows found in {path}')
    return rows, fieldnames


def parse_float(value, path, column):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f'Cannot parse {column}={value!r} in {path}') from exc


def load_combined_derivatives(path):
    rows, fieldnames = read_csv_rows(path)
    derivative_columns = {}
    derivative_kinds = set()
    for column in fieldnames:
        info = derivative_column_info(column)
        if info is None:
            continue
        _, derivative_kind, parameter = info
        derivative_columns[parameter] = column
        derivative_kinds.add(derivative_kind)

    if not derivative_columns:
        raise RuntimeError(f'No derivative columns found in {path}')
    if len(derivative_kinds) != 1:
        raise RuntimeError(f'Mixed derivative kinds found in {path}: {sorted(derivative_kinds)}')

    records = []
    seen_fields = set()
    for row in rows:
        field = row.get('field', '').strip()
        if field not in DATA_FIELDS:
            continue
        if field in seen_fields:
            raise RuntimeError(f'Expected one full-range derivative row for {field} in {path}.')
        seen_fields.add(field)
        values = {parameter: parse_float(row[column], path, column)
                  for parameter, column in derivative_columns.items()}
        records.append(DerivativeRecord(field=field,
                                        k_h_mpc=parse_float(row['k_h_mpc'], path, 'k_h_mpc'),
                                        k_min_h_mpc=parse_float(row['k_min_h_mpc'], path, 'k_min_h_mpc'),
                                        k_max_h_mpc=parse_float(row['k_max_h_mpc'], path, 'k_max_h_mpc'),
                                        values=values))
    if not records:
        raise RuntimeError(f'No environment derivative rows found in {path}')
    return records, list(derivative_columns), next(iter(derivative_kinds))


def load_matrix_derivatives(path):
    rows, fieldnames = read_csv_rows(path)
    columns_by_field_param = {}
    derivative_kinds = set()
    parameters = []
    fields = []
    for column in fieldnames:
        info = split_field_derivative_column(column)
        if info is None:
            continue
        _, derivative_kind, parameter, field = info
        columns_by_field_param[(field, parameter)] = column
        derivative_kinds.add(derivative_kind)
        if parameter not in parameters:
            parameters.append(parameter)
        if field not in fields:
            fields.append(field)

    if not columns_by_field_param:
        raise RuntimeError(f'No field derivative columns found in {path}')
    if len(derivative_kinds) != 1:
        raise RuntimeError(f'Mixed derivative kinds found in {path}: {sorted(derivative_kinds)}')
    if len(rows) != 1:
        raise RuntimeError(f'Expected one full-range derivative row in {path}, found {len(rows)}.')

    records = []
    for field in [item for item in DATA_FIELDS if item in fields]:
        for row in rows:
            values = {}
            for parameter in parameters:
                column = columns_by_field_param.get((field, parameter))
                if column is not None:
                    values[parameter] = parse_float(row[column], path, column)
            if values:
                records.append(DerivativeRecord(
                    field=field,
                    k_h_mpc=parse_float(row['k_h_mpc'], path, 'k_h_mpc'),
                    k_min_h_mpc=parse_float(row['k_min_h_mpc'], path, 'k_min_h_mpc'),
                    k_max_h_mpc=parse_float(row['k_max_h_mpc'], path, 'k_max_h_mpc'),
                    values=values))
    return records, parameters, next(iter(derivative_kinds))


def load_derivatives(deriv_dir, derivatives_file, combined_derivatives_file,
                     derivative_family):
    deriv_dir = Path(deriv_dir).expanduser().resolve()
    matrix_name, combined_name = DERIVATIVE_FILENAMES[derivative_family]
    combined_path = (Path(combined_derivatives_file).expanduser().resolve()
                     if combined_derivatives_file
                     else deriv_dir / combined_name)
    matrix_path = (Path(derivatives_file).expanduser().resolve()
                   if derivatives_file
                   else deriv_dir / matrix_name)

    if matrix_path.is_file():
        records, parameters, derivative_kind = load_matrix_derivatives(matrix_path)
        return records, parameters, derivative_kind, matrix_path
    if combined_path.is_file():
        records, parameters, derivative_kind = load_combined_derivatives(combined_path)
        return records, parameters, derivative_kind, combined_path

    raise RuntimeError(f'Cannot find derivative inputs. Tried {combined_path} and {matrix_path}.')


def derivative_metadata_path(derivative_path):
    path = Path(derivative_path)
    stem = path.stem.replace('_combined_env', '')
    metadata_path = path.parent / f'{stem}_metadata.json'
    return metadata_path if metadata_path.is_file() else None


def load_derivative_metadata(derivative_path):
    metadata_path = derivative_metadata_path(derivative_path)
    if metadata_path is None:
        return {}, None
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f), metadata_path


def find_matching_derivative_record(target, records):
    for record in records:
        if record.field != target.field:
            continue
        if (np.isclose(record.k_min_h_mpc, target.k_min_h_mpc,
                       rtol=1.0e-8, atol=1.0e-12, equal_nan=True)
                and np.isclose(record.k_max_h_mpc, target.k_max_h_mpc,
                               rtol=1.0e-8, atol=1.0e-12, equal_nan=True)):
            return record
    raise RuntimeError(f'No derivative row matches {target.field}, '
                       f'k range [{target.k_min_h_mpc}, {target.k_max_h_mpc}].')


def merge_derivative_records(inputs):
    if not inputs:
        raise RuntimeError('No derivative inputs were loaded.')

    base = inputs[0]
    merged_records = []
    parameters = []
    derivative_kind = base['derivative_kind']
    for item in inputs:
        if item['derivative_kind'] != derivative_kind:
            raise RuntimeError('Cannot combine derivative files with different derivative kinds: '
                               f"{derivative_kind} and {item['derivative_kind']}.")
        for parameter in item['parameters']:
            if parameter in parameters:
                raise RuntimeError(f'Duplicate derivative parameter: {parameter}')
            parameters.append(parameter)

    for base_record in base['records']:
        values = dict(base_record.values)
        for item in inputs[1:]:
            record = find_matching_derivative_record(base_record, item['records'])
            for parameter, value in record.values.items():
                if parameter in values:
                    raise RuntimeError(f'Duplicate derivative parameter: {parameter}')
                values[parameter] = value
        merged_records.append(DerivativeRecord(
            field=base_record.field,
            k_h_mpc=base_record.k_h_mpc,
            k_min_h_mpc=base_record.k_min_h_mpc,
            k_max_h_mpc=base_record.k_max_h_mpc,
            values=values))

    return merged_records, parameters, derivative_kind


def load_pipeline_derivatives(args):
    cosmo_deriv_dir = args.deriv_dir or args.cosmo_deriv_dir
    inputs = []

    records, parameters, derivative_kind, path = load_derivatives(
        cosmo_deriv_dir, args.derivatives_file,
        args.combined_derivatives_file, 'cosmo')
    metadata, metadata_path = load_derivative_metadata(path)
    inputs.append({'family': 'cosmo',
                   'records': records,
                   'parameters': parameters,
                   'derivative_kind': derivative_kind,
                   'path': path,
                   'metadata': metadata,
                   'metadata_path': metadata_path})

    if args.include_hod_derivatives:
        records, parameters, derivative_kind, path = load_derivatives(
            args.hod_deriv_dir, args.hod_derivatives_file,
            args.hod_combined_derivatives_file, 'hod')
        metadata, metadata_path = load_derivative_metadata(path)
        inputs.append({'family': 'hod',
                       'records': records,
                       'parameters': parameters,
                       'derivative_kind': derivative_kind,
                       'path': path,
                       'metadata': metadata,
                       'metadata_path': metadata_path})

    merged_records, parameters, derivative_kind = merge_derivative_records(inputs)
    return merged_records, parameters, derivative_kind, inputs


def load_components(path):
    rows, _ = read_csv_rows(path)
    components = []
    seen_fields = set()
    for row in rows:
        field = row['field']
        if field in seen_fields:
            raise RuntimeError(f'Expected one covariance component for {field} in {path}.')
        seen_fields.add(field)
        components.append(Component(
            component_index=int(row['component_index']),
            global_component_index=int(row.get('global_component_index', row['component_index'])),
            field=field,
            k_h_mpc=float(row['k_h_mpc']),
            k_min_h_mpc=float(row['k_min_h_mpc']),
            k_max_h_mpc=float(row['k_max_h_mpc']),
            label=row.get('label') or field))
    return components


def derivative_records_by_field(records):
    by_field = {}
    for record in records:
        by_field.setdefault(record.field, []).append(record)
    return by_field


def find_derivative_record(component, records_by_field):
    candidates = records_by_field.get(component.field, [])
    for record in candidates:
        if (np.isclose(record.k_min_h_mpc, component.k_min_h_mpc,
                       rtol=1.0e-8, atol=1.0e-12, equal_nan=True)
                and np.isclose(record.k_max_h_mpc, component.k_max_h_mpc,
                               rtol=1.0e-8, atol=1.0e-12, equal_nan=True)):
            return record
    raise RuntimeError(f'No derivative row matches component {component.label} '
                       f'({component.field}, k range '
                       f'[{component.k_min_h_mpc}, {component.k_max_h_mpc}]).')


def build_derivative_matrix(records, components, parameters):
    by_field = derivative_records_by_field(records)
    derivative = np.empty((len(components), len(parameters)), dtype=np.float64)
    matched_records = []
    for i, component in enumerate(components):
        record = find_derivative_record(component, by_field)
        matched_records.append(record)
        for j, parameter in enumerate(parameters):
            if parameter not in record.values:
                raise RuntimeError(f'Missing derivative for {parameter} at component {component.label}.')
            derivative[i, j] = record.values[parameter]
    if not np.all(np.isfinite(derivative)):
        raise RuntimeError('Derivative matrix contains non-finite values.')
    return derivative, matched_records


def load_covariance_inputs(cov_dir, case):
    cov_dir = Path(cov_dir).expanduser().resolve()
    name = case_name(case)
    cov_path = cov_dir / f'cov_{name}.npy'
    components_path = cov_dir / f'components_{name}.csv'
    if case == 'combined' and not cov_path.is_file():
        cov_path = cov_dir / 'cov_all.npy'
    if case == 'combined' and not components_path.is_file():
        components_path = cov_dir / 'components_all.csv'
    if not cov_path.is_file():
        raise RuntimeError(f'Missing covariance file: {cov_path}')
    if not components_path.is_file():
        raise RuntimeError(f'Missing component file: {components_path}')
    cov = np.load(cov_path)
    components = load_components(components_path)
    if cov.shape != (len(components), len(components)):
        raise RuntimeError(f'Covariance shape {cov.shape} does not match '
                           f'{len(components)} components for {name}.')
    if not np.all(np.isfinite(cov)):
        raise RuntimeError(f'Covariance contains non-finite values: {cov_path}')
    return cov, components, cov_path, components_path


def load_cov_metadata(cov_dir):
    path = Path(cov_dir).expanduser().resolve() / 'cov_metadata.json'
    if not path.is_file():
        return {}, None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f), path


def covariance_precision(cov, inverse_method, rcond):
    rank = int(np.linalg.matrix_rank(cov))
    n_components = int(cov.shape[0])
    full_rank = rank == n_components
    if inverse_method == 'inv' or (inverse_method == 'auto' and full_rank):
        if not full_rank:
            raise RuntimeError(f'Covariance is singular: rank={rank}, n_components={n_components}. '
                               'Use --inverse-method pinv or auto.')
        precision = np.linalg.inv(cov)
        method_used = 'inv'
    else:
        precision = np.linalg.pinv(cov, rcond=rcond)
        method_used = 'pinv'
    precision = 0.5 * (precision + precision.T)
    return precision, {'rank': rank,
                       'n_components': n_components,
                       'is_singular': bool(not full_rank),
                       'inverse_method_used': method_used}


def hartlap_factor_from_metadata(metadata, name):
    diagnostics = metadata.get('diagnostics') or {}
    case_diagnostics = diagnostics.get(name) or {}
    factor = case_diagnostics.get('hartlap_factor')
    if factor is None:
        return None
    return float(factor)


def fisher_from_derivatives(derivative, precision):
    fisher = derivative.T @ precision @ derivative
    return 0.5 * (fisher + fisher.T) #!make sure the matrix is exactly symmetric to avoid errors


def parameter_covariance_from_fisher(fisher, rcond):
    rank = int(np.linalg.matrix_rank(fisher))
    n_params = int(fisher.shape[0])
    if rank == n_params:
        covariance = np.linalg.inv(fisher)
        method = 'inv'
    else:
        covariance = np.linalg.pinv(fisher, rcond=rcond)
        method = 'pinv'
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, {'rank': rank,
                        'n_params': n_params,
                        'is_singular': bool(rank < n_params),
                        'inverse_method_used': method}


def correlation_from_covariance(cov):
    diag = np.diag(cov)
    scale = np.sqrt(np.outer(diag, diag))
    corr = np.full_like(cov, np.nan, dtype=np.float64)
    mask = np.isfinite(scale) & (scale > 0.0)
    corr[mask] = cov[mask] / scale[mask]
    return corr


def safe_sigma_from_precision_diag(values):
    values = np.asarray(values, dtype=np.float64)
    sigma = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values) & (values > 0.0)
    sigma[mask] = 1.0 / np.sqrt(values[mask])
    return sigma


def safe_sigma_from_cov_diag(values):
    values = np.asarray(values, dtype=np.float64)
    sigma = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values) & (values >= 0.0)
    sigma[mask] = np.sqrt(values[mask])
    return sigma


def write_named_matrix_csv(path, matrix, labels, index_name):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([index_name] + list(labels))
        for label, row in zip(labels, matrix):
            writer.writerow([label] + list(row))


def write_derivative_matrix_csv(path, derivative, components, parameters):
    headers = ['component_index', 'global_component_index', 'field',
               'k_h_mpc', 'k_min_h_mpc', 'k_max_h_mpc', 'label']
    headers += [f'dd_d{parameter}' for parameter in parameters]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for component, row_values in zip(components, derivative):
            row = [component.component_index,
                   component.global_component_index,
                   component.field,
                   component.k_h_mpc,
                   component.k_min_h_mpc,
                   component.k_max_h_mpc,
                   component.label]
            row.extend(row_values)
            writer.writerow(row)


def write_constraints_csv(path, parameters, fisher, parameter_covariance):
    unmarginalized = safe_sigma_from_precision_diag(np.diag(fisher))
    marginalized = safe_sigma_from_cov_diag(np.diag(parameter_covariance))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter', 'sigma_marginalized', 'sigma_unmarginalized',
                         'fisher_diagonal'])
        for i, parameter in enumerate(parameters):
            writer.writerow([parameter, marginalized[i], unmarginalized[i], fisher[i, i]])
    return marginalized, unmarginalized


def write_case_outputs(outdir, name, parameters, components, derivative,
                       precision, fisher, parameter_covariance,
                       parameter_correlation):
    paths = {'derivatives': str(outdir / f'derivatives_{name}.csv'),
             'precision_npy': str(outdir / f'precision_{name}.npy'),
             'fisher_csv': str(outdir / f'fisher_{name}.csv'),
             'fisher_npy': str(outdir / f'fisher_{name}.npy'),
             'parameter_covariance_csv': str(outdir / f'parameter_cov_{name}.csv'),
             'parameter_covariance_npy': str(outdir / f'parameter_cov_{name}.npy'),
             'parameter_correlation_csv': str(outdir / f'parameter_corr_{name}.csv'),
             'parameter_correlation_npy': str(outdir / f'parameter_corr_{name}.npy'),
             'constraints': str(outdir / f'constraints_{name}.csv')}
    write_derivative_matrix_csv(paths['derivatives'], derivative, components, parameters)
    np.save(paths['precision_npy'], precision)
    np.save(paths['fisher_npy'], fisher)
    np.save(paths['parameter_covariance_npy'], parameter_covariance)
    np.save(paths['parameter_correlation_npy'], parameter_correlation)
    write_named_matrix_csv(paths['fisher_csv'], fisher, parameters, 'parameter')
    write_named_matrix_csv(paths['parameter_covariance_csv'], parameter_covariance,
                           parameters, 'parameter')
    write_named_matrix_csv(paths['parameter_correlation_csv'], parameter_correlation,
                           parameters, 'parameter')
    marginalized, unmarginalized = write_constraints_csv(
        paths['constraints'], parameters, fisher, parameter_covariance)
    return paths, marginalized, unmarginalized


def write_summary_constraints(path, case_results, parameters):
    headers = ['case']
    headers += [f'sigma_marg_{parameter}' for parameter in parameters]
    headers += [f'sigma_unmarg_{parameter}' for parameter in parameters]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for name, result in case_results.items():
            row = [name]
            row.extend(result['sigma_marginalized'])
            row.extend(result['sigma_unmarginalized'])
            writer.writerow(row)


def write_metadata(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    t0 = time.time()

    cases = normalize_cases(args.cases)
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    records, available_parameters, derivative_kind, derivative_inputs = (
        load_pipeline_derivatives(args))
    if derivative_kind == 'log' and not args.allow_log_derivatives:
        raise RuntimeError(
            'The derivative file contains logarithmic derivatives (dlnO/dtheta or dlnP/dtheta). '
            'For the requested Fisher formula with data vector P(k) or O(k), rerun '
            'src/calc_deriv_cosmo.py with --derivative-kind linear, or pass '
            '--allow-log-derivatives only if your covariance is also in log-data space.')

    parameters = normalize_params(args.params, available_parameters)
    cov_metadata, cov_metadata_path = load_cov_metadata(args.cov_dir)
    derivative_paths = {item['family']: str(item['path'])
                        for item in derivative_inputs}
    derivative_metadata_paths = {
        item['family']: str(item['metadata_path']) if item['metadata_path'] else None
        for item in derivative_inputs}
    parameter_groups = {
        'cosmo': [parameter for parameter in parameters
                  if parameter in PARAMETER_GROUPS['cosmo']],
        'hod': [parameter for parameter in parameters
                if parameter in PARAMETER_GROUPS['hod']]}
    fiducials = {}
    for item in derivative_inputs:
        if item['family'] != 'hod':
            continue
        hod_metadata = item['metadata'] or {}
        fiducials['hod'] = {
            'name': hod_metadata.get('fiducial_hod'),
            'parameters': hod_metadata.get('fiducial_parameters') or {}}

    for family, path in derivative_paths.items():
        print(f'---> {family} derivative file: {path}')
    print(f'---> derivative kind: {derivative_kind}')
    print(f'---> covariance directory: {Path(args.cov_dir).expanduser().resolve()}')
    print(f'---> output directory: {outdir}')
    print(f'---> cases: {[case_name(case) for case in cases]}')
    print(f'---> parameters: {parameters}')

    outputs = {}
    diagnostics = {}
    case_results = {}
    for case in cases:
        name = case_name(case)
        cov, components, cov_path, components_path = load_covariance_inputs(args.cov_dir, case)
        derivative, _ = build_derivative_matrix(records, components, parameters)
        precision, cov_diag = covariance_precision(cov, args.inverse_method, args.rcond)

        hartlap_factor = None
        if args.hartlap_correction:
            hartlap_factor = hartlap_factor_from_metadata(cov_metadata, name)
            if hartlap_factor is None:
                raise RuntimeError(f'Hartlap correction requested but unavailable for case {name}.')
            precision *= hartlap_factor

        fisher = fisher_from_derivatives(derivative, precision)
        parameter_covariance, fisher_diag = parameter_covariance_from_fisher(fisher, args.rcond)
        parameter_correlation = correlation_from_covariance(parameter_covariance)

        paths, sigma_marginalized, sigma_unmarginalized = write_case_outputs(
            outdir, name, parameters, components, derivative, precision, fisher,
            parameter_covariance, parameter_correlation)
        outputs[name] = paths
        diagnostics[name] = {'covariance_path': str(cov_path),
                             'components_path': str(components_path),
                             'n_components': len(components),
                             'n_parameters': len(parameters),
                             'covariance': cov_diag,
                             'fisher': fisher_diag,
                             'hartlap_factor_applied': hartlap_factor}
        case_results[name] = {'sigma_marginalized': list(sigma_marginalized),
                              'sigma_unmarginalized': list(sigma_unmarginalized)}
        print(f'---> {name}: components={len(components)}, '
              f"cov_inverse={cov_diag['inverse_method_used']}, "
              f"fisher_inverse={fisher_diag['inverse_method_used']}")

    summary_constraints_path = outdir / 'constraints_summary.csv'
    write_summary_constraints(summary_constraints_path, case_results, parameters)
    outputs['constraints_summary'] = str(summary_constraints_path)

    metadata_path = outdir / 'fisher_metadata.json'
    outputs['metadata'] = str(metadata_path)
    metadata = {'derivative_file': str(next(iter(derivative_paths.values()))),
                'derivative_files': derivative_paths,
                'derivative_metadata_files': derivative_metadata_paths,
                'derivative_kind': derivative_kind,
                'covariance_directory': str(Path(args.cov_dir).expanduser().resolve()),
                'covariance_metadata': str(cov_metadata_path) if cov_metadata_path else None,
                'outdir': str(outdir),
                'cases': [case_name(case) for case in cases],
                'parameters': parameters,
                'available_parameters': available_parameters,
                'parameter_groups': parameter_groups,
                'include_hod_derivatives': args.include_hod_derivatives,
                'fiducials': fiducials,
                'inverse_method_requested': args.inverse_method,
                'rcond': args.rcond,
                'hartlap_correction': args.hartlap_correction,
                'allow_log_derivatives': args.allow_log_derivatives,
                'diagnostics': diagnostics,
                'outputs': outputs}
    write_metadata(metadata_path, metadata)

    summary = {'elapsed_sec': time.time() - t0,
               'cases': [case_name(case) for case in cases],
               'parameters': parameters,
               'outputs': outputs}
    summary_path = outdir / f'fisher_summary_{int(time.time())}.json'
    write_metadata(summary_path, summary)

    print(f'---> wrote: {summary_constraints_path}')
    print(f'---> wrote: {metadata_path}')
    print(f'---> wrote: {summary_path}')
    print(f"---> elapsed: {summary['elapsed_sec']:.2f} s")


if __name__ == '__main__':
    main()