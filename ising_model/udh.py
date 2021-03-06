import struct
import numpy
import pandas
import json
from ising_model import udh_pb2


def _read_next_chunk(stream):
    byte_array = stream.read(8)
    if len(byte_array) < 8:
        return None
    chunk_size = struct.unpack('Q', byte_array)[0]
    byte_array = stream.read(chunk_size)
    if len(byte_array) < chunk_size:
        return None
    return byte_array


def _read_next_protobuf(stream, ProtobufClass):
    chunk = _read_next_chunk(stream)
    if chunk is None:
        return None
    result = ProtobufClass()
    result.ParseFromString(chunk)
    return result


def _read_protobuf_array(stream, ProtobufClass):
    result = list()
    while True:
        next_protobuf = _read_next_protobuf(stream, ProtobufClass)
        if next_protobuf is None:
            break
        result.append(next_protobuf)
    return result


def load_params(file_name):
    with open(file_name, 'rb') as stream:
        return _read_next_protobuf(stream, udh_pb2.UdhParameters)


def load_observables(file_name):
    observables_list = []
    with open(file_name, 'rb') as stream:
        if _read_next_chunk(stream) is None:
            return []
        return _read_protobuf_array(stream, udh_pb2.UdhObservables)


def load_autocorrelation_table(file_name):
    with open(file_name, 'rb') as stream:
        ac_points = _read_protobuf_array(
            stream, udh_pb2.UdhAutocorrelationPoint)
    table = []
    for ac_point in ac_points:
        table.append({
            "J": ac_point.j,
            "mu": ac_point.mu,
            "L0": ac_point.shape[0],
            "n_wolff": ac_point.n_wolff,
            "n_metropolis": ac_point.n_metropolis,
            "metropolis_stride": ac_point.metropolis_stride,
            "measure_every": ac_point.measure_every,
            "file_group": ac_point.file_group,
            "count": ac_point.count,
            "tau": ac_point.tau,
            "ud_ac": ac_point.ud_autocorrelation,
            "ud_ac_std": ac_point.ud_autocorrelation_std,
            "h_ac": ac_point.h_autocorrelation,
            "h_ac_std": ac_point.h_autocorrelation_std,
            "t_wolff": ac_point.t_wolff,
            "t_metropolis": ac_point.t_metropolis,
            "t_measure": ac_point.t_measure,
            "t_serialize": ac_point.t_serialize,
            "t_residual": ac_point.t_residual})
    return pandas.DataFrame(table)


def load_aggregate_observables_table(file_name):
    with open(file_name, 'rb') as stream:
        agg_obs_list = _read_protobuf_array(
            stream, udh_pb2.UdhAggregateObservables)
    table = []
    for agg_obs in agg_obs_list:
        table.append({
            "J0": agg_obs.j0,
            "mu0": agg_obs.mu0,
            "L0": agg_obs.shape[0],
            "n_wolff": agg_obs.n_wolff,
            "n_metropolis": agg_obs.n_metropolis,
            "metropolis_stride": agg_obs.metropolis_stride,
            "measure_every": agg_obs.measure_every,
            "file_group": agg_obs.file_group,
            "J": agg_obs.j,
            "mu": agg_obs.mu,
            "count": agg_obs.count,
            "susceptibility": agg_obs.susceptibility,
            "susceptibility_std": agg_obs.susceptibility_std,
            "binder_cumulant": agg_obs.binder_cumulant,
            "binder_cumulant_std": agg_obs.binder_cumulant_std,
            "hole_density": agg_obs.hole_density,
            "hole_density_std": agg_obs.hole_density_std,
            "sum_si_sj": agg_obs.sum_si_sj,
            "sum_si_sj_std": agg_obs.sum_si_sj_std})
    return pandas.DataFrame(table)


def load_params_table(file_names):
    table = []
    for file_name in file_names:
        params = load_params(file_name)
        table.append({
            "J": params.j,
            "mu": params.mu,
            "L0": params.shape[0],
            "n_wolff": params.n_wolff,
            "n_metropolis": params.n_metropolis,
            "metropolis_stride": params.metropolis_stride,
            "measure_every": params.measure_every,
            "n_measure": params.n_measure,
            "seed": params.seed,
            "id": params.id,
            "file_name": file_name})
    return pandas.DataFrame(table)


def load_file_groups_table(path):
    with open(path + "/file-groups.json", 'r') as stream:
        json_db = json.load(stream)

    table = []
    for group_id, file_name in json_db['file_names']['data']:
        params = load_params(path + "/" + file_name)
        table.append({
            "group": group_id,
            "J": params.j,
            "mu": params.mu,
            "L0": params.shape[0],
            "dim": len(params.shape),
            "n_wolff": params.n_wolff,
            "n_metropolis": params.n_metropolis,
            "metropolis_stride": params.metropolis_stride,
            "quenched_holes": params.quenched_holes,
            "measure_every": params.measure_every,
            "n_measure": params.n_measure,
            "seed": params.seed,
            "id": params.id,
            "file_name": file_name})
    return pandas.DataFrame(table)


def get_chebyshev_matrix(x, n):
    x = numpy.array(x)
    A = [[1.0] * len(x), x]
    while len(A) < n:
        A.append(2.0 * x * A[-1] - A[-2])
    return numpy.vstack(A[:n]).transpose()


def get_reg_matrix(n):
    A = numpy.zeros((n - 2, n))
    for i in range(0, n - 2):
        A[i, i + 2] = (i + 1) * (i + 2)
    return A


def evaluate_chebyshev_polynomial(c, sc, x):
    A = get_chebyshev_matrix(x, len(c))
    return (numpy.matmul(A, c),
            numpy.matmul(numpy.matmul(A, sc), numpy.transpose(A)))


def fit_chebyshev_polynomial(x, y, sy, reg, n):
    x = numpy.array(x)
    y = numpy.array(y)
    sy = numpy.array(sy)
    A = get_chebyshev_matrix(x, n)
    A = A / numpy.reshape(sy, (sy.size, 1))
    A = numpy.concatenate([A, reg * get_reg_matrix(n)])
    b = numpy.concatenate([y / sy, numpy.zeros(n-2)])
    return (
        numpy.linalg.lstsq(A, b, rcond=-1)[0],
        numpy.linalg.inv(numpy.matmul(numpy.transpose(A), A)))


def get_critical_points_table(dim):
    assert dim in [3, 4]
    if dim == 3:
        return numpy.array((
            (-1.0e128, 0.22165, 0.0001),
            (0.00, 0.31288, 0.0001),
            (1.00, 0.44566, 0.0001),
            (1.50, 0.55755, 0.0001),
            (1.75, 0.62635, 0.0001),
            (2.00, 0.70325, 0.0001)))
    elif dim == 4:
        return numpy.array((
            (-1.0e128, 0.149700, 0.000100),
            (0.000000, 0.215750, 0.002000),
            (1.000000, 0.316700, 0.001000),
            (1.500000, 0.407470, 0.000010),
            (1.625000, 0.435490, 0.000010),
            (1.656250, 0.442830, 0.000010),
            (1.687500, 0.450340, 0.000010),
            (1.703125, 0.454150, 0.000010),
            (2.000000, 0.530000, 0.005000)))


def mu_to_x(mu):
    if isinstance(mu, float):
        y = mu - 1.0
        if abs(y) < 1.0e-7:
            return y
        else:
            return (numpy.sqrt(1 + 4 * y * y) - 1) / (2 * y)
    else:
        return numpy.array([mu_to_x(mu_i) for mu_i in mu])


def get_critical_J(dim, mu, reg):

    if isinstance(mu, float):
        return get_critical_J(dim, numpy.array([mu]), reg)[:, 0]
    elif isinstance(mu, list):
        return get_critical_J(dim, numpy.array(mu), reg)

    critical_points = get_critical_points_table(dim)
    x_fit = mu_to_x(critical_points[:, 0])
    y_fit = (1 - x_fit) * critical_points[:, 1]
    sy_fit = (1 - x_fit) * critical_points[:, 2]
    coeff, coeff_cov = fit_chebyshev_polynomial(x_fit, y_fit, sy_fit, reg, 32)

    x_ev = mu_to_x(mu)
    y_ev, y_ev_cov = evaluate_chebyshev_polynomial(coeff, coeff_cov, x_ev)

    J_ev = y_ev / (1 - x_ev)
    sJ_ev = numpy.sqrt(numpy.diagonal(y_ev_cov)) / (1 - x_ev)

    return numpy.stack([J_ev, sJ_ev])
