import struct
import numpy
import pandas
from udh_pb2 import UdhParameters, UdhObservables

def load_params(file_name):
    with open(file_name, 'rb') as in_file:
        n_read = struct.unpack('Q', in_file.read(8))[0]
        assert n_read < (1 << 20), "Corrupt file"
        params = UdhParameters()
        params.ParseFromString(in_file.read(n_read))
    return params

def load_observables(file_name):
    observables_list = []
    with open(file_name, 'rb') as in_file:
        n_read = struct.unpack('Q', in_file.read(8))[0]
        assert n_read < (1 << 20), "Corrupt file"
        in_file.read(n_read)
        while True:
            chunk = in_file.read(8)
            if len(chunk) < 8:
                break
            n_read = struct.unpack('Q', chunk)[0]
            assert n_read < (1 << 20), "Corrupt file"
            observables = UdhObservables()
            chunk = in_file.read(n_read)
            if len(chunk) < n_read:
                break
            observables.ParseFromString(chunk)
            observables_list.append(observables)
    return observables_list

def load_params_table(file_names):
    table = []
    for file_name in file_names:
        params = load_params(file_name)
        table.append((
            params.j,
            params.mu,
            params.shape[0],
            params.n_wolff,
            params.n_metropolis,
            params.measure_every,
            params.n_measure,
            params.seed,
            params.id))
    table = pandas.DataFrame(
        table, 
        columns=[
            'J', 'mu', 'L0', 'n_wolff', 'n_metropolis', 
            'measure_every', 'n_measure', 'seed', 'id'])
    return table

def get_chebyshev_matrix(x, n):
    x = numpy.array(x)
    A = [[1.0] * len(x), x]
    while len(A) < n:
        A.append(2.0 * x * A[-1] - A[-2])
    return numpy.vstack(A[:n]).transpose()

def get_reg_matrix(n):
    A = numpy.zeros((n - 2, n));
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
        numpy.linalg.lstsq(A,b , rcond=-1)[0],
        numpy.linalg.inv(numpy.matmul(numpy.transpose(A), A)))

def get_critical_points_table(dim):
    assert dim in [3, 4]
    if dim == 3:
        return numpy.array((
            (-1.0e128, 0.22165, 0.0001),
            (    0.00, 0.31288, 0.0001),
            (    1.00, 0.44566, 0.0001),
            (    1.50, 0.55755, 0.0001),
            (    1.75, 0.62635, 0.0001),
            (    2.00, 0.70325, 0.0001)))
    elif dim == 4:
        return numpy.array((
            (  -1.0e128, 0.149700, 0.000100),
            (  0.000000, 0.215750, 0.002000),
            (  1.000000, 0.316700, 0.001000),
            (  1.500000, 0.407470, 0.000010),
            (  1.625000, 0.435490, 0.000010),
            (  1.656250, 0.442830, 0.000010),
            (  1.687500, 0.450340, 0.000010),
            (  1.703125, 0.454150, 0.000010),
            (  2.000000, 0.530000, 0.005000)))

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
        return get_critical_J(dim, numpy.array([mu]), reg)[:,0]
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

# Phase diagram related stuff #################################################################

def load_pd_params(file_name):
    with open(file_name, 'rb') as in_file:
        n_read = struct.unpack('Q', in_file.read(8))[0]
        assert n_read < (1 << 20), "Corrupt file"
        params = PhaseDiagramParams()
        params.ParseFromString(in_file.read(n_read))
    return params

def make_pd_params_table(file_names):
    table = []
    for file_name in file_names:
        params = load_pd_params(file_name)
        table.append((
            file_name, 
            params.j, 
            params.mu,
            params.shape[0],
            params.n_wolff,
            params.n_metropolis))
    table = pandas.DataFrame(
        table, 
        columns=[
            'file_name',
            'J',
            'mu',
            'L0',
            'wolf',
            'metr'])
    return table
        
def load_pd_data(file_name):
    with open(file_name, 'rb') as in_file:
        n_read = struct.unpack('Q', in_file.read(8))[0]
        assert n_read < (1 << 20), "Corrupt file"
        params = PhaseDiagramParams()
        params.ParseFromString(in_file.read(n_read))
        n_read = struct.unpack('Q', in_file.read(8))[0]
        assert n_read < (1 << 20), "Corrupt file"
        pd = PhaseDiagram()
        pd.ParseFromString(in_file.read(n_read))
        n_J = params.j_end - params.j_begin
        n_mu = params.mu_end - params.mu_begin
        J_increment = numpy.power(2.0, params.log2_j_increment)
        mu_increment = numpy.power(2.0, params.log2_mu_increment)
        return {
            'J0': [params.j],
            'mu0': [params.mu],
            'J_begin': params.j_begin,
            'J_end': params.j_end,
            'log2_J_increment': params.log2_j_increment,
            'mu_begin': params.mu_begin,
            'log2_mu_increment': params.log2_mu_increment,
            'mu_end':params.mu_end,
            'J_range': ((params.j_begin - 0.5) * J_increment, 
                        (params.j_end - 0.5) * J_increment),
            'mu_range': ((params.mu_begin - 0.5) * mu_increment, 
                         (params.mu_end - 0.5) * mu_increment),
            'distance': numpy.array(pd.distance).reshape((n_mu, n_J)),
            'susc': numpy.array(pd.susceptibility).reshape((n_mu, n_J)),
            'susc_std': numpy.array(pd.susceptibility_std).reshape((n_mu, n_J)),
            'binder': numpy.array(pd.binder_cumulant).reshape((n_mu, n_J)),
            'binder_std': numpy.array(pd.binder_cumulant_std).reshape((n_mu, n_J)),
            'hole_density': numpy.array(pd.hole_density).reshape((n_mu, n_J)),
            'hole_density_std': numpy.array(pd.hole_density_std).reshape((n_mu, n_J)),
            'si_sj': numpy.array(pd.si_sj).reshape((n_mu, n_J)),
            'si_sj_std': numpy.array(pd.si_sj_std).reshape((n_mu, n_J)),
        }

def load_pd_data_as_table(file_name):
    with open(file_name, 'rb') as in_file:
        n_read = struct.unpack('Q', in_file.read(8))[0]
        assert n_read < (1 << 20), "Corrupt file"
        params = PhaseDiagramParams()
        params.ParseFromString(in_file.read(n_read))
        n_read = struct.unpack('Q', in_file.read(8))[0]
        assert n_read < (1 << 20), "Corrupt file"
        pd = PhaseDiagram()
        pd.ParseFromString(in_file.read(n_read))
        n_J = params.j_end - params.j_begin
        n_mu = params.mu_end - params.mu_begin
        J_increment = numpy.power(2.0, params.log2_j_increment)
        mu_increment = numpy.power(2.0, params.log2_mu_increment)
        table = []
        for i_mu in range(params.mu_begin, params.mu_end):
            for i_J in range(params.j_begin, params.j_end):            
                k = (i_mu - params.mu_begin) * n_J + i_J - params.j_begin
                table.append([
                    params.j,
                    params.mu,
                    i_J * J_increment,
                    i_mu * mu_increment,
                    params.shape[0],
                    pd.susceptibility[k],
                    pd.susceptibility_std[k],
                    pd.binder_cumulant[k],
                    pd.binder_cumulant_std[k],
                    pd.hole_density[k],
                    pd.hole_density_std[k],
                    pd.si_sj[k],
                    pd.si_sj_std[k]
                ])
        return pandas.DataFrame(table, columns=[
            'J0', 'mu0', 'J', 'mu', 'L0', 'susc', 
            'susc_std', 'binder', 'binder_std', 'hole_density',
            'hole_density_std', 'si_sj', 'si_sj_std'])

def merge_pds(pds):
    assert len(pds) > 0
    res = {
        "J0": pds[0]['J0'],
        "mu0": pds[0]['mu0'],
        "J_begin": pds[0]['J_begin'],
        "J_end": pds[0]['J_end'],
        "log2_J_increment": pds[0]['log2_J_increment'],
        "mu_begin": pds[0]['mu_begin'],
        "mu_end": pds[0]['mu_end'],
        "log2_mu_increment": pds[0]['log2_mu_increment'],
    }

    for pd in pds[1:]:
        assert pd['log2_J_increment'] == res['log2_J_increment']
        assert pd['log2_mu_increment'] == res['log2_mu_increment']
        res['J0'].extend(pd['J0'])
        res['mu0'].extend(pd['mu0'])
        res['J_begin'] = min(res['J_begin'], pd['J_begin'])
        res['mu_begin'] = min(res['mu_begin'], pd['mu_begin'])
        res['J_end'] = max(res['J_end'], pd['J_end'])
        res['mu_end'] = max(res['mu_end'], pd['mu_end'])
        
    J_increment = numpy.power(2.0, res['log2_J_increment'])
    mu_increment = numpy.power(2.0, res['log2_mu_increment'])
    res['J_range'] = ((res['J_begin'] - 0.5) * J_increment, 
                      (res['J_end'] - 0.5) * J_increment)
    res['mu_range'] = ((res['mu_begin'] - 0.5) * mu_increment, 
                       (res['mu_end'] - 0.5) * mu_increment)
    
    n_J = res['J_end'] - res['J_begin']
    n_mu = res['mu_end'] - res['mu_begin']
    keys = [
        ['susc', 'susc_std'],
        ['susc_std', 'susc_std'],
        ['binder', 'binder_std'],
        ['binder_std', 'binder_std'],
        ['hole_density', 'hole_density_std'],
        ['hole_density_std', 'hole_density_std'],
        ['si_sj', 'si_sj_std'],
        ['si_sj_std','si_sj_std'],
        ['distance', 'distance']]
    
    for key, _ in keys:
        res[key] = numpy.full((n_mu, n_J), numpy.inf)

    for pd in pds:
        J0 = pd['J_begin'] - res['J_begin']
        J1 = pd['J_end'] - res['J_begin']
        mu0 = pd['mu_begin'] - res['mu_begin']
        mu1 = pd['mu_end'] - res['mu_begin']

        for key, std_key in keys:
            #std_key = 'distance'
            res[key][mu0:mu1, J0:J1] = numpy.where(
                res[std_key][mu0:mu1, J0:J1] > pd[std_key],
                pd[key], res[key][mu0:mu1, J0:J1])
    return res
