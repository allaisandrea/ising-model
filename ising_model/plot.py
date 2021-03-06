from matplotlib import pyplot
import collections
import numpy
import ising_model


def plot_autocorrelation(
        data, groupby=['mu', 'J', 'L0', 'n_wolff', 'file_group'], fig_size=(9, 5),
        label_format='g:{file_group:3} mu:{mu:9.6f} L:{L0:3} J:{J:9.6f} wolff:{n_wolff:4}',
        show_cpu_time=False):
    groups = data.groupby(groupby)
    figure, axes = pyplot.subplots(1, 2)
    axes = numpy.reshape(axes, (axes.size,))
    figure.set_size_inches(*fig_size)
    for key, group in groups:
        label = label_format.format(**(next(group.itertuples())._asdict()))
        abscissae = group['tau'] * group['measure_every']
        if show_cpu_time:
            abscissae = abscissae * \
                (group['t_wolff'] + group['t_metropolis']) * 1e-3

        axes[0].errorbar(
            abscissae, group['ud_ac'], group['ud_ac_std'],
            capsize=4, label=label, linestyle='', marker='x')
        axes[0].set_ylabel('ud_ac')
        axes[1].errorbar(
            abscissae, group['h_ac'], group['h_ac_std'],
            capsize=4, linestyle='', marker='x')
        axes[1].set_ylabel('hole_ac')

    if show_cpu_time:
        axes[0].set_xlabel('time (s)')
        axes[1].set_xlabel('time (s)')
    else:
        axes[0].set_xlabel('seq_index')
        axes[1].set_xlabel('seq_index')
        max_x_range = (data['tau'] * data['measure_every']).max()
        axes[0].set_xticks([(max_x_range // 8) * i for i in range(0, 9)])
        axes[1].set_xticks([(max_x_range // 8) * i for i in range(0, 9)])

    axes[0].grid()
    axes[1].grid()

    figure.legend(
        loc='upper left',
        bbox_to_anchor=(0, 1, 1, 0.05 * len(groups)), ncol=1,
        mode='expand', prop={'family': 'Monospace'})
    figure.tight_layout()
    return figure, axes


def plot_J_cut(data, observable, xlim=None, ylim=None,
               groupby=['mu', 'L0', 'J0', 'n_wolff', 'file_group'], fig_size=(9, 5),
               label_format='g:{file_group:3} mu:{mu:9.6f} L:{L0:3} J0:{J0:10.7f} wolff:{n_wolff:4}',
               show_cpu_time=False):
    groups = data.groupby(groupby)

    figure, axes = pyplot.subplots(1, 1, squeeze=False)
    axes = axes.reshape(axes.size)
    figure.set_size_inches(*fig_size)
    for key, group in groups:
        label = label_format.format(**(next(group.itertuples())._asdict()))
        axes[0].plot(group['J'], group[observable], label=label)
        axes[0].fill_between(
            group['J'],
            group[observable] - group[observable + '_std'],
            group[observable] + group[observable + '_std'],
            alpha=0.7)

    if xlim:
        axes[0].set_xlim(xlim)
    if ylim:
        axes[0].set_ylim(ylim)

    if observable == 'binder_cumulant':
        axes[0].axhline(0.27052, linewidth=0.5, color='r')

    for J0 in data['J0'].unique():
        axes[0].axvline(J0, linestyle='--', linewidth=0.5)

    axes[0].set_title(observable)
    axes[0].grid()

    figure.legend(
        loc='upper left',
        bbox_to_anchor=(0, 1, 1, 0.3), ncol=1,
        mode='expand', prop={'family': 'Monospace'})
    figure.tight_layout()
    return figure, axes


def get_J_range(dim, mu):
    J_range_table = {
        (4, -1024.000): (0.1496938, 0.000002),
        (4, -1.000000): (0.1750000, 0.005000),
        (4,  0.000000): (0.2156000, 0.000500),
        (4,  1.000000): (0.3165000, 0.000500),
        (4,  1.500000): (0.4074800, 0.000050),
        (4,  1.625000): (0.4354900, 0.000050),
        (4,  1.656250): (0.4428600, 0.000050),
        (4,  1.687500): (0.4503415, 0.000005),
        (4,  1.703125): (0.4541500, 0.000050),
        (4,  1.718750): (0.4579800, 0.000050),
        (4,  1.750000): (0.4650000, 0.005000),
        (4,  2.000000): (0.5280000, 0.010000),
    }
    J_range_table = {key: (J[0] - J[1], J[0] + J[1])
                     for key, J in J_range_table.items()}
    return J_range_table[(dim, mu)]


def plot_time_series_single(file_name, max_length, use_stamp,  axes):
    params = ising_model.udh.load_params(file_name)
    vol = 1
    for L in params.shape:
        vol *= L
    observables_array = ising_model.udh.load_observables(file_name)
    time_series = collections.defaultdict(lambda: list())
    i = 0
    stamps = []
    for observables in observables_array:
        #time_series['flip_cluster'].append(observables.flip_cluster_duration)
        #time_series['clear_flag'].append(observables.clear_flag_duration)
        #time_series['metropolis'].append(observables.metropolis_sweep_duration)
        #time_series['measure'].append(observables.measure_duration)
        #time_series['serialize'].append(observables.serialize_duration)
        time_series['phi2'].append(numpy.square(
            (observables.n_up - observables.n_down) / vol))
        time_series['n_holes'].append(observables.n_holes / vol)
        stamps.append(observables.stamp)
        i += 1
        if i >= max_length:
            break

    for i, (key, value) in enumerate(time_series.items()):
        if use_stamp:
            abscissae = stamps
        else:
            abscissae = range(len(stamps))
        axes[i, 0].plot(abscissae, value, marker='o', linestyle='None', markersize=1)
        axes[i, 0].set_title(key)
        axes[i, 1].hist(value, bins=32, alpha=0.5)


def plot_time_series(file_names, max_length=8192, use_stamp=False):
    figure, axes = pyplot.subplots(2, 2, squeeze=False)
    figure.set_size_inches(9, 4)
    for file_name in file_names:
        plot_time_series_single(file_name, max_length, use_stamp, axes)
    figure.tight_layout()
    return figure, axes
