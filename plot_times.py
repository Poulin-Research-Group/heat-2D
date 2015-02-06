import matplotlib.pyplot as plt
import numpy as np

M = range(8, 13)

par_step_s  = [0.67669319000000006, 4.9837427200000013, 19.886271460000003, 73.424382310000013, 279.66571850000003]
par_step_p2 = [0.66937611000000008, 3.5186517000000004, 14.595639349999999, 55.64336428999998, 255.78881990000005]
par_step_p4 = [0.75947747000000054, 3.561731819999999, 13.540444619999999, 52.69146482, 249.24051960000003]


def eff(t1, tP, p):
    # return np array of values that correspond to the efficiencies of parallel operations
    # efficiency is defined as: E_p = t1 / (p * tP)
    # t1 = list of serial times
    # tP = list of parallel times
    # p  = number of processors
    return np.array(t1) / (p*np.array(tP))

eff_step_p2 = eff(par_step_s, par_step_p2, 2)
eff_step_p4 = eff(par_step_s, par_step_p4, 4)


def eff_plot(op, title, eff_p2, eff_p4):
    m = M[:len(eff_p2)]
    plt.xlim([5.5, 14.5])
    plt.xlabel('$m$ ($M=2^m$, grid size is $(M+2)\\times(M+2)$ )')
    plt.ylabel('Efficiency')
    plt.title('Efficiency of %s' % title)
    plt.plot(m, eff_p2, '.-', label='p=2')
    plt.plot(m, eff_p4, '.-', label='p=4')
    plt.legend(loc='lower right')
    for M_eff in zip(m, eff_p2):
        plt.annotate('%.2f' % M_eff[1], xy=M_eff)
    for M_eff in zip(m, eff_p4):
        plt.annotate('%.2f' % M_eff[1], xy=M_eff)
    plt.savefig('./pics/eff_%s.pdf' % op)
    plt.clf()


def timing_plot(op, title, ylim, data_s, data_p2, data_p4):
    plt.xlim([6.5, 14.5])
    # plt.ylim(ylim)
    plt.xlabel('$m$ ($M=2^m$, grid size is $(M+2)\\times(M+2)$ )')
    plt.ylabel('Time (s)')
    plt.title(title)
    plt.plot(M, data_p2, label='2 cores')
    plt.plot(M, data_p4, label='4 cores')
    plt.plot(M, data_s, label='serial')
    plt.legend(loc='upper left')
    plt.savefig('./pics/%s.pdf' % op)
    plt.clf()

eff_plot('stepping', 'stepping solver of diffusion equation', eff_step_p2, eff_step_p4)

timing_plot('stepping', 'stepping solver of diffusion eqn', [], par_step_s, par_step_p2, par_step_p4)
