import numpy as np
import scipy.special


def T_to_Z(statistic, dof, verbose=False):
    """Convert T statistics to Z statistics, taking care of underflow situations using method outlined in FMRIB Technical Report TR00MJ1"""

    # first do all together
    astat = abs(statistic)
    stat_sgn = statistic/astat
    P = scipy.special.betainc(dof/2.0,0.5,dof/(dof + astat*astat))/2
    zstat = scipy.special.erfinv(-(P-0.5)*2)*np.sqrt(2)

    # then look for values which may cause a problem
    if dof >= 15:          # (value from FMRIB Technical Report)
        # find the indices of all values >= 7.5 (value from FMRIB Technical Report)
        astat_as = astat.argsort()
        b = (astat[astat_as]>=7.5).searchsorted(True)
        large_vals_indx = astat_as[b:]

        # replace overflowed z-statistic value with asymptotic approximate value for each index
        if len(large_vals_indx) > 0:
            large_vals = astat[large_vals_indx]
            large_vals_2 = large_vals*large_vals
            log_p = -0.5*np.log(2*np.pi) - 1.0/(4*dof) - np.log(large_vals) - (dof-1.0)*np.log(1.0+large_vals_2/dof)/2.0 \
                        + np.log(1.0-dof/((dof+2.0)*large_vals_2) + 3*dof*dof/((dof+2.0)*(dof+4.0)*large_vals_2*large_vals_2))
            z0 = np.sqrt(-2*log_p - np.log(2*np.pi))
            z0_2 = z0*z0
            z1 = np.sqrt(-2*log_p - np.log(2*np.pi) - 2*np.log(z0) + 2*np.log(1 - 1.0/(z0_2) + 3.0/(z0_2*z0_2)))
            z1_2 = z1*z1
            z2 = np.sqrt(-2*log_p - np.log(2*np.pi) - 2*np.log(z1) + 2*np.log(1 - 1.0/(z1_2) + 3.0/(z1_2*z1_2)))
            z2_2 = z2*z2
            z3 = np.sqrt(-2*log_p - np.log(2*np.pi) - 2*np.log(z2) + 2*np.log(1 - 1.0/(z2_2) + 3.0/(z2_2*z2_2)))
            zstat[large_vals_indx] = z3
#        if verbose:
#            print 'Fixed %d values' % len(large_vals_indx)

    else:
        print ('WARNING: have not checked z-statistics for spurious values (not implemented for dof < 15 yet).')

    statistic = zstat * stat_sgn

    return statistic

