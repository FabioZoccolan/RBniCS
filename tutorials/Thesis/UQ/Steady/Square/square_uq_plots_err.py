import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

font14 = {'size': 16,
        }

font16 = {'size': 18,
        }


error_y_0 = genfromtxt('Numerical_Results/Beta0304_0402/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_montecarlo/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_montecarlo/solution_y.csv', delimiter=';', skip_header=1)[:, 3]
error_y_1 =  genfromtxt('Numerical_Results/Beta0304_0402/error_analysis_GIUSEPPE/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_montecarlo/solution_y.csv', delimiter=';', skip_header=1)[:, 3]
error_y_2 = genfromtxt('Numerical_Results/Beta0304_0402/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_montecarlo_OLD/solution_y.csv', delimiter=';', skip_header=1)[:, 3]


basis0 = genfromtxt('Numerical_Results/Beta0304_0402/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_montecarlo/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_montecarlo/solution_y.csv', delimiter=';', skip_header=1)[:, 0]



basis1 =  genfromtxt('Numerical_Results/Beta0304_0402/error_analysis_GIUSEPPE/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_montecarlo/solution_y.csv', delimiter=';', skip_header=1)[:, 0]

basis2 =  genfromtxt('Numerical_Results/Beta0304_0402/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_montecarlo_OLD/solution_y.csv', delimiter=';', skip_header=1)[:, 0]


#basis = np.concatenate([basis0, basis1, basis2])

#error_y = np.concatenate([error_y_0, error_y_1, error_y_2])



plt.xlim(1,50)
# plt.ylim(5e4, 2e5)
#print(min(error_v))
plt.semilogy(basis0, error_y_0, marker = "o", linestyle = "solid", label = "montecarlo_mio_testing_set_Giuseppe")
plt.semilogy(basis1, error_y_1, marker = "v", linestyle = "solid", label = "montecarlo_giuseppe")
plt.semilogy(basis2, error_y_2, marker = "^", linestyle = "solid", label = "montecarlo_mio_NO_TESTING_GIUS")

plt.yticks(fontsize=13)
plt.xlabel('N', fontdict=font14)
plt.ylabel('log-error', fontdict=font16)
plt.title("FE vs ROM averaged relative error", fontdict=font16)
plt.legend(fontsize = 16)
plt.gcf().set_size_inches(11, 9)
plt.grid()
plt.savefig('Quadratura-Err-Square_mio_vs_giuseppe')
