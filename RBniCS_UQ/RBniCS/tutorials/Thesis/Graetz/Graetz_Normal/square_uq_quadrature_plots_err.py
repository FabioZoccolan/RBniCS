import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

font14 = {'size': 16,
        }

font16 = {'size': 18,
        }


error_y_0 = genfromtxt('Numerical_Results/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_montecarlo/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_montecarlo/solution_y.csv', delimiter=';', skip_header=1)[:, 3]
error_y_1 =  genfromtxt('Numerical_Results/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_gaussjacobi_tensor/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_gaussjacobi_tensor/solution_y.csv', delimiter=';', skip_header=1)[:, 3]
error_y_2 = genfromtxt('Numerical_Results/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_gaussjacobi_smolyak/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_gaussjacobi_smolyak/solution_y.csv', delimiter=';', skip_header=1)[:, 3]
error_y_3 = genfromtxt('Numerical_Results/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_haltonbeta/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_haltonbeta/solution_y.csv', delimiter=';', skip_header=1)[:, 3]
error_y_4 = genfromtxt('Numerical_Results/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_ucc_tensor/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_ucc_tensor/solution_y.csv', delimiter=';', skip_header=1)[:, 3]
error_y_5 = genfromtxt('Numerical_Results/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_ucc_smolyak/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_ucc_smolyak/solution_y.csv', delimiter=';', skip_header=1)[:, 3]

basis0 = genfromtxt('Numerical_Results/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_montecarlo/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_montecarlo/solution_y.csv', delimiter=';', skip_header=1)[:, 0]



basis1 = genfromtxt('Numerical_Results/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_gaussjacobi_tensor/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_gaussjacobi_tensor/solution_y.csv', delimiter=';', skip_header=1)[:, 0]

basis2 = genfromtxt('Numerical_Results/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_gaussjacobi_smolyak/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_gaussjacobi_smolyak/solution_y.csv', delimiter=';', skip_header=1)[:, 0]

basis3 = genfromtxt('Numerical_Results/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_haltonbeta/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_haltonbeta/solution_y.csv', delimiter=';', skip_header=1)[:, 0]

basis4 = genfromtxt('Numerical_Results/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_ucc_tensor/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_ucc_tensor/solution_y.csv', delimiter=';', skip_header=1)[:, 0]

basis5 = genfromtxt('Numerical_Results/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta0304_0402_ucc_smolyak/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01_beta0304_0402_ucc_smolyak/solution_y.csv', delimiter=';', skip_header=1)[:, 0]



#basis = np.concatenate([basis0, basis1, basis2])

#error_y = np.concatenate([error_y_0, error_y_1, error_y_2])



plt.xlim(1,50)
# plt.ylim(5e4, 2e5)
#print(min(error_v))
plt.semilogy(basis0, error_y_0, marker = "o", linestyle = "solid", label = "montecarlo")
plt.semilogy(basis1, error_y_1, marker = "v", linestyle = "solid", label = "gauss_tensor")
plt.semilogy(basis2, error_y_2, marker = "^", linestyle = "solid", label = "gauss_smolyak")
plt.semilogy(basis3, error_y_3, marker = "<", linestyle = "solid", label = "haltonbeta")
plt.semilogy(basis4, error_y_4, marker = ">", linestyle = "solid", label = "ucc_tensor")
plt.semilogy(basis5, error_y_5, marker = "1", linestyle = "solid", label = "ucc_smolyak")
plt.yticks(fontsize=13)
plt.xlabel('N', fontdict=font14)
plt.ylabel('log-error', fontdict=font16)
plt.title("FE vs ROM averaged relative error", fontdict=font16)
plt.legend(fontsize = 16)
plt.gcf().set_size_inches(11, 9)
plt.grid()
plt.savefig('Quadratura-Err-_Square_mu_1e4_3e4')
