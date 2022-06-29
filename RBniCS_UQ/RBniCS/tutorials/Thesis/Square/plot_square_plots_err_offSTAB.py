import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import math

font14 = {'size': 16,
        }

font16 = {'size': 18,
        }
font22 = {'size': 22,
        }
        
font15 = {'size': 15,
        }


error_y_0 = genfromtxt('Article2/Beta1010_1010/error_analysis/error_analysis_OC_Square3_h_0.025_OffSTAB_mu_2.4_1.2_alpha_0.01_beta1010_1010_standardPOD/solution_p.csv', delimiter=';', skip_header=1)[:, 3]

error_y_1 = genfromtxt('Article2/Beta1010_1010/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffSTAB_mu_2.4_1.2_alpha_0.01_beta1010_1010_w_montecarlo/solution_p.csv', delimiter=';', skip_header=1)[:, 3]

error_y_2 = genfromtxt('Article2/Beta1010_1010/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta1010_1010_gauss_tensor/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffSTAB_mu_2.4_1.2_alpha_0.01_beta1010_1010_gauss_tensor/solution_p.csv', delimiter=';', skip_header=1)[:, 3]

error_y_3 = genfromtxt('Article2/Beta1010_1010/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta1010_1010_gauss_smolyak/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffSTAB_mu_2.4_1.2_alpha_0.01_beta1010_1010_gauss_smolyak/solution_p.csv', delimiter=';', skip_header=1)[:, 3]

error_y_4 = genfromtxt('Article2/Beta1010_1010/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta1010_1010_ucc_tensor/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffSTAB_mu_2.4_1.2_alpha_0.01_beta1010_1010_ucc_tensor/solution_p.csv', delimiter=';', skip_header=1)[:, 3]

error_y_5 = genfromtxt('Article2/Beta1010_1010/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta1010_1010_ucc_smolyak/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffSTAB_mu_2.4_1.2_alpha_0.01_beta1010_1010_ucc_smolyak/solution_p.csv', delimiter=';', skip_header=1)[:, 3]

error_y_6 = genfromtxt('Article2/Beta1010_1010/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta1010_1010_haltonbeta/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffSTAB_mu_2.4_1.2_alpha_0.01_beta1010_1010_haltonbeta/solution_p.csv', delimiter=';', skip_header=1)[:, 3]

basis0 = genfromtxt('Article2/Beta1010_1010/UQ_OCSquare3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01_d_2.1_beta1010_1010_ucc_tensor/error_analysis/error_analysis_UQ_OC_Square3_h_0.025_OffSTAB_mu_2.4_1.2_alpha_0.01_beta1010_1010_ucc_tensor/solution_p.csv', delimiter=';', skip_header=1)[:, 0]

#basis = np.concatenate([basis0, basis1, basis2])

#error_y = np.concatenate([error_y_0, error_y_1, error_y_2])


plt.tick_params(axis='x', labelsize=27)
plt.tick_params(axis='y', labelsize=27)
#xint = range(0, 20, 2)
plt.xlim(1,50)
#plt.xticks([2,4,6,8,10,12,14,16,18,20])
#plt.ylim(1e-5, 100)
#print(min(error_v))
plt.semilogy(basis0, error_y_0, marker = "o", linestyle = "solid", label = "Standard POD", linewidth = 3)
plt.semilogy(basis0, error_y_1, marker = "o", linestyle = "solid", label = "Weighted POD Monte-Carlo", linewidth = 3)
plt.semilogy(basis0, error_y_2, marker = "o", linestyle = "solid", label = "GaussJacobi - tensor", linewidth = 3)
plt.semilogy(basis0, error_y_3, marker = "o", linestyle = "solid", label = "GaussJacobi - Smolyak", linewidth = 3)
plt.semilogy(basis0, error_y_4, marker = "o", linestyle = "solid", label = "ClenshawCurtis - tensor", linewidth = 3)
plt.semilogy(basis0, error_y_5, marker = "o", linestyle = "solid", label = "ClenshawCurtis - Smolyak", linewidth = 3)
plt.semilogy(basis0, error_y_6, marker = "o", linestyle = "solid", label = "PseudoRandom - Halton", linewidth = 3)

#plt.yticks(fontsize=13)
plt.xlabel('N', fontsize= 28) #fontdict=font15)
plt.ylabel('Relative Log-Error', fontsize= 28) # fontdict=font15)
plt.title("FEM vs ROM averaged relative error - p (adjoint)", fontsize=30)
plt.legend(fontsize = 15)

plt.gcf().set_size_inches(12, 10)
plt.grid(color='b', linestyle='--',linewidth=0.6)
plt.savefig('Plot_Square_offStab_h_0_025_error_p')

