import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

font14 = {'size': 16,
        }

font16 = {'size': 22,
        }


error_y_0 = genfromtxt('Article/AdvectionOCGraetzPOD2_h_0.029_STAB_mu_1e5_alpha_0.01/error_analysis/error_analysis_OCGraetz2_h_0.029_OffONSTAB_mu_1e5_alpha_0.01/solution_u.csv', delimiter=';', skip_header=1)[:, 3]
error_y_1 = genfromtxt('Article/AdvectionOCGraetzPOD2_h_0.029_STAB_mu_1e5_alpha_0.01/error_analysis/error_analysis_OCGraetz2_h_0.029_OffSTAB_mu_1e5_alpha_0.01/solution_u.csv', delimiter=';', skip_header=1)[:, 3]


basis0 = genfromtxt('Article/AdvectionOCGraetzPOD2_h_0.029_STAB_mu_1e5_alpha_0.01/error_analysis/error_analysis_OCGraetz2_h_0.029_OffONSTAB_mu_1e5_alpha_0.01/solution_u.csv', delimiter=';', skip_header=1)[:, 0]
basis1 = genfromtxt('Article/AdvectionOCGraetzPOD2_h_0.029_STAB_mu_1e5_alpha_0.01/error_analysis/error_analysis_OCGraetz2_h_0.029_OffSTAB_mu_1e5_alpha_0.01/solution_u.csv', delimiter=';', skip_header=1)[:, 0]


#basis = np.concatenate([basis0, basis1, basis2])

#error_y = np.concatenate([error_y_0, error_y_1, error_y_2])


plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.xlim(1,8)
# plt.ylim(5e4, 2e5)
#print(min(error_v))
plt.semilogy(basis0, error_y_0, marker = "o", linestyle = "solid", label = "Online stab.", linewidth = 3)
plt.semilogy(basis1, error_y_1, marker = "o", linestyle = "solid", label = "Online not stab.", linewidth = 3)
#plt.yticks(fontsize=13)
plt.xlabel('N', fontdict=font16)
plt.ylabel('Relative Log-Error', fontdict=font16)
plt.title("FEM vs ROM averaged relative error - u (control)", fontdict=font16)
plt.legend(fontsize = 16)
plt.gcf().set_size_inches(10, 8)
plt.grid(color='b', linestyle='--',linewidth=0.6)
plt.savefig('Article/Graetz_h_0_029_error_u_art')
