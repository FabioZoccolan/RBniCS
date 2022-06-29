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

error_y_0 = genfromtxt('Article/AdvectionOCSquarePOD3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01/error_analysis/error_analysis_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01/solution_u.csv', delimiter=';', skip_header=1)[:, 3]

error_y_1 = genfromtxt('Article/AdvectionOCSquarePOD3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01/error_analysis/error_analysis_OC_Square3_h_0.025_OffSTAB_mu_2.4_1.2_alpha_0.01/solution_u.csv', delimiter=';', skip_header=1)[:, 3]

basis0 = genfromtxt('Article/AdvectionOCSquarePOD3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01/error_analysis/error_analysis_OC_Square3_h_0.025_OffONSTAB_mu_2.4_1.2_alpha_0.01/solution_u.csv', delimiter=';', skip_header=1)[:, 0]

basis1 = genfromtxt('Article/AdvectionOCSquarePOD3_h_0.025_STAB_mu_2.4_1.2_alpha_0.01/error_analysis/error_analysis_OC_Square3_h_0.025_OffSTAB_mu_2.4_1.2_alpha_0.01/solution_u.csv', delimiter=';', skip_header=1)[:, 0]
#basis = np.concatenate([basis0, basis1, basis2])

#error_y = np.concatenate([error_y_0, error_y_1, error_y_2])



xint = range(0, 20, 2)
plt.tick_params(axis='x', labelsize=23)
plt.tick_params(axis='y', labelsize=23)
plt.xlim(1,50)
#plt.xticks([2,4,6,8,10,12,14,16,18,20])
#plt.ylim(1e-5, 100)
#print(min(error_v))
plt.semilogy(basis0, error_y_0, marker = "o", linestyle = "solid", label = "Online stab.", linewidth = 3)
plt.semilogy(basis1, error_y_1, marker = "o", linestyle = "solid", label = "Online not stab.", linewidth = 3)
#plt.yticks(fontsize=13)
plt.xlabel('N', fontsize= 24) #fontdict=font15)
plt.ylabel('Relative Log-Error', fontsize= 24) # fontdict=font15)
plt.title("FEM vs ROM averaged relative error - u (control)", fontsize=24)
plt.legend(fontsize = 16)

plt.gcf().set_size_inches(12, 9)
plt.grid(color='b', linestyle='--',linewidth=0.6)
plt.savefig('Article/square_h_0_025_error_u')



