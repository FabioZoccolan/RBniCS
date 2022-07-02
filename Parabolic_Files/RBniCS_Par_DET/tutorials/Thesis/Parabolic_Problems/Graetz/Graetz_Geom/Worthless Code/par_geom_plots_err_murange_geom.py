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

error_y_0 = genfromtxt('Parabolic_OCGraetzPOD3_GEOM_h_0.043_STAB_mu_1e5_2.0_alpha_0.01_T_3_t_0.1d_1_/error_analysis/error_analysis_Par_OCGraetz2_GEOM_h_0.043_OffONSTAB_mu_1e5_2.0_alpha_0.01_T_3_t_0.1/solution_u.csv', delimiter=';', skip_header=1)[:, 3]

error_y_1 = genfromtxt('Parabolic_OCGraetzPOD3_GEOM_h_0.043_STAB_mu_1e5_2.0_alpha_0.01_T_3_t_0.1d_1_/error_analysis/error_analysis_Par_OCGraetz2_GEOM_h_0.043_OffSTAB_mu_1e5_2.0_alpha_0.01_T_3_t_0.1/solution_u.csv', delimiter=';', skip_header=1)[:, 3]

basis0 = genfromtxt('Parabolic_OCGraetzPOD3_GEOM_h_0.043_STAB_mu_1e5_2.0_alpha_0.01_T_3_t_0.1d_1_/error_analysis/error_analysis_Par_OCGraetz2_GEOM_h_0.043_OffONSTAB_mu_1e5_2.0_alpha_0.01_T_3_t_0.1/solution_u.csv', delimiter=';', skip_header=1)[:, 0]

basis1 = genfromtxt('Parabolic_OCGraetzPOD3_GEOM_h_0.043_STAB_mu_1e5_2.0_alpha_0.01_T_3_t_0.1d_1_/error_analysis/error_analysis_Par_OCGraetz2_GEOM_h_0.043_OffSTAB_mu_1e5_2.0_alpha_0.01_T_3_t_0.1/solution_u.csv', delimiter=';', skip_header=1)[:, 0]
#basis = np.concatenate([basis0, basis1, basis2])

#error_y = np.concatenate([error_y_0, error_y_1, error_y_2])



xint = range(0, 20, 2)
plt.xlim(1,30)
#plt.xticks([2,4,6,8,10,12,14,16,18,20])
#plt.ylim(1e-5, 100)
#print(min(error_v))
plt.semilogy(basis0, error_y_0, marker = "o", linestyle = "solid", label = "online-stab")
plt.semilogy(basis1, error_y_1, marker = "o", linestyle = "solid", label = "online-notstab")
#plt.yticks(fontsize=13)
plt.xlabel('N', fontsize= 18) #fontdict=font15)
plt.ylabel('Relative Log-Error', fontsize= 18) # fontdict=font15)
plt.title("FEM vs ROM averaged relative error (control)", fontsize=18)
plt.legend(fontsize = 16)

plt.gcf().set_size_inches(12, 9)
plt.grid(color='b', axis='both', linestyle='dashed',linewidth=0.5)
plt.savefig('Par_square_h_0_036_error_u_murange_geom')



