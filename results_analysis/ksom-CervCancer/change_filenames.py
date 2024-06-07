# -*- coding: utf-8 -*-
"""
Change filename in order to put it in a new pattern
"""

import os

############# For Kernel SOM

# cervicalCancer_2_ksomef_hold_1_hpo_1_norm_3_lbl_1_nn_1_Nep_50_Nprot_30_Kt_1.mat

old_file_1 = "cervicalCancer_2_ksomgd_hold_1_hpo_b_norm_3_lbl_"
old_file_2 = ["1","2","3"]
old_file_3 = "_nn_1_Nep_50_Nprot_30_Kt_"
old_file_4 = ["1","2","3","4","5","6","7","8"]
old_file_5 = ".mat"

new_file_1 = "cervicalCancer_2_ksomgd_hold_1_norm_3_hpo_b_lbl_"

for lbl in old_file_2:
    for ktype in old_file_4:
        old_file_name = old_file_1 + lbl + old_file_3 + ktype + old_file_5
        print(old_file_name)
        new_file_name = new_file_1 + lbl + old_file_3 + ktype + old_file_5
        print(new_file_name)
        try:
            os.rename(old_file_name, new_file_name)
        except FileNotFoundError:
            print(f"{old_file_name} does not exist.")

############# For Spark / Spok

"""
old_file_1 = "motorFailure_isk2nn_hpo1_norm3_Dm"
old_file_2 = ["1","2"]
old_file_3 = "_Ss"
old_file_4 = ["1","2","3","4"]
old_file_5 = "_Us0_Ps0_"
old_file_6 = ["cau","exp","gau","kmod","lin","log","pol","sig"]
old_file_7 = "_"
old_file_8 = ["1","2"]
old_file_9 = "nn.mat"

new_file_1 = "motorFailure_2_spok_hold_2_norm_3_hpo_1_Dm"

for Dm in old_file_2:
    for Ss in old_file_4:
        for Kt in old_file_6:
            for NN in old_file_8:
                old_file_name = old_file_1 + Dm + old_file_3 + Ss + \
                                old_file_5 + Kt + old_file_7 + NN + \
                                old_file_9
                print(old_file_name)

                new_file_name = new_file_1 + Dm + old_file_3 + Ss + \
                                old_file_5 + Kt + old_file_7 + NN + \
                                old_file_9
                print(new_file_name)
                
                try:
                    os.rename(old_file_name, new_file_name)
                except FileNotFoundError:
                    print(f"{old_file_name} does not exist.")
"""