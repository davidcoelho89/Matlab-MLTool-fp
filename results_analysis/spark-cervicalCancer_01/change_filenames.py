# -*- coding: utf-8 -*-
"""
Change filename in order to put it in a new pattern
"""

import os

old_file_1 = "cervicalCancer_spok_hpo1_norm3_Dm"
old_file_2 = ["1","2"]
old_file_3 = "_Ss"
# old_file_4 = ["1","2","3","4"]
old_file_4 = ["1","2"]
old_file_5 = "_Us0_Ps0_"
old_file_6 = ["cau","exp","gau","kmod","lin","log","pol","sig"]
old_file_7 = "_nn"
old_file_8 = ["1","2"]
old_file_9 = ".mat"

# cervicalCancer_spok_hpo1_norm3_Dm1_Ss1_Us0_Ps0_cau_nn1.mat

new_file_1 = "cervicalCancer_1_spok_hpo1_norm3_Dm"

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
                
