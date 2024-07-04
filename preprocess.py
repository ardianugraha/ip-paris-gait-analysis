import pandas as pd
from importlib import reload
import dataset.data_preprocessing as dp
reload(dp)


def get_data(directory, disabled):
    df_meta = dp.get_metadata_df(directory=directory, disabled=disabled)
    left, right, rate = dp.get_data_raw_cycles(directory=directory,
                                               list_joints=['Knee'],
                                               disabled=disabled)
    
    df_rate = pd.DataFrame(rate)
    df_rate.columns = ['Rate']
    df_meta = pd.concat([df_meta, df_rate], axis=1)

    return df_meta, left, right

def mean_centering(dictionary):
    for key, value in dictionary.items():
        mean_value = sum(value) / len(value)
        dictionary[key] = [item - mean_value for item in value]

def process_data(df_meta, left, right):
    left_dict = {}
    right_dict = {}

    for i in range(len(left)):
        id_patient = df_meta['id_patient'][i]
        if id_patient not in left_dict:
            left_dict[id_patient] = []
            count = 0
        if count < 3:
            for j in range(len(left[i])):
                left_dict[id_patient].append(left[i][j][1])
        count += 1
        
    for i in range(len(right)):
        id_patient = df_meta['id_patient'][i]
        if id_patient not in right_dict:
            right_dict[id_patient] = []
            count = 0
        if count < 3:
            for j in range(len(right[i])):
                right_dict[id_patient].append(right[i][j][1])
        count += 1

    mean_centering(left_dict)
    mean_centering(right_dict)

    df_lc_length = df_meta.groupby(['id_patient'])['Left cycle length'].sum()
    df_lc_length.name = 'Left signal length'
    df_rc_length = df_meta.groupby(['id_patient'])['Right cycle length'].sum()
    df_rc_length.name = 'Right signal length'
    df_disease = df_meta.groupby(['id_patient'])['Disease'].first()
    df_rate = df_meta.groupby(['id_patient'])['Rate'].first()
    
    df_meta_signal = pd.concat([df_disease,
                                 df_lc_length,
                                 df_rc_length,
                                 df_rate,
                                 ],
                                 axis=1)
    
    return df_meta_signal, left_dict, right_dict