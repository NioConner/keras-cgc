#!/usr/bin/env python3
# encoding: utf-8
'''
@author: NioConner
@contact: 798225589@qq.com
@file: DataGenerater.py
@time: 2022-01-14 16:36
@desc:
'''
import pandas as pd
from tensorflow.keras.utils import to_categorical

SEED = 1024


def data_preparation():
    '''
    /data数据参考
    https://www2.1010data.com/documentationcenter/prod/Tutorials/MachineLearningExamples/CensusIncomeDataSet.html
    categrical 特征使用one-hot结构
    :return: data:pd.data lable:[array([1. 0.],[]...),array([],[]...)]
    '''

    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    # Load the dataset in Pandas
    train_df = pd.read_csv(
        'data/census-income.data.gz',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )
    test_df = pd.read_csv(
        'data/census-income.test.gz',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )
    # 不同的目标可以理解为最终的label
    label_columns = ['income_50k', 'marital_stat']

    # categorical 特征
    categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                           'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                           'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                           'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                           'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                           'vet_question']
    train_raw_labels = train_df[label_columns]
    test_raw_labels = test_df[label_columns]
    transformed_train = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    transformed_test = pd.get_dummies(test_df.drop(label_columns, axis=1), columns=categorical_columns)

    transformed_test['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0


    train_income = to_categorical((train_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    train_marital = to_categorical((train_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)
    test_income = to_categorical((test_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    test_marital = to_categorical((test_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)

    dict_outputs = {
        'income': train_income.shape[1],
        'marital': train_marital.shape[1]
    }
    dict_train_labels = {
        'income': train_income,
        'marital': train_marital
    }
    dict_test_labels = {
        'income': test_income,
        'marital': test_marital
    }
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    # 将testdata以1:1分为test和validation样本 多目标类型的label不是数组形式的[0,1] 而是dic下的数组{aim1:[0,1]...}
    validation_indices = transformed_test.sample(frac=0.5, replace=False, random_state=SEED).index
    test_indices = list(set(transformed_test.index) - set(validation_indices))
    validation_data = transformed_test.iloc[validation_indices]
    validation_label = [dict_test_labels[key][validation_indices] for key in sorted(dict_test_labels.keys())]
    test_data = transformed_test.iloc[test_indices]
    test_label = [dict_test_labels[key][test_indices] for key in sorted(dict_test_labels.keys())]
    train_data = transformed_train
    train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

    return train_data, train_label, validation_data, validation_label, test_data, test_label, output_info
