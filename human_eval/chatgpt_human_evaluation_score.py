#!/usr/bin/env python
# !-*-coding:utf-8 -*-
from scipy import stats
import pandas as pd
import krippendorff

if __name__ == '__main__':

    human_annotation = pd.read_excel('chatgpt_score.xlsx')
    human1 = human_annotation["annotator A"].tolist()
    human2 = human_annotation["annotator B"].tolist()
    human3 = human_annotation["annotator C"].tolist()
    human_label = [human1, human2, human3]
    human_label1 = human1+human2+human3
    print(f"Average evaluation score:{sum(human_label1)/len(human_label1)}")
    print("Krippendorff's alpha for ordinal metric: {}".format(
        krippendorff.alpha(reliability_data=human_label, level_of_measurement='ordinal')))