import pandas as pd
from scipy import stats
import krippendorff


def compute_line_number(code):
    "compute line number of code"
    non_blank_number = len(code.rstrip().split('\n'))
    blank_number = len(code)-len(code.rstrip('\n'))
    return non_blank_number+blank_number

# read_excel
excel_filename = 'classifier_score.xlsx'
df = pd.read_excel(excel_filename)

#read annotated resultes
annotated_code_list = df['annotator A'].tolist()
annotated_code_list1 = df['annotator B'].tolist()
annotated_code_list2 = df['annotator C'].tolist()
end_line_list = [ compute_line_number(code) for code in annotated_code_list]
end_line_list1 = [ compute_line_number(code) for code in annotated_code_list1]
end_line_list2 = [ compute_line_number(code) for code in annotated_code_list2]
    
line = [end_line_list, end_line_list1, end_line_list2]
print("Krippendorff's alpha for ordinal metric: {}".format(
        krippendorff.alpha(reliability_data=line, level_of_measurement='ordinal')))