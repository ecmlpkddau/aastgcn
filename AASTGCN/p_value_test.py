from scipy import stats
# stu test for two series
def p_test(timeseries1,timeseries2):
    # First check the variance of the two samples.
    # If the output p-value is greater than 0.05, it means that the variance is equal;
    # if the output p-value is less than 0.05, it means that the variance is not equal.
    results_lev = stats.levene(timeseries1, timeseries2)
    len_p_value = results_lev[1]
    equal_val = True
    if len_p_value > 0.05:
        equal_val = False
    # If the variance of the two samples is the same, then equal_val=True;
    # If the variance of the two samples is different, then equal_val=False
    results = stats.ttest_ind(timeseries1, timeseries2, equal_var=equal_val)
    p_value = results[1]
    return p_value

