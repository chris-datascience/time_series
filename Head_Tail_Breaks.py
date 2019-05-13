import numpy as np

"""
Python implementation of the head/tail breaks algorithm for classifying heavy-tailed distributions.

sources:
    https://github.com/chad-m/head_tail_breaks_algorithm
    head tail breaks on wikipedia

Article reference:
http://arxiv.org/ftp/arxiv/papers/1209/1209.2801.pdf

Example:
    pareto_data = [(1/_)**1.16 for _ in range(1,100)]  # pareto distribution: x_min=1, a=1.16 (80/20)
htb_pareto = htb(pareto_data)
print(htb_pareto)
[0.03883742394002349, 0.177990388624465, 0.481845351678573]
JavaScript implementation for reference:
function htb(data) {
    var data_mean = data.reduce(function(a, b){return a + b})/data.length;
    var head = data.filter(function(d){return d > data_mean});
    console.log(data_mean);
    while (head.length > 1 && head.length/data.length < 0.40) {
        return htb(head);
    };
}
"""

def htb(data):
    """
    Function to compute the head/tail breaks algorithm on an array of data.
    Params:
    -------
    data (list): array of data to be split by htb.
    Returns:
    --------
    outp (list): list of data representing a list of break points.
    """
    # test input
    assert data, "Input must not be empty."
    assert all(isinstance(_, int) or isinstance(_, float) for _ in data), "All input values must be numeric."

    outp = []  # array of break points

    def htb_inner(data):
        """
        Inner ht breaks function for recursively computing the break points.
        """
        data_length = float(len(data))
        data_mean = sum(data) / data_length
        head = [_ for _ in data if _ > data_mean]
        outp.append(data_mean)
        while len(head) > 1 and len(head) / data_length < 0.40:
            return htb_inner(head)
    htb_inner(data)
    return outp

# =============================================================================
# Functions added by me:
# =============================================================================
def split_list(L, split_value):
    return [x for x in L if x<=split_value], [x for x in L if x>split_value]

def perform_head_tail_break(L):
    """
    Wrapper for head tail break routine, returning ht_index and list of lists with broken down
    sections of original list L, according to break points.
    This function is by me!
    
    Need to be sure elements in L are pure Python objects (ints or floats), i.e. not numpy ints etc.
    
    classified_list is list of same length as original, having classification indexes according to assigned classes.
    """
    L = L[:]
    if type(L[0])==np.int64:
        try:
            L = [int(x) for x in L]
        except Exception as e:
            print(e)
            print('\nFailed to convert to integers. Check for NaNs etc.\n')
    else:
        try:
            L = [float(x) for x in L]
        except Exception as e:
            print(e)
            print('\nFailed to convert to floats. Check for NaNs etc.\n')
    breakpoints = htb(L)
    ht_index = len(breakpoints) + 1  # defined as number of identified classes
    breakpoints = sorted(breakpoints)  # essential this is ascending sorted list
    breaks = []  # this becomes list of lists
    breaks_classificated = []
    unassigned_elements = L[:]
    break_index = 0
    while break_index<len(breakpoints):
        lower, upper = split_list(unassigned_elements, breakpoints[break_index])
        breaks.append(lower)
        breaks_classificated.append([break_index for _ in lower])
        unassigned_elements = upper[:]
        break_index += 1
        if break_index==len(breakpoints):
            breaks.append(upper)
            breaks_classificated.append([break_index for _ in upper])
    # Collect classes and map back to input data
    classified_list = []
    for bc in breaks_classificated:
        classified_list += bc
    classified_list = np.array(classified_list)[np.argsort(L)]  # restoring original order!
    """
        CHECK RESULTS: something still needs fixing.
    """
    return breaks, ht_index, breakpoints, classified_list


if __name__=='__main__':
    print('Test1')
    pareto_data = [(1/_)**1.16 for _ in range(1,100)]  # pareto distribution: x_min=1, a=1.16 (80/20)
    htb_pareto = htb(pareto_data)
    print(htb_pareto)
    
    print('\nTest2')
    testdata = [88,20,19,19,8,7,6,2,1,1,1,1,0]
    brokendown, HT_index, b_pts, classified_list = perform_head_tail_break(testdata)
    print([np.round(x, 2) for x in b_pts])
    print(classified_list)
    
    