# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:43:40 2018

@author: erdbrca
"""

"""
GOAL:
    Apply regular expressions to manipulate and extract patterns from binary signals
    
See also https://www.youtube.com/watch?v=kWyoYtvJpe4&list=PL123FD827C7984559&index=4
"""

import re
import numpy as np


teststring = '00100110001110001111100101110'

# =============================================================================
# (0) Most basic: return first match found
# =============================================================================
match = re.search('iig', 'called piiig')
match.group() # returns the match

# =============================================================================
# (1) Replacement depending on returned object
# =============================================================================

def X_replace(matchobj):
#     print matchobj.group(0)
    return 'X'*len(matchobj.group(0))
print(teststring)
print(re.sub(r'111+',X_replace,teststring))

# =============================================================================
# (2) Get lengths of events and generate list of tuples with start/end indices of event
# =============================================================================
def get_long_event(signal,N):
    # ---Identify long events---
    if type(signal)==list:
        signal = str(signal).replace("0.0","0").replace("1.0","1").replace(", ", "").replace('[','').replace(']','')
    elif type(signal)==np.ndarray:
        signal = str(list(signal)).replace("0.0","0").replace("1.0","1").replace(", ", "").replace('[','').replace(']','')
    I = [(m.start(0), m.end(0)) for m in re.finditer(r'1'*(N-1)+'1+', signal)]
    event_lenghts = [I[i][1] - I[i][0] for i in range(len(I))]
    return I, event_lenghts
print(teststring)
(start_end_ind,lengths) = get_long_event(teststring,3) # returns start and end indices of events of length >=3
print(start_end_ind)
print(lengths)

s = np.array([1,1,0,0,1,1,1,0,0,1,0,1])
print(get_long_event(s,3))

# =============================================================================
# (3) Check whether certain pattern exists or not
# =============================================================================
def Find(pat, text): # typical function you'll use
    match = re.search(pat, text)
    if match: 
        print(match.group())
    else:
        print("not found")
print(teststring)
Find('10+1',teststring)

# =============================================================================
# (4) Return all occurring instances of desired pattern
# =============================================================================
re.findall('10+1', teststring)

print(dir(re.findall)) # show other options of findall command


