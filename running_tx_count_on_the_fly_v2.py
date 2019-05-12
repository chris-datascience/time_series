# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:48:53 2018

@author: erdbrca
"""

import os
import sys, time
from datetime import datetime,  timedelta
import csv
from inplace_original import inplace

"""
Illustrative example showing the main idea of a tracker for a single identifier with integer minute timestamps.

def freq_track(T, ws):
    L = []
    for t in T:
        L.append(t)
        while L[-1] - L[0] > ws:
            L = L[1:]
        yield len(L)

Timestamps = [1,3,4,6,6,6,7,9,9,10,11,12,16,20] # all for the same identifier, let's assume
q = freq_track(Timestamps, ws=5)
print(list(q))
"""

def get_freq(ID_list, windowsize):
    """
       See illustrative example above.
       windowsize should be datetime timedelta object.
    """
    #ID_list.append(new_timestamp)
    while ID_list[-1] - ID_list[0] > windowsize:
        ID_list = ID_list[1:]
    return ID_list
    #return len(ID_list)
    
    
if __name__=='__main__':
    if len(sys.argv)!=2:
        print('\nInputfile required.')
        sys.exit(1)
    else:
        datafile = sys.argv[1]

    # --Manual inputs--
#    datafile = 'txNOV2017_to_account_100k.csv'
#    outputfile = 'txNOV2017_TEST.csv'

    TIMEDELTAS = {'minute': timedelta(minutes=1),  \
                  'hour': timedelta(minutes=60),   \
                  'halfday': timedelta(minutes=720)}
    TimestampRecords = {'minute': {}, 'hour': {}, 'halfday':{}}
    dateformat = '%d-%b-%y %H.%M.%S.%f'

    # [1] USING INPLACE EDITING
    start_time = time.time()
    with inplace(datafile, 'rb') as (infh, outfh):
        reader = csv.reader(infh)
        writer = csv.writer(outfh)
        writer.writerow(['tran_ref','tran_type','timestamp','to_account_id','tx_per_minute','tx_per_hour','tx_per_halfday'])
        for row in reader:
            tran_ref, tran_type, ts, accID = row
            ts_now = datetime.strptime(ts[:-3], dateformat)  # from str to datetime
            tx_counts = []
            for windowsize in ['minute', 'hour', 'halfday']:
                TimestampRecords[windowsize].setdefault(accID, []).append(ts_now)
                updatedRecord = TimestampRecords[windowsize][accID]            
            
                updatedTimestampList = get_freq(updatedRecord, TIMEDELTAS[windowsize])
                tx_counts.append( len(updatedTimestampList) )
                TimestampRecords[windowsize][accID] = updatedTimestampList
            
            # Write result for this line
            #writer.writerow(row + [minute_count, hour_count, day_count])
            writer.writerow(row + [tx_counts[0], tx_counts[1], tx_counts[2]])
    
    print('\nStopwatch: %2.2f minutes.\n'%((time.time() - start_time) / 60.))
