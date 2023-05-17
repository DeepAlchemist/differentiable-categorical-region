# -*- coding: utf-8 -*-
# yangwenjie05@kuaishou.com.com @2022-08-06 15:33:12
# Last Change:  2022-11-05 21:26:42

import matplotlib.pyplot as plt

def custom_line_chart():
    x_val = list(range(1, 9))
    rank1 = [83.8, 85.7, 86.7, 86.2, 85.3, 85.1, 83.4, 82.7] 
    rank1 = [t+1.8 for t in rank1]
    mAP = [70.0, 72.8, 74.2, 73.4, 72.0, 71.1, 70.1, 68.2]
    mAP = [t+2.0 for t in mAP]
    y_val = mAP
    #create curve
    fig, ax = plt.subplots()
    ax.plot(x_val, y_val, color='blue')  # purple
    for _r, _p in zip(x_val, y_val):
        #ax.text(_r, _p+0.001, f"({_r:.3f},{_p:.3f})", ha="center", va="center", fontsize=7, rotation=45)
        ax.text(_r, _p+0.1, f"{_p:.1f}", ha="center", va="center", fontsize=12, rotation=0)
    
    #add axis labels to plot
    #ax.set_title('xxx')
    ax.set_ylabel('mAP-1 (%)', fontsize=12)
    ax.set_xlabel('Number of parts', fontsize=12)
    
    #display plot
    plt.savefig("./nparts_to_map.pdf", dpi=150)
    return

if __name__=="__main__":
    custom_line_chart()
