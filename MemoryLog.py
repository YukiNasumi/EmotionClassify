import torch
import time
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
def used_Memory():
    free_memory, total_memory = torch.cuda.mem_get_info(0)
    free_memory = free_memory / 1024**2
    total_memory = total_memory / 1024**2
    return free_memory
def CurrentTime():
    return datetime.now().strftime("%H:%M")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--period',type=int,default=12)
    parser.add_argument('--imgpath',type=str,default=CurrentTime())
    parser.add_argument('--gap',type=int,default=300)
    args = parser.parse_args()
    period = args.period
    imgpath = args.imgpath
    gap = args.gap
    log = []
    xticks = list(range(period))
    xticks_labels = []
    for _ in xticks:
        current_memory = used_Memory()
        log.append(current_memory)
        xticks_labels.append(CurrentTime())
        time.sleep(gap)
    plt.plot(xticks,log,'-')
    plt.xlabel('Time')
    plt.ylabel('Used Memory')
    plt.xticks(xticks,xticks_labels)
    plt.grid()
    plt.savefig(f'{imgpath}.png')