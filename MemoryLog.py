import subprocess
import time
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
def used_Memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE,
        text=True
    )
    free_memory = float(result.stdout.strip()) / 1024  # 转换为 GB
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
    for i in range(int(period*(3600/gap))):
        current_memory = used_Memory()
        log.append(current_memory)
        xticks_labels.append(CurrentTime())
        plt.plot(xticks[:i+1],log[:i+1],'o-')
        plt.xlabel('Time')
        plt.ylabel('Used Memory')
        plt.xticks(xticks[:i+1],xticks_labels[:i+1])
        plt.grid()
        plt.savefig(f'{imgpath}.png')
        time.sleep(gap)