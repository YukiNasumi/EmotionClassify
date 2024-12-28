import os
import time
import torch

THRESHOLD = 40000  

# 要启动的 Python 脚本路径
TARGET_SCRIPT = "/workspace/zzy/zzy2/EmotionClassify/BERT.py"

# 检查显存的函数
def check_free_memory():
    free_memory, total_memory = torch.cuda.mem_get_info(0)
    free_memory = free_memory / 1024**2
    total_memory = total_memory / 1024**2
    print(f'当前可用显存{free_memory}MB')
    if(free_memory>THRESHOLD) :
        return True
    else:
        return False
    


# 启动目标脚本的函数
def launch_script():
    print(f"显存充足，启动脚本: {TARGET_SCRIPT}")
    os.system(f"python {TARGET_SCRIPT} --name model17")

if __name__ == "__main__":
    #time.sleep(1800)
    for _ in range(4):
        
        if check_free_memory():
            flag=True
            for _ in range(5):
                print('显存暂时充足，接下来每隔3分钟检查一次，共检查5次，确认显卡空闲再运行任务')
                time.sleep(180)
                flag = check_free_memory()
                if not flag:
                    break
            if flag:
                start = time.time()
                launch_script()
                end = time.time()
                print(f'执行完成，退出程序,本次任务耗时{end-start:.4f}s')
                break
        else:
            print("显存不足，继续等待...")
        time.sleep(1800)
