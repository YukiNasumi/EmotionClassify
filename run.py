import os
zipName = '~/LastExp1.zip'
os.system(f'python GPT2.py --name model20')
os.system(f'zip -r {zipName} /root/EmotionClassify')
os.system(f'oss cp {zipName} oss://')
os.system('shutdown')