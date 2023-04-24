import os


dir = os.listdir('./texts/')
files = []
for f in dir:
  if '.txt' in f:
    files.append('./texts/'+f)

texts = []
for file in files:
  chinese_text = []
  with open(file) as f:
      chinese_text = f.readlines()
  texts += chinese_text

f=open('./text_file.txt','w')
for ele in texts:
    f.write(ele)
f.close()