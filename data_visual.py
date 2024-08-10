import re
import os
import pandas as pd
import matplotlib.pyplot as plt

app_gL = 0.000
highperL = 0.000
styL = 0.000
content = []
label = ['l1', 'highperPL', 'sty_loss']
with open('../learn_test/loss_log.txt', 'r') as file:
    for line in file:
        match = re.search(r'^\s*\((.*?)\)(.*?)$', line)
        if match:
            content.append((match.group(1).strip(), match.group(2).strip()))
    # print(content)
    # print(len(content))
    # print(content[0][0].split(':')[1].split(',')[0])
    # print(content[0][1].split(':')[4].split()[0])

dic = {}
for index in range(len(content)):
    d1 = {}
    for k, v in re.findall(r'(\w+):\s+(\S+)', content[index][0]):
        d1[k] = float(v.split(',')[0]) #if re.match(r'^[0-9.]+$', v) else v

    d2 = {}
    for k, v in re.findall(r'(\w+):\s+([0-9.]+)', content[index][1]):
        d2[k] = float(v.split(',')[0])

    result = {**d1, **d2}
    dic[index] = result
print(dic)

count = 0.0
with open("evaluations.txt", "w") as log_file:
    for i in range(len(dic)):
        if dic[i]['epoch'] == 1 or dic[i]['epoch'] == dic[i-1]['epoch'] :
            #print(dic[0]["app_g"])
            app_gL += float(dic[i]["app_g"])
            highperL += float(dic[i]["highper"])
            styL += float(dic[i]["sty"])
            float(app_gL)
            float(highperL)
            float(styL)
            count += 1.0
        else:
            app_gL = '%.3f' % (app_gL / count)
            highperL = '%.3f' % (highperL / count)
            styL = '%.3f' % (styL / count)
            log_file.write('%s\t%s\t%s\n' % (app_gL, highperL, styL))
            app_gL = 0.000
            highperL = 0.000
            styL = 0.000
            count = 0.0

# 读取数据，并将每列数据存储在一个列表中
data = []
with open('evaluations.txt', 'r') as f:
    for line in f:
        data.append([float(x) for x in line.strip().split('\t')])

# 绘制折线图
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # 定义颜色列表
labels = ['loss:{}'.format(label[i]) for i in range(len(label))]  # 定义标签列表
for i in range(len(data[0])):
    plt.plot([row[i] for row in data], color=colors[i % len(colors)], label=labels[i])
plt.legend()

plt.show()