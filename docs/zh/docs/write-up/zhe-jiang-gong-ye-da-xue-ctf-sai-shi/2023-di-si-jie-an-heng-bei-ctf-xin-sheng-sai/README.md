# 2023第四届“安恒杯”CTF新生赛

## Misc

### Exif

下载图片后右键查看属性，发现图片的注释就是flag

&#x20;![20231224164016](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224164016.png)

### 是谁在搞渗透？

下载图片后用010 Editor打开，发现最后有一段PHP代码，是一句话木马，POST参数就是flag内容 ![20231224164233](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224164233.png)

### 加密的压缩包1

通过阅读题面，怀疑是压缩包伪加密。使用010 Editor打开，发现"struct ZIPFILERECORD record"的"enum COMPTYPE frCompression"（全局加密标志）与"struct ZIPDIRENTRY dirEntry"的"ushort deFlags"（单个文件加密标志）不一致，将后者改为8，保存后解压，得到flag

详细请参考：[https://blog.csdn.net/xiaozhaidada/article/details/124538768](https://blog.csdn.net/xiaozhaidada/article/details/124538768)

![20231224165441](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224165441.png)

### 我唱片怎么坏了

听音频，发现有一段声音有问题，用Audacity或Adobe Audition打开，使用频谱图直接看到flag

![20231224170114](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224170114.png)

### 黑铁的鱼影

使用010 Editor打开，发现模板运行报CRC校验错误

![20231224171426](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224171426.png)

结合图片观察（使用Windows自带的照片应用可以打开，其他软件可能会直接报错打不开），发现图片被截取了一部分，猜测是手动改变了高度导致看图软件无法完全显示，将高度改大，或使用CRC爆破脚本，恢复原来的高度，即可看到完整的图片。

PS: 如果做别的题目看到图片是撕裂的且CRC校验错误，也有可能是改变了宽度。

```python
import binascii
import struct

print("PNG宽高检查&爆破")
FileName=input("输入文件地址：")
if(FileName[0]=="\"" and FileName[len(FileName)-1]=="\""):
    FileName=FileName[1:len(FileName)-1]

crcbp = open(FileName, "rb").read()
data_f = crcbp[:12]
data_r = crcbp[29: ]
crc32frombp = int(crcbp[29:33].hex(),16)

w=int(crcbp[16:20].hex(),16)
h=int(crcbp[20:24].hex(),16)

print("宽："+str(w))
print("高："+str(h))

def check_size(data):
    crc32 = binascii.crc32(data) & 0xffffffff
    if(crc32 == crc32frombp):
        return True

data = crcbp[12:16] + \
    struct.pack('>i', w)+struct.pack('>i', h)+crcbp[24:29]

if check_size(data):
    print("校验正确，无需爆破")
    exit(0)
    
print("校验不正确，开始爆破")
OutFileName=FileName[0:len(FileName)-4]+"_fixed.png"

while True:
    minw=int(input("最小宽："))
    maxw=int(input("最大宽："))
    minh=int(input("最小高："))
    maxh=int(input("最大高："))
    print("爆破中...")
    for i in range(minw,maxw+1):
        for j in range(minh,maxh+1):
            data = crcbp[12:16] + \
                struct.pack('>i', i)+struct.pack('>i', j)+crcbp[24:29]
            if check_size(data):
                output=open(OutFileName,'wb')
                output.write(data_f + data + data_r)
                print("爆破成功！")
                print("宽：",i)
                print("高：",j)
                print("文件已输出至",OutFileName)
                exit(0)
    print("爆破失败，请重试")
```

### 凡凡的照片

使用Wireshark打开下载的pcapng文件，发现有一段HTTP POST流量，导出后发现是一张图片，打开后发现是flag

导出文件的办法：

1. 显示分组字节，自动渲染图片

&#x20;![20231224172519](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224172519.png)

1. 使用binwalk或foremost提取pcapng中的文件

### OSINT1

打开图片，仔细观察，发现车次和发车时间信息

![20231224172906](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224172906.png)

搜索D3135时刻表，发现11:54发车的是海宁西站

### QRCode.txt

下载附件得到一串疑似RGB信息的文本，行数为29\*29，编写脚本将其转换为图片，得到缺失的二维码，补上三个定位点后扫码即可得到flag

![20231224173321](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224173321.png)

![20231224173330](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224173330.png)

```python
from PIL import Image

def load_pixels_from_txt(txt_path, image_path, image_size):
    # 创建一个新的空白图片
    img = Image.new("RGB", image_size)
    pixels = img.load()

    # 读取文本文件中的像素数据
    with open(txt_path, "r") as file:
        # 将文件中的每行转换为RGB值，并设置对应的像素
        for y in range(image_size[1]):
            for x in range(image_size[0]):
                line = file.readline().strip()
                if line:
                    r, g, b = map(int, line.strip("()").split(","))
                    pixels[x, y] = (r, g, b)

    # 保存图片
    img.save(image_path)

load_pixels_from_txt("QRCode.txt", "reconstructed_image.jpg", (29, 29))
```

### OSINT2

首先阅读题面，搜索2023深圳1024程序员节CTF比赛，发现比赛地点在“深圳市龙华区北站中心公园”。 观察图片，发现对面有一座中国石化加油站。在比赛地点附近搜索，并比对图片与卫星图中的马路、天桥、周围建筑等特征，找到拍摄人所处的楼宇是“鸿荣源·天俊D栋”

![20231224173938](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224173938.png)

### 聪明的小明

阅读密码提示，使用python脚本生成字典

```python
import itertools
import random
import string
import datetime

# 小明的名字拼音
base_name = "xiaoming"

# 最小和最大密码长度
min_length = 17
max_length = 20
# 特殊字符集合
special_characters = "!@#$%^&*()_+-=[]{}|;:,.<>?/\\"

# 生成密码
passwords = []
for index in range(0,8):
    lowername_list = list(base_name)
    lowername_list[index] = lowername_list[index].upper()
    name = ''.join(lowername_list)
    for year in range(2021, 2022):  # 仅包括2021年
        for month in range(1, 13):  # 1到12月
            for day in range(1, 32):  # 1到31日
                try:
                    # 尝试创建日期对象，如果日期不存在会引发异常
                    date_suffix = datetime.date(year, month, day).strftime("%Y%m%d")
                    for char1 in special_characters:
                        for char2 in special_characters:
                            password = name+date_suffix+char1+char2
                            passwords.append(password)
                except ValueError:
                    # 日期不存在，跳过
                    continue

# 将密码保存到文件
with open('password_dictionary.txt', 'w') as file:
    file.write('\n'.join(passwords))

print(f"已生成密码字典，保存到 'password_dictionary.txt' 文件中，共有 {len(passwords)} 个密码。")
```

使用john破解（office2john），得到密码：xiaoMing20210818()

打开PPT后发现需要寻找flag，按下Ctrl+F搜索“flag”或用Ctrl+A全选或打开选择窗格发现有一段隐藏的文本，即为flag

![20231224174543](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224174543.png)
