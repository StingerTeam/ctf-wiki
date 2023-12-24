# 2023第四届“安恒杯”CTF新生赛

## Misc

### Exif

下载图片后右键查看属性，发现图片的注释就是flag

![Exif](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224164016.png)

### 是谁在搞渗透？

下载图片后用010 Editor打开，发现最后有一段PHP代码，是一句话木马，POST参数就是flag内容

![是谁在搞渗透？](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224164233.png)

### 加密的压缩包1

通过阅读题面，怀疑是压缩包伪加密。使用010 Editor打开，发现"struct ZIPFILERECORD record"的"enum COMPTYPE frCompression"（全局加密标志）与"struct ZIPDIRENTRY dirEntry"的"ushort deFlags"（单个文件加密标志）不一致，将后者改为8，保存后解压，得到flag

详细请参考：[https://blog.csdn.net/xiaozhaidada/article/details/124538768](https://blog.csdn.net/xiaozhaidada/article/details/124538768)

![加密的压缩包1](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224165441.png)

### 我唱片怎么坏了

听音频，发现有一段声音有问题，用Audacity或Adobe Audition打开，使用频谱图直接看到flag

![我唱片怎么坏了](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224170114.png)

### 黑铁的鱼影

使用010 Editor打开，发现模板运行报CRC校验错误

![黑铁的鱼影](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224171426.png)

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

![凡凡的照片](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224172519.png)

2. 使用binwalk或foremost提取pcapng中的文件

### OSINT1

打开图片，仔细观察，发现车次和发车时间信息

![OSINT1](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224172906.png)

搜索D3135时刻表，发现11:54发车的是海宁西站

### QRCode.txt

下载附件得到一串疑似RGB信息的文本，行数为29\*29，编写脚本将其转换为图片，得到缺失的二维码，补上三个定位点后扫码即可得到flag

![QRCode.txt-1](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224173321.png)

![QRCode.txt-2](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224173330.png)

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

首先阅读题面，搜索2023深圳1024程序员节CTF比赛，发现比赛地点在“深圳市龙华区北站中心公园”。

观察题目给的图片，发现对面有一座中国石化加油站。在比赛地点附近搜索，并比对图片与卫星图中的马路、天桥、周围建筑等特征，找到拍摄人所处的楼宇是“鸿荣源·天俊D栋”

![OSINT2](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224173938.png)

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

![聪明的小明](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224174543.png)

## Web

### 被挡住了捏

按F12打开浏览器开发者工具或Ctrl+U查看源码即可拿到flag

![被挡住了捏](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224152216.png)

### Dino

抓包修改数值，可使用Hackbar或BurpSuite或浏览器开发者工具-网络，大于规定值即可得到flag

![Dino-1](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224152530.png)

![Dino-2](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224152713.png)

### sqli1

使用万能密码('or 1=1#)即可登录

![sqli1-1](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224152958.png)

![sqli1-2](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224153005.png)

源码如下：

```php
<?php
    error_reporting(0);
    session_start();
    include "connect.php";

    if ($_SERVER["REQUEST_METHOD"] == "POST") {
        $username = $_POST["username"];
        $password = $_POST["password"];

        $sql = "SELECT * FROM users WHERE username = '" . $username . "' AND password = '" . $password . "'";
        $result = $conn->query($sql);

        if ($result->num_rows > 0) {
            // User is authenticated, store user information in session
            $_SESSION["username"] = $username;
            $_SESSION["password"] = $password;
            $_SESSION["loggedin"] = true;

            // Redirect to dashboard or home page
            header("Location: dashboard.php");
            exit();
        } else {
            $error = "Invalid username or password";
        }
    }
?>
```

username=admin' or '1'='1'#拼接后的SQL语句如下：

```sql
SELECT * FROM users WHERE username = 'admin' or '1'='1'#' AND password = 'xxx';
```

也可以用username=admin\&password=' or '1'='1，拼接后的SQL语句如下：

```sql
SELECT * FROM users WHERE username = 'admin' AND password = '' or '1'='1';
```

执行后都可以select到admin的信息，登录成功。

### easyHTTP

考点： HTTP协议方法以及结构

第一步：修改GET参数中npc的值为alice

![easyHTTP-1](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224153508.png)

第二步：修改GET参数中npc的值为bob

![easyHTTP-2](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224153547.png)

第三步：POST传参，使用Hackbar或其他工具POST传递指定参数和内容

![easyHTTP-3](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224153723.png)

得到信息

![easyHTTP-4](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224153745.png)

第四步：转到jack并修改HTTP请求头部

![easyHTTP-5](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224153856.png)

![easyHTTP-6](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224153906.png)

### 魔法猫咪

PHP反序列化漏洞

这里是我们可以利用的类

![魔法猫咪-1](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224154139.png)

这里告诉我们可传入进行反序列化的参数名是lawn

![魔法猫咪-2](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224154148.png)

这里理一下pop链 unserialize-> sunflower.\_\_wakeup() -> eggplant.\_\_debugInfo() -> cat.toString() ->flag

编写代码获得序列化结果

![魔法猫咪-3](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224155620.png)

构造paylod: ?lawn=O:9:"sunflower":1:{s:3:"sun";O:8:"eggplant":3:{s:3:"egg";b:1;s:5:"plant";O:3:"cat":0:{}s:6:"zombie";N;\}}\`

![魔法猫咪-4](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224155413.png)

这里附上php的魔法函数（记得保存）

#### 魔术方法

魔术方法是会在某种条件下发生自行调用的方法

| 魔术方法(magic method) | 说明                                                 |
| ------------------ | -------------------------------------------------- |
| `__construct()`    | 当对象创建（new）时会自动调用。但在 unserialize() 时是不会自动调用的。（构造函数） |
| `__destruct()`     | 当对象被销毁时会自动调用。（析构函数）                                |
| `__wakeup()`       | 使用 unserialize 反序列化时自动调用                           |
| `__sleep()`        | 使用 serialize 序列化时自动调用                              |
| `__set()`          | 在给未定义的属性赋值时自动调用                                    |
| `__get()`          | 调用未定义的属性时自动调用                                      |
| `__isset()`        | 使用 isset() 或 empty() 函数时自动调用                       |
| `__unset()`        | 使用 unset() 时自动调用                                   |
| `__call()`         | 调用一个不存在的方法时自动调用                                    |
| `__callStatic()`   | 调用一个不存在的静态方法时自动调用                                  |
| `__toString()`     | 把对象转换成字符串时自动调用                                     |
| `__invoke()`       | 当尝试把对象当方法调用时自动调用                                   |
| `__set_state()`    | 当使用 var\_export() 函数时自动调用，接受一个数组参数                 |
| `__clone()`        | 当使用 clone 复制一个对象时自动调用                              |
| `__debugInfo()`    | 使用 var\_dump() 打印对象信息时自动调用                         |

### 坤言坤语

考点：目录爆破

题目给了很明确的提示

![坤言坤语-1](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224155921.png)

爆破之后发现了 有备份压缩包

![坤言坤语-2](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224160547.png)

下载下来

![坤言坤语-3](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224160752.png)

阅读源码发现是一个简单的加密函数

![坤言坤语-4](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224160834.png)

简单编写解码函数 获得四个密文的明文

![坤言坤语-5](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224165241.png)

![坤言坤语-6](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224161128.png)

构造payload传参，进行蚁剑连接（或者手搓命令执行）

?sing=jI\&dance=Ni\&rap=TaI\&basketball=Mei

![坤言坤语-7](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224161424.png)

找到flag

![坤言坤语-8](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224161451.png)

### babybabyweb

考点：javaweb web.xml泄露

啥都不输都能登录 说明这个登录框肯定没用

![babybabyweb-1](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224161844.png)

点开登录框下的链接

发现是java后端和一个file攻击点

尝试读取javaweb的配置文件web.xml

![babybabyweb-2](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224162122.png)

读取成功

看到了源码地址 尝试继续下载

![babybabyweb-3](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224162250.png)

读取成功

![babybabyweb-4](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224162543.png)

class文件用ide反编译一下即可

![babybabyweb-5](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224162732.png)

### 新人爆照

考点：文件上传漏洞，.user.ini利用

先随便传个东西 发现疑似有过滤

![新人爆照-1](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224162915.png)

F12检查后发现是 前端验证

![新人爆照-2](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/Pasted%20image%2020231224163141.png)

![新人爆照-3](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224165839.png)

控制台直接写个同名函数给他覆盖掉

成功绕过，抓个包研究一下

![新人爆照-4](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224170225.png)

发现还有后端检测

![新人爆照-5](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224170253.png)

再尝试php,php3,php4,php5,phtml,pht等等一系列后缀后发现全部被过滤了

通过返回包的请求标头或使用浏览器插件Wappalyzer可以发现后端服务器是Nginx

尝试上传.user.ini文件恶意修改配置文件

```txt
auto_append_file=attack.jpg
```

修改一下文件类型和文件头绕过后端验证（仅需文件开头是图片头就可以，这里用GIF是为了方便输入）

![新人爆照-6](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224171025.png)

参考连接：[浅析.user.ini的利用](https://blog.csdn.net/cosmoslin/article/details/120793126)

发现上传成功

![新人爆照-7](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224171124.png)

再传一句话木马，同样加上图片头

![新人爆照-8](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224171241.png)

成功

![新人爆照-9](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224171205.png)

此时访问该文件夹的任意PHP文件，发现我们上传的图片马已经被附加到页面中了

![新人爆照-10](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224171305.png)

直接蚁剑链接拿到flag

![新人爆照-11](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224172159.png)

### sqli2

还没写好，明天再来看看吧\~

### 黑心商店

考点：任意文件读取 逻辑漏洞

查看url并尝试改变参数发现，图片数据是以base64的形式传输到前端的 尝试利用这个先读取一下index.php的内容

![黑心商店-1](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224172407.png)

发现可以读取

![黑心商店-2](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224172756.png)

![黑心商店-3](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224172811.png)

研究一下源码看看还可以读什么（或者尝试爆破） 发现两个 读取一下

![黑心商店-4](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224173002.png)

分析后发现loginServer.php没用 但是register.php 里有passcode的格式

![黑心商店-5](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224173201.png)

按照正则表达式直接构造一个 passcode=as1as1as1aa11aa11a|/|/1111 正则表达式解析网站： [https://c.runoob.com/front-end/7625/#!flags=\&re=%5E(%5Ba-z%5D%2B%5B0-5%5D)%7B3%7D(%5Cw%7B2%7D%5Cd%7B2%7D)%7B2%7D%5Ba-zA-Z%5D%2B(%5C%7C%5C%2F)%7B2%7D%5Cd%7B4%7D%24](https://c.runoob.com/front-end/7625/#!flags=\&re=%5E\(%5Ba-z%5D%2B%5B0-5%5D\)%7B3%7D\(%5Cw%7B2%7D%5Cd%7B2%7D\)%7B2%7D%5Ba-zA-Z%5D%2B\(%5C%7C%5C%2F\)%7B2%7D%5Cd%7B4%7D%24) （但还是建议大家学会人工分析）

![黑心商店-6](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/9860911ea23fb1ad62ba8fb926a53b7.png)

注册成功

![黑心商店-7](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224173440.png)

然后进行登录 发现打工时总是会出现把我们金币重制（所以脚本暴力发包法失效）

![黑心商店-8](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224173527.png)

先把服务端源码读下来

![黑心商店-9](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224173655.png)

这一块发现了逻辑漏洞没有检测传入数量的合法性直接计算，那么如果我们的数量参数为负数，那这个价格就变成负数了，下面对数据库修改我们金币的时候就会减去一个负数，使我们的金币变多

![黑心商店-10](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224173721.png)

传负数的时候发现还是有前端验证，可以先输入一个正数再抓包进行修改

![黑心商店-11](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224173909.png)

![黑心商店-12](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224174107.png)

获得金币后直接购买flag，得到答案

![黑心商店-13](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224174134.png)

![黑心商店-14](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224174201.png)

## Crypto

### 胡言乱语

简单替换密码，通过比对翻译后的文本和题目给的密文，可以得到字母的对应关系，即可解密。

这题也可以用[quipqiup](https://www.quipqiup.com/)快速解密，输入密文即可破解得到明文。

### 摩西摩西

考点：摩斯密码

观察文本内容，发现共有3种字词，分别是摩西、喂、?，并且?疑似作为分隔符，结合题目名，猜测是摩斯密码，将摩西和喂分别替换为-和.，?替换为空格，摩斯解码后即可得到明文。

下面是使用[CyberChef](https://github.com/gchq/CyberChef)的示例

![摩西摩西](https://raw.githubusercontent.com/StingerTeam/img\_bed/main/20231224185844.png)

### Vigenere

维吉尼亚密码，发现没有给密钥，搜索维吉尼亚爆破，即可找到[Vigenère Solver](https://www.guballa.de/vigenere-solver)

破解后即为flag

### rot13

简单的rot13解码，使用工具解码后得到flag内容

### easyCaeser

考点：变异凯撒

```txt
cj`dyd>3x\A`0Q`O]U^p0>Rhll|
flag{.....................}
```

根据题面，将"cj\`dy"与"flag{"对照，发现偏移量为3213...，猜测偏移的规律为321循环，编写脚本解密

```python
c = "cj`dyd>3x\A`0Q`O]U^p0>Rhll|"
add = 3
m = ""
# 解密
for i, char in enumerate(c):
    m += chr(ord(char) + add)
    add -= 1
    if add <= 0:
        add = 3
print(m)
```

### easyRSA

已知p, q, c, 编写RSA解密脚本即可获得flag

```python
from Crypto.Util.number import long_to_bytes

p = 9266056543660540596894853230433714137277477768240817161109767150943725091483376412440366423393090810696352884199521954839288680938321937402144565250668173
q = 8051467402050481462499163607796111674774708671074076046306978426538900731802961937312040570043878089847179385039113681399358308676045964255604069136971199
c = 43941854467939299468268964271726313579657450705314752718510302430415954106542679833030670731953196670055236704623370877982820274247752507416770874350886013221434598673187512882046247451530730137450366205462959748656327653512362501405361695417575283039143792891937365951751255206943780791642745314441009143924
n = p*q
phi = (p-1)*(q-1)
e = 65537
d = pow(e,-1,phi)
m = pow(c,d,n)
print(long_to_bytes(m))
```

### easyXOR

简单的异或加密（与前一字节异或得到当前字节），解密脚本如下

```python
hex = ['0x66', '0xa', '0x6b', '0xc', '0x77', '0x20', '0x48', '0x31', '0x6e', '0x0', '0x6f', '0x1b', '0x44', '0x31', '0x42', '0x27', '0x78', '0x31', '0x75', '0x34', '0x49']
hex = [int(hex_str,16) for hex_str in hex]
flag = "f"
i = 1
while i < len(hex):
    flag += chr(hex[i] ^ hex[i-1])
    i += 1
print(flag)
```

## 其他题目将在明天上传（歇会\~）
