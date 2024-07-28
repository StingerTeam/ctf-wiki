# Crypto

## 常用模块或工具

### Crypto.Util.number

#### 安装

```bash
pip install pycryptodome
```

#### 常用函数

| 函数名 | 作用 | 返回值 |
| :-: | :-: | :-: |
| `long_to_bytes(n, blocksize=0)` | 使用大端序 (Big Endian) 编码将正整数转换为字节字符串 | `bytes` |
| `bytes_to_long(s)` | 使用大端序 (Big Endian) 编码将字节字符串转换为长整数 | `long` |
| `isPrime(N)` | 测试数字 $N$ 是否为素数(先查素数表，查不到再使用 Miller-Rabin 素性测试) | `bool` |
| `getPrime(N)` | 返回一个随机的 $N$ 位(二进制)素数。 | `int` |

> 更多用法详见 [Crypto.Util package](https://pycryptodome.readthedocs.io/en/latest/src/util/util.html)

### gmpy2

#### 安装

```bash
pip install gmpy2
```

#### 常用类/函数

| 类/函数名 | 作用 | 返回值 |
| :-: | :-: | :-: |
| `mpz(n=0/s, base=0)` | 大整数类 | `mpz` |
| `powmod(x, y, m)` | 幂运算取模，即求 $x^y \space mod \space m$ | `mpz` |
| `invert(x, m)` | 求 $x$ 在模 $m$ 意义下的乘法逆元 $y$ , 即求满足 $xy \equiv 1 \pmod m$ 的 $y$ | `mpz` |
| `gcd(a, b)` | 欧几里得算法求 $a, b$ 的最大公约数 | `mpz` |
| `lcm(a, b)` | 欧几里得算法求 $a, b$ 的最小公倍数 | `mpz` |
| `gcdext(a, b)` | 扩展欧几里得算法求不定方程 $ax + by = \gcd(a, b)$ 的一组特解, 返回值依次为 $\gcd(a, b), x_0, y_0$ | `tuple[mpz, mpz, mpz]` |
| `iroot(x, n)` | 求 $x$ 的 $N$ 次方根 , 同时告知是否开尽 | `tuple[mpz, bool]` |

> 更多 gmpy2 用法详见 [gmpy2 Tutorial](https://gmpy2.readthedocs.io/en/latest/tutorial.html)

### SymPy

#### 安装

```bash
pip install sympy
```

#### 常用类/函数

##### 方程求解

| 类名 | 作用 | 使用示例 |
| :-: | :-: | :-: |
| `Symbol()` | 声明数学表达式和多项式的变量 | `x = Symbol("x")` |
| `symbols()` | 声明数学表达式和多项式的变量 | `x, y = symbols("x y")` |
| `Eq(A, B)` | 声明符号方程式 | `Eq(x**2, 1)` 代表等式 $x^2=1$ |

| 函数名 | 作用 | 使用示例 |
| :-: | :-: | :-: |
| `solve(f, *Symbols)` | 求解方程组 | `solve([Eq1, Eq2], x, y)` |

###### 使用示例

求解方程组

$$
\begin{cases}
    x + y = 2 \\
    x \times y = 1    
\end{cases}
$$

```python
from sympy.solvers import solve
from sympy import symbols, Eq

x, y = symbols("x y")
Eq1 = Eq(x + y, 2)
Eq2 = Eq(x * y, 1)

res = solve([Eq1, Eq2], x, y)

print(res)
# [(1, 1)]
```

##### 数论相关

| 类名 | 作用 | 返回值 |
| :-: | :-: | :-: |
| `crt(m, v)` | 中国剩余定理求解线性同余方程组，给定模数数组 $m$ 和余数数组 $v$ 表示方程组 $x \equiv v_i \pmod {m_i}$ , 求得模方程组的解 $x \equiv b \pmod M$ | `Tuple[int, int]` 表示 $(b, M)$ |

> 更多 SymPy 用法详见 [SymPy API Reference](https://docs.sympy.org/latest/reference/index.html)

### z3-solver

#### 安装

```bash
pip install z3-solver
```

> 注意: 包名为 `z3-solver` 而不是 `z3`

#### 常用类/函数

| 类名 | 作用 | 使用示例 |
| :-: | :-: | :-: |
| `Int(name)` / `Ints(names)` | 创建一个/多个整型变量 | `x = Int('x')` / `x, y = Int('x y')` |
| `IntVal(val)` | 创建整型常量 | `IntVal(233)` |
| `Real(name)` / `Reals(names)` | 创建一个/多个实数类型变量 | `x = Real('x')` / `x, y = Reals('x y')` |
| `RealVal(val)` | 创建实数类型常量 | `RealVal(2.33)` |
| `BitVec(name, bv)` / `BitVecs(names, bv)` | 创建一个/多个位向量 | `x = Real('x')` / `x, y = Reals('x y')` |
| `BitVecVal(val, bv)` | 创建位向量常量 | `RealVal(2.33)` |
| `Solver()` | 创建一个通用求解器，用于约束条件，进行下一步的求解 |

| 函数名 | 作用 |
| :-: | :-: |
| `solve(Equations)` | 用于处理较少的约束条件，简化 `Solver()` 的声明 |
| `add(Equation)` | 添加约束条件，通常在Solver()命令之后，添加的约束条件通常是一个逻辑等式 |
| `check()` | 判断在添加完约束条件后，来检测解的情况，有解的时候会回显 `sat`，无解的时候会回显 `unsat` |
| `model()` | 在存在解的时候，该函数会将每个限制条件所对应的解集的交集，进而得出正解 |

###### 求解器使用示例

求解约束条件

$$
\begin{cases}
    x^2 + y^2 = 100 \\
    x + y = 5
\end{cases}
$$

- 使用 `solve()` 函数求解

```python
from z3 import Reals, solve

x, y = Reals("x y")
solve(x ** 2 + y ** 2 == 100, x + y == 5)
# [y = -4.1143782776?, x = 9.1143782776?]
```

- 使用求解器求解

```python
from z3 import Reals, Solver, sat

x, y = Reals("x y")
s = Solver()
s.add(x ** 2 + y ** 2 == 100)
s.add(x + y == 5)
if s.check() == sat:
    print(s.model())
# [y = -4.1143782776?, x = 9.1143782776?]
```

> 更多 z3-solver 用法详见 [Programming Z3](https://z3prover.github.io/papers/programmingz3.html)

### libnum

#### 安装

```bash
pip install libnum
```

#### 常用函数

| 函数名 | 作用 | 返回值 |
| :-: | :-: | :-: |
| `s2n()` | 数字转字符串 | str |
| `n2s()` | 字符串转数字 | int |
| `b2s()` | 二进制转字符串 | str |
| `s2b()` | 字符串转二进制 | bytes |

> 更多 libnum 用法详见 [libnum PyPI](https://pypi.org/project/libnum/)

### os, pwn, base64, random 库

#### 安装

```bash
# os 无需安装
pip install pwntools # pwn
pip install base64 # base64
# random 无需安装
```

#### 常用函数

| 函数名 | 作用 | 返回值 |
| :-: | :-: | :-: |
| `os.urandom(size)` | 生成长度为 size 的随机字节字符串 | bytes |
| `pwm.xor(*args, cut='max')` | 按位异或 | bytes |
| `base64.b64encode(str)` | 将字符串编码为Base64编码格式 | bytes |
| `base64.b64decode(str)` | 将Base64编码格式的字符串解码 | bytes |
| `random.randint(start, stop)` | 生成随机数 | int |
| `random.randrange(start, stop+1)` | 生成随机数 | int |

### 随机数预测

#### 安装

```bash
pip install randcrack
```

#### 使用示例

```python
from randcrack import RandCrack
rc = RandCrack()
for i in range(624):
    rc.submit(randint(0, 2**32-1))
print(rc.predict_getrandbits(32))
print(randint(0, 2**32-1))
```

### Openssl 工具

#### 使用方法

```bash
openssl rsa -pubin -text -modulus -in pubkey.pem
```

### rsautl工具

#### 使用方法

```bash
rsautl -decrypt -in flag.enc -inkey pubkey.pem
```

## 解题前置数论知识

### 欧几里得定理

> 对于求解整数 $a, b$ 的最大公约数 , 我们使用欧几里得算法(辗转相除法) , 也可以使用更相减损术

#### 原理

$$
\gcd(a,b)=\gcd(b,a \bmod b)
$$

### 扩展欧几里得算法

> 扩展欧几里得算法，常用于求 $ax+by=\gcd(a,b)$ 的一组可行解

#### 求解过程

设 

$$
ax_1+by_1=\gcd(a,b) \\
bx_2+(a\bmod b)y_2=\gcd(b,a\bmod b)
$$

由欧几里得定理可知：

$$
\gcd(a,b)=\gcd(b,a\bmod b)
$$

所以 

$$
ax_1+by_1=bx_2+(a\bmod b)y_2
$$

又因为

$$
a\bmod b=a-(\lfloor\frac{a}{b}\rfloor\times b)
$$

所以 

$$ 
ax_1+by_1=bx_2+(a-(\lfloor\frac{a}{b}\rfloor\times b))y_2 \\
 
ax_1+by_1=ay_2+bx_2-\lfloor\frac{a}{b}\rfloor\times by_2=ay_2+b(x_2-\lfloor\frac{a}{b}\rfloor y_2)
$$

因为 $a=a,b=b$ ，所以 $x_1=y_2,y_1=x_2-\lfloor\frac{a}{b}\rfloor y_2$

将 $x_2,y_2$ 不断代入递归求解直至 $\gcd$（最大公约数，下同）为 $0$ 递归 $x=1,y=0$ 回去求解

### 乘法逆元

> 如果一个线性同余方程 $ax \equiv 1 \pmod b$，则 $x$ 称为 $a \bmod b$ 的逆元，记作 $a^{-1}$ 

### 费马小定理

> 若 $p$ 是素数, $a$ 与 $p$ 互质, 则有
> $$
> a ^ {p-1} \equiv 1 \pmod p
> $$

在求解与模数互质的数在模意义下的乘法逆元时, 使用该定理的变形 $a \cdot a^{p - 2} \equiv 1 \pmod p$ 可以求得 $a^{-1} \equiv a^{p - 2} \pmod p$.

### 欧拉定理

> 当 $a$ 与 $n$ 互质时, 有
> $$
> a ^ {\varphi(n)} \equiv 1 \pmod n
> $$

这个定理可以看成对费马小定理的推广

当 $n$ 为素数时，$\varphi(n) = n - 1$，代入欧拉定理可以得到费马小定理

### 威尔逊定理

> 对于素数 $p$ 有 
> $$
> (p-1)! \equiv -1 \pmod p
> $$

如果题目出现 **阶乘** 的形式，可以考虑使用威尔逊定理

### 裴蜀定理

> 设 $a,b$ 是不全为零的整数，对任意整数 $x,y$ ，满足 $\gcd(a,b)\mid ax+by$ ，且存在整数 $x,y$ , 使得 $ax+by=\gcd(a,b)$

### 中国剩余定理

> 对于如下形式的 **一元线性同余方程组** ( 其中 $m_1, m_2, \cdots, m_k$ 两两互质 ) , 我们使用中国剩余定理求解
> $$
> \begin{cases}
> x & \equiv & a_1 \pmod {m_1} \\
> x & \equiv & a_2 \pmod {m_2} \\
>   & \vdots & \\
> x & \equiv & a_k \pmod {m_k}
> \end{cases}
> $$

#### 求解过程

1. 计算所有模数的积 $M = m_1 \cdot m_2 \cdots m_k$
2. 对于第 $i$ 个方程 :
    1. 计算 $M_i=\frac{M}{m_i}$
    2. 计算 $M_i$ 在模 $m_i$ 意义下的 **乘法逆元** $M_i^{-1}$
    3. 计算 $c_i=M_iM_i^{-1}$（**不要对 $m_i$ 取模**）
3. 方程组在模 $n$ 意义下的唯一解为：$x=\sum_{i=1}^k a_ic_i \pmod n$

### 扩展中国剩余定理

> 对于如下形式的 **一元线性同余方程组** ( 其中 $m_1, m_2, \cdots, m_k$ 两两 **不一定** 互质 ) , 我们使用拓展中国剩余定理求解
> $$
> \begin{cases}
> x & \equiv & a_1 \pmod {m_1} \\
> x & \equiv & a_2 \pmod {m_2} \\
>   & \vdots & \\
> x & \equiv & a_k \pmod {m_k}
> \end{cases}
> $$

#### 求解思路

每次合并两个方程, 直到剩下一个同余方程为止

#### 求解过程

考虑其中两个同余方程组成的方程组

$$
\begin{cases}
x \equiv a_1 \pmod {m_1} \\
x \equiv a_2 \pmod {m_2}
\end{cases}
$$

化为一般方程, 得到

$$
\begin{cases}
x = a_1 + k_1 \cdot m_1 & k_1 \in \mathbb{Z} \\
x = a_2 + k_2 \cdot m_2 & k_2 \in \mathbb{Z}
\end{cases}
$$

两式相减 , 得到

$$
k_1 \cdot m_1 - k_2 \cdot m_2 = a_2 - a_1
$$

对于该式 , 由裴蜀定理可知当 $a_2-a_1$ 不能被 $\gcd(m_1,m_2)$ 整除时，无解

对于其他情况 , 可以使用拓展欧几里得算法得到一组可行解 $(p, q)$

则原两同余方程组成的同余方程组的解为 $x \equiv b \pmod M$ , 其中 $b = m_1 p + a_1 , M = \text{lcm} (m_1, m_2)$

> 除此之外, 我们也可以使用以下方法直接计算出一组可行解
> $$
> \begin{aligned}
> & k_1 \cdot m_1 - k_2 \cdot m_2 = a_2 - a_1 \\
> \Longrightarrow & k_1 \cdot m_1 = a_2 - a_1 + k_2 \cdot m_2 \\
> \Longrightarrow & k_1 \cdot m_1 \equiv a_2 - a_1 & \pmod {m_2} \\
> \stackrel{\textcircled{1}}{\Longrightarrow} & k_1 \cdot \frac{m_1}{\gcd(m_1, m_2)} \equiv \frac{a_2 - a_1}{\gcd(m_1, m_2)} & \pmod {\frac{m_2}{\gcd(m_1, m_2)}} \\
> \Longrightarrow & k_1 \equiv \frac{a_2 - a_1}{\gcd(m_1, m_2)} \cdot (\frac{m_1}{\gcd(m_1, m_2)})^{-1} & \pmod {\frac{m_2}{\gcd(m_1, m_2)}}
> \end{aligned} \\
> $$
> 
> 将上面求得的 $k_1$ 代入 $x = a_1 + k_1 \cdot m_1$ 中，就可以得到 $x$ 的值
> 
> $$
> x \equiv a_1 + \frac{a_2 - a_1}{\gcd(m_1, m_2)} \cdot (\frac{m_1}{\gcd(m_1, m_2)})^{-1} \cdot m_1 \pmod {\frac{m_1 \cdot m_2}{\gcd(m_1, m_2)}}
> $$
> 
> 则原两同余方程组成的同余方程组的解为
> 
> $$
> x \equiv a' \pmod {m'} \\
> (a' = a_1 + \frac{a_2 - a_1}{\gcd (m_1, m_2)} \cdot (\frac{m_1}{\gcd (m_1, m_2)})^{-1} \cdot m_1, m' = \frac{m_1 \cdot m_2}{\gcd (m_1, m_2)} = \text{lcm} (m_1, m_2())
> $$

继续两两合并即可得到同余方程组的解.

## RSA

### RSA的基本原理

#### RSA算法的基本流程

1. 选择两个不相等的素数 $p$ 和 $q$， 计算 $n = p \cdot q$，$n$ 称为公共模数
2. 计算 $\varphi(n) = \varphi(p \cdot q) = \varphi(p) \cdot \varphi(q) = (p-1) \cdot (q-1)$
3. 随机选择一个整数 $e$，$1 < e < \varphi(n)$，且 $e$ 与 $\varphi(n)$ 互质
4. 计算 $e$ 关于 $\varphi(n)$ 的乘法逆元 $d$，使得 $d \cdot e \equiv 1 \pmod {\varphi(n)}$
5. 公钥为 $(n, e)$ , 私钥为 $(n, d)$

#### RSA的加密和解密过程

对于 $0 \le m < n$

1. 加密：$c = m^e \pmod n$
2. 解密：$m = c^d \pmod n$

### 常见的RSA题型整理

#### 已知 $n, d, c$

##### 特征

已知 $n, d, c$

##### 解法

按照定义求解

$$m \equiv c^d \pmod n$$

##### 代码实现

```python
from gmpy2 import powmod
from Crypto.Util.number import long_to_bytes

n = XXX
d = XXX
c = XXX

m = powmod(c, d, n)
print(long_to_bytes(m))
```

#### 已知 $n, e, c, p, q$

##### 特征

已知 $n, e, c, p, q$

##### 解法

1. $n = pq$
2. $\varphi(n) = \varphi(pq) = \varphi(p) \varphi(q) = (p - 1)(q - 1)$
3. $d \equiv e^{-1} \pmod n$
4. $m \equiv c^d \pmod n$

##### 代码实现

```python
from gmpy2 import powmod, invert
from Crypto.Util.number import long_to_bytes

p = XXX
q = XXX
n = XXX
e = XXX
c = XXX

phi = (p - 1) * (q - 1)
d = invert(e, phi)
m = powmod(c, d, n)
print(long_to_bytes(m))
```

#### lcm 泄漏

##### 特征

已知 $n, e, c, \text{lcm}(p - 1, q - 1)$, 求解 $m$

##### 解法

在 RSA 中 $\varphi(n) = \varphi(pq) = \varphi(p) \cdot \varphi(q) = (p - 1)(q - 1) =  \text{lcm}(p - 1, q - 1)$

将 $\text{lcm}(p - 1, q - 1)$ 视作 $\varphi(n)$ 按定义求解即可

##### 代码实现

```python
from gmpy2 import powmod, invert
from Crypto.Util.number import *

n = XXX
e = XXX
lcm = XXX
c = XXX

d = invert(e, lcm)
m = powmod(c, d, n)
print(long_to_bytes(m))
```

#### 大数分解 (已知 $n, e, c$, 且 $p$ 和 $q$ 接近或 $n$ 比较小)

主要有 FactorDB分解、 yafu分解 和 费马分解

##### FactorDB (需联网)

```python
from gmpy2 import powmod, invert
from Crypto.Util.number import long_to_bytes
from factordb.factordb import FactorDB

n = XXX
e = XXX
c = XXX

f = FactorDB(n)
f.connect()
p, q = f.get_factor_list()

phi = (p - 1) * (q - 1)
d = invert(e, phi)
m = powmod(c, d, n)
print(long_to_bytes(m))
```

##### yafu 分解 (无脑分解，比较看人品)

```bash
yafu-x64.exe "factor(XXX)"
```

```python
from gmpy2 import powmod, invert
from Crypto.Util.number import long_to_bytes

p = XXX # 分解得到的p
q = XXX # 分解得到的q
n = XXX
e = XXX
c = XXX

phi = (p - 1) * (q - 1)
d = invert(e, phi)
m = powmod(c, d, n)
print(long_to_bytes(m))
```

##### 费马分解 (当 $p$ 和 $q$ 接近时)

```python
from gmpy2 import invert, powmod
from Crypto.Util.number import long_to_bytes

def isqrt(n):
  x = n
  y = (x + n // x) // 2
  while y < x:
    x = y
    y = (x + n // x) // 2
  return x

def fermat(n):
    a = isqrt(n)
    b2 = a*a - n
    b = isqrt(n)
    count = 0
    while b * b != b2:
        a = a + 1
        b2 = a * a - n
        b = isqrt(b2)
        count += 1
    p = a + b
    q = a - b
    assert n == p * q
    return p, q

n = XXX
e = XXX
c = XXX

pq = fermat(n)
p = pq[0]
q = pq[1]
phi_n = (p - 1) * (q - 1)
d = invert(e, phi_n)
m = pow(c, d, n)
print(long_to_bytes(m))
```

#### 共模攻击

##### 特征

已知 $n, e_1, e_2, c_1, c_2$ 且 $e_1, e_2$ 互质

##### 原理

由两指数互质 $gcd(e_1, e_2) = 1 \Rightarrow e_1 \times s_1 + e_2 \times s_2 = 1$ ，使用扩展欧几里得算法可以求得 $s_1, s_2$

因为

$$
\begin{cases}
    c_1 \equiv m^{e_1} \pmod n \\
    c_1 \equiv m^{e_2} \pmod n
\end{cases}
$$

所以有

$$
    c_1^{d_1} \times c_2^{d_2} \equiv (m^{e_1})^{s_1} \times (m^{e_2})^{s_2} \equiv m^{e_1 \times s_1 + e_2 \times s_2} \pmod n
$$

又

$$
    e_1 \times s_1 + e_2 \times s_2 = 1
$$

所以

$$
    c_1^{s_1} \times c_2^{s_2} \equiv m \pmod n
$$

$s_1,s_2$一定是一个正数一个负数，因为 $e_1, e_2$ 中一定有一个大于 $1$ , 所以不能同时为负数也不能同时为正数

##### 代码实现

```python
from gmpy2 import gcdext, invert, powmod
from Crypto.Util.number import long_to_bytes
s = gcdext(e1, e2)
s1 = s[1]
s2 = s[2]
if s1 < 0:
    s1 = - s1
    c1 = invert(c1, n)
elif s2 < 0:
    s2 = - s2
    c2 = invert(c2, n)
m = (powmod(c1,s1,n) * powmod(c2 ,s2 ,n)) % n
print(long_to_bytes(m))
```

脚本中要注意的是, 在密码学中, $x^{-1}$实际上是$x$的模逆元, 即满足$x \cdot x^{-1} \equiv 1 \pmod n$的数, 而不是$x$的倒数, 但是在计算中和倒数的计算方法是一样的, 所以在计算时, 如果$s_1$或$s_2$为负数, 需要将$c_1$或$c_2$取逆

#### 共模攻击的扩展

##### 特征

已知 $n, e_1, e_2, c_1, c_2$, 且 $e_1, e_2$ **不** 互质

##### 原理

由共模攻击的公式

$$
m \equiv c_1^{s_1} \cdot c_2^{s_2} \pmod n
$$

当 $e_1, e_2$ 不互质, 但是我们可以构造出满足条件的 $e_1, e_2$

$$
\begin{cases}
c_1 \equiv m^{e_1} \pmod n \\
c_2 \equiv m^{e_2} \pmod n
\end{cases}
\Rightarrow
\begin{cases}
c_1 \equiv {m^{gcd(e_1, e_2)}}^{\frac{e_1}{gcd(e_1, e_2)}} \pmod n \\
c_2 \equiv {m^{gcd(e_1, e_2)}}^{\frac{e_2}{gcd(e_1, e_2)}} \pmod n
\end{cases}
$$

很明显 $\frac{e_1}{gcd(e1, e2)}$ 和 $\frac{e_2}{gcd(e1, e2)}$ 互质, 所以可以使用共模攻击的方法解密

原来的 $m$ 对应现在的 $m^{gcd(e_1, e_2)}$

脚本和共模攻击的脚本基本一致，这里不再赘述

#### 共享素数

##### 特征

已知 $e, c, n_1, n_2, ...$ , 其中 $n_1 = p \cdot q_1 , n_2 = p \cdot q_2 , ...$

##### 解法

可以尝试使用欧几里德算法求得 $p = \gcd(n_1, n_2, ...)$ , 于是 $q_i = \frac{n_i}{p}$ ,之后按定义正常求解即可

##### 代码实现

```python
from gmpy2 import powmod, invert, gcd
from Crypto.Util.number import long_to_bytes

n1 = XXX
n2 = XXX
e = XXX
c = XXX

p = gcd(n1, n2)
q1 = n1 // p
q2 = n2 // p
phi1 = (p - 1) * (q1 - 1)
phi2 = (p - 1) * (q2 - 1)
d1 = invert(e, phi1)
d2 = invert(e, phi2)
m = powmod(c, d1, n1)
m = powmod(m, d2, n2)
```

#### 低加密指数攻击

##### 特征

已知 $n, e, c$，且 $e$ 很小, 可以使用低加密指数攻击

##### 解法

$$
\begin{aligned}
& c \equiv m^e \pmod n \\
\Rightarrow & c = m^e + kn \\
\Rightarrow & m = \sqrt[e]{c - kn}
\end{aligned}
$$

##### 代码实现

```python
from gmpy2 import iroot, powmod, invert
from Crypto.Util.number import long_to_bytes

k = 0
while True:
    mm = c + n * k
    m, flag = iroot(mm, e)
    if flag:
        break

print(long_to_bytes(m))
```

#### dp、dq 泄漏

##### 特征

已知 $dp, dq, p, q, c$

##### dp、dq 的定义

在 RSA 中常使用 $dp, dq, coefficient$ 加速RSA的运算

$$
d_p = d \pmod {p-1} \\
d_q = d \pmod {q-1} \\
coefficient = (q - 1) \mod p
$$

其性质包括

1. $d_p = d - k \cdot (p - 1)$
2. $d_p < (p - 1) < p$

##### 解法

对于公式 $m \equiv c^{d} \pmod{n}$

我们可以将其化为一般等式 $$m = c^d + k_0 n, k_0 \in \mathbb{Z}$$

又 $$n = p ∗ q$$

所以可以进一步写成 $$m = c^d + k_0 n = c^d + k_0 pq, k_0 \in \mathbb{Z}$$

对等式两边分别同时对 $p, q$ 取模 , 消去 $kpq$ , 即可得到

$$
\begin{equation}
\begin{cases}
m_1 \equiv c^d \pmod p \\
m_2 \equiv c^d \pmod q
\end{cases}
\end{equation}
$$

由于

$$
\begin{cases}
d_p \equiv d \pmod {p-1} \\ 
d_q \equiv d \pmod {q-1}
\end{cases}
$$

将其化为等式得到

$$
\begin{cases}
d = k_1 (p - 1) + d_p & , k_1 \in \mathbb{Z} \\ 
d = k_2 (q - 1) + d_q & , k_2 \in \mathbb{Z}
\end{cases}
$$

代入 $(1)$ 得到

$$
\begin{cases}
m_1 \equiv c^{k_1 (p - 1) + d_p} \pmod p & , k_1 \in \mathbb{Z}\\
m_2 \equiv c^{k_2 (q - 1) + d_q} \pmod q & , k_2 \in \mathbb{Z}
\end{cases}
$$

又由 **费马小定理** 可知

$$
\begin{cases}
c^{k_1 (p - 1)} \equiv (c^{k_1})^{(p - 1)} \equiv 1 \pmod p & , k_1 \in \mathbb{Z}\\
c^{k_2 (q - 1)} \equiv (c^{k_2})^{(q - 1)} \equiv 1 \pmod q & , k_2 \in \mathbb{Z}
\end{cases}
$$

代入后得到

$$
\begin{equation}
\begin{cases}
m_1 \equiv c^{d_p} \pmod p \\
m_2 \equiv c^{d_q} \pmod q
\end{cases}
\end{equation}
$$

由 $(1)$ 中两式相减可得 $kp \equiv m_2-m_1 \pmod q$ , $k \in \mathbb{Z} $。

又因为 $p, q$ 互素，所以 $k \equiv (m_2-m_1) p^{-1} \pmod q$ ，其中 $p^{-1}$ 是 $p$ 关于 $q$ 的乘法逆元。

将这个 $k$ 的表达式代入 $m_1 \equiv c^d \pmod p$，得到 $c^d = m_1 + [(m_2-m_1) p^{-1} \mod q] \times p $ , 其中的 $m_1$ 和 $m_2$ ​可以通过 $(2)$ 式计算得到

#### 代码实现

```python
from gmpy2 import invert, powmod
from Crypto.Util.number import long_to_bytes

p = XXX
q = XXX
dp = XXX
dq = XXX
c = XXX

m1 = powmod(c, dp, p)
m2 = powmod(c, dq, q)
inv_p = invert(p, q)
m = m1 + ((m2 - m1) % q) * inv_p * p
print(long_to_bytes(m))
```

#### $e$ 和 $\varphi(n)$ 不互素

##### 特征

已知 $p, q, e, c$ 但是 $e$ 与 $\varphi(n) = (p - 1)(q - 1)$ 不互素

##### 情况一

$e$ 和 $p - 1$ 或者 $q - 1$ 互素

###### 解法

假设 $e$ 与 $p - 1$互素

构造 $ed \equiv 1 \pmod {p - 1}$

使用 $m \equiv c^d \pmod p$ 求解

###### 代码实现

```python
from gmpy2 import invert, powmod

p = XXX
q = XXX
e = XXX
c = XXX

d = invert(e, q - 1)
m = powmod(c, d, q)
```

##### 情况二

###### 解法

$e$ 和 $\varphi(n)$ 不互素，意味着他们存在不为 $1$ 的最大公约数, 记 $t = \gcd(e, \varphi(n))$

当 $t$ 不大时, 
考虑将该式化为指数与模数互质的形式，将指数中与模数的最大公因数提出来:

记 $e' = \frac{e}{t}$ ，则有 $d' = td_i$

$$
\begin{aligned}
    & e \times d \equiv 1 \pmod {\varphi(n)} & \Rightarrow & e' \times {d'} \equiv 1 \pmod {\varphi(n)} \\
    & (m ^ t) ^ {e'} \equiv c \pmod n & \Rightarrow & m ^ t \equiv c ^ {d'} \pmod n \cdots \textcircled{1} \\
\end{aligned}
$$

将 $\textcircled{1}$ 式转化为以 $p, q$ 为模的同余方程组, 得到

$$
\begin{cases}
m ^ t \equiv c ^ {d'} \pmod p \\
m ^ t \equiv c ^ {d'} \pmod q
\end{cases}
$$

用中国剩余定理求解后开平方得到 $m$

###### 代码实现

```python
from Crypto.Util.number import long_to_bytes
from sympy import gcd, mod_inverse, root
from sympy.ntheory.modular import crt

e = XXX
p = XXX
q = XXX
c = XXX

phi = (p - 1) * (q - 1)
n = p * q
t = gcd(e, phi)
d_ = mod_inverse(e // t, phi)
m = root(crt([p, q], [pow(c, d_, p), pow(c, d_, q)])[0], t)

print(long_to_bytes(m))
```

##### 使用 AMM 算法求解

```python
from Crypto.Util.number import *
import gmpy2
import time
import random
from tqdm import tqdm
e = 1531793
p = XXX
q = XXX
n = p * q
c = XXX

def AMM(o, r, q):
    start = time.time()

    g = GF(q)
    o = g(o)
    p = g(random.randint(1, q))
    while p ^ ((q-1) // r) == 1:
        p = g(random.randint(1, q))
    print('[+] Find p:{}'.format(p))
    t = 0
    s = q - 1
    while s % r == 0:
        t += 1
        s = s // r
    print('[+] Find s:{}, t:{}'.format(s, t))
    k = 1
    while (k * s + 1) % r != 0:
        k += 1
    alp = (k * s + 1) // r
    print('[+] Find alp:{}'.format(alp))
    a = p ^ (r**(t-1) * s)
    b = o ^ (r*alp - 1)
    c = p ^ s
    h = 1
    for i in range(1, t):
        d = b ^ (r^(t-1-i))
        if d == 1:
            j = 0
        else:
            print('[+] Calculating DLP...')
            j = - discrete_log(d, a)
            print('[+] Finish DLP...')
        b = b * (c^r)^j
        h = h * c^j
        c = c^r
    result = o^alp * h
    end = time.time()
    print("Finished in {} seconds.".format(end - start))
    print('Find one solution: {}'.format(result))
    return result

def onemod(p,r): 
    t=p-2 
    while pow(t,(p-1) // r,p)==1: 
        t -= 1 
    return pow(t,(p-1) // r,p) 

def solution(p,root,e):  
    g = onemod(p,e) 
    may = set() 
    for i in range(e): 
        may.add(root * pow(g,i,p)%p) 
    return may
def union(x1, x2):
    a1, m1 = x1
    a2, m2 = x2
    d = gmpy2.gcd(m1, m2)
    assert (a2 - a1) % d == 0
    p1,p2 = m1 // d,m2 // d
    _,l1,l2 = gmpy2.gcdext(p1,p2)
    k = -((a1 - a2) // d) * l1
    lcm = gmpy2.lcm(m1,m2)
    ans = (a1 + k * m1) % lcm
    return ans,lcm


def excrt(ai,mi):
    tmp = zip(ai,mi)
    return reduce(union, tmp)

cp = c % p
mp = AMM(cp,e,p)

mps = solution(p,mp,e)
for mpp in tqdm(mps):
     ai = [int(mpp)]]
     mi = [p]
     m = CRT_list(ai,mi)
     flag = long_to_bytes(m)
     if b'flag' in flag:
         print(flag)
         exit(0)
```

#### 费马小定理

通过费马小定理定理

$$
a^{p - 1} \equiv 1 \pmod p
$$ 

可以得到 

$$
a^{p - 1} = kp + 1, k \in \mathbb{Z}
$$

从而实现对某些式子的降幂 , 用于简化同余方程

#### 中国剩余定理

##### 特征

已知 $e$ 和多组 $n_i$ 和 $c_i$ , 其中 $n_i$ 两两互素

##### 解法

本质上是解形式如下的一个一元线性同余方程组

$$
\begin{cases}
m^e & \equiv & c_1 \pmod {n_1} \\
m^e & \equiv & c_2 \pmod {n_2} \\
    & \vdots & \\
m^e & \equiv & c_n \pmod {n_n}
\end{cases}
$$

可以使用中国剩余定理进行求解

##### 代码实现(不依赖 SymPy 库)

```python
from gmpy2 import invert, powmod, iroot
from Crypto.Util.number import long_to_bytes

e = n # 一般e和方程的数量是一样的
n_list = [n1, n2, n3, ...]
c_list = [c1, c2, c3, ...]

M = 1
for n in n_list:
    M *= n

m_e = 0
for i in range(len(n_list)):
    M_i = M // n_list[i]
    y_i = invert(M_i, n_list[i])
    m_e += c_list[i] * y_i * M_i

m = iroot(m_e, e)[0]
print(long_to_bytes(m))
```

#### 扩展中国剩余定理

##### 特征

已知 $e$ 和多组 $n_i$ 和 $c_i$ , 其中 $n_i$ 两两 **不一定** 互素

##### 解法

脚本编写和中国剩余定理类似，这里介绍使用 SymPy 中的 `crt()` 函数的写法

##### 代码实现

```python
from sympy.ntheory.modular import crt
from sympy import root
from Crypto.Util.number import long_to_bytes

e = n # 一般e和方程的数量是一样的
n_list = [n1, n2, n3, ...]
c_list = [c1, c2, c3, ...]

m = root(crt(n_list, c_list)[0], e)
print(long_to_bytes(m))
```

#### 维纳攻击

##### 特征

$e$过大或$e$过小

##### 解法

模数 $N \equiv p \cdot q$, 其中 $q < p < 2q$
若$d < \frac{1}{3}N^{\frac{1}{4}}$时, 给定公钥$(N, e)$, 且 $e \cdot d \equiv 1 \pmod {\lambda(N)}$

$$
\lambda(N) = lcm(p-1, q-1) = \frac{(p-1) \cdot (q-1)}{gcd(p-1, q-1)}
$$

##### 代码实现

```python
from gmpy2 import powmod, isqrt
from Crypto.Util.number import long_to_bytes

def continuedFra(x, y):
    cf = []
    while y:
        cf.append(x // y)
        x, y = y, x % y
    return cf
def gradualFra(cf):
    numerator = 0
    denominator = 1
    for x in cf[::-1]:
        numerator, denominator = denominator, x * denominator + numerator
    return numerator, denominator
def solve_pq(a, b, c):
    par = isqrt(b * b - 4 * a * c)
    return (-b + par) // (2 * a), (-b - par) // (2 * a)
def getGradualFra(cf):
    gf = []
    for i in range(1, len(cf) + 1):
        gf.append(gradualFra(cf[:i]))
    return gf


def wienerAttack(e, n):
    cf = continuedFra(e, n)
    gf = getGradualFra(cf)
    for d, k in gf:
        if k == 0: continue
        if (e * d - 1) % k != 0:
            continue
        phi = (e * d - 1) // k
        p, q = solve_pq(1, n - phi + 1, n)
        if p * q == n:
            return d

n = XXX
e = XXX
c = XXX

d = wienerAttack(e, n)
m = powmod(c, d, n)
print(long_to_bytes(m))
```

#### Rabin 密码体制

##### 条件

已知 $p, q, c$, 其中 $p$ 和 $q$ 都是大素数，且 $p \equiv q \equiv 3 \pmod 4$ **(满足二次剩余的特殊情况)**

模数 $n = pq$

公钥 $(2, n)$

私钥 $(p, q)$

记明文为 $m$ , 密文为 $c$

##### 加密过程

$$c \equiv m^2 \pmod n$$

##### 解密过程

1. 根据二次剩余的特殊情况, 得知方程有两组解

$$
\begin{cases}
m_p \equiv c^{\frac{p + 1}{4}} \pmod p \\
m_q \equiv c^{\frac{q + 1}{4}} \pmod q
\end{cases}
$$

2. 使用扩展欧几里得算法求得不定方程 $p \cdot y_p + q \cdot y_q = 1$ 的一组特解 $y_p, y_q$
   
3. 依次求得四个可能的根

$$
\begin{cases}
m_1 = (y_p \cdot p \cdot m_q + y_q \cdot q \cdot m_p) \mod p \\
m_2 = n - m_1 \\
m_3 = (y_p \cdot p \cdot m_q - y_q \cdot q \cdot m_p) \mod p \\
m_4 = n - m_3 \\
\end{cases}
$$

最后根据验证文本来获取真正的原文

##### 代码实现

```python
from gmpy2 import powmod, gcdext
from Crypto.Util.number import long_to_bytes

p = XXX    
q = XXX    
c = XXX

n = p * q
mp = powmod(c, (p + 1) // 4, p)
mq = powmod(c, (q + 1) // 4, q)
k, yp, yq = gcdext(p, q)
assert k == 1 # 如果k不等于1，说明p和q不互质, 这时可以考虑共模攻击扩展类似的方法
m1 = (yp * p * mq + yq * q * mp) % n
m2 = n - m1
m3 = (yp * p * mq - yq * q * mp) % n
m4 = n - m3
print(long_to_bytes(m1))
print(long_to_bytes(m2))
print(long_to_bytes(m3))
print(long_to_bytes(m4))
```

#### Schmidt-Samoa 密码系统

##### 条件

$p, q$ 为大质数，$N = p ^ 2 * q$

公钥 $N$

私钥 $d = N^{-1} \pmod {\varphi(p \times q)}$

##### 加密过程

$$c \equiv m ^ N \pmod N$$

##### 解密过程

$$m \equiv c ^ d \pmod {p \times q}$$

具体的代码实现可以仿照RSA的解法

#### 广播攻击

##### 特征

已知 $e, c_1, c_2, \cdots, c_k, n_1, n_2, \cdots, n_k$

##### 解法

因为 $n_i$ 是不同的，所以 $n_i$ 之间很有可能有公约数，我们可以通过求解 $n_i$ 的公约数来得到 $p, q$，然后再按照定义解密

##### 代码实现

```python
from gmpy2 import gcd, invert
from Crypto.Util.number import long_to_bytes

e = 65537
n = [int(line.replace("n = ", "")) for line in open('n_output.txt').readlines()] # n1, n2, n3, ...
c = [int(line.replace("c = ", "")) for line in open('c_output.txt').readlines()] # c1, c2, c3, ...

for i in range(len(n)):
    for j in range(len(n)):
        if i != j:
            p = gcd(n[i], n[j])
            if p != 1:
                print(i, j, p)

p_17 = XXX # 在上面的输出结果中得到的
q_17 = n[17] // p_17

phi_n = (p_17 - 1) * (q_17 - 1)
d = invert(e, phi_n)

m = pow(c[17], d, n[17])
print(long_to_bytes(m))
```

#### SageMath多项式环相关的解法

[在线运行SageMath脚本](https://sagecell.sagemath.org/)

##### m的高位泄漏 (`(m >> number) << number`)

求解方程组

$$
\begin{cases}
m = m_{high} + m_{low} \\
m ^ e - c = 0 \pmod n
\end{cases}
$$

```python
from Crypto.Util.number import long_to_bytes
from sage.all import *

def high_m_solve(high_m, n, e, c):
    R = PolynomialRing(Zmod(n), implementation='NTL', names=('x',))
    (x,) = R._first_ngens(1)
    m = high_m + x
    M = m((m ** e - c).small_roots()[0])
    print(long_to_bytes(int(M)))

e = 3
high_m = XXX
n = XXX
c = XXX

high_m_solve(high_m, n, e, c)
```

##### m的低位泄漏 (`m % number`)

求解方程组

$$
\begin{cases}
m_{low} = m \pmod {number} \Rightarrow m = m_{low} + k * number \\
m ^ e - c = 0 \pmod n
\end{cases}
$$

```python
from Crypto.Util.number import long_to_bytes
from sage.all import *

def low_m_solve(low_m, n, e, c, r):
    R = PolynomialRing(Zmod(n), implementation='NTL', names=('k',))
    (k,) = R._first_ngens(1)
    m = low_m + k * r
    M = m((m ** e - c).small_roots()[0])
    print(long_to_bytes(int(M)))

e = 3
low_m = XXX
n = XXX
c = XXX
r = XXX # number

low_m_solve(low_m, n, e, c, r)
```

##### p的高位泄漏 (`(p >> number) << number`)

Coppersmith 可以解决多项式在模 $n$ 的某个因数下的根。我们设 $p = pHigh + x$，然后拿去求解方程

$$
    p = 0 \pmod {\text{sth divides n}}
$$

得到 $p$ 之后即可推出私钥。

```python
from sage.all import *
def high_p_solve(high_p, n, r):
    R = PolynomialRing(Zmod(n), implementation='NTL', names=('x',))
    (x,) = R._first_ngens(1)
    p = high_p + x
    x0 = p.small_roots(X = 2 ** r, beta = 0.1)[0]

    P = int(p(x0))
    Q = n // P
    print(P)
    print(Q)
    assert n == P*Q

p4 = XXX
n = XXX
c = XXX
r = XXX

high_p_solve(p4, n, r)
```

##### d的低位泄漏 (`d >> number`)

既然已知 $d$ 的低位，也就是已知 $d$ 在模 $2 ^ {512}$ 意义下的值，又有 $e=3$，我们考虑等式

$$
\begin{aligned}
    & ed \equiv 1 \pmod {(p - 1)(q - 1)} \\
    & 3d  = 1 + k \cdot (p - 1)(q - 1) & where \space {k < 3}
\end{aligned}
$$

两边对 $2 ^ {512}$ 取模，有

$$
    3\cdot dLow \equiv 1 + k\cdot (n - p - q + 1) \pmod{2 ^ {512}}
$$


以 $\frac{n}{p}$ 代替 $q$，使上面的方程成为单变量的：

$$
    3\cdot dLow \cdot p \equiv p + k\cdot (np - p^2 - n + p) \pmod {2^{512}}
$$

这个方程是模意义下的一元二次方程，是可解的。解出来之后得到了 $p$ 的低位，通过与前一题型类似的方式可以得到 $p, q$.

```python
from sage.all import *
from Crypto.Util.number import long_to_bytes, inverse
from gmpy2 import powmod

def getFullP(low_p, n):
    R = PolynomialRing(Zmod(n), implementation='NTL', names=('x',))
    (x,) = R._first_ngens(1)
    p = x * 2 ** 512 + low_p
    root = (p - n).monic().small_roots(X=2 ** 128, beta=0.4)
    if root:
        return p(root[0])
    return None

def low_d_solve(low_d, n, c, e):
    maybe_p = []
    for k in range(1, 4):
        p = var('p')
        p0 = solve_mod([e * p * low_d == p + k * (n * p - p ** 2 - n + p)], 2 ** 512)
        maybe_p += [int(x[0]) for x in p0]
    for x in maybe_p:
        P = getFullP(x, n)
        if P:
            break
    P = int(P)
    Q = n // P
    assert P * Q == n
    print(f"P={P}")
    print(f"Q={Q}")

e = 3
n = XXX
c = XXX
low_d = XXX

low_d_solve(low_d, n, c, e)
```

## LCG 线性同余

### 定义

$$
X_{n+1} = (aX_n + b) \mod m \\
$$

其中 , $X_n$ 代表第 $n$ 个生成的随机数, $X_0$ 被称为 **种子 (Seed)** , $a$ 被称为 **乘数** , $b$ 被称为 **增量** , $a, b$ 均与 $m$ 互素.

### 常见题型

#### 由 $X_{n + 1}$ 反推 $X_n$

$$
X_n = ((X_{n+1} - b) \cdot a^{-1}) \mod m
$$

其中 $a^{-1}$ 是 $a$ 关于 $m$ 的乘法逆元，即$a \cdot a^{-1} \equiv 1 \pmod m$

#### 由 $X_{n - 1}$ , $X_n$ , $X_{n + 1}$  推导参数 $a$

由定义式得到

$$
\begin{cases}
X_{n+1} & = (aX_n & + & b) \mod m \\
X_{n} & = (aX_{n-1} & + & b) \mod m \\
\end{cases}
$$

将两式相减, 得到

$$
X_{n+1} - X_{n}  = a (X_n - X_{n-1}) \mod m
$$

等式两边同乘 $(X_n - X_{n - 1})$ 关于 $m$ 的乘法逆元, 得到

$$
a = (X_{n+1} - X_n) \cdot (X_n - X_{n-1})^{-1} \mod m
$$

#### 由 $X_n$ , $X_{n + 1}$  推导参数 $b$

$$
X_{n+1} = (aX_n + b) \mod m \Rightarrow b = (X_{n+1} - aX_n) \mod m
$$

代码实现较为简单这里不再赘述

## 梅森旋转算法

‌梅森旋转算法（Mersenne Twister）是一种用于产生伪随机数序列的算法，由‌松本真和‌西村拓士于1997年开发。该算法以梅森素数命名，是一种32位或64位的伪随机数生成器。‌

梅森旋转算法的特点包括：

* 周期长：梅森旋转算法的周期是 $2^{19937}-1$，这是一个非常大的数，即使用十进制表示也超过了 $6000$ 位。
* 多维均匀分布：在 $1 \leq k \leq 623$ 的维度之间都可以均等分布。‌
* 随机性强：连续的随机数之间的相关关系较小，产生的随机数质量高。‌

梅森旋转算法被广泛应用于各种软件包中，包括‌ Ruby 、‌ Python 、‌ R 等。在‌ C++11 及更高版本中，也可以使用这种算法。此外，许多数值计算库如‌ Boost C++ 、‌ Glib 和‌ NAG 数值库等都提供了该算法的实现。

与其他算法相比，梅森旋转算法的优点在于其周期长且随机性强。例如，与线性同余法相比，梅森旋转算法的周期更长，且产生的随机数质量更高。线性同余法的周期较短，且容易产生重复的随机数序列。

以下是梅森旋转算法进行明文加密的 Python 实现参考:

```python
def convert(m):
    m = m ^ m >> 13
    m = m ^ m << 9 & 2029229568
    m = m ^ m << 17 & 2245263360
    m = m ^ m >> 19
    return m
def transform(message):
    cipher = b''
    for i in range(len(message) // 4):
        block = message[i * 4 : i * 4 + 4]
        block = number.bytes_to_long(block)
        block = convert(block)
        block = number.long_to_bytes(block, 4)
        cipher=cipher+block
    return cipher
```

由于梅森旋转算法具有周期性, 为了求解此类加密算法的明文, 我们可以将密文不断的加密，找到周期中密文的前一位即为明文, 代码实现如下:

```python
def decode(cipher):
	temp = cipher
	while True:
		message = temp
		temp = transform(temp)
		if temp == cipher:
			return message
```

## AES

AES（高级加密标准）是一种对称加密算法，广泛用于数据加密。AES能够通过不同的模式工作，其中最常见的两种模式是 ECB（电子密码本模式）和 CBC（密码块链接模式）。这两种模式在加密数据时的处理方式有所不同。

### ECB（电子密码本模式）

ECB 模式将明文分成固定大小的块，每个块独立加密。这个模式的主要优点是简单，易于实现。然而，由于每个明文块在加密时都使用相同的密钥，因此相同的明文块会产生相同的密文块。这使得 ECB 模式在某些情况下容易受到模式重放攻击和频率分析攻击。

**优点**：
- 实现简单，适合并行处理。

**缺点**：
- 相同的明文块会产生相同的密文块，容易受到攻击。

### CBC（密码块链接模式）

CBC 模式在加密时将每个明文块与前一个密文块进行异或操作后再进行加密。第一个明文块与一个随机的初始化向量（IV）进行异或。由于每个明文块的加密依赖于前一个密文块，因此即使明文中有相同的块，它们也会产生不同的密文。

**优点**：
- 提高了安全性，相同的明文生成不同的密文。

**缺点**：
- 不能并行处理，速度较慢。

### Python中的Crypto.Cipher进行AES加密解密

我们可以使用 `Crypto.Cipher` 库实现 ECB 和 CBC 模式加密解密。

#### 示例代码

```python
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

# 定义密钥和数据
key = get_random_bytes(16)  # AES 128位密钥
data = b'This is a secret message.'

# ECB模式加密
cipher_ecb = AES.new(key, AES.MODE_ECB)
ciphertext_ecb = cipher_ecb.encrypt(pad(data, AES.block_size))
print(f'ECB Mode Ciphertext: {ciphertext_ecb.hex()}')

# ECB模式解密
decrypted_ecb = unpad(cipher_ecb.decrypt(ciphertext_ecb), AES.block_size)
print(f'Decrypted Data (ECB): {decrypted_ecb.decode()}')

# CBC模式加密
iv = get_random_bytes(16)  # 生成初始化向量
cipher_cbc = AES.new(key, AES.MODE_CBC, iv)
ciphertext_cbc = cipher_cbc.encrypt(pad(data, AES.block_size))
print(f'CBC Mode Ciphertext: {ciphertext_cbc.hex()}')

# CBC模式解密
cipher_cbc_decrypt = AES.new(key, AES.MODE_CBC, iv)
decrypted_cbc = unpad(cipher_cbc_decrypt.decrypt(ciphertext_cbc), AES.block_size)
print(f'Decrypted Data (CBC): {decrypted_cbc.decode()}')
```

## Shamir 门限方案

Shamir 门限方案是由以色列密码学家阿迪·沙米尔（Adi Shamir）在 1979 年提出的一种秘密共享方法。该方案使得一个秘密可以被分割成若干份（或称为“份额”），并且只有在收集到一定数量的份额（即门限）后，才能恢复出原始秘密。该方案在安全性、灵活性和可靠性方面具有广泛的应用，尤其在分布式系统和密钥管理中。

#### 方案原理

Shamir 门限方案的核心思想是使用多项式插值。具体步骤如下：

1. **选择参数**：
   - 设定一个秘密 \( S \)，并选择一个门限 \( k \)（需要的最小份额数量）和总份额数量 \( n \)（将生成的份额数量）。条件是 \( k \leq n \)。

2. **构造多项式**：
   - 构造一个 \( k-1 \) 次的随机多项式 \( f(x) \)，使得 \( f(0) = S \)（即常数项为秘密）。多项式的形式为：
   \[
   f(x) = a_0 + a_1 x + a_2 x^2 + \ldots + a_{k-1} x^{k-1}
   \]
   其中 \( a_0 = S \) 是秘密，\( a_1, a_2, \ldots, a_{k-1} \) 是随机选择的系数。

3. **生成份额**：
   - 计算 \( n \) 个份额 \( (i, f(i)) \) 对于 \( i = 1, 2, \ldots, n \)。每个用户获得一个独特的份额。

4. **恢复秘密**：
   - 只有当至少 \( k \) 个用户联合使用他们的份额时，才能通过拉格朗日插值法恢复原始秘密：
   \[
   S = f(0) = \sum_{j=1}^{k} f(j) \cdot L_j(0)
   \]
   其中 \( L_j(x) \) 是拉格朗日基函数，表示为：
   \[
   L_j(x) = \prod_{\substack{1 \leq m \leq n \\ m \neq j}} \frac{x - m}{j - m}
   \]

### 属性

1. **安全性**：
   - 未满足门限的份额不能恢复原始秘密。即使攻击者获得了 \( k-1 \) 份额，无法推断出秘密。

2. **灵活性**：
   - 可以根据需要自由设置 \( k \) 和 \( n \)，允许在不同的应用场景中进行调整。

3. **抗篡改性**：
   - 多项式的性质确保任何未授权的攻击者无法获得任何信息。

### 应用场景

- **密钥管理**：在分布式密钥生成和存储中，通过将密钥分割成多个份额，增强安全性。
- **数据保护**：重要数据（如密码、证书）可以分散存储，确保即使部分数据丢失，仍然可以恢复。
- **投票系统**：在投票和决策过程中，确保只有达到一定参与者的同意才能揭示结果。

### Python 实现示例

下面是一个简单的 Python 实现 Shamir 门限方案的示例，使用 `numpy` 和 `Random` 模块：

```python
import random
import numpy as np

# 生成多项式的随机系数
def generate_polynomial(secret, k):
    coefficients = [random.randint(0, 100) for _ in range(k-1)]
    coefficients.insert(0, secret)  # 插入秘密作为常数项
    return coefficients

# 计算份额
def generate_shares(secret, n, k):
    coefficients = generate_polynomial(secret, k)
    shares = [(i, evaluate_polynomial(coefficients, i)) for i in range(1, n + 1)]
    return shares

def evaluate_polynomial(coefficients, x):
    return sum(c * (x ** i) for i, c in enumerate(coefficients))

# 拉格朗日插值法恢复秘密
def lagrange_interpolation(shares, x):
    total = 0
    for i, (xi, yi) in enumerate(shares):
        term = yi
        for j, (xj, _) in enumerate(shares):
            if i != j:
                term *= (x - xj) / (xi - xj)
        total += term
    return int(total)

# 示例
secret = 12345  # 需要分享的秘密
n = 5  # 总份额
k = 3  # 门限

# 生成份额
shares = generate_shares(secret, n, k)
print("Generated Shares:", shares)

# 恢复秘密
selected_shares = shares[:k]  # 选取前k个份额来恢复
recovered_secret = lagrange_interpolation(selected_shares, 0)
print("Recovered Secret:", recovered_secret)
```