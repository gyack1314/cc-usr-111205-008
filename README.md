# 多站雨型的受限最小平均代价配准

这是独立的防汛演练资料Python项目。`data/`保存虚构人数、观测、时序或方案，`response_data.io`提供严格数值读取与来源字节SHA-256，`response_data.domain`给出已有原始摘要，`response_data.cli`输出JSON。字段、量纲与分析口径见[资料约定](docs/data-contract.md)。这些数据用于离线复盘，不连接真实人员、传感器或指挥系统。

配准模块比较冻结的参考过程与候选过程：同名通道用锁定的中心与正scale标准化，缺测不补零，局部代价是共同通道上的加权平方差除以共同权重和；路径受索引带与`run_limit`伸缩限制，目标是全局最小平均代价（不是先求最小总和再除长度），并用增广长度DP加Dinkelbach分式规划精确求解。逐点对报告有效通道、累计成本、复用次数与时间伸缩解释，但只复用原始采样点、不做插值，也不据配准好坏判断降雨因果。

Python 3.11及以上即可直接使用，无第三方Python依赖。在项目根目录执行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q response_data tests
python -m response_data.cli inspect data/normal.json
python -m response_data.cli register data/normal.json
python -m response_data.cli bruteforce data/normal.json
python -m response_data.demo
```

`inspect`显示原始资料概况，不覆盖源文件；`register`输出完整配准报告（整数资料默认精确有理模式，`--float`强制浮点）；`bruteforce`穷举全部合法路径并与分式规划胜者核对；`demo`运行穷举核对、“先求最小和再除长度”选错反例、同分决胜、通道重命名/等价换单位不变性与缺测覆盖五组演示。`invalid-`前缀资料为不合法输入，其余变体使用相同字段保留不同业务边界。数值null须按字段语义区别于0，不在读取时自动填补。

Dockerfile固定基础镜像，可用`docker build -t response-data .`及`docker run --rm response-data`读取资料；`docker run --rm response-data python -m unittest discover -s tests -v`执行已有检查。运行所需资料全部在本地，无需登录或实时来源。
