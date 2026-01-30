【**pyqpanda_alg.QARM文档注释补充**】

<u>注意：官方算法文档请参考：https://qcloud.originqc.com.cn/document/pyqpanda-algorithm/index.html</u>

补充原pyqpanda_alg.QARM文档中 “Package Contents” 模块下的 “Examples” 中的注释： 

```python
#导入模块
import os
from pyqpanda_alg.QARM import QuantumAssociationRulesMining
from pyqpanda_alg import QARM

#数据读取
def read(file_path):
    if os.path.exists(file_path):
        trans_data = []
        with open(file_path, 'r', encoding='utf8') as f:
            data_line = f.readlines()
            if data_line:
                for line in data_line:
                    if line:
                        data_list = line.strip().split(',')
                        trans_data.append([data.strip() for data in data_list])
            else:
                raise ValueError("The file {} has no any data!".format(file_path))#异常处理，若文件为空或无有效行，抛出 ValueError
    else:
        raise FileNotFoundError('The file {} does not exists!'.format(file_path))#异常处理，若文件不存在，抛出 FileNotFoundError
    return trans_data


if __name__ == '__main__':
    data_path = QARM.__path__[0]#获取 QARM 模块的安装路径，用于定位数据集文件夹。
    data_file = os.path.join(data_path, 'dataset/data2.txt')#拼接数据文件完整路径。
    trans_data = read(data_file)
    support = 0.2
    conf = 0.5
    qarm = QuantumAssociationRulesMining(trans_data, support, conf)#实例化量子关联规则挖掘类，传入交易数据、支持度和置信度阈值。
    result = qarm.run()
    print(result)
```

  
