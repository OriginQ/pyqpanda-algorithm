import math  # 用于floor函数
from typing import Dict, List, Optional, Tuple
from pyqpanda3.core import QCircuit, QProg, QGate, H, X, CR, SWAP, measure

class hadamard_circuit(QCircuit):
    """
    自定义Hadamard电路类，继承自QCircuit。
    功能：为输入的Python列表（含量子比特索引）中的每个量子比特应用Hadamard门。
    """
    def __init__(self, qubit_list: list[int]) -> None:
        """
        初始化Hadamard电路，确保输入为size_t类型列表（非负整数）。
        
        参数:
            qubit_list: Python列表，元素为量子比特索引（整数）。
                        例如：[0, 1, 2] 表示对第0、1、2个量子比特应用H门。
        """
        # 调用父类QCircuit的构造函数
        super().__init__()
        
        # 遍历列表中的每个量子比特索引，应用Hadamard门
        for qubit_idx in qubit_list:
            # 验证输入为整数（量子比特索引必须是整数）
            if not isinstance(qubit_idx, int):
                raise TypeError(f"量子比特索引必须是整数，而非 {type(qubit_idx)}")
            
            # 验证是否为非负数（符合size_t无符号特性）
            if qubit_idx < 0:
                raise ValueError(f"size_t类型不允许负数，输入索引为 {qubit_idx}")
            
            # 将H门添加到电路中（H门接受整数索引作为参数）
            self << H(qubit_idx)
            
            
def apply_QGate(qubit_list: list[int], func_obj) -> QCircuit:
    """
    批量为量子比特列表中的每个量子比特应用自定义量子门函数，生成并返回组合后的量子电路。

    功能说明：
        遍历输入的量子比特列表，对每个量子比特应用`func_obj`生成的量子门（QGate），
        所有门操作按顺序组合成一个QCircuit对象，便于嵌入主量子程序中执行。
        支持PyQpanda3内置门函数（如H、X、RX等）或自定义门生成函数。

    参数：
        qubit_list (list[int]): 量子比特索引列表，元素为非负整数（size_t类型），
            表示需要应用门操作的量子比特。例如：[0, 1, 2] 对应第0、1、2个量子比特。
        func_obj: 门生成函数，接收一个量子比特索引（int）作为输入，返回一个QGate对象。
            要求：
            - 输入必须为整数（量子比特索引）；
            - 返回值必须是PyQpanda3的QGate类型（如H门、RX门等）。
            示例：pq.H（Hadamard门）、lambda q: pq.RX(q, 0.5)（带角度的RX门）。

    返回：
        QCircuit: 包含所有量子门操作的电路对象，可通过<<操作符嵌入主程序（QProg）。

    异常：
        TypeError: 若`qubit_list`中元素不是整数，或`func_obj`返回值不是QGate类型。
        ValueError: 若`qubit_list`中包含负数（不符合size_t无符号整数特性）。

    使用示例：
        1. 批量应用Hadamard门：
            >>> q_list = [0, 1, 2]
            >>> circuit = apply_QGate(q_list, H)
            >>> # 电路包含：H(0) << H(1) << H(2)

        2. 批量应用带参数的RX门：
            >>> q_list = [0, 1]
            >>> rx_func = lambda q: RX(q, 0.314)  # 自定义带固定角度的RX门函数
            >>> circuit = apply_QGate(q_list, rx_func)
            >>> # 电路包含：RX(0, 0.314) << RX(1, 0.314)

        3. 动态生成门（根据量子比特索引调整参数）：
            >>> q_list = [0, 1, 2]
            >>> def dynamic_gate(q):
            ...     angles = [0.1, 0.2, 0.3]  # 为不同qubit设置不同角度
            ...     return RY(q, angles[q])
            >>> circuit = apply_QGate(q_list, dynamic_gate)
            >>> # 电路包含：RY(0, 0.1) << RY(1, 0.2) << RY(2, 0.3)
    """
    cir = QCircuit()

    if not isinstance(qubit_list, list) and not isinstance(qubit_list, int) :
        raise TypeError("量子比特索引必须为整数")
    
    if isinstance(qubit_list, int) :
        gate = func_obj(qubit_list)
        cir << gate
    else :
        for qubit in qubit_list:
            if not isinstance(qubit, int):
                raise TypeError(f"量子比特索引必须为整数，而非 {type(qubit)}")
            gate = func_obj(qubit)
            if not isinstance(gate, QGate):
                raise TypeError(f"func_obj必须返回QGate类型，而非 {type(gate)}")
            cir << gate
        
    return cir


def measure_all(qubit_list: list[int], cbit_list: list[int]) -> QProg:
    """
    批量创建量子比特到经典比特的测量操作，组合为量子程序（QProg）。
    
    功能说明：
        为输入的量子比特索引列表和经典比特索引列表创建一一对应的测量操作，
        所有测量操作按顺序添加到QProg中，用于读取量子态的测量结果。
        要求量子比特列表和经典比特列表长度必须相等。
    
    参数：
        qubit_list (list[int]): 量子比特索引列表（非负整数），表示需要测量的量子比特。
            例如：[0, 1, 2] 对应第0、1、2个量子比特。
        cbit_list (list[int]): 经典比特索引列表（非负整数），用于存储测量结果。
            例如：[0, 1, 2] 对应第0、1、2个经典比特。
    
    返回：
        QProg: 包含所有测量操作的量子程序，可直接嵌入主量子程序中执行。
    
    异常：
        ValueError: 若qubit_list与cbit_list的长度不相等。
        TypeError: 若列表中包含非整数元素。
    
    使用示例：
        >>> # 测量量子比特0、1到经典比特0、1
        >>> q_list = [0, 1]
        >>> c_list = [0, 1]
        >>> measure_prog = measure_all(q_list, c_list)
        >>> # 测量程序包含：Measure(0, 0) << Measure(1, 1)
    """
    # 验证输入列表元素为整数
    for i, qubit in enumerate(qubit_list):
        if not isinstance(qubit, int):
            raise TypeError(f"量子比特列表第{i}个元素必须为整数，而非 {type(qubit)}")
    for i, cbit in enumerate(cbit_list):
        if not isinstance(cbit, int):
            raise TypeError(f"经典比特列表第{i}个元素必须为整数，而非 {type(cbit)}")
    
    # 验证列表长度相等
    if len(qubit_list) != len(cbit_list):
        raise ValueError(f"量子比特列表长度（{len(qubit_list)}）与经典比特列表长度（{len(cbit_list)}）不相等")
    
    # 初始化空量子程序
    measure_prog = QProg()
    
    # 逐个添加测量操作
    for qubit, cbit in zip(qubit_list, cbit_list):
        # 验证索引非负（量子/经典比特索引不能为负）
        if qubit < 0:
            raise ValueError(f"量子比特索引不能为负数，传入{qubit}")
        if cbit < 0:
            raise ValueError(f"经典比特索引不能为负数，传入{cbit}")
        
        # 添加测量操作：Measure(量子比特, 经典比特)
        measure_prog << measure(qubit, cbit)
    
    return measure_prog


def qft(qubit_list: list[int]) -> QCircuit:
    """
    构建量子傅里叶变换（QFT）电路，对输入的量子比特列表应用QFT操作。
    
    功能说明：
        量子傅里叶变换是经典傅里叶变换的量子版本，通过Hadamard门和受控旋转门（CR）的组合实现，
        可在O(n²)时间内完成n个量子比特的傅里叶变换，远快于经典算法的O(n·2ⁿ)。
        本实现遵循标准QFT电路结构：从最后一个量子比特开始，逐层应用H门和CR门。
    
    参数：
        qubit_list (list[int]): 量子比特索引列表（非负整数），表示参与QFT的量子比特。
            例如：[0, 1, 2] 表示对3个量子比特执行QFT。
    
    返回：
        QCircuit: 包含完整QFT操作的量子电路，可嵌入主程序执行。
    
    异常：
        TypeError: 若qubit_list中包含非整数元素。
        ValueError: 若qubit_list为空或包含负数索引。
    
    使用示例：
        >>> # 对3个量子比特（0,1,2）执行QFT
        >>> q_list = [0, 1, 2]
        >>> qft_circuit = qft(q_list)
        >>> # 电路结构：
        >>> # H(2) << CR(1, 2, π) << CR(0, 2, π/2) 
        >>> # << H(1) << CR(0, 1, π) 
        >>> # << H(0)
        >>> 
        >>> # 嵌入主程序执行
        >>> qvm = pq.QMachine(pq.QMachineType.CPU)
        >>> main_prog = pq.QProg()
        >>> main_prog << qft_circuit  # 添加QFT电路
        >>> qvm.run(main_prog)
    """
    pi = 3.141592653589793238462643383279502884
    
    # 输入验证
    if not qubit_list:
        raise ValueError("量子比特列表不能为空")
    for i, qubit in enumerate(qubit_list):
        if not isinstance(qubit, int):
            raise TypeError(f"量子比特列表第{i}个元素必须为整数，而非{type(qubit)}")
        if qubit < 0:
            raise ValueError(f"量子比特索引不能为负数，第{i}个元素为{qubit}")
    
    # 初始化空电路
    qft_circuit = QCircuit()
    n = len(qubit_list)  # 量子比特数量
    
    # 外层循环：遍历每个目标比特层（i从0到n-1）
    for i in range(n):
        # 计算当前目标比特的索引（从最后一个比特开始）
        target_idx = n - 1 - i  # 对应C++的qvec.size()-1 - i
        target_qubit = qubit_list[target_idx]
        
        # 1. 对目标比特应用Hadamard门
        qft_circuit << H(target_qubit)
        
        # 内层循环：应用受控旋转门（CR门）
        for j in range(i + 1, n):
            # 计算控制比特的索引
            control_idx = n - 1 - j  # 对应C++的qvec.size()-1 - j
            control_qubit = qubit_list[control_idx]
            
            # 计算旋转角度：2π / 2^(j - i + 1)
            angle = 2 * pi / (1 << (j - i + 1))  # 1<<n 等价于2^n
            
            # 应用CR门（控制比特，目标比特，角度）
            # CR门功能：控制比特为|1>时，对目标比特应用R(angle)旋转
            qft_circuit << CR(control_qubit, target_qubit, angle)
    
    return qft_circuit


def QFT(qubit_list: list[int]) -> QCircuit:
    """
    实现完整的量子傅里叶变换（QFT）电路，包含相位变换和比特交换步骤。
    
    功能说明：
        量子傅里叶变换是经典离散傅里叶变换的量子版本，通过H门、CR门实现相位变换，
        最后通过SWAP门修正比特顺序，输出符合标准傅里叶变换的量子态。
        适用于量子算法中需要频率域分析的场景（如Shor算法、量子相位估计等）。
    
    参数：
        qubit_list (list[int]): 量子比特索引列表（非负整数），按从低到高顺序传入。
            例如：[0, 1, 2] 表示3个量子比特，0为最低位，2为最高位。
    
    返回：
        QCircuit: 包含完整QFT操作的量子电路（含SWAP交换）。
    
    异常：
        TypeError: 若列表中包含非整数元素。
        ValueError: 若列表为空或包含负数索引。
    
    使用示例：
        >>> q_list = [0, 1, 2]  # 3个量子比特
        >>> qft_circuit = qft(q_list)
        >>> # 电路结构：
        >>> # 相位变换部分：H(2) << CR(1,2,π/2) << CR(0,2,π/4) 
        >>> #            << H(1) << CR(0,1,π/2) 
        >>> #            << H(0)
        >>> # 交换部分：SWAP(0,2)
        >>> 
        >>> # 嵌入主程序执行
        >>> qvm = pq.QMachine(pq.QMachineType.CPU)
        >>> main_prog = pq.QProg() << qft_circuit
        >>> qvm.run(main_prog)
    """
    pi = 3.141592653589793238462643383279502884
    
    # 输入验证
    if not qubit_list:
        raise ValueError("量子比特列表不能为空")
    for i, qubit in enumerate(qubit_list):
        if not isinstance(qubit, int):
            raise TypeError(f"量子比特列表第{i}个元素必须为整数，而非{type(qubit)}")
        if qubit < 0:
            raise ValueError(f"量子比特索引不能为负数，第{i}个元素为{qubit}")
    
    qft_circuit = QCircuit()
    n = len(qubit_list)  # 量子比特数量

    # 1. 相位变换部分：H门 + CR门
    for i in range(n):
        # 目标比特索引：从最后一个比特开始（qvec.size()-1 - i）
        target_idx = n - 1 - i
        target_qubit = qubit_list[target_idx]
        
        # 应用H门
        qft_circuit << H(target_qubit)
        
        # 应用CR门序列
        for j in range(i + 1, n):
            control_idx = n - 1 - j  # 控制比特索引
            control_qubit = qubit_list[control_idx]
            
            # 计算旋转角度：2π / 2^(j - i + 1)
            angle = 2 * pi / (1 << (j - i + 1))  # 1<<n 等价于2^n
            
            # 添加CR门（控制比特，目标比特，角度）
            qft_circuit << CR(control_qubit, target_qubit, angle)
    
    # 2. 交换比特：修正输出顺序（完成完整QFT）
    for i in range(math.floor(n / 2)):
        # 交换第i个比特和第n-1-i个比特
        qft_circuit << SWAP(qubit_list[i], qubit_list[n - 1 - i])
    
    return qft_circuit


def bind_nonnegative_data(value: int, qubit_list: list[int]) -> QCircuit:
    """
    将非负整数绑定到量子比特列表，生成初始化电路（将量子比特从全|0⟩态转换为对应整数的量子态）。
    
    功能说明：
        根据输入整数的二进制表示，通过X门翻转量子比特列表中对应位置的比特（1对应|1⟩，0对应|0⟩），
        实现经典数据到量子态的映射。量子比特列表的初始状态需为全|0⟩，且不考虑符号位（仅支持非负数）。
    
    参数：
        value (int): 待绑定的非负整数（≥0）。
        qubit_list (list[int]): 量子比特索引列表（非负整数），用于存储量子态。
    
    返回：
        QCircuit: 初始化电路，包含X门操作，使量子比特列表表示value的二进制形式。
    
    异常：
        ValueError: 若value为负数，或qubit_list长度不足（无法存储value的二进制表示）。
        TypeError: 若qubit_list中包含非整数元素。
    
    使用示例：
        >>> # 将整数5（二进制101）绑定到3个量子比特[0,1,2]
        >>> circuit = bind_nonnegative_data(5, [0, 1, 2])
        >>> # 电路包含：X(0) << X(2)（对应二进制101，低位在前）
        >>> 
        >>> # 验证：量子比特0和2被翻转为|1⟩，1保持|0⟩
        >>> qvm = pq.QMachine(pq.QMachineType.CPU)
        >>> main_prog = pq.QProg() << circuit
        >>> qvm.run(main_prog)
    """
    # 输入验证：value必须为非负整数
    if not isinstance(value, int):
        raise TypeError(f"value必须为整数，而非{type(value)}")
    if value < 0:
        raise ValueError(f"仅支持非负整数，传入value={value}")
    
    # 输入验证：qubit_list必须为非负整数列表
    for i, qubit in enumerate(qubit_list):
        if not isinstance(qubit, int):
            raise TypeError(f"量子比特列表第{i}个元素必须为整数，而非{type(qubit)}")
        if qubit < 0:
            raise ValueError(f"量子比特索引不能为负数，第{i}个元素为{qubit}")
    
    # 特殊情况：value=0，返回空电路（全|0⟩已符合要求）
    if value == 0:
        return QCircuit()
    
    # 计算存储value所需的最小量子比特数（二进制位数）
    # 公式：floor(log2(value) + 1)，等价于value的二进制长度
    qnum = math.floor(math.log2(value) + 1)
    
    # 检查量子比特数量是否足够
    if len(qubit_list) < qnum:
        raise ValueError(f"量子比特数量不足（需要至少{qnum}个，实际传入{len(qubit_list)}个）")
    
    # 构建初始化电路
    circuit = QCircuit()
    cnt = 0  # 量子比特索引（从0开始，对应二进制低位）
    remaining_value = value  # 剩余值，用于提取二进制位
    
    while remaining_value > 0:
        # 提取当前最低位（0或1）
        current_bit = remaining_value % 2
        if current_bit == 1:
            # 若为1，对第cnt个量子比特应用X门（|0⟩→|1⟩）
            circuit << X(qubit_list[cnt])
        
        # 右移一位，处理下一个高位
        remaining_value = remaining_value // 2
        cnt += 1
    
    return circuit


def parse_quantum_result_dict(result: Dict[str, float], qubit_list: List[int], select_max: int = -1) -> Dict[str, float]:
    """
    解析量子程序测量结果字典，支持筛选Top-N概率结果，保持与prob_run_dict一致的功能逻辑。
    
    功能说明：
        对量子测量结果（键为二进制字符串，值为概率）进行处理，根据select_max筛选结果，
        同时验证结果的合法性（如概率和为1），并保持与原量子比特顺序的对应关系。
    
    参数：
        result (Dict[str, float]): 原始测量结果字典，键为二进制字符串（如"0110"），
            值为对应结果的概率（浮点数，范围[0,1]）。

        qubit_list (List[int]): 测量的量子比特索引列表，用于验证二进制字符串长度是否匹配。

        select_max (int, 可选): 限制返回结果的数量，默认为-1（返回所有结果）。
            取值范围：[-1, 2^len(qubit_list)]，其中-1表示无限制，正整数n表示返回前n个最高概率结果。
    
    返回：
        Dict[str, float]: 处理后的测量结果字典，按概率从高到低排序（若select_max>0）。
    
    异常：
        ValueError: 若输入参数不合法（如select_max超出范围、二进制字符串长度与qubit_list不匹配、概率和不为1等）。
        TypeError: 若输入result不是字典或概率不是浮点数。
    
    使用示例：
        >>> # 原始结果：3个量子比特的测量概率
        >>> raw_result = {"000": 0.1, "111": 0.8, "010": 0.1}
        >>> qubit_list = [0, 1, 2]
        >>> 
        >>> # 1. 返回所有结果
        >>> parse_quantum_result(raw_result, qubit_list)
        {'000': 0.1, '111': 0.8, '010': 0.1}
        >>> 
        >>> # 2. 返回概率最高的1个结果
        >>> parse_quantum_result(raw_result, qubit_list, select_max=1)
        {'111': 0.8}
        >>> 
        >>> # 3. 返回概率最高的2个结果
        >>> parse_quantum_result(raw_result, qubit_list, select_max=2)
        {'111': 0.8, '000': 0.1, '010': 0.1}  # 概率相同则保留原始顺序
    """
    # 输入类型验证
    if not isinstance(result, dict):
        raise TypeError(f"result必须是字典类型，而非{type(result)}")
    for bit_str, prob in result.items():
        if not isinstance(bit_str, str) or not all(c in {'0', '1'} for c in bit_str):
            raise TypeError(f"result的键必须是二进制字符串（如'0110'），而非{bit_str}")
        # if not isinstance(prob, (int, float)) or prob < 0 or prob > 1:
        #     raise TypeError(f"result的值必须是[0,1]范围内的数字，'{bit_str}'对应值为{prob}")
    
    # 验证二进制字符串长度与量子比特数量匹配
    n_qubits = len(qubit_list)
    for bit_str in result:
        if len(bit_str) != n_qubits:
            raise ValueError(f"二进制字符串'{bit_str}'长度为{len(bit_str)}，与量子比特数量{n_qubits}不匹配")
    
    # 验证概率和为1（允许微小浮点误差）
    total_prob = sum(result.values())
    if not abs(total_prob - 1.0) < 1e-6:
        raise ValueError(f"测量概率总和为{total_prob}，不符合归一化条件（应为1.0）")
    
    # 验证select_max合法性
    max_possible = 1 << n_qubits  # 2^n_qubits，所有可能的结果数
    if select_max != -1 and (not isinstance(select_max, int) or select_max < -1 or select_max > max_possible):
        raise ValueError(f"select_max必须在[-1, {max_possible}]范围内，实际传入{select_max}")
    
    # 处理select_max：返回所有结果或Top-N结果
    if select_max == -1:
        return result.copy()  # 不筛选，返回所有结果
    else:
        # 按概率从高到低排序（概率相同则保留原始插入顺序）
        sorted_items = sorted(result.items(), key=lambda x: (-x[1], x[0]))
        # 取前select_max个结果，转换为字典
        top_items = sorted_items[:select_max]
        return dict(top_items)
    
    
def parse_quantum_result_list(result: List[float], qubit_list: List[int], select_max: int = -1) -> List[float]:
    """
    解析量子程序测量结果列表，支持筛选Top-N概率结果，功能逻辑与字典版本保持一致。
    
    功能说明：
        对量子测量结果列表（元素为概率值）进行处理，根据select_max筛选结果，
        验证结果的合法性（如概率和为1），保持与原量子比特顺序的对应关系。
        列表索引对应二进制结果（如索引0对应"000...", 索引1对应"000...1"等）。
    
    参数：
        result (List[float]): 原始测量结果列表，元素为概率值（范围[0,1]），
            索引对应二进制结果（按整数解析，如索引3对应二进制"11"）。

        qubit_list (List[int]): 测量的量子比特索引列表，用于确定总结果数。

        select_max (int, 可选): 限制返回结果的数量，默认为-1（返回所有结果）。
            取值范围：[-1, 2^len(qubit_list)]，其中-1表示无限制，正整数n表示返回前n个最高概率结果。
    
    返回：
        List[float]: 处理后的测量结果列表，按概率从高到低排序（若select_max>0）。
    
    异常：
        ValueError: 若输入参数不合法（如select_max超出范围、列表长度不符合量子比特数、概率和不为1等）。
        TypeError: 若输入result不是列表或元素不是浮点数。
    """
    # 输入类型验证
    if not isinstance(result, list):
        raise TypeError(f"result必须是列表类型，而非{type(result)}")
    for idx, prob in enumerate(result):
        if not isinstance(prob, (int, float)) or prob < 0 or prob > 1:
            raise TypeError(f"result的元素必须是[0,1]范围内的数字，索引{idx}对应值为{prob}")
    
    # 验证列表长度与量子比特数量匹配（应等于2^n_qubits）
    n_qubits = len(qubit_list)
    expected_length = 1 << n_qubits  # 2^n_qubits
    if len(result) != expected_length:
        raise ValueError(f"结果列表长度为{len(result)}，与量子比特数量{n_qubits}不匹配（应为{expected_length}）")
    
    # 验证概率和为1（允许微小浮点误差）
    total_prob = sum(result)
    if not abs(total_prob - 1.0) < 1e-6:
        raise ValueError(f"测量概率总和为{total_prob}，不符合归一化条件（应为1.0）")
    
    # 验证select_max合法性
    if select_max != -1 and (not isinstance(select_max, int) or select_max < -1 or select_max > expected_length):
        raise ValueError(f"select_max必须在[-1, {expected_length}]范围内，实际传入{select_max}")
    
    # 处理select_max：返回所有结果或Top-N结果
    if select_max == -1:
        return result.copy()  # 不筛选，返回所有结果
    else:
        # 结合索引排序（按概率从高到低，概率相同则按原索引升序）
        indexed_probs = [(prob, idx) for idx, prob in enumerate(result)]
        sorted_probs = sorted(indexed_probs, key=lambda x: (-x[0], x[1]))
        # 提取前select_max个概率值
        top_probs = [prob for prob, idx in sorted_probs[:select_max]]
        return top_probs