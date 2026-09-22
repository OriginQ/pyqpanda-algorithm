# 选题调研快照（2026-09-21）

基线 develop：`5f973efccb84bc193157d1ccebe32137e307293b`。
阅读官方 README、main/develop CONTRIBUTING、QUBO/QAOA 模块、现有测试与 CI。
检索源码中的 schedule / scheduling / 排程 / 排班 / 调度 / 实验室 / 考试，
没有发现已实现的同类应用。

GitHub 匿名 REST API 返回限流，网页工具的缓存计数与实时页面不同，故改为
直接读 GitHub HTML：开放 PR 第 1–3 页，共 **70** 项；第 4 页为空。
[完整标题与链接快照](research_pr_snapshot.json)保留可审阅清单。
查阅关闭 PR 第 1 页（13 项），未发现同题。对相关 PR/Issues 进一步阅读正文。

| 对象 | 已有范围 | 对本应用的决策 |
|---|---|---|
| [#60](https://github.com/OriginQ/pyqpanda-algorithm/pull/60) | 小麦灌溉预测、土壤水量平衡、取水预算及 QUBO/QAOA | 同用调度技术但业务模型不同；选择设备故障、清洗、多时隙实验、前序和预约变更，避免农业/水预算方向 |
| [#45](https://github.com/OriginQ/pyqpanda-algorithm/pull/45) | QUBO_QAOA 默认 layer=None 导致崩溃 | 不重复修库，不依赖该封装默认参数 |
| [#69](https://github.com/OriginQ/pyqpanda-algorithm/pull/69) | QUBO 多项式规范化和高阶验证 | 直接编译二次系数、显式 Ising 索引，不依赖待合并修复 |
| [#40](https://github.com/OriginQ/pyqpanda-algorithm/pull/40) | QAOA 风险损失函数修复 | 应用使用标准期望能量，不修改 CVaR/Gibbs |
| [#38](https://github.com/OriginQ/pyqpanda-algorithm/issues/38) | QmRMR 导出、部分示例和文档问题 | 不扩大范围处理 |
| [#8](https://github.com/OriginQ/pyqpanda-algorithm/issues/8) | 旧硬件上的 QPanda3 应用案例 | 不涉及约束实验排程 |
| [#13](https://github.com/OriginQ/pyqpanda-algorithm/issues/13) | 比赛任务与提交流程 | 完整创新应用，PR 到 develop，规范标题 |

网页主页初次读取只返回 JavaScript 提示。进一步读取该页面实际引用的 main
bundle 和 2026ccf 路由 chunk，核对可见文案：开源创新赛道不限制学生/社会人士，
采用 GitHub PR 提交，报名与 PR 窗口均为 4 月 10 日至 9 月 30 日；
[主页](https://qcloud.originqc.com.cn/learning/zh/2026ccf)指向
[开源赛道详情](https://qcloud.originqc.com.cn/learning/zh/contest/list/60/contest:introduction)。
没有登录或代填报名信息，不能确认用户的个人报名状态。

**结论：** LabReschedule 符合完整创新应用任务；在本次可见的开放 PR、Issue
及 develop 中，没有发现同题或高度相似的实验室重排应用。QUBO/QAOA 通用方法
和 XY 技术本身不属于本项目原创。创新定位是场景约束组合、预约稳定性、候选
压缩、可解释约束报告和端到端复现。不能凭检索证明全球新颖性。
在用户批准发布之前还应刷新 PR 清单，以覆盖此快照之后的新投稿。
