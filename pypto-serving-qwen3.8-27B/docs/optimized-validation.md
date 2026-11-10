# 最近优化源码的验证范围

本次迁入已经取得真机数值证据的模型图，以及长预填充与解码轮换、保持未调度请求状态的服务实现。历史精度验证针对各自冻结的源码组合；本次独立包完成的源码、CPU 编译和 ABI 等价检查，不能替代迁移后整服务的真机验收。

本次整理没有启动新的 NPU 或 HTTP 测试。迁移后的独立服务尚未重新完成全生命周期、原模型 44+3、全 18 格及每格最终 200 请求验收，因此不宣称达到最终发布条件或超越 vLLM。

## 合入范围与默认配置

`--backend auto` 在 TP1/DP1 时使用 `serving/optimized`，其他原有拓扑保留 legacy 实现。优化后端包含 P16/P64 批量预填充、单请求 P512、fused greedy decode、局部状态 compaction 和预填充/解码轮换；总上下文为 2048，prefix reuse 关闭。

最新 P128 FFN 混部图随源码提供，`--mixed-prefill 1` 才启用。Host lookahead prepare 通过 `--async-prepare 1` 启用。两个开关默认均为 `0`；它们的新服务组合尚未完成设备生命周期验证。独立包按源码正常编译图，不依赖 harness、开发机编译目录或预编译设备产物。

## 历史真机证据

下表的调用数属于不同冻结作业，不能相加为本次新包的验证调用数。部分后续作业明确复用了此前证据。

| 冻结版本 | 实际通过的检查 | 结论与适用边界 |
| --- | --- | --- |
| de59 | 原模型 44 个数值案例及 3 个诊断，1980+195=2175 次调用；8 图、832 functions；66 层、完整 consumed vocab、全部 8 个状态 owner、只读与 sentinel 检查 | 原模型基座有完整 44+3 数值依据。收据中的 serving、release、vLLM parity 均未通过，不代表独立服务已验收。 |
| rotation180 | 10 个案例、648 次调用；完整状态/KV、hidden、词表、sampler、5 token continuation；682 次未调度状态与 KV metadata 检查 | 轮换及保持未调度 owner 的组合精度通过。此版本 fused decode 与后续 decode264 的源码不同，不能将其 HTTP 数据直接标为当前新组合性能。 |
| rotation185–187 | 全 18 个输入/输出/并发格，1728 个计分 HTTP 请求；18/18 共享筛选门限通过；复用 rotation180 数值证据 | 是原冻结服务组合的 HTTP 筛选结果，未完成每格最终 200 请求与新的匹配 vLLM 终验。 |
| integrated353 | 1396 次调用，包含原 648 检查及 28 个 mixed 案例；14 图、1658 functions；14 个完整输出、53 项 Host 字段、66 层、全部 8 owner/物理 KV、完整词表/greedy 和 5 token continuation | mixed334/short348 对应的模型路径精度通过。真实 admission/cancel/EOS/reuse/HTTP 全生命周期未验收。 |
| FFN373 | 1396 次调用，同样覆盖原 648 检查及 28 个 mixed 案例；14 图、1656 functions；保留上述完整字段检查 | 最新 mixed371 FFN 图有实际精度依据；CID12 为 ABI50、119 functions。不能因此默认启用原启动器内的旧 feedback323。 |
| FFN381 | 在 373 证据上新增 294 次调用、6 个 mixed 数值案例及 72 个恢复相同状态的 ABBA legs；6/6 诊断性能门限通过 | 相对 mixed334，整个 mixed step 延迟下降 6.25%–8.19%。该指标属于设备 mixed step，不能换算成 HTTP TTFT、TPOT 或吞吐收益。 |

收据名称、原始 SHA256、关键原字段和 381 的六格延迟见 [历史证据摘要](evidence/optimized-history.json)。摘要为审核索引，原始收据仍是完整证据来源；摘要没有复制本机绝对路径、权重或编译产物。

## 本次源码迁移检查

导出阶段的 12 个模型/辅助源码仅调整包导入；逆变换后与原源码逐字节一致，AST 一致。由导出源码重新 CPU 编译的 **10/10 图**，完整 `distributed_meta.json` 与实际验证时的 metadata 逐字节一致，函数数量一致。随后把图模块放入 `serving/optimized/kernels`，只相应调整包路径；参数顺序、shape、dtype、方向和 ND/NZ layout 保持不变。

所有 **575 个生成 PTO 文件**在仅移除调试源码位置、对 SSA 名称做一一对应重命名后严格一致。没有归一化数学常数、运算顺序、精度转换、rounding、layout、循环或依赖；修改常数的负控被拒绝。CPU 编译没有构造 DeviceWorker，没有运行 NPU。具体检查收据及每图 metadata SHA 见 [迁移证据摘要](evidence/optimized-source-export.json)。

最终独立包及全部 10 个图入口的导入检查通过，17 项 CPU 测试通过，覆盖调度、owner 保留和入口配置等行为。导入 `torch_npu` 时可能查询设备信息；这些检查没有创建 DeviceWorker 或执行模型，不能计为真机模型精度验证。

| 图 | 原 CID | 参数数 | functions | 原输入 shape |
| --- | ---: | ---: | ---: | --- |
| prefill_small16 | 0 | 37 | 57 | `[8,16]` |
| prefill_wide64 | 8 | 37 | 520 | `[8,64]` |
| prefill_chunk512 | 7 | 37 | 61 | `[1,512]` |
| prefill_single128 | 13 | 37 | 61 | `[1,128]` |
| decode_greedy | 9 | 41 | 60 | `[8,8]` |
| mixed_ffn | 12 | 50 | 119 | owner 0 的 P128 与已选择 decode owner |
| compact_local | 10 | 4 | 2 | int32 状态字视图与 copy plan |
| zero_state | 2 | 3 | 2 | 完整状态 owner |
| compact_scratch | 3 | 7 | 4 | 完整状态 owner 与 scratch |
| greedy | 4 | 7 | 2 | 完整 logits 与 sampler 输出 |

P512 是 CID7，CID8 是 wide64，两者不可混淆。P128、P512 虽为单请求输入，仍共享完整的 8 个物理状态 owner。局部 compactor 的 int32 view 是按位视图：conv 为 `[8,983040]`，FP32 recurrent 为 `[8,37748736]`，copy plan 为 `[12,3]`；不得改为数值转换或覆盖未调度 owner。

## PyPTO 与 NZ 兼容合同

`pypto.pin` 固定 **98752784548b777b2377a9084411c3716eb341d6**。优化后端要求 Python 3.11 或以上，并使用该 PyPTO 版本所配套的 Simpler/PTOAS；本摘要引用的选定终审收据没有单独冻结 Simpler/PTOAS 版本号，因而不对其他组合做兼容性承诺。应用依赖版本以 `requirements.txt` 为准。

历史设备验证使用隔离 NZ binding overlay，`compiled_program.py` 的 SHA256 为 `41116deca9f5c64fa4b5755cdf3fc745685d0fd2c6e10d67293d4611d20f025b`。对应未修复文件为 `a830c800b24986b279045f0c0ea98be5047472e56e3e0d5e39d15d786f63b32a`。

本包的 `nz_binding.py` 只接受这两个已知文件版本：对原文件在进程内补充相同的 logical NZ shape 规则，遇到已有修复则直接使用。它不修改安装目录；保留 dynamic-batch rank 限制和静态维度校验，其余情况交回原 binding。dtype 校验仍由原 runtime 执行。这是绑定兼容处理，历史数值比较器和容差没有改变。进程内适配与历史 overlay 的整服务设备等价性仍应随新包验收。

本次已完成 448 种 shape/layout/dtype 行为的 CPU 对照，与历史 overlay 逐项相同；实际 DeviceTensor validator 对错误 dtype/shape 的拒绝行为保留。适配安装幂等，库文件字节未变，新增 DeviceWorker 和模型执行均为 0。相应收据摘要及适配源码 SHA 已纳入迁移证据。

NZ 图 metadata 的物理维度为 `[layers,N/16,K/16,16,16]`，编译 meta tensor 恢复逻辑维度 `[layers,K,N]`。权重必须由当前 loader 打包为 NZ；不能将 legacy ND 权重直接绑定到这些图。

## 明确排除的候选

- owned feedback323 已在 `budget1_noop` 真实失败，报 `Complete49 ABI logical inputs differ from original264 replay`；242 个已观察调用、0 次采纳。当前后端不注册该图。
- prepare397/398v2 只有 31 个修复正例及 31 个旧实现负例的组件字节验证，完整模型与服务生命周期未通过；不随本后端注册。
- persistent multi-step、packed-prefill 及其试验启动器不在本次执行路径中；组件/CPU 通过不能充当完整服务验收。
- GDN396 虽完成 1394 次数值/诊断调用，但 12/12 性能组均回退，延迟增加约 3.36%–6.56%，不采纳该候选。这里保留的是已验证的 wide64 GDN 源码。

迁移后的完整 NPU 精度、异构请求调度生命周期、HTTP 18 格和新匹配 vLLM 性能仍需在相同冻结负载下验证。源码 PR 的验证状态与部署验收应分别记录。

## 运行 Host 检查

最终打包检查摘要见 [打包检查](evidence/packaging-checks.json)。

CPU 合同检查（Python 3.11，状态测试需要 PyTorch）：

```bash
python -B tests/test_optimized_contract.py -v
python -B tests/test_optimized_entrypoints.py -v
```
