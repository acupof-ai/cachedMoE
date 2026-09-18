# P4 / Track X — cache and prefetch policy study on the real routing trace

> `tools/cache_policy_study.py` · `bench/results/cache_policy/policy_study.json` ·
> trace `traces/mixed` (27,399 token × 40 层，`docs/route_trace.md` §7 的 schema)
> 全部离线、纯 CPU，不启动任何 GPU / engine 进程。

## 0. 一句话结论

**在 4.5 GB/s 的单盘上，没有任何可实现的预取策略能赢过 demand-only LRU。**
盘的空闲时间确实有（每 token 80 ms，占一半），但把它用起来所需的预测精度是
**1.00**——不是 0.9，是 1.00。原因是 demand miss 本身已经要 95–106 ms 盘时间，
超过 80 ms 的计算窗口；浪费预算因此是**负数**。

能赢的只有两件事，都不是预测：

1. **换淘汰策略**（Belady 上限 +43%），
2. **加带宽**（第二块盘 9 GB/s → +40%）。

---

## 1. 这份研究要回答什么

owner 的问题是"streaming 还有很多玩法——最后四层总能提前取到，*只要我们知道它要什么*；
去看更多 cache 算法"。把它拆成三个可测的问题：

- **天花板在哪**：如果在 layer 0 就知道这个 token 全部 40 层要哪 240 个 expert，
  能跑多快？（clairvoyant prefetch = streaming ceiling）
- **容量天花板在哪**：如果淘汰是完美的（Belady），能跑多快？
- **可实现的预测器够不够好**：不需要隐状态的预测器（expert 转移 Markov、token id 表、
  token 内共现）和已有的隐状态 lookahead，各自的 per-layer recall / precision 是多少，
  放进带宽模型后净收益是正是负？

