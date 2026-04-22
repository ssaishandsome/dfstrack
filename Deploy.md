## 第一天
新建了一个slot_parser.py文件，实现了一个SlotParser类

## 第二天
+ slot_attention_diversity_loss(assign)
输入 assign: [B, N, K]
把每个 slot 看成一张“对 N 个 token 的注意力图”
如果不同 slots 的注意力图很像，这个 loss 就会变大
目标是让不同 slots 关注不同 token

+ slot_orthogonality_loss(slots)
输入 slots: [B, K, C]
先把每个 slot 向量归一化
再算 slot-slot 的 Gram matrix
希望它接近单位阵，也就是：
自己和自己相似度高
不同 slots 之间尽量正交
目标是让不同 slot 表示更独立

+ slot_balance_loss(assign)
输入 assign: [B, N, K]
先在整个 batch 和 token 维度上统计每个 slot 收到的总质量
得到 batch-level slot mass
再和均匀分布做 MSE
目标是避免所有 token 都塌到少数几个 slots 上
