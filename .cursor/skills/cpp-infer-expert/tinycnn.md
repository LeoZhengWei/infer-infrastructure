# M1 tiny-cnn：算子抽象

上游是 https://github.com/tiny-dnn/tiny-dnn （文档里的 tiny-cnn）。克隆到 `third_party/tiny-dnn`。

这门课要拿走的是框架形状，不是 MNIST 精度：

- 张量类型和数据布局（NCHW 与 NHWC 差在哪一次内层循环连续）
- 层的接口：如何注册、前向怎么被串起来
- 卷积或全连接的 SIMD 写在哪一层
- 热路径用模板做类型分发的位置；虚函数留在不烫的地方

## L14

跑上游 examples 里的 MNIST。路径以仓库现状为准，导师帮学员定位文件，不替他改构建。PROGRESS 记下示例名称和它打印的精度或损失。

## L15

笔记 `notes/L15-framework.md` 必须回答：

1. 张量持有数据还是视图？
2. 指出一条卷积或内积的热循环，说明它没有通过虚函数进入。
3. 换布局会让哪一个下标变成最内层连续维。

## L16 新增 RMSNorm

按该框架已有层的写法增加 RMSNorm，不要在 `engine/` 里再抄一份就算完成。

公式与 S1 相同：`y = x / sqrt(mean(x^2) + eps) * weight`。

测试用固定的几个数。对拍对象二选一，写进 PROGRESS：

- 与 `engine/` 里已经对拍过的 RMSNorm，同一输入的 max abs diff
- 或长度 ≤ 8 的手算

导师不把算子类的完整实现写出来。可以指出该框架里最接近的归一化层文件名，让学员照那个文件的注册方式加。
