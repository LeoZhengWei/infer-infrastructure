# 导读：先说这份指南怎么用

用 ROS 2 做产品，真正麻烦的不是「怎么创建一个节点」，而是三件事：**消息到底怎么从 A 到 B**、**回调到底在哪个线程里跑**、**延迟高了以后按什么顺序去查**。这份指南就围绕这三件事写，按「原理 → 工程 → 调优 → 速查」的顺序排。

建议第一遍顺着读，把分层和链路打通；之后当手册用——「跨机不通」翻第 1、2 章，「回调卡住」翻第 3 章，「延迟抖动」翻第 4 章。第 5 章放了一条完整链路的设计参考、要点速记和命令清单，平时查起来最快。

最容易出问题的地方集中在四处：QoS 不兼容导致静默不通、单线程 Executor 里的 service 死锁、零拷贝看起来开了其实没生效、1 kHz 控制环的抖动来源。这四处在正文里都配了失效场景和排查动作。

> 一句话抓住 ROS 2 的性格：它不是一个通信库，而是把工业级的 DDS（一种自带服务发现、自带服务质量约定的发布订阅标准）包成了机器人好用的接口。理解这句话，后面很多设计就不别扭了。

<!-- PAGEBREAK -->

# 第 1 章 ROS 2 为什么长这样

## 1.1 ROS 1 撞上的四堵墙

ROS 1 的核心是一个叫 roscore 的中心节点。所有节点先找它登记，再互相连。这套东西在教学和实验室里够用，放到产品上会撞墙：

- **单点故障**：master 挂了，已经连上的节点还能聊，但新节点进不来，整个系统慢慢僵掉。
- **实时性没保证**：TCP 传输、消息在用户态多次拷贝、回调由各节点自己的线程随便跑，没法和控制周期对齐。
- **网络不友好**：无线链路丢包时 TCP 会重传、会队头阻塞，激光雷达一帧旧数据能把新数据堵在后面。
- **没有质量和安全**：谁都能连上来发消息，既不能约定「这包丢了就算了」，也没法认证对端是谁。

ROS 2 的答案是把通信这层整个换掉：去掉中心节点，改用 DDS 的自动发现；把「这包要不要可靠」「队列留几帧」变成可协商的 QoS；把执行回调的权力收拢到 Executor，方便做实时调度。

## 1.2 五层结构，一层都别跳过

从上往下看，一次 `publish()` 会穿过这些层：

| 层 | 名字 | 干什么 | 容易混淆的点 |
| --- | --- | --- | --- |
| 应用 | rclcpp / rclpy | 你写的 Node、Publisher、Subscription | 客户端库的线程安全、消息所有权 |
| 抽象 | rcl | 与语言无关的 C 层，统一概念与生命周期 | rcl 与 rclcpp 的分工 |
| 接口 | rmw | ROS Middleware Interface，把中间件抽象成统一函数 | 换 RMW 不改业务代码 |
| 中间件 | DDS 实现 | Fast DDS / Cyclone DDS / Zenoh 等，负责发现、QoS、可靠传输 | QoS 在哪一层生效 |
| 传输 | UDP / TCP / 共享内存 | 真正发字节，RTPS 是 DDS 的线协议 | 端口、组播、MTU、零拷贝 |

几个容易背反的点：

- **QoS 是 DDS 层的东西**，rclcpp 只是把参数传下去。所以「为什么两边 QoS 不兼容就连不上」的答案在 DDS 的匹配规则里，不在 ROS 代码里。
- **进程内通信（intra-process）反而不经过 rmw 的数据通道**，它是 rclcpp 自己用 IntraProcessManager 做的捷径。所以开了进程内通信以后，DDS 的 QoS 事件、网络抓包都可能看不到这条链路。
- **序列化发生在 rmw / DDS 交界处**。`.msg` 编译后生成 IDL 描述和 C/C++ 类型支持，发送时按 CDR（Common Data Representation，DDS 规定的二进制编码）打好包。跨语言能互通，靠的是这套中立的线格式。

## 1.3 DDS：为什么是「买」不是「造」

DDS 是 OMG 的标准，在军工、航电、金融里用了二十年。它给 ROS 2 白送了三样自建中间件最难做对的东西：

1. **自动发现**：节点启动后自己广播「我是谁、我有哪些端点、我的 QoS」，无需中心注册。分两步——SPDP（Simple Participant Discovery Protocol）先发现「域参与者」，SEDP（Simple Endpoint Discovery Protocol）再交换发布/订阅端点的详细信息。
2. **QoS 协商**：连接建立前先校验双方的服务质量能不能对上，对不上就不连，而不是运行时悄悄丢数据。
3. **可靠传输与存活检测**：基于 UDP 自己实现 ACK/NACK 重传、心跳、租约，比直接用 TCP 更适合「宁可丢一帧也不能堵住」的传感器流。

RTPS（Real-Time Publish-Subscribe）是 DDS 的互操作线协议，不同厂商的 DDS 能互相通信就是靠它。分清「DDS 是标准、RTPS 是线协议、Fast DDS 是实现」这三层关系，后面读文档不容易乱。

## 1.4 RMW：换发动机不改车

RMW 是一层很薄的 C 接口，把「创建 publisher」「写数据」「等事件」这些操作抽象成函数指针。业务代码只依赖 rclcpp，运行时用环境变量选实现：

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp      # Fast DDS，多数发行版默认
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp    # Cyclone DDS，轻量、延迟稳
export RMW_IMPLEMENTATION=rmw_zenoh_cpp         # Zenoh，Kilted 起 Tier 1，适合跨网段/弱网
```

选型上的经验：单机、功能要多（共享内存、Discovery Server、XML 细调）用 Fast DDS；资源紧、想要低而稳的延迟用 Cyclone DDS；多机器人跨网络、需要穿透和路由时看 Zenoh。切换 RMW 不用改代码，但 **QoS 的边界行为、发现方式、零拷贝支持程度会变**，这部分必须重新压测。

## 1.5 节点是怎么找到彼此的

这是「跨机通信为什么不通」类问题的答案所在。

**域隔离**：`ROS_DOMAIN_ID` 决定 DDS 域，不同域默认互相看不见。同一网段里跑多台机器人，不给每台分配不同域，就会互相发现、topic 串台、带宽被打满。实践中常用 0~101，避开保留端口段。

**端口是算出来的**：DDSI-RTPS 规定了「知名端口」公式，Fast DDS 默认参数是端口基数 PB=7400、域增益 DG=250、参与者增益 PG=2，四个偏移 d0=0、d1=10、d2=1、d3=11。

| 流量 | 端口公式 | 域 0、参与者 0 的例子 |
| --- | --- | --- |
| 发现流量组播 | 7400 + 250×域 + 0 | 7400 |
| 用户数据组播 | 7400 + 250×域 + 1 | 7401 |
| 发现流量单播 | 7400 + 250×域 + 10 + 2×参与者ID | 7410 |
| 用户数据单播 | 7400 + 250×域 + 11 + 2×参与者ID | 7411 |

防火墙只放行 7400 是跨机不通的高频原因；同一台机器上开很多进程时，参与者 ID 递增会占掉后面的端口，Fast DDS 有 `mutation_tries`（默认 100）来重试，超了会警告并退化成没有单播 locator。

**组播不通怎么办**：很多云环境、VPN、企业交换机关了 IGMP 组播。三条路——配 initial peers（静态单播对端列表，按上面的单播发现端口写）、用 Fast DDS Discovery Server 做集中式发现、或者换 Zenoh 走 router。Iron 起引入 `ROS_AUTOMATIC_DISCOVERY_RANGE`（`LOCALHOST` / `SUBNET` / `OFF` / `SYSTEM_DEFAULT`）控制发现范围，配 `ROS_STATIC_PEERS` 指定静态对端；Humble 及更早用 `ROS_LOCALHOST_ONLY`，这个旧变量在 Kilted 已移除。

**发现风暴**：N 个节点、每个有 M 个端点，发现流量大致按 N×M 量级增长，几十个节点、几百个 topic 的启动瞬间能把 CPU 和交换机打满。治理手段是减少端点数量（合并 topic、组件化）、静态发现、Discovery Server、限制发现范围，以及错峰启动。

## 1.6 版本地图（2026 年 11 月）

| 发行版 | 发布 | 类型 / EOL | 主平台 | 值得记住的变化 |
| --- | --- | --- | --- | --- |
| Lyrical Luth | 2026-05 | LTS / 2031-05 | Ubuntu 26.04 | callback-group events executor、rclpy AsyncNode、Gazebo Jetty |
| Kilted Kaiju | 2025-05 | 非 LTS / 2026-12 | Ubuntu 24.04 | Zenoh 升 Tier 1、EventsExecutor 转正、移除 ROS_LOCALHOST_ONLY |
| Jazzy Jalisco | 2024-05 | LTS / 2029-05 | Ubuntu 24.04 | 默认 MCAP bag、ros2_control 4.x，新项目常用 LTS |
| Humble Hawksbill | 2022-05 | LTS / 2027-05 | Ubuntu 22.04 | 存量最多的生产版本、bag 默认 SQLite、仍用 ROS_LOCALHOST_ONLY |
| Iron Irwini | 2023-05 | 已 EOL | Ubuntu 22.04 | 首次把 MCAP 设为默认 bag，引入发现范围变量 |
| Galactic | 2021-05 | 已 EOL | Ubuntu 20.04 | 唯一默认 Cyclone DDS 的版本 |

选版本时别只看代号。Humble 是存量最多的生产版本，新项目在 Jazzy / Lyrical 之间选；切版本要注意 bag 格式、发现变量和 ros2_control API 的破坏性变化。

> **小结（ROS 2 相比 ROS 1 变了什么）**：三处——架构上去了 master、靠 DDS 自动发现；通信上把可靠性/历史/存活变成可协商的 QoS，传感器能用 best effort 躲开队头阻塞；执行上把回调调度收拢到 Executor 与回调组。代价也很实在：QoS 不匹配会「静默不通」，发现流量和 XML 配置成了新的运维负担。

<!-- PAGEBREAK -->

# 第 2 章 通信机制：一条消息的完整旅程

## 2.1 一次 publish 到底发生了什么

把链路记成一条线，查延迟时顺着它逐段排除：

```text
回调里构造消息
  → rclcpp::Publisher::publish()
  → 判断有没有进程内订阅者：有则走 IntraProcessManager（可能零拷贝），
    没有则交给 rmw_publish
  → CDR 序列化（跨进程时）
  → DDS DataWriter：按 QoS 写进 history 缓存，RELIABLE 时维护 ACK/NACK
  → RTPS 打包，可配 UDP / TCP / 共享内存；大消息按 MTU 分片
  → 网络 / 共享内存
  → DDS DataReader：收包、重组、去重、按 history 排队、检查 lifespan
  → rmw_take → rcl_take
  → Executor 的 waitset 收到「有数据」事件，把对应 callback 放进就绪队列
  → 回调组判并发，Executor 挑一个线程执行订阅回调
```

三段最容易被忽略的开销：

- **序列化 + 拷贝**：大消息（4 MB 图像、几 MB 点云）在默认路径上会来回拷贝好几次，CPU 和内存带宽先于网络成为瓶颈。
- **DDS 缓存**：history 不是免费的。KEEP_ALL 又不限资源，慢消费者会让缓存无限涨；RELIABLE 重传还会把积压放大。
- **Executor 排队**：数据已经到本机了，但 callback 排在长耗时回调后面，端到端延迟照样高。很多人只盯网络，漏了这一段。

## 2.2 Topic 与类型系统

`.msg` 编译后生成三样东西：各语言的消息结构体、类型支持（typesupport，负责序列化）、IDL 描述。跨语言、跨厂商互通靠的是 IDL + CDR 这套中立表示。

新版（Iron 起逐步引入，Jazzy 后可用）加了 **type hash 与 type description**：除了 topic 名和类型名，还会校验类型定义的哈希。好处是能发现「两边都叫 `sensor_msgs/msg/Image` 但字段定义其实不同」这类悄悄改了 .msg 的事故；代价是老节点和新节点混跑时可能因为哈希不一致连不上，排查时要记得看这一项。

topic 命名与重映射：节点内用相对名，靠 namespace 和 remap 在 launch 里改，别把 `/camera/front/image_raw` 硬编码进代码——这是复用性和多机器人部署的常见坑。

## 2.3 QoS 全景：七个策略各管什么

QoS（Quality of Service，服务质量）就是发布方和订阅方之间的一份「约定」。ROS 2 暴露的常用策略：

| 策略 | 取值 | 管什么 | 典型用途 |
| --- | --- | --- | --- |
| Reliability | RELIABLE / BEST_EFFORT | 丢包要不要重传 | 图像、点云用 BEST_EFFORT；指令、参数、服务用 RELIABLE |
| Durability | VOLATILE / TRANSIENT_LOCAL | 晚加入的订阅者能不能拿到历史样本 | 地图、/tf_static、配置用 TRANSIENT_LOCAL（近似 ROS 1 的 latched） |
| History + Depth | KEEP_LAST(N) / KEEP_ALL | 缓存保留几条 | 传感器 depth 1~5；KEEP_ALL 必须配资源上限 |
| Deadline | 周期 | 期望的最大发布/接收间隔，违约触发事件 | 传感器掉线监控、控制周期监控 |
| Lifespan | 时长 | 样本多久算过期，过期直接丢 | 宁可丢旧帧也别把过期图像送进规控 |
| Liveliness | AUTOMATIC / MANUAL_BY_TOPIC + Lease Duration | 判定发布者还活着没有 | 节点假死检测、主备切换 |
| TimeBasedFilter | minimum separation | 订阅端限频（不是所有 RMW 都支持） | 下游只想按 10 Hz 拿 100 Hz 的数据 |

另外还有 ResourceLimits（max_samples 等）控制缓存上限，LatencyBudget 之类属于提示性策略，很多 RMW 直接忽略，别把它当有效手段。

## 2.4 QoS 兼容：订阅方「要求」不能高于发布方「承诺」

这是最容易踩的坑。规则叫 **RxO（Requested / Offered）**：订阅方给出的是「我能接受的最低质量」，发布方给出的是「我最多能提供到这个程度」，任何一项订阅方要求更严，DDS 就不建立关联，**不报错、不传数据、静默不通**。

| 策略 | 兼容条件 | 不通的组合（记住这个就够） |
| --- | --- | --- |
| Reliability | offered ≥ requested | 发布 BEST_EFFORT + 订阅 RELIABLE |
| Durability | offered ≥ requested | 发布 VOLATILE + 订阅 TRANSIENT_LOCAL |
| Liveliness kind | offered ≥ requested | 发布 AUTOMATIC + 订阅 MANUAL_BY_TOPIC |
| Deadline | offered 周期 ≤ requested 周期 | 发布承诺 100 ms + 订阅要求 10 ms |
| Liveliness lease | offered ≤ requested | 发布租约 5 s + 订阅要求 1 s |
| History / Depth | 不参与匹配 | 各自本地生效，不会导致连不上 |
| Lifespan | 不参与匹配 | 本地丢弃行为 |

记忆口诀：**发布方要「更可靠、更持久、更勤快」，订阅方要「更宽松」**。

要分清两件事：Deadline 在匹配阶段参与 RxO 校验（不兼容会报 incompatible_qos）；而运行期「实际超过周期」触发的是 deadline_missed 事件，那时连接已经建好了。很多人把这两个混为一谈。

**排查顺序**（照着做一遍就行）：

```bash
ros2 topic info /image_raw -v        # 看发布端和订阅端各自的 QoS 配置，直接比对
ros2 topic hz /image_raw             # 确认到底有没有数据
ros2 node info /my_node              # 看这个节点挂了哪些 pub/sub
```

再在代码里注册 QoS 事件回调：`on_incompatible_qos`、`on_deadline_missed`、`on_liveliness_changed`、`on_message_lost`。产品代码里应该把这些事件记日志或上报，否则线上「没数据」只能靠猜。

**经典事故**：用 `rclcpp::SensorDataQoS()` 发布相机（BEST_EFFORT），订阅端写 `create_subscription<Image>(..., 10)` 用了默认 RELIABLE，于是永远收不到。反过来，`ros2 topic echo` 默认也可能因为 QoS 不同看不到数据，要用 `--qos-profile sensor_data` 或 `--qos-reliability best_effort`。录 bag 时同理，录制端 QoS 不匹配会录到空文件。

**QoS 覆盖**：实在改不了代码，可以用 launch 里的 `qos_overrides` 参数，或者给 DDS 写 XML profile（`FASTRTPS_DEFAULT_PROFILES_FILE` / `CYCLONEDDS_URI`）来改默认值。这是「先跑通、后重构」的过渡手段，别当正规解法讲。

## 2.5 预设 QoS 与怎么选

| 预设 | 典型配置 | 用在哪 |
| --- | --- | --- |
| SensorDataQoS | BEST_EFFORT + VOLATILE + KEEP_LAST(5) | 相机、雷达、IMU，宁丢不堵 |
| SystemDefaultsQoS / QoS(10) | RELIABLE + VOLATILE + KEEP_LAST(10) | 一般 topic，rclcpp 默认 |
| ServicesQoS | RELIABLE + VOLATILE + KEEP_LAST(10) | 服务，必须送达、不要历史请求 |
| ParametersQoS | RELIABLE + TRANSIENT_LOCAL + 较大 depth | 参数，晚启动的节点也要拿到当前值 |
| ClockQoS | BEST_EFFORT + 特定 depth | /clock，仿真时钟高频广播 |
| /tf 与 /tf_static | /tf 用 RELIABLE + 较大 depth；/tf_static 加 TRANSIENT_LOCAL | 坐标变换，静态 TF 必须能被后来者拿到 |

选型的判断只有一句：**这条数据丢了能不能用下一帧补上**。能补（传感器流、TF 流）就 BEST_EFFORT 加小队列；不能补（指令、配置、任务目标）就 RELIABLE；新节点加入时必须补上历史（地图、静态 TF、参数）才加 TRANSIENT_LOCAL。

RELIABLE 不是「更好」：网络一抖，重传会让队列堆积，新数据排在旧数据后面，端到端延迟可能从几毫秒涨到几百毫秒。自动驾驶里「宁可要 30 ms 前的新帧，也不要 300 ms 前的完整帧」，就是这个道理。

## 2.6 Service、Action、Parameter 的底下还是 topic

**Service** 是一对请求/响应，底层是两个 DDS topic（request 与 reply）。请求里带上客户端标识和序列号，服务端用来把响应送回正确的客户端。默认 RELIABLE + VOLATILE，避免服务重启后收到一堆过期请求。

- 只适合**低频、小数据、要结果**的调用：开关传感器、查询状态、触发一次标定。
- 高频或大数据别用 service：一次调用要来回两跳、序列化两次，还会占着回调线程。
- **同步调用是死锁之王**：在单线程 Executor 的回调里 `future.get()` 或 `spin_until_future_complete()` 等一个由同一 Executor 处理响应的 service，会永远等下去。第 3 章细讲。

**Action** 是给「跑几分钟还要能取消」的长任务用的：导航到某点、机械臂抓取。底层是 **3 组 service + 2 个 topic**——发送目标、取消目标、取结果各一组服务，加上进度反馈（feedback）和状态广播（status）两个 topic。有人笼统说「Action 是 5 个 topic」，指的是这些通道，讲的时候说清 3+2 更准。它比 service 多出来的价值是：可抢占、能持续回报进度、有明确状态机（ACCEPTED / EXECUTING / CANCELING / SUCCEEDED 等）、目标有唯一 ID。

**Parameter** 是每节点自带的一组参数服务（get / set / list / describe）加一个 `parameter_events` topic。参数要 `declare_parameter()` 才能用，类型一旦声明就固定；`on_set_parameters` 回调可以拒绝不合法的修改。参数事件 topic 用较大 depth + TRANSIENT_LOCAL，保证监控节点晚启动也能拿到当前值。产品里所有可调项都应该走参数 + YAML，而不是散落在代码常量里——这是「能不能灰度调参」的分水岭。

## 2.7 进程内通信与零拷贝：别把「开了」当成「生效了」

这一块最容易判断错。三种层次要分清楚：

**层次一：进程内通信（intra-process）**。多个节点放进同一个进程（组件容器），且 `NodeOptions().use_intra_process_comms(true)`，rclcpp 的 IntraProcessManager 会绕过 rmw 的数据通道，直接把消息交给同进程的订阅者。省掉序列化和 DDS 那圈开销。

**层次二：真正零拷贝**。哪怕在同一进程，也要**发布 `std::unique_ptr<T>`、订阅端也用 unique_ptr 回调**，消息所有权才会被 move 过去，地址不变。如果发布 `const T&` 或 `shared_ptr`，或者同一个消息有多个订阅者，rclcpp 就得复制一份给每个订阅者——零拷贝退化成拷贝。想验证，最简单就是打印发布和订阅两边消息的地址，看是否相同。

**层次三：跨进程零拷贝**。靠 DDS 的共享内存传输（Fast DDS DataSharing / SHM、Cyclone 的 iceoryx 集成）配合 **loaned message**：先向中间件借一块共享内存，直接往里填数据，再发布，接收方读同一块内存。

```cpp
// 借一块中间件的内存，填完直接发布，全程不经过用户态拷贝
if (pub_->can_loan_messages()) {
    auto loaned = pub_->borrow_loaned_message();
    fill_pointcloud(loaned.get());          // 直接写共享内存
    pub_->publish(std::move(loaned));
} else {
    pub_->publish(*make_message());        // 退回普通路径
}
```

**生效条件（容易漏掉的是这几条）**：

- 节点选项开了进程内通信，或 RMW 支持并开启了共享内存；
- 两端 QoS 兼容，且订阅方在同一进程 / 同一共享内存域；
- 消息类型能被 loan——通常是定长、无变长字段的类型。**含 `string`、无界 `sequence` 的消息一般借不到内存**，会退回普通发布；
- 用 `borrow_loaned_message()`，检查 `can_loan_messages()`，别静默假设成功；
- 借来的内存在 `publish` 之后就不能再碰，生命周期由中间件管；多个订阅者共享时还要考虑谁先释放。

再补一个常被忽略的点：即使数据零拷贝了，如果发布端和订阅端跑在**两个独立 Executor**，中间仍有线程切换和队列调度开销。同进程 + 单线程 Executor 的内联执行才是最顺的那条路。

## 2.8 边上但常用的四块

**时间与仿真时钟**：`use_sim_time=true` 后，节点的时间来源从系统时钟改成 `/clock` topic。TF 查找、定时器、bag 回放都依赖它。典型事故：仿真里 TF 报错「extrapolation into the future」，十有八九是某个节点忘了设 `use_sim_time`，或者回放时没加 `--clock`。

**TF2**：维护一棵坐标变换树，`tf2_ros::Buffer` 缓存一段时间内的变换，支持按时间戳插值。`/tf` 是动态变换流，`/tf_static` 是静态变换（TRANSIENT_LOCAL，晚加入也能拿到）。高频 `lookupTransform` 会成为 CPU 热点，控制环里应该缓存或降频。

**rosbag2**：Iron 起默认 MCAP 格式，Humble 及更早是 SQLite。录制时每个 topic 都要协商 QoS，**录制端的 QoS 与发布端不兼容就录不到数据**；`--all` 会为所有 topic 单独建订阅，topic 多时磁盘 IO 和订阅开销都不小。回放用 `--clock` 发布仿真时间、可变速、可暂停，是复现线上问题的主力工具。

**SROS2 / DDS Security**：通过 `ROS_SECURITY_ENABLE`、`ROS_SECURITY_ENCLAVE`、`ROS_SECURITY_STRATEGY` 启用，底层用 DDS Security 的三个能力：认证（对端是谁）、加密（链路）、访问控制（这个节点能不能发这个 topic）。每个 enclave 一套证书与权限文件，产品里应按最小权限生成，别一把全开。多机器人同网段时，域隔离解决「串台」，安全机制解决「伪造」。

## 2.9 通信故障速查

| 症状 | 最可能的原因 | 先查什么 |
| --- | --- | --- |
| 订阅收不到，发布端明明在发 | QoS 不兼容 | `ros2 topic info -v` 对比两端，看 incompatible_qos 事件 |
| 本机正常、跨机不通 | 组播被禁 / 防火墙只放了一个端口 / 域不同 | 端口公式、initial peers、`ROS_DOMAIN_ID`、IGMP |
| 前几帧总是丢 | 发布早于订阅建立，且 VOLATILE | 改 TRANSIENT_LOCAL，或在 on_activate 后延迟发布 |
| 延迟随运行时间涨 | RELIABLE 重传积压 / KEEP_ALL 缓存堆积 | 队列深度、丢包率、监控 RSS |
| 偶发拿到很旧的数据 | Lifespan 没设，慢消费者读历史 | 设 Lifespan、改 KEEP_LAST 小深度 |
| 启动瞬间 CPU 与网络飙高 | 发现风暴 | 减少端点、静态发现、Discovery Server、错峰启动 |
| 多个机器人互相干扰 | 同一域、没隔离 | 分域、限制发现范围、Zenoh router |

> **小结（QoS 为什么是核心设计）**：它把可靠性从传输层搬到了应用可配的策略里。代价是不兼容会静默不通，所以工程上要显式指定 QoS、注册 QoS 事件、把 `ros2 topic info -v` 写进排障手册。

<!-- PAGEBREAK -->

# 第 3 章 节点架构与执行模型

## 3.1 一个 Node 里到底有什么

Node 不是「一段能收发的程序」，而是一个实体容器：发布者、订阅者、服务端、客户端、定时器、参数、回调组、时钟、日志器，都挂在它名下。它同时是 DDS 里的一个参与者（或共享参与者）和命名空间边界。

两个工程习惯值得养成：

- **实体在构造函数或 `on_configure()` 里一次性建好**，别在回调里动态建订阅、建定时器。既容易泄漏，也会让 StaticSingleThreadedExecutor 这类实现「看不见」新实体。
- **别在构造函数里立刻 publish**：订阅端可能还没发现你、还没建好关联，头几帧会丢。要么放到 `on_activate()`，要么用 TRANSIENT_LOCAL。

## 3.2 组合式节点（Composition）：一个进程装多个节点

默认的 `ros2 run` 是「一个节点一个进程」。组件化把节点编译成动态库，运行时由 `component_container` 装载进同一个进程。

它值钱的地方不只是省几个进程：**只有同进程才可能走进程内通信和零拷贝**，所以大消息链路（相机 → 预处理 → 推理）组件化往往能拿到成倍的收益。此外启动更快、发现端点更少、内存占用更低。

代价也要说：一个节点崩溃可能带走整个容器，隔离性变差；所有节点共享同一个进程内 Executor 或需要显式分组；全局变量、静态单例在组件化后会互相污染。产品上的常见做法是**按故障域分组**——把强耦合、大数据量的节点放一个容器，把安全相关的控制节点单独放。

## 3.3 生命周期节点：让启动顺序可控

普通节点构造完就开始收发，几十个节点同时启动时，「谁先谁后」「依赖没就绪怎么办」全靠 sleep 猜。Lifecycle Node（托管节点）给了一个明确状态机：

| 状态 | 含义 | 典型动作 |
| --- | --- | --- |
| Unconfigured | 刚创建，只有实体，没占资源 | 什么都没打开 |
| Inactive | 已配置，资源就绪但不处理数据 | on_configure：读参数、开设备、分配内存、建实体 |
| Active | 正常工作 | on_activate：开始 publish、启用控制输出 |
| Finalized | 已销毁，只能重建 | on_shutdown / on_error |

过渡回调：`on_configure` / `on_activate` / `on_deactivate` / `on_cleanup` / `on_shutdown` / `on_error`。外部用 `ros2 lifecycle set /node configure|activate` 驱动，nav2 的 lifecycle manager 就是按依赖顺序把整栈节点逐个拉起来的。

最要紧的一句：**去激活必须先让机器人进安全状态**。`on_deactivate` 里要发零速度、关执行器，否则节点停了电机还保持着上一条速度指令，这是真实会伤人的坑。

## 3.4 Executor：谁来跑回调

Executor 是 ROS 2 里最容易被低估的组件。它做三件事：

1. 收集节点的订阅、定时器、服务、客户端等实体，交给中间件的 **waitset** 等待；
2. 有事件就绪时，把对应的回调放进就绪队列；
3. 按回调组规则和线程数，挑线程执行回调。

关键认知：**消息到达 ≠ 回调执行**。数据可能已经在本机内存里了，但回调还在排队。所以「端到端延迟」必须包含 Executor 排队时间，调优时不能只盯网络。

四种实现：

| Executor | 线程 | 特点 | 适用 |
| --- | --- | --- | --- |
| SingleThreadedExecutor | 1 | 所有回调串行，无数据竞争；一个慢回调卡住全部 | 简单节点、确定性优先 |
| MultiThreadedExecutor | N | 按回调组并发；每次循环重建 waitset，开销较大 | 有耗时回调的通用场景 |
| StaticSingleThreadedExecutor | 1 | spin 前扫描一次实体，省掉反复重建；动态新增实体不会被发现 | 拓扑固定的低延迟链路 |
| EventsExecutor | 可配 | 事件驱动，不再反复重建 waitset，支持回调组粒度调度；Kilted 起转正 | 高频、低抖动、新项目 |

EventsExecutor 值得单独记一笔：传统 Executor 每轮都要重新收集实体、重建 waitset，节点数一多这块开销很可观；EventsExecutor 用事件队列推送就绪通知，把复杂度从「每轮 O(实体数)」降下来，是 Kilted / Lyrical 在实时方向的主要改进。

## 3.5 回调组：并发的最小单位

每个回调都归属于一个 CallbackGroup，两种类型：

- **MutuallyExclusive（默认）**：同一组内，同一时刻只能跑一个回调。
- **Reentrant**：同一组内允许多个回调并发。

有三条隐含规则常被问到：

1. 没显式指定组的实体，全落在节点的**默认互斥组**。所以在 MultiThreadedExecutor 里，一个节点内的普通订阅、定时器、服务依然互相串行——多线程不等于并行。想要并行，得显式建组。
2. 跨组之间可以并发，不管组是互斥还是可重入。
3. Reentrant 组意味着你的回调必须自己保证线程安全（成员变量加锁、无锁结构）。

配置方式：订阅/服务/客户端通过 `SubscriptionOptions.callback_group` 指定，定时器用 `create_wall_timer(..., group)`。经验法则是**按「资源」和「耗时」分组**：抢同一硬件或同一状态对象的放互斥组；耗时图像处理放独立组或独立节点；控制环独占一组，别和日志、IO 混在一起。

## 3.6 五个经典坑

**坑一：单线程 Executor 里同步等 service，死锁。** 响应的回调也要靠同一个 Executor 执行，而 Executor 正被 `future.get()` 占着。

```cpp
// 错：响应回调没机会跑，永久阻塞
void on_timer() {
  auto future = client_->async_send_request(req);
  auto resp = future.get();
}

// 对：异步回调，马上让出 Executor
void on_timer() {
  client_->async_send_request(req,
    [this](rclcpp::Client<MySrv>::SharedFuture f) { use(f.get()); });
}
```

**坑二：以为换成 MultiThreadedExecutor 就解决了。** 如果 client 和发起调用的服务回调挂在同一个默认互斥组，发起方占着组锁等响应，响应回调进不了这个组，照样死锁。要把 client 放进独立的互斥组。

**坑三：回调里 `sleep`、同步 IO、锁等待。** 占住的是 Executor 线程，不只是自己的回调。改用 `create_wall_timer`、独立工作线程，或者把耗时任务丢到线程池，回调只做投递。

**坑四：回调里动态创建实体。** 资源泄漏 + StaticSingleThreadedExecutor 感知不到。所有实体在构造 / on_configure 阶段建好。

**坑五：线程数越多越快。** 线程数超过实际并发度后只剩上下文切换和锁竞争。先按「CPU 核数 × 1~1.5」起步，用 tracing 看回调等待时间再调。

还有一个不那么经典但值得提的：MultiThreadedExecutor 早期版本存在**回调饥饿**——某些就绪任务长期抢不到线程。遇到某条链路偶发长延迟但 CPU 不满，可以用 ros2_tracing 看回调排队时长，必要时换 EventsExecutor 或把关键链路拆到独立 Executor。

## 3.7 ROS 2 能硬实时吗

诚实答案：**ROS 2 本身不保证硬实时，它只是「实时友好」**。它提供了可预测的前提——静态类型、可预分配、Executor 可定制、大部分 API 不在回调里偷偷分配内存，但通用 Linux、通用 DDS、动态发现、日志 IO 都会引入抖动。

工程上的做法是把硬实时那段**挪出中间件**：

- 1 kHz 控制环放在独立的实时线程里（ros2_control 的 controller_manager 就是这么干的），用无锁环形缓冲或实时安全的队列和 ROS 通信线程交换数据；控制环内不 `publish()`、不拿锁、不写日志、不分配内存。
- 系统层配 PREEMPT_RT 内核、`SCHED_FIFO` 优先级、`mlockall(MCL_CURRENT|MCL_FUTURE)` 锁内存避免缺页、CPU 隔离（`isolcpus` / `nohz_full`）、`performance` 调频、关 swap。
- 内存全部预分配并预热（先跑一遍让页表就位），避免运行期 `new`、`std::vector` 扩容、`std::string` 拼接。
- 用 `cyclictest` 量内核调度抖动，用 ros2_tracing 量回调端到端抖动，别只测平均延迟。

平均延迟没意义，要看 p99 和最坏情况——这一条比记一堆配置管用。

## 3.8 一段可以直接用的代码骨架

```cpp
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
using namespace std::chrono_literals;

class FusionNode : public rclcpp::Node {
public:
  explicit FusionNode(const rclcpp::NodeOptions & opts)
  : Node("fusion", opts) {
    // 高频传感器：宁可丢帧，也不要被重传堵住
    auto qos = rclcpp::SensorDataQoS();

    // 控制回调与耗时回调分组，避免互相拖慢
    control_group_ = create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    heavy_group_ = create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);

    rclcpp::SubscriptionOptions sub_opt;
    sub_opt.callback_group = heavy_group_;

    // unique_ptr 回调 + 进程内通信，才能真正零拷贝
    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "points", qos,
        [this](std::unique_ptr<sensor_msgs::msg::PointCloud2> msg) {
          process(std::move(msg));
        }, sub_opt);

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "fused", qos);

    timer_ = create_wall_timer(10ms, [this] { tick(); },
                               control_group_);
  }
private:
  void tick() { /* 只做投递，不做耗时计算 */ }
  void process(std::unique_ptr<sensor_msgs::msg::PointCloud2> m) { /* ... */ }

  rclcpp::CallbackGroup::SharedPtr control_group_, heavy_group_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions opts;
  opts.use_intra_process_comms(true);   // 组件化装载时才有意义
  auto node = std::make_shared<FusionNode>(opts);
  rclcpp::executors::MultiThreadedExecutor exec(
      rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
}
```

这段代码把 QoS、回调组、unique_ptr、进程内通信、多线程 Executor 五处设置串在一起，可以当作写节点时的模板。

> **小结（Executor 解决了什么）**：ROS 1 里每个节点自己起线程，没法统一控制优先级和并发度。ROS 2 把「等事件」和「执行回调」收拢到一个可替换的 Executor，于是能选单线程/多线程、用回调组声明并发约束、换成事件驱动实现。代价是单线程里一个慢回调卡全场，回调组配错会死锁。

<!-- PAGEBREAK -->

# 第 4 章 性能调优与资源管理

## 4.1 先量化，再动手

调优最忌讳「感觉慢，先开个共享内存」。先把指标和工具摆齐：

| 要看什么 | 具体指标 | 工具 |
| --- | --- | --- |
| 时延 | 端到端 p50 / p95 / p99、最坏值、抖动分布 | `ros2 topic delay`、消息内打时间戳、ros2_tracing |
| 完整性 | 丢包率、乱序、deadline miss 次数 | QoS 事件回调、序列号统计 |
| 吞吐 | 频率、带宽 | `ros2 topic hz` / `bw`、iperf3 |
| 计算 | 各核 CPU、回调执行与排队时长、上下文切换 | ros2_tracing + LTTng / TraceCompass、perf、eBPF |
| 内存 | RSS 增长曲线、分配次数、DDS 缓存占用 | ps / smem、valgrind massif、DDS 内部统计 |
| 系统 | 内核调度抖动、网卡丢包、socket 缓冲溢出 | cyclictest、`ss -ulpn`、ethtool -S、`netstat -su` |

`ros2_tracing`（LTTng）是这一层最有价值的东西：它能把「发布 → DDS 写入 → 网络 → 取数据 → 回调开始 → 回调结束」整条链上的时间戳打出来，直接看出时间花在哪一段。没有它，多数人只能靠猜。

**基准要可复现**：固定消息大小、频率、topic 数、机器负载，跑够时长（分钟级），记录 p99 而不是平均。改一个变量测一次，别一次改五项然后说「快了 30%」。

## 4.2 七层调优清单（按收益从高到低）

**第一层：架构与数据路径。** 收益最大，改起来也最贵。
把强耦合的大数据链路组件化到同一进程，开 `use_intra_process_comms(true)`；发布/订阅都用 `unique_ptr`；跨进程的大消息走共享内存 + loaned message；合并过细的 topic，减少跳数；图像能用压缩或 ROI 就别发全帧；把 4 MB 图像从「每帧都发」改成「按需发」，比任何 DDS 参数都管用。

**第二层：QoS。**
高频传感器一律 BEST_EFFORT + KEEP_LAST 小深度（1~5）；给传感器数据设 Lifespan，宁丢旧帧；控制指令、参数、服务保持 RELIABLE；静态数据用 TRANSIENT_LOCAL 但**别对大消息用**，否则历史样本会把发布端内存撑爆；用 Deadline 和 Liveliness 做监控，而不是当调速手段。

**第三层：DDS / RMW。**
按场景换实现（Cyclone 轻、Fast DDS 功能全、Zenoh 适合跨网）；打开共享内存传输；用 XML profile 限制 DDS 线程数量与优先级、设 `max_samples` 等资源上限、配 reliable 的心跳与重传间隔；组播不可用时配静态 peer 或 Discovery Server；大消息关注分片（fragmentation）与异步发布模式，避免 publish 调用被网络 IO 卡住。

**第四层：网络与操作系统。**
增大 UDP 收发缓冲（`net.core.rmem_max` / `wmem_max`、`SO_RCVBUF`），否则突发大消息会在内核里被丢；核对 RTPS 知名端口放行范围；条件允许时用巨帧 MTU 9000 减少分片，但要全链路一致；网卡中断绑核、关掉 CPU 节能调频；跨机用 PTP/NTP 对齐时钟，否则延迟测量和 TF 插值都不准。

**第五层：执行器与线程。**
用 MultiThreadedExecutor 或 EventsExecutor；按资源/耗时拆回调组；把阻塞回调踢出 Executor 线程；控制链路单独一个 Executor 甚至单独进程；线程数贴合核数；定时器别用 `sleep` 模拟周期；拓扑固定时考虑 StaticSingleThreadedExecutor。

**第六层：内存与分配。**
消息预分配、复用 buffer、避免 `std::vector` 扩容和 `std::string` 拼接；用 loaned message 直接填中间件内存；定长类型优先（变长类型借不到 loan）；DDS history 用 KEEP_LAST 并设 ResourceLimits；实时线程 `mlockall` 锁内存、启动时预热；长期运行看 RSS 曲线，实体反复创建销毁是典型泄漏。

**第七层：资源治理。**
限制单节点 topic / 实体数量；关闭不用的 topic；日志设级别、用异步、别在高频回调里 `RCLCPP_INFO`；bag 录制单独磁盘或限流，避免写盘抖动打进控制环；用 cgroup 给非关键节点限额；发布端给 RELIABLE 队列设上限，防止慢订阅者把整机内存吃光。

> 调优别一次改五项。挑一条链路，按「先量 → 定位到某一段 → 改这一层 → 复测 p99」走一遍，比一次列二十个参数管用。

## 4.3 大数据专题：图像与点云

4 MB 的图、几 MB 的点云，默认路径上会经历「用户态拷贝 → 序列化 → DDS 缓存 → 网络/共享内存 → 反序列化 → 再拷贝」，CPU 常常先于带宽饱和。

按效果排序的做法：

1. **别发**：降分辨率、降帧率、发 ROI、发压缩格式（image_transport）、下游按需订阅。
2. **同进程**：组件化 + 进程内通信 + `unique_ptr` 发布订阅，直接把所有权 move 过去。零拷贝这时最稳。
3. **跨进程**：共享内存传输 + `borrow_loaned_message()`，但要确认类型可 loan、RMW 支持、缓冲池够大；含 string / 无界数组的类型会退回拷贝路径。
4. **传输**：巨帧、增大 UDP buffer、避免 RELIABLE 重传放大；多订阅者场景注意每个订阅者各有缓存。
5. **别在回调里做图像处理**：把耗时计算放到工作线程或 GPU 流水线，回调只负责投递指针。

一个好用的验证方法：在消息里塞一个字段打印地址，或者在两个节点里同时打印 `msg.get()`，地址相同才是真零拷贝。同时用 `ros2 topic bw` 和 CPU profile 交叉验证——开了共享内存但 CPU 没降，多半是某个环节悄悄复制了。

## 4.4 1 kHz 控制环专题

抖动通常来自这五处，按排查顺序：

1. **内核调度**：非 RT 内核、节能调频、其他进程抢核 → cyclictest 量，上 PREEMPT_RT + CPU 隔离 + SCHED_FIFO。
2. **缺页与分配**：第一次访问新内存触发页错误，`new` / vector 扩容 → 预分配、`mlockall`、启动预热。
3. **Executor 排队**：控制回调和图像回调同一个组 → 拆组、拆 Executor、拆进程。
4. **中间件在环内**：控制环里调 `publish()`、拿 DDS 锁、写日志 → 控制环只写实时安全缓冲，由通信线程转发。
5. **DDS 与网卡中断**：DDS 后台线程、网卡 IRQ 和 RT 线程抢同一个核 → 绑核、限制 DDS 线程数与优先级。

目标要具体：比如「1 kHz 周期，p99 抖动 < 100 µs，最坏 < 1 ms」，`cyclictest` 测内核底噪、tracing 测回调抖动、示波器或硬件时间戳测真实输出。ros2_control 的 controller_manager 已经把「实时线程 + 非实时 ROS 接口」这层分离做好了，能讲清它的分层比手写 RT 代码更讨喜。

## 4.5 发现风暴与资源治理

节点数一多，SPDP/SEDP 交换的端点信息量按「节点 × 端点」增长，启动瞬间组播风暴、CPU 尖峰、部分节点发现不全，是集群部署最常见的「玄学问题」。

治理顺序：减少端点（组件化、合并 topic、关掉不用的发布者）→ 缩小发现范围（`ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET`、`ROS_STATIC_PEERS`）→ 静态发现 / Discovery Server / Zenoh router → 错峰启动 → 必要时拆分域。

资源侧的几根红线：KEEP_ALL 不设上限、TRANSIENT_LOCAL 存大消息、RELIABLE 队列无上限、回调里反复建实体、日志在高频路径同步写。它们共同的表现是「跑几小时后内存涨、延迟变糟、重启又好了」。产品里应该给 DDS 缓存、线程数、日志速率、bag 写盘速率都设上限，并把 RSS、丢包数、deadline miss 做成常驻监控。

## 4.6 瓶颈速查表

| 症状 | 大概率位置 | 定位动作 | 手段 |
| --- | --- | --- | --- |
| 大消息时 CPU 高、带宽没满 | 序列化与拷贝 | perf 看 CDR / memcpy 占比 | 组件化、loaned message、SHM |
| 延迟 p99 远大于 p50 | Executor 排队或 GC/缺页 | tracing 看回调等待 | 拆回调组、独立 Executor、预分配 |
| 网络一抖延迟暴涨 | RELIABLE 重传积压 | 看丢包与队列深度 | BEST_EFFORT、Lifespan、小 depth |
| 跨机丢包但带宽不高 | UDP 缓冲溢出 | `netstat -su`、`ss` 看 drop | 增大 rmem_max / wmem_max |
| 启动瞬间卡顿，之后正常 | 发现风暴 | 抓组播包、看 CPU 尖峰 | 静态发现、减端点、错峰启动 |
| 跑久了越来越慢 | 缓存 / 内存泄漏 | RSS 曲线、DDS 统计 | KEEP_LAST + ResourceLimits、修实体泄漏 |
| 仿真里 TF 报未来时间 | 时钟没统一 | 查各节点 use_sim_time | 全局设 use_sim_time，回放加 --clock |

## 4.7 环境变量与配置速查

| 变量 | 作用 |
| --- | --- |
| `ROS_DOMAIN_ID` | DDS 域隔离，多机器人必配 |
| `RMW_IMPLEMENTATION` | 选中间件实现 |
| `ROS_AUTOMATIC_DISCOVERY_RANGE` | Iron 起引入、Jazzy 后成为主流：`LOCALHOST` / `SUBNET` / `OFF` / `SYSTEM_DEFAULT` |
| `ROS_STATIC_PEERS` | 组播不通时指定静态单播对端 |
| `FASTRTPS_DEFAULT_PROFILES_FILE` | Fast DDS XML profile 路径 |
| `CYCLONEDDS_URI` | Cyclone DDS 配置路径 |
| `ROS_SECURITY_ENABLE` / `ROS_SECURITY_ENCLAVE` / `ROS_SECURITY_STRATEGY` | SROS2 安全开关、安全域、策略 |
| `RCUTILS_CONSOLE_OUTPUT_FORMAT` | 日志格式（加时间戳便于对延迟） |
| `ROS_LOCALHOST_ONLY` | 旧版变量，Humble 用，Kilted 已移除 |

> **小结（延迟高怎么查）**：先定义延迟是哪一段，再用时间戳或 tracing 分段；按「应用构造 → 序列化 → DDS 缓存 → 网络 → Executor 排队 → 回调执行」逐段排除；定位后按七层清单挑对应手段，改一项复测一次 p99。动手前先排除 QoS 不兼容和测量方法本身的问题。

<!-- PAGEBREAK -->

# 第 5 章 工程实践与速查

## 5.1 一条「感知 → 规控」链路的设计参考

拿「一台车上，相机 + 激光雷达 + 定位 + 规控」这条链做例子，把 topic、QoS、进程划分和调优手段摆在一起看：

```text
[相机驱动 30Hz]  ──/image_raw  BEST_EFFORT depth 2 ──┐
[雷达驱动 10Hz]  ──/points     BEST_EFFORT depth 2 ──┼→ [感知进程] ──/objects
[定位 /tf]       ──/tf        RELIABLE, /tf_static TRANSIENT_LOCAL ─┘     │
                                                                           │
   /objects  RELIABLE? 通常 BEST_EFFORT + Lifespan 100ms + depth 1           │
                                                                           ▼
                                                        [规控进程] ──/cmd
                                                                           │
   /cmd   RELIABLE + depth 1（指令不能丢）                                  ▼
                                                        [底盘控制 1kHz RT 线程]
```

按四层拆开看：

1. **进程划分**：相机驱动 + 预处理 + 推理放一个组件容器，走进程内通信和零拷贝；规控单独进程，底盘控制独立进程内放 RT 线程；故障域隔离。
2. **QoS**：原始传感器 BEST_EFFORT + 小 depth + Lifespan；目标列表 BEST_EFFORT + Lifespan（过期目标比丢包更危险）；控制指令 RELIABLE + depth 1；静态 TF 和地图 TRANSIENT_LOCAL；参数 RELIABLE + TRANSIENT_LOCAL。
3. **执行器**：感知容器用 MultiThreadedExecutor 或 EventsExecutor，耗时推理单独回调用独立组；规控用单线程或独立 Executor 保确定性；RT 线程不碰中间件，通过实时安全缓冲与 ROS 线程交换。
4. **调优与兜底**：大消息组件化 + loaned message / SHM；Deadline 与 Liveliness 监控传感器掉线并触发降级；ros2_tracing 量 p99；bag 录制限流到单独磁盘；多车部署分域 + SROS2。

还有两个容易忽略的地方：感知输出别用 RELIABLE（旧目标堆在队列里比丢一帧更危险）；控制指令别用 BEST_EFFORT，但也不能让重传把队列堆起来，所以 depth 要小、要有超时兜底。

## 5.2 十条要点速记

1. 五层：rclcpp → rcl → rmw → DDS → 传输；DDS 是标准，RTPS 是线协议。
2. RxO 口诀：发布更可靠、更持久、更勤快，订阅更宽松；不兼容静默不通。
3. 传感器 BEST_EFFORT，指令 RELIABLE，静态数据 TRANSIENT_LOCAL。
4. 零拷贝三条件：同进程 / 共享内存、QoS 兼容、`unique_ptr` 或 loan；变长类型借不到内存。
5. 消息到达 ≠ 回调执行，延迟里有 Executor 排队。
6. 默认回调组是节点级互斥组，多线程 Executor 里单节点仍串行。
7. 单线程 Executor 同步等 service 必死锁，异步 + 独立组。
8. Action = 3 组 service + feedback / status 两个 topic。
9. ROS 2 不保证硬实时，RT 环要移出中间件，看 p99 不看平均。
10. 调优顺序：先量 → 分段 → 架构/进程内 → QoS → DDS → 网络 → 执行器 → 内存 → 治理。

## 5.3 常用命令

```bash
# 看拓扑与 QoS
ros2 node list
ros2 node info /node_name
ros2 topic list -t
ros2 topic info /image_raw -v          # 排 QoS 不兼容的第一命令
ros2 topic hz /image_raw
ros2 topic bw /image_raw
ros2 topic delay /objects
ros2 interface show sensor_msgs/msg/Image
ros2 doctor --report

# 用指定 QoS 订阅 / 录制
ros2 topic echo /image_raw --qos-reliability best_effort
ros2 bag record --topics /image_raw /points -o run01
ros2 bag play run01 --clock
ros2 bag info run01

# 生命周期与组件
ros2 lifecycle list /map_server
ros2 lifecycle set /map_server activate
ros2 component types
ros2 component load /ComponentManager composition composition::Talker

# 系统与网络
cyclictest -p 99 -i 200 -l 10000
ss -ulpn | grep 74
netstat -su
ethtool -S eth0
iperf3 -s / iperf3 -c <host> -u -b 1G
chrt -f 90 ./control_node
taskset -c 2,3 ./control_node
```

## 5.4 延伸阅读（官方优先）

- ROS 2 官方文档 `docs.ros.org`：Concepts 章节把 QoS、Executor、生命周期讲得最准。
- ROS 2 Design `design.ros2.org`：为什么这样设计，讲「为什么不自己写」这类设计取舍时好用。
- REP-2000：发行版与平台支持周期。
- Fast DDS Documentation：Listening Locators / Well Known Ports、Discovery Server、共享内存与 XML profile。
- Cyclone DDS / Zenoh 官方文档：跨网与轻量场景。
- `ros2_tracing` 与 ROSCon 的 real-time / performance 议题：调优与实测数据的主要来源。

---

这份指南刻意不堆 API 用法。工程里真正麻烦的，几乎都是「为什么这样设计」「不这么做会怎样」「你怎么验证」。

把每个结论都配上一条失效场景，答案就站得住了。
