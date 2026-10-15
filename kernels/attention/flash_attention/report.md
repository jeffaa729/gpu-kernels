### Reproducing FlashAttention 3-4 Kernel in H100 with raw CUDA and PTX instructions.

Before this tutorial , you should familiar with all the fundamental of flashattention 1-4 , online softmax, tensor core and cuda (better to know a bit about ptx instructions)

The Hopper optimized Flash Attention kernel can be also break down to different optimization techinique, it is also similar to gemm in h100.

- target and boundaries
- Online Softmax
- Tensor core
- CTA
- TMA
- shared descriptor
- barrier
- tma preload
- qk : shared wgmma
- pv : register wgmma
- async fetch , wait
- causal mask and skipping
- online softmax constant
- warp specification
- ping pong 
- L2, static longest first
- launch kernel , output , epligoue

### 1 target input and output, what should be fused into kernel?
Attention : Softmax(QKT) V
where Q with shape (B,T,Hq,D). K,V with shape(B,T,Hkv,D)
where B : batch , T : seq length, D : head dimension, output type should be bf16. Note that LSE LogSumExp is FP32.

To simipify the implementaito, we currently only consider MHA where Hq = Hkv
skipping explaination the basics of FA, a lot of online sources are here. 
At the end, FA aims to fuse all these operations into 1 kernel launch : 
- QKT
- scale
- causal mask
- online softmax 
-> the original naive softmax is the main reason why the attention cannot fuse into 1 kernel
- PV
- BF16 output and FP32 logsumexp
so the kernel wont materialize a full S and P matrix.
for Q tile : load K0V0 -> QK0T -> update softmax -> accumulate P0V0
            load K1V1 -> QK1T -> update softmax -> accumulate P1V1
            load K2V2 -> QK2T -> update softmax -> accumulate P1V2
            normalize accumulated output
Note for fp32 LSE output : xxxxxx
kernel fused and unfused:

进入 kernel 前
────────────────────────
Q/K/V linear projection
RoPE
Q/K/V tensor preparation
────────────────────────
FlashAttention kernel
QKᵀ
scale
causal mask
online softmax
PV
output normalization
LSE
────────────────────────
离开 kernel 后
output projection
residual connection
RMSNorm
MLP/MoE

### 2 how these state qkv, managed by thread, registers 
to understand the tiling stragtegy in flashattention , we need to know CTA, Warp Group, and the registers.
for 1 batch and 1 query head, input Q is Q (T, 128)
each row is 1 token's query vector, One CTA take the consecutive 128 row -> so Qcta-> (128, 128) -> so each CTA do 128 query token, and each token has 128 D head dimensions;

In a CTA, each read a 128 Key/Value token , for computing Qcta * KtileT = S

A consumer Warp Groups seperate by query rows. so CTA : C0 -> 0-63 query and C1 : 64-128 output U[0:64, :] and U[64:128, ] respectively.

Inside this 128x128 tile, how they actually distributed into threads?

recall now 
CTA : 128 x 128
-> C0 : 64x128, 128 threads -> 4warps : warp 0 : 16x 128 .... warp 3 : 16x128
-> C1 : 64 x 128 , 128 threads -> same 
so each thread get 64 element(16 x 128 / 32 = 64)
1 warp 32 lanes also can divide as 8 quad where Quad 1 : lane 0-3, .... Quad 7 : Lane 28-31
for example, for warp 0, r0 = warp 8 16 + lane / 4, so for quad 0, r0 = 0, r1 = 8,.... quad 7 r0 = 7, r1 = 15.

Note : the element that 1 thread got is not continuous. Why seperate 8 rows? this is related to a later instructoion WGMMA Accumulator layout. 

for cols : c = g * 16 + 2 * (lane & 3);
note : g * 16 : select which 16 cols
        2 * (lane & 3) : start of col of current lane in a Quad
example : warp = 0, quad = 0, g = 0
Lane	\(c\)	每一行持有的列
0	0	0、1、8、9
1	2	2、3、10、11
2	4	4、5、12、13
3	6	6、7、14、15     

combine 4 lane, they compute 16 col. each lane has 4 col in row 0 and row 8. so each lane now has 2 x 4 = 8. 

以 Lane 0 为例，s[0][slot] 的实际坐标是：
slot	矩阵坐标 (row, col)
0	(0, 0)
1	(0, 1)
2	(8, 0)
3	(8, 1)
4	(0, 8)
5	(0, 9)
6	(8, 8)
7	(8, 9)

当 g 增加，列坐标整体增加 16。例如 Lane 0：
g=0：  0,  1,  8,  9
g=1： 16, 17, 24, 25
g=2： 32, 33, 40, 41
...
g=7：112,113,120,121
so at the end, each lane has 8 x 2 x 4 = 64 elements


### 3 Q, K , V how they aligned in Shared Memory ? how 
Here, Bm = BN = 128, HALF_D = 64 (why need half D? seperate D = 128 as two 64 dimension feature ), STAGES = 2 (2 time buffer)
```cuda
struct Shared Storage {
    alignas(SMEM_ALIGNMENT) bf16 Q[2][BM * HALF_D];
    alignas(SMEM_ALIGNMENT) bf16 K[STAGES][2][BN * HALF_D];
    union {
        alignas(SMEM_ALIGNMENT) bf16 V[STAGES][2][BN * HALF_D];
        alignas(SMEM_ALIGNMENT) bf16 O[2][BM * HALF_D];
    }
}
```
Note : understanding Q [2][BM* half_d]  : recall that 1 CTA process 128x 128 Q tile. Cutting into half by dimension D ->we got Q[0] 128x64 , Q[1] 128x64

understanding K[Stages][2][BN* half_d] here : why need [STAGES] ? 
Stage 0 K[0] : K[0][0] , K[0][1]
Stage 1 K[1] : K[1][0], K[1][1] 
to support double buffering , where buffer interchange loading during computing to hide loading latency. e.g. Consumer compute stage 0 data , while producer can load stage 1 data.

so 的结构和大小相同，也占 64 KiB。Q 在遍历所有 KV 块时保持不变，所以这里不需要为 Q 分配两个时间槽。
于是主循环的数据存储为：
数据	大小
Q：一个 tile	32 KiB
K：两个时间槽	64 KiB
V：两个时间槽	64 KiB
合计	160 KiB

Union : 2 memeber start from same address.这里：
{sizeof}(V)=64\text{ KiB}
{sizeof}(O)=32\text{ KiB}

because 主循环期间：
┌─────────────────────────────────┐
│            V：64 KiB            │
└─────────────────────────────────┘

输出阶段：
┌────────────────┬────────────────┐
│   O：32 KiB    │  剩余 32 KiB   │
└────────────────┴────────────────┘
O 覆盖的是原来 V 区域的前 32 KiB。Union 不会自动复制数据，也不会自动等待正在读 V 的指令完成，它只是让两个名字指向重叠的存储。
after last load of V, we need a consumer syncronaizations. 
1. each consumer finish wgmma and confirm no more V is loaded
2. each consumer confirm the other consumer no more V loaded

### 4 how TMA load get a complete tile from the original input [B,T,H,128] based on a descriptor.

```cuda
CUtensorMap make_attnetion_map(const bf16* pointer, int batch_size, int sequence_length, int heads) {
    CUtensorMap map;
    void* address = const_cast<bf16*>(pointer);
    // Transfer a complete Q/K/V tile per instruction. TMA groups D into 64-value bands
    // so a single transfer produces the canonical SW128 descriptor layout.
    const uint64_t global_shape[5] = {HALF_D, static_cast<uint64_t>(sequence_length), static_cast<uint64_t>(heads), 2, static_cast<uint64_t>(batch_size)};
    const uint64_t global_stride[4] = {static_cast<uint64_t>(heads) * D * sizeof(bf16), D * sizeof(bf16), HALF_D * sizeof(bf16), static_cast<uint64_t>(sequence_length) * heads * D * sizeof(bf16)};
    const uint32_t box_shape[5] = {HALF_D, BM, 1, 2, 1};
    const uint32_t element_stride[5] = {1, 1, 1, 1, 1};
    const CUresult result = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, address, global_shape, global_stride, box_shape, element_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return map;
}

```
CUtensorMap : Tensor Descriptor given to TMA to obtain information to define data layout
cuTensorMapEncodeTiled() : create the tensor Descriptor with defined api
1. Original input : B,T,H,128
2. Tensor Map : TMA use it to obtain dimension , step, tile size and swizzle rule
3. TMA load : given start address and shared memory target, launch the data loading.

假设张量连续存储：
X[B][T][H][128]
访问：
X[b][t][h][feature]
它相对于起始地址的元素偏移是：
\[
\operatorname{offset}_{\text{elements}}
=
((bT+t)H+h)\times128+\text{feature}
\]
每个 BF16 占 2 字节，因此字节偏移为：
\[
\operatorname{offset}_{\text{bytes}}
=
256THb+256Ht+256h+2\,\text{feature}
\]
各项的意思是：
- 换一个 Batch，跳过 \(T\times H\times128\) 个 BF16。
- 换一个 Token，跳过 \(H\times128\) 个 BF16。
- 换一个 Head，跳过 128 个 BF16。
- 换一个特征元素，前进 2 字节。

split feature into 2 coordinate
define feature = 64u + d
u {0,1}, d {0,63}
\[
\operatorname{offset}_{\text{bytes}}
=
256THb+256Ht+256h+2\,\text{feature}
\]

global shape : `const uint64_t global_shape[5] = (HALF_D, static_cast<uint64_t>(sequence_length),static_cast<uint64_t>(heads), 2, static_cast<uint64_t>(batch_size)};`
坐标： [ d, token, head, band, batch ]
大小： [64,   T,    H,    2,     B   ]

global stride : `const uint64_t global_stride[4] = {static_cast<uint64_t>(heads) * D * sizeof(bf16), D * sizeof(bf16), HALF_D * sizeof(bf16), static_cast<uint64_t>(sequence_length) * heads * D * sizeof(bf16)};`

| 坐标增加 1 | 字节步长 | 原因 |
|---|---:|---|
| `d` | 2，隐含 | 下一个 BF16 元素 |
| `token` | \(256H\) | 跳过一个 Token 的所有 Head |
| `head` | 256 | 跳过一个完整 Head 向量 |
| `band` | 128 | 从前 64 维移到后 64 维 |
| `batch` | \(256TH\) | 跳过一个完整序列 |

box_shape : `const uint32_t box_shape[5] = {HALF_D, BM, 1, 2, 1};`
in this case [64, 128, 1, 2, 1]
| 维度 | 一次传输覆盖 |
|---|---|
| `d` | 带内全部 64 个元素 |
| `token` | 连续 128 个 Token |
| `head` | 一个 Head |
| `band` | 两个特征带 |
| `batch` | 一个 Batch |
element_stride : ` const uint32_t element_stride[5] = {1, 1, 1, 1, 1};`
for this repo , we chose `CU_TENSOR_MAP_SWIZZLE_128B`

```
cuTensorMapEncodeTiled(
    &map,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    5,
    address,
    global_shape,
    global_stride,
    box_shape,
    element_stride,
    ...
);
```
这段调用把以下信息编码到 map：
数据从哪里开始
+ 数据是什么类型
+ 如何将坐标转换成地址
+ 每次加载多大的 Tile
+ 如何组织共享内存中的数据
后面的 TMA 加载只需要使用这张 Map，并提供当前 Tile 的坐标、目标地址以及相应同步信息

### 5 Barrier
as mentioned before, we need barrier to syncronized such that the producer dont keep loading to overwrite the Kv while consumer shouldnt read the kv that are not loaded.

Note : two barrier , FULL and Empty. Producer load data to shared memory through TMA and two consumer use C0 and C1 to compute the loaded data. there are two risks if we dont use barrier : 1. consumer start compute WGMMA before TMA load finish 2. Producer load data with TMA and overwrite data while Consumer hvnt finish its computation. 

| 屏障 | 谁完成它？ | 谁等待它？ | 完成意味着什么？ |
|---|---|---|---|
| `Full` | 生产者登记到达，TMA 报告搬运完成 | 消费者 | 当前数据可以读取 |
| `Empty` | C0、C1 各归还一次 | 生产者 | 旧数据不再使用，可以覆盖 |

Here we use PTX instruction,
```cuda
__device__ __forceinline__ uint32_t barrier_address(const uint64_t* barrier) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
}
```
```cuda
__device__ __forceinline__ void barrier_init(uint64_t* barrier, uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(barrier_address(barrier)), "r"(arrivals) : "memory");
}
barrier_init(&full, 1);
barrier_init(&empty, 2);
```
arrive : count arrival
expected bytes : count bytes return
wait : 

### 6 tma load
combining previous 2 sections, 
```cuda
template <bool Query = false>
__device__ __forceinline__ void tma_load(
    bf16* destination,
    const CUtensorMap* map,
    uint64_t* barrier,
    int token, int head, int batch)
```
tma load use the tensor map, and barrier we created before. based on the rule of the tensormap, take 1 tile from a token/head/batch dimension, move this tile to the destination (shared memory), report to barrier after this load is done.

so for 1 whole loading 
```cuda
// 1. 确保这个槽里的旧数据已经用完。
barrier_wait(&k_empty[stage], phase);

// 2. 登记本轮需要搬完 32 KiB。
barrier_expect_bytes(&k_full[stage], 32768);

// 3. 发起搬运，返回时数据可能尚未就绪。
tma_load(
    shared.K[stage][0],
    &K_map,
    &k_full[stage],
    token, head, batch);

// 4. 更新生产者下一次要使用的槽。
advance_stage(stage, phase);
```

### 7 How WGMMA find its corrosponding element to compute
| | `CUtensorMap` | WGMMA 共享描述符 |
|---|---|---|
| 使用者 | TMA 搬运单元 | WGMMA 矩阵计算单元 |
| 描述对象 | 全局张量及搬运规则 | 共享内存中的矩阵操作数 |
| 主要内容 | 全局地址、维度、步长、box、Swizzle | 共享起点、布局偏移、Swizzle |
| 当前代码中的类型 | `CUtensorMap` | `uint64_t` |

全局 Q/K/V
    │
    │ TMA 读取 CUtensorMap
    ▼
按 SW128 排布的共享内存
    │
    │ WGMMA 读取 64 位共享描述符
    ▼
寄存器中的矩阵乘法结果

### 8 cuda core and tensor cores
| 动作 | 作用 |
|---|---|
| 编译器约束 `asm("" : "+f"/"+r" ...)` | 明确寄存器依赖，约束编译器优化 |
| `wgmma.fence` | 建立先前寄存器访问与后续 WGMMA 的硬件顺序 |
| `wgmma.commit_group` | 将此前未提交的 WGMMA 操作划为一个批次 |
| `wgmma.wait_group<N>` | 等待较早批次完成，最多保留最新 N 个批次未完成 |

等待 Full
    ↓
共享输入已经搬到位
    ↓
编译器操作数约束 + wgmma.fence
    ↓
发射 WGMMA，commit
    ↓
wait_group 等相关计算完成
    ↓
允许读取结果、修改寄存器、归还共享槽

### 9 wgmma qk
m64n128k16 : A(64x16) B(16x128)  + C (64x128) -> S(64x128)
A：从共享内存读取
B：从共享内存读取
C：保存在寄存器
```
__device__ __forceinline__ void wgmma_qk(float (&accumulator)[GROUPS][8], uint64_t A, uint64_t B)
```
| PTX 参数 | C++ 来源/固定值 | 含义 |
|---|---|---|
| `%0..%63` | `accumulator[g][slot]` | 当前线程的 64 个 FP32 输出累加器 |
| `%64` | `A` | A 的共享描述符 |
| `%65` | `B` | B 的共享描述符 |
| `%66` | `ScaleD` | 是否保留旧累加器 |
| 第一个 `1` | `imm-scale-a` | A 不取负 |
| 第二个 `1` | `imm-scale-b` | B 不取负 |
| 第一个 `0` | `imm-trans-a` | A 使用 K-major 共享布局 |
| 第二个 `0` | `imm-trans-b` | B 使用 K-major 共享布局 |

### 10 wgmma pv 
| 对比项 | `wgmma_qk` | `wgmma_pv` |
|---|---|---|
| 左操作数 A | Q，共享内存 | \(\widetilde P\)，寄存器 |
| 右操作数 B | K，共享内存 | V，共享内存 |
| 输入形式 | 共享/共享（SS） | 寄存器/共享（RS） |
| 输出位置 | FP32 寄存器 | FP32 寄存器 |
| `k16` 归约的是什么 | 16 个头特征 | 16 个 Key Token |

这一节的核心：将 Softmax 留在寄存器里的 P，直接交给第二次 Tensor Core 矩阵乘法；V 从共享内存读取，结果继续累加在 FP32 寄存器中

### 11 issue_qk, issue_pv
this chapter pack the previous 2 wgmma instruction into a really gemm operation.
wgmma_qk / wgmma_pv
    一次处理归约维度上的 16 个元素
                ↓
issue_qk / issue_pv
    连续发射 8 次，覆盖完整的 128 个元素
                ↓
外层消费者循环
    遍历不同 K/V 块，执行在线 Softmax 和流水线调度

### 12 pack probabilities
connect online softmax and PV gemm
QK WGMMA
    ↓ 等待完成
scores：FP32 点积分数
    ↓ online_softmax()
scores：FP32 未归一化指数权重
    ↓ pack_probabilities()
P：打包后的 BF16 指数权重
    ↓ issue_pv()
U：FP32 输出分子累加器

### 13 online softmax
First part of online softmax :
1. get the Q/K position of each register
2. masked future token 
3. Get the Row Max Candidate

template <bool First, bool Masked>
First : is it the first K/V tile that this consumer process
Masked : if this tile need Causal Mask

```
__device__ __forceinline__ void online_softmax(
    float (&scores)[GROUPS][8],
    float (&row_max)[2],
    float (&row_sum)[2],
    float (&previous_scale)[2],
    int query_start,
    int key_start,
    int consumer_thread,
    float scale_log2)
```

| 参数 | 含义 |
|---|---|
| `scores` | 当前线程的 64 个 FP32 分数；后半段原位改成指数权重 |
| `row_max[2]` | 当前线程对应两行的旧指数基准 |
| `row_sum[2]` | 两行的累计分母状态；当前实现此时保留线程局部部分和 |
| `previous_scale[2]` | 基准变化时，对旧输出和旧分母进行修正的系数 |
| `query_start` | 当前消费者第一行 Q 的全局 Token 位置 |
| `key_start` | 当前 K 块第一行的全局 Token 位置 |
| `consumer_thread` | 消费者 Warp Group 内的线程号，0～127 |
| `scale_log2` | `scale × log2(e)`，供后续 `exp2` 使用 |

```
const int lane = consumer_thread & 31;
const int warp = consumer_thread >> 5;
const int top_row = warp * 16 + lane / 4;
const int bottom_row = top_row + 8;
```
based on the thread number to find which 2 rows it process, refer to chapter xx

```
for (int group = 0; group < 8; ++group) {
    const int column = group * 16 + 2 * (lane & 3);
}
```
column is determined by this code based on which grp it is currently and which lane .

After masking, each thread process and get its local max.

current max is processed from warp shuffle reduce within 4 lanes

init next_max , this used to compare the next and previous mas to determine a new max need to replace the old max or not

reduce sum :  same as max, but different operation
整章可以记成：选基准 → 算指数 → 求局部和 → 修正并累计分母 → 保存基准。真正除以分母，要等所有 K/V 块处理完

### 14 scale output

### 15 finialize_rows 
after all k/v tile is processed . combine all , compute LSE.
until now , we have row_max, row_sum, O for each Query row.

so now we need to combine all result, reduce the final row sum and max

```
for (int r = 0; r < 2; ++r) {
    row_sum[r] += __shfl_xor_sync(0xffffffff, row_sum[r], 1);
    row_sum[r] += __shfl_xor_sync(0xffffffff, row_sum[r], 2);
}
```
then compute lse `lse_values[r] = row_max[r] * scale + __logf(row_sum[r]);`

### 16 reduction , stmatrix and lse write back

FP32 输出分子寄存器
        ↓ 乘分母倒数
归一化后的输出
        ↓ 转换并打包
BF16 寄存器
        ↓ stmatrix
SW128 布局的共享内存
        ↓ 外层 TMA store
全局输出张量

### 17 main kernel

in all previous section, we defined all the utilties , function we need.

now the overall flow is 
准备共享内存
      ↓
初始化屏障
      ↓
让所有线程确认初始化完成
      ↓
划分生产者和消费者角色

what 384 threads responsible to? when shared buffer can write or read?

recall that | `threadIdx.x` | `group` | 角色 | 工作 |
|---|---:|---|---|
| 0～127 | 0 | 生产者（Producer） | 发起 TMA 搬运 |
| 128～255 | 1 | 消费者 C0 | 计算前 64 行 Query |
| 256～383 | 2 | 消费者 C1 | 计算后 64 行 Query |

### 18 静态 CTA 映射与 GQA 头选择
给定一个 blockIdx.x，这个 CTA 应该计算哪个 Batch、哪个 Query Head、哪 128 行 Query，以及使用哪个 KV Head？

1 CTA process 1 Batch, 1 query head, 128 query row

query_blocks：
    每个 Query Head 有多少个 Query 块

section_tiles：
    一个区段中总共有多少个 CTA 任务

section：
    当前 CTA 属于第几个区段

section_tile：
    当前 CTA 在该区段内的编号

| 模式 | Hq | Hkv | Query Head 到 KV Head 的映射 |
|---|---:|---:|---|
| MHA | 8 | 8 | 每个 Query Head 使用同编号 KV Head |
| GQA | 8 | 2 | 每四个 Query Head 共享一个 KV Head |
| MQA | 8 | 1 | 全部 Query Head 使用 KV Head 0 |

### 19 Producer
Q 加载一次并保留；K/V 分块加载，其中下一块 K 优先于当前块 V 发射。
\
### 20 消费者前导：启动协议与首块 QK
### 21 
### 22
### 23