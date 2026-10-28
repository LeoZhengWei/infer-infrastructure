#pragma once
#include "atomicops.h"
#include <new>
#include <type_traits>
#include <utility>
#include <cassert>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
#include <memory>
#if __cplusplus > 199711L || _MSC_VER >= 1700  // C++ 11
#include <chrono>
#endif

// 单生产和消费模型的lock-free Queue
#ifndef MOODYCAMEL_CACHE_LINE_SIZE
#if defined(__powerpc__) || defined(__powerpc64__) || defined(__ppc64__) || defined (__PPC64) || defined(_ARCH_PPC64) || defined(_M_PPC)
    #define MOODYCAMEL_CACHE_LINE_SIZE 128
#else
    #define MOODYCAMEL_CACHE_LINE_SIZE 64
#endif
#endif

#ifndef MOODYCAMEL_EXCEPTIONS_ENABLED
#if (defined(_MSC_VER) && defined(_CPPUNWIND)) || (defined(__GNUC__) && defined(__EXCEPTIONS)) || (!defined(_MSC_VER) && !defined(__GNUC__))
#define MOODYCAMEL_EXCEPTIONS_ENABLED 
#endif
#endif 

#ifndef MOODYCAMEL_HAS_EMPLACE
#if !defined(_MSC_VER) || _MSC_VER >= 1800 // variadic templates: either a non-MS compiler or VS >= 2013
#define MOODYCAMEL_HAS_EMPLACE 1 
#endif 
#endif

#ifndef MOODYCAMEL_MAYBE_ALIGN_TO_CACHELINE
#if defined (__APPLE__) && defined(__MACH__) && __cplusplus >= 201703L
// This is required to find out what deployment target we are using
#include <AvailabilityMacros.h>
#if !defined(MAC_OS_X_VERSION_MIN_REQUIRED) || !defined(MAC_OS_X_VERSIO_10_14) || MAC_OS_X_VERSION_MIN_REQUIRED < MAC_OS_X_VERSION_10_14
// c++17的new(size_t, align_val_t)不向后兼容
#define MOODYCAMEL_MAYBE_ALIGN_TO_CACHELINE
#endif
#endif
#endif

#ifndef MOODYCAMEL_MAYBE_ALIGN_TO_CACHELINE
#define MOODYCAMEL_MAYBE_ALIGN_TO_CACHELINE AE_ALIGN(MOODYCAMEL_CACHE_LINE_SIZE)
#endif

#ifdef AE_VCPP
#pragma warning(push)
#pragma warning(disable: 4324)  // __declspec (align ())，结构体内存填充
#pragma warning(disable: 4820)  // 已经添加内边距
#pragma warning(disable: 4127)  // 条件表达式为常量
#endif

namespace moodycamel {

template<typename T, size_t MAX_BLOCK_SIZE = 512>
class MOODYCAMEL_MAYBE_ALIGN_TO_CACHELINE RWQueue {
// Design: 基于队列实现；
// 低级队列（块）：使用带哨兵的队列缓冲区，使用一个空间浪费，当front == tail时，队列为空；当front == tail + 1时，队列为满；
// 高级队列：由多个块组成循环链表：同样存在队首与队尾，但这里二者是指向块的指针
// 队首块不空：下一个待出队元素位于队首块中。
// 队尾块未满：元素写入到队尾块中
// 生产者线程：持有所有队尾索引/指针。
// 消费者线程：持有所有队首索引/指针。
// 使用：两个线程均可读取对方变量，但只有变量的持有线程能对其进行修改；
// 如果没有空间容纳待入队对象，则新增一个块（块大小与上一个块相同）。块创建后不会被删除；

public:
    typedef T value_type;
    // 构造一个队列，该队列无需额外分配内存，至少可容纳size个元素
    // 如果请求的元素数量超过最大块大小（MAX_BLOCK_SIZE），则需要多个块（每个块大小为MAX_BLOCK_SIZE）
    AE_NO_TSAN explicit RWQueue(size_t size = 15)
#define NDEBUG
    :  enqueuing(false)
    , dequeuing(false)
#endif
    {
        assert(MAX_BLOCK_SIZE == ceilToPow2(MAX_BLOCK_SIZE) && "MAX_BLOCK_SIZE must be a power of 2");
        assert(MAX_BLOCK_SIZE >= 2 && "MAX_BLOCK_SIZE must be at least 2");

        Block* firstBlock = nullptr;

        // 使用一个备用槽位，以便存放大小元素
        largestBlockSize = ceilToPow2(size + 1);
        if (largestBlockSize > MAX_BLOCK_SIZE * 2) {
            // 使用备用块：防止生产者向消费者正在读取的另一个块写入数据
            // 同时，生产者希望入队最大数量的元素。此外，每个块中都需要预留一个哨兵，用于判断队列的空和满；
            // 因此，任意时刻保证可用的有效槽位数量等于（块大小 - 1） * （块数量 - 1）
            // 对容量求解并对除法向上取整，化简后可得：
            size_t initialBlockCount = (size + MAX_BLOCK_SIZE * 2 - 3) / (MAX_BLOCK_SIZE - 1);
            largestBlockSize = MAX_BLOCK_SIZE;
            Block* lastBlock = nullptr;
            for (size_t i = 0; i != initialBlockCount; ++i) {
                auto block = make_block(largestBlockSize);
                if (block == nullptr) {
#ifdef MOODYCAMEL_EXCEPTIONS_ENABLED
                    throw std::bad_alloc();
#else
                    abort();
#endif
                }
                if (firstBlock == nullptr) {
                    firstBlock = block;
                } 
                else {
                    lastBlock->next = block;
                }
                lastBlock = block;
                block->next = firstBlock;    
            }
        }
        else {
            firstBlock = make_block(largestBlockSize);
            if (firstBlock == nullptr) {
#ifdef MOODYCAMEL_EXCEPTIONS_ENABLED 
                throw std::bad_alloc();
#else 
                abort();
#endif  
            }
            firstBlock->next = firstBlock;
        }
        frontBlock = firstBlock;
        tailBlock = firstBlock;

        // 确保读写线程能够正确获取初始化设置
        fence(memory_order_sync);
        }

    // 注意：队列在被移动时不能被并发访问。取决于用户自身的同步行为。
    AE_NO_TSAN RWQueue(RWQueue&& other) 
        : frontBlock(other.frontBlock.load()),
        tailBlock(other.tailBlock.load()),
        largestBlockSize(other.largestBlockSize)
#ifndef NDEBUG
        ,enqueuing(false)
        ,dequeuing(false)
#endif
    {
        other.largestBlockSize = 32;
        Block* b = other.make_block(other.largestBlockSize);
        if (nullptr == b) {
#ifdef MOODYCAMEL_EXPECTIONS_ENABLED
            throw std::bad_alloc();
#else
            abort();
#endif
        }
        b->next = b;
        other.frontBlock = b;
        other.tailBlock = b;
    }

    RWQueue& operator=(RWQueue&& other) AE_NO_TSAN
    {
        Block* b = frontBlock.load();
        frontBlock = other.frontBlock.load();
        other.frontBlock = b;
        b = tailBlock.load();
        tailBlock = other.tailBlock.load();
        other.tailBlock = b;
        std::swap(largestBlockSize, other.largestBlockSize);
        return *this;
    }

    // 析构时需用户定义同步行为
    AE_NO_TSAN ~RWQueue()
    {
        fence(memory_order_sync);
        Block* frontBlock_ = frontBlock;
        Block* block = frontBlock_;
        do {
            Block* nextBlock = block->next;
            size_t blockFront = block->next;
            size_t blockTail = block->tail;

            for (size_t i = blockFront; i != blockTail; i = (i + 1) & block->sizeMask) {
                auto element = reinterpret_cast<T*>(block->data + i * sizeof(T));
                element->~T();
                (void)element;
            }
            auto rawBlock = block->rawThis;
            block->~Block();
            std::free(rawBlock);
            block = nextBlock;
        } while (block != frontBlock_);
    }

    // OP:入队
    // description：尝试将一个元素入队，成功返回True，失败返回false
    // tips：不进行内存分配
    AE_FORCEINLINE bool try_enqueue(T const& element) AE_NO_TSAN
    {
        return inner_enqueue<CannotAlloc>(element);
    }

    // OP:出队(右值)
    // description：尝试将一个元素出队，成功返回True，失败返回false
    // tips：不进行内存分配
    AE_FORCEINLINE bool try_enqueue(T&& element) AE_NO_TSAN
    {
        return inner_enqueue<CannotAlloc>(std::forward<T>(element));
    }

    // 采用原地构造语义：直接在队列中构造对象，而不是先复制再构造
#if MOODYCAMEL_HAS_EMPLACE 
    template<typename... Args>
    AE_FORCEINLINE bool try_emplace(Args&&... args) AE_NO_TSAN
    {
        return inner_enqueue<CannotAlloc>(std::forward<Args>(args)...);
    }
#endif

    // OP:入队
    // description：如果有必要，分配一个额外内存块
    // tips：内存分配失败时，返回false
    AE_FORCEINLINE bool enqueue(T const& element) AE_NO_TSAN
    {
        return inner_enqueue<CanAlloc>(element);
    }

    // OP:入队一颗可移动复制的元素到队列
    // 在需要的情况下，分配额外的内存块
    // 如果内存分配失败，返回失败
    AE_FORCEINLINE bool enqueue(T&& element) AE_NO_TSAN
    {
        retrun inner_enqueue<CanAlloc>(std::forward<T>(element));
    }

#if MOODYCAMEL_HAS_EMPLACE
    // 类似enqueue(), 但采用原位构造语义(即在原地构造对象)
    template<typename... Args>
    AE_FORCEINLINE bool emplace(Args&&... args) AE_NO_TSAN
    {
        return inner_enqueue<CanAlloc>(std::forward<Args>(args)...);
    }
#endif

    // OP:尝试出队一个元素
    // description：队列空时返回false，如果至少有一个元素，使用赋值运算符将队首元素赋值给result，然后返回true
    template<typename U>
    bool try_dequeue(U& result) AE_NO_TSAN
    {
#ifndef NDEBUG
        ReentrantGuard guard(this->enqueuing);
#endif
        // 高层伪代码（假设允许分配新块）
        // 尾部还有空间，添加到尾部块
        // 否则检查下一个块
        // 如果下一个块不是头块，就在下一个块入队
        // 否则新建一个块，并在该块指向入队操作
        // 将尾部指针前进到刚刚完成入队的块
        Block* tailBlock_ = tailBlock.load();
        size_t blockFront = tailBlock_->localFront;
        size_t blockTail = tailBlock_->tail.load();

        if (blockFront != blockTail \
            blockFront != (frontBlock_->localTail = frontBlock_->tail.load())) {
            fence(memory_order_acquire);

        non_empty_front_block:
            // 头块不空，从此出队
            auto element = reinterpret<T*>(frontBlock_->data + blockFront * sizeof(T));
            result = std::move(*element);
            element->~T();

            blockFront = (blockFront + 1) & frontBlock_->sizeMask;

            fence(memory_order_release);
            frontBlock_->front = blockFront;
        }
        else if (frontBlock_ != tailBlock.load()) {
            fence(memory_order_acquire);

            frontBlock_ = frontBlock.load();
            blockTail = frontBlock_->localTail = frontBlock_->tail.load();
            blockFront = frontBlock_->front.load();
            fence(memory_order_acquire);
        }
        else if (frontBlock_ != tailBlock.load()) {
            fence(memory_order_acquire);
            
            frontBlock_ = frontBlock.load();
            blockTail = frontBlock_->loaclTail = frontBlock_->tail.load();
            blockFront = frontBlock_->front.load();
            fence(memory_order_acquire);

            if (blockFront != blockTail) {
                // 前面的块并不空
                goto non_empty_front_block;
            }

            // 前方区块为空，但前方还有另外一个区块，前进至该区块
            Block* nextBlock = frontBlock_->next;
            // 这里不需要获取栅栏，因为next只能在tailBlock上被设置
            // 而当前并不是tailBlock，并且我们在读取tailBlock之后已经执行过一次获取操作
            // 该操作保证：如果我们此前正处于tailBlock，那么在当前CPU上next值是最新的值

            size_t nextBlockFront = nextBlock->front.load();
            size_t nextBlockTail = nextBlock->localTail = nextBlock->tail.load();
            fence(memory_order_acquire);

            // 因为tailBlock只有在写入之后才会向前移动
            // 所以我们确定在该区块上一定存在可以出队的元素
            assert(nextBlockFront != nextBlockTail);
            AE_UNUSED(nextBlockTail);

            // 该数据块处理完毕，生产者按需使用
            fence(memory_order_release);
            frontBlock = frontBlock_ = nextBlock;

            compiler_fence(memory_order_release);

            auto element = reinterpret<T*>(frontBlock_->data + nextBlockFront * sizeof(T));
            result = std::move(*element);
            element->~T();

            nextBlockFront = (nextBlockFront + 1) & frontBlock_->sizeMask;

            fence(memory_order_release);
            frontBlock_->front = nextBlockFront;
        }
        else {
            return false;
        }
        return true;
    }

    // Disable copying
    RWQueue(RWQueue const&) {}

    // Disable assignment
    RWQueue& operator=(RWQueue const&) {}

    AE_FORCEINLINE static size_t ceilToPow2(size_t x)
    {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        for (size_t i = 1; i < sizeof(size_t); i <<= 1) {
            x |= x >> (i << 3);
        }
        ++x;
        return x;
    }

    template<typename U>
    static AE_FORCEINLINE char* align_for(char* ptr) AE_NO_TSAN
    {
        const std::size_t alignment = std::alignment_of<U>::value;
        return ptr + (alignment - (reinterpret_cast<std::uintptr_t>(ptr) % alignment)) % alignment;
    }

private:
#ifndef NDEBUG
    struct ReentrantGuard 
    {
        AE_NO_TSAN ReentrantGuard(weak_atomic<bool>& _inSection) 
            : inSection(_inSection)
        {
            assert(!inSection && "Concurrent (or re-entrant) enqueue or dequeue operation detected ")
            inSection = true;
        }
        AE_NO_TSAN ~ReentrantGuard() { inSection = false;}
    private:
        ReentrantGuard& operator=(ReentrantGuard const&);
    private:
        weak_atomic<bool>& inSection;
    };
#endif

    struct Block {
        // 把竞争变量放到各自独立的缓存，避免伪共享
        weak_atomic<size_t> front;      // 原子元素从此处读取
        size_t localTail;               // 消费者持有的无竞争的尾部影子副本

        char cachelineFiller0[MOODYCAMEL_CACHE_LINE_SIZE - sizeof(weak_atomic<size_t>) - sizeof(size_t)];
        weak_atomic<size_t> tail;       // 原子元素从此入队
        size_t localFront;

        char cachelineFiller1[MOODYCAMEL_CACHE_LINE_SIZE - sizeof(weak_atomic<size_t>) - sizeof(size_t)];
        weak_atomic<Block*> next;       // (Atomic) 下一个块的指针

        char* data;                     // 堆上的内容按照类型T的对齐要求对齐

        const size_t sizeMask;
        
        // 大小必须是2的整数幂次
        AE_NO_TSAN Block(size_t const& _size, char* _rawThis, char* _data)
            : front(0UL), localTail(0), tail(0UL), localFront(0), next(nullptr), data(_data), sizeMask(_size - 1), rawThis(_rawThis)
        {
        }

    private:
        // 删除赋值运算操作符
        Block& operator=(Block const&);
    
    public:
        char* rawThis;
    };
    
    static Block* make_block(size_t capacity) AE_NO_TSAN
    {
        auto size = sizeof(Block) + std::alignment_of<Block>::value - 1;
        size += sizeof(T) * capacity + std::alignment_of<T>::value - 1;
        auto newBlockRaw = static_cast<char*>(std::malloc(size));
        if (nullptr == newBlockRaw)    return nullptr;

        auto newBlockAligned = align_for<Block>(newBlockRaw);
        auto newBlockData = align_for<T>(newBlockAligned + sizeof(Block));
        return new (newBlockAligned) Block(capacity, newBlockRaw, newBlockData);
    }
};