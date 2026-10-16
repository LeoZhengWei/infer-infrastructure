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

# 单生产和消费模型的lock-free Queue
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
                throw std::bad_alloc()
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


#if MOODYCAMEL_HAS_EMPLACE
    template<AllocationMode canAlloc, typename... Args>
    bool inner_enqueue(Args&&... args) AE_NO_TSAN
#else 
    template<AllocationMode canAlloc, typename U>
    bool inner_enqueue(U&& element) AE_NO_TSAN
#endif
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

        size_t nextBlockTail = (blockTail + 1) & tailBlock_->sizeMask;
        if (nextBlockTail != blockFront || nextBlockTail != (tailBlock_->localFront = tailBlock_->front.load()) {
            fence(memory_order_acquire);
            char* location = tailBlock_->data + blockTail * sizeof(T);
#if MOODYCAMEL_HAS_EMPLACE
            new (location) T(std::forward<Args>(args)...);
#else 
            new (location) T(std::forward<U>(element));
#endif 
            fence(memory_order_release);
            tailBlock_->tail = nextBlockTail;
        } 
        else {
            fence(memory_order_acquire);
            if (tailBlock_->next.load() != frontBlock) {
                // 不能再切换到前区块并在该区块中新增条目
                // 原因：如果这么做，出队操作会停留在这个区块中，最终读取新增的数据
                // 而不会切换到下一个已满的区块（该区块内的数据是最先入队的，应该优先被取出）
                fence(memory_order_acquire);
            
                // 尾块满，但是在之前有空闲的块，使用
                Block* tailBlockNext = tailBlock_->next.load();
                size_t nextBlockFront = tailBlockNext->localFront = tailBlockNext->front.load();
                nextBlockTail = tailBlockNext->tail.load();
                fence(memory_order_acquire);

                // 这个块由于不是头块，必须为空，并且我们会循环遍历各个块
                assert(nextBlockFront == nextBlocktail);
                tailBlockNext->localFront = nextBlockFront;

                char* location = tailBlockNext->data + nextBlockTail * sizeof(T);
#if MOODYCAMEL_HAS_EMPLACE
                new (location) T(std::forward<Args>(args)...);
#else   
                new (location) T(std::forward<U>(element));
#endif  
                tailBlockNext->tail = (nextBlockTail + 1) & tailBlockNext->sizeMask;

                fence(memory_order_release);
                tailBlock = tailBlockNext;
            }
            else if (canAlloc == CanAlloc) {
                auto newBlockSize = largestBlockSize >= MAX_BLOCK_SIZE ? largestBlockSize : largestBlockSize * 2;
                auto newBlock = make_block(newBlockSize);
                if (nullptr == newBlock) {
                    return false;
                }
                largestBlockSize = newBlockSize;
            }
#if MOODYCAMEL_HAS_EMPLACE 
                new (newBlock->data) T(std::forward<Args>(args)...);
#else       
                new (newBlock->data) T(std::forward<U>(element));
#endif
                assert(newBlock->front == 0);
                newBlock->tail = newBlock->localTail= 1;

                newBlock->next = tailBlock_->next.load();

                // 发布所有写操作到*newBlock在它可以通过tailBlock_->next访问前
                fence(memory_order_release);
                tailBlock_->next = newBlock;

                // 确保观察到新尾块的读取操作也能看到tailBlock_->next此前的写入发布
                fence(memory_order_release);
                tailBlock = newBlock;
            }
            else if (canAlloc == CannotAlloc) {
                return false;
            }
            else {
                assert(false && "Should be unreachable code");
                return false;
            }
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
    
};