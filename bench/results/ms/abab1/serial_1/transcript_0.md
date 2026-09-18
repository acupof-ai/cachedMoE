
### turn 0
**User:** Explain in detail how a Vulkan timeline semaphore orders submissions across queues, and what happens to a wait that is submitted before its signal.

**Assistant:** # Vulkan Timeline Semaphores: Cross-Queue Ordering and Out-of-Order Waits

## 1. What a Timeline Semaphore Is

A timeline semaphore is a synchronization primitive that holds a **monotonically increasing 64-bit counter**. Unlike a binary semaphore (which is either signaled or
