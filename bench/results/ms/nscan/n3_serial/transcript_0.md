
### turn 0
**User:** Explain in detail how a Vulkan timeline semaphore orders submissions across queues, and what happens to a wait that is submitted before its signal.

**Assistant:** # Vulkan Timeline Semaphores: Cross-Queue Ordering and Out-of-Order Waits

## 1. What a Timeline Semaphore Is

A timeline semaphore is a synchronization primitive that holds a **monotonically increasing

### turn 1
**User:** Continue: what does that mean for a command buffer that both waits on and signals the same timeline value, and how should a decoder loop use it?

**Assistant:** # Timeline Semaphores: Self-Wait/Self-Signal and Decoder Loops

## 1. Recap: The Monotonic Counter Model

A timeline semaphore holds a single 64-bit unsigned integer that only ever increases
