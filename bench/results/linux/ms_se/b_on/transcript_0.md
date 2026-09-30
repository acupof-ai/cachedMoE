
### turn 0
**User:** Explain in detail how a Vulkan timeline semaphore orders submissions across queues, and what happens to a wait that is submitted before its signal.

**Assistant:** # Vulkan Timeline Semaphores: Cross-Queue Ordering and Out-of-Order Waits

## 1. What a Timeline Semaphore Is

A timeline semaphore is a synchronization primitive that holds a **monotonically increasing 64-bit counter**. Unlike a binary semaphore (which is either signaled or

### turn 1
**User:** Continue: what does that mean for a command buffer that both waits on and signals the same timeline value, and how should a decoder loop use it?

**Assistant:** # Timeline Semaphores: Self-Referential Waits and Decoder Loop Patterns

## 1. Recap: The Ordering Guarantee

A timeline semaphore's counter is a single 64-bit value that advances monotonically. A `vkQueueSubmit` (or `vkQueueSubmit2`) can carry

### turn 2
**User:** Continue: summarise the rules for host waits versus device waits on that timeline, and the failure modes of each.

**Assistant:** # Host Waits vs. Device Waits on a Timeline Semaphore

## 1. The Two Kinds of Wait

A timeline semaphore can be waited on from two very different places, and the rules, guarantees, and failure modes are not the same.

**Device wait** — a wait encoded into a

### turn 3
**User:** Given that, how should a runtime size its expert cache so that a long conversation keeps its working set resident?

**Assistant:** # Sizing an Expert Cache for Long-Conversation Working-Set Residency

## 1. The Problem Restated

A mixture-of-experts (MoE) runtime routes each token to a small subset of experts. Over a long conversation, the *set* of experts that get touched is not the whole
