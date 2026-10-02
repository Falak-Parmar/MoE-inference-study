## Axis 2: Eviction Policy & Expert Locality

**Research Question:** Does expert routing exhibit strong enough temporal and spatial locality to justify complex eviction logic over simple FIFO / LRU?  
**Hypothesis:** Least Frequently Used (LFU) significantly outperforms LRU on homogeneous, sustained tasks by anchoring core "syntactic" experts. However, during multi-turn domain shifts, un-decayed LFU causes stale experts to remain resident, causing hit rates to degrade below LRU.
