## Axis 1: Cache Capacity vs Working Set

**Research Question:** At what capacity does the working set saturate, and when does adding memory yield diminishing returns?  
**Hypothesis:** At low cache sizes ($K < 8$), the system is purely I/O bound. Beyond a critical knee ($K \approx 24$), the hit rate plateaus because tail experts are rarely activated. At this saturation threshold, the primary performance bottleneck abruptly shifts from storage/bus bandwidth to GPU compute.