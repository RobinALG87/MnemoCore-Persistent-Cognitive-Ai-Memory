## 2025-05-27 - [Synapse Iteration Bottleneck]
**Learning:** `HAIMEngine.query` performance degrades linearly with total synapse count (O(S)) multiplied by nodes searched (O(N)), causing O(N*S) complexity. This is because `get_node_boost` iterated over all synapses to find connections.
**Action:** Always use an Adjacency List for graph traversals or lookups. Implemented `adjacency_list` in `HAIMEngine` to reduce lookup to O(D) (degree), resulting in ~9x speedup for 5000 synapses.

## 2025-05-27 - [Immediate Tier Demotion]
**Learning:** Default configuration (`ltp.initial_importance=0.5` < `tiers.hot.ltp_threshold_min=0.7`) causes new memories to be immediately demoted from HOT to WARM upon first access in `get_memory`. This can break tests expecting HOT tier residency.
**Action:** Ensure LTP initialization exceeds promotion/demotion thresholds in test configurations, or adjust default config values to avoid immediate churn.
