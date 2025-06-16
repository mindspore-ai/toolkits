GLOBAL_COMMUNICATION_OTHER = "communication_other"
GLOBAL_OP_TYPE_OTHER = "other"

TOTAL_TIME = "total time (ms)"
OVERLAP_TIME = "overlap time (ms)"
OVERLAP_RATIO = "overlap ratio"
NOT_OVERLAP_TIME = "non-overlap time (ms)"
RATIO_OF_TOTAL_TIME = "ratio of total time(non-overlap time)"
OP_COUNT = "operators count"
RECOMPUTE_COUNT = "recompute operators count"
RECOMPUTE_TIME = "recompute time (ms)"
RECOMPUTE_OVERLAP_TIME = "recompute overlap time (ms)"
RECOMPUTE_NOT_OVERLAP_TIME = "recompute non-overlap time (ms)"
RECOMPUTE_RATIO = f"recompute_ratio({RECOMPUTE_NOT_OVERLAP_TIME}/{NOT_OVERLAP_TIME})"

OVERLAP_PRIORITY = {"fa_forward": 0, "fa_back": 1, "grouped": 2, "matmul": 3, "vector": 4,
                                          "broadcast": 5, "alltoall": 6, "alltoallv": 7, "alltoallc": 8, "allGather": 9,
                                          "reduceScatter": 10, "allReduce": 11, GLOBAL_COMMUNICATION_OTHER: 12, GLOBAL_OP_TYPE_OTHER: 13,
                                          "receive": 14, "send": 15}
COMMUNICATION_OVERLAP_PRIORITY_RE = ["send", "receive", "allReduce", "reduceScatter","allGather","alltoallc", "alltoallv", "alltoall", "broadcast"]

COMMUNICATION_PARALLEL = {"ep": ["alltoall"], "tp": ["allGather", "reduceScatter"], "pp": ["send", "receive"], "dp":["allReduce"], "broadcast_op": ["broadcast"]}

US_TO_MS = 1000
