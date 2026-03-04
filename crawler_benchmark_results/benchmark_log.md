## Step 2 - Baseline Tweaked (timeouts 3s)
Timestamp: 2026-03-04T12:46:05
Total Time: 27.23s
Pages: 38
Status: success
## Step 4 - Simulation Removed
Timestamp: 2026-03-04T12:47:11
Total Time: 25.04s
Pages: 38
Status: success
## Step 3 - Baseline Optimized (HTTP selector short-circuit)
Timestamp: 2026-03-04T13:02:08
Total Time: 25.79s
Pages: 38
Status: success
## Step 4 - Simulation Removed (benchmark)
Timestamp: 2026-03-04T13:03:31
Total Time: 37.49s
Pages: 38
Status: success
## Step 5 - Split Queue (failed)
Timestamp: 2026-03-04T13:10:14
Result: benchmark timed out >180s
Action: reverted to single-queue version
## Step 6 - Split Queue (HTTP wait + browser timeout 60s)
Timestamp: 2026-03-04T13:20:32
Total Time: 65.22s
Pages: 38
Status: stopped
Note: browser queue drain timed out
## Step 7 - Reverted to Stable Single-Queue
Timestamp: 2026-03-04T13:21:52
Total Time: 25.63s
Pages: 38
Status: success
