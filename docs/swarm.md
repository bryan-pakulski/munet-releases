# Training swarm: owner and native nodes

The training swarm trains one model across machines. One owner holds the weights and applies every optimizer
update. Nodes compute gradients for chunks of each global batch and send them back. The result equals single-machine
SGD on the global batch, up to floating-point differences.

## Scope and limits

- Dense, deterministic models whose outputs are independent across samples.
- Loss: mean squared error. Optimizer: SGD without momentum.
- Nodes run the C++ forward/backward graph on Vulkan or on the CPU reference backend.
- Owner and node run on Linux/POSIX.
- Not supported: BatchNorm, detection losses and RT-DETR. They need a global normalization policy the swarm does
  not have.

How training proceeds:

- The dataset is shuffled once per epoch into fixed global batches.
- One global batch is one round and one optimizer update, however many workers connect.
- The owner weights each chunk's gradient by its sample count, accumulates in float64 and applies a float32 update.
- Workers joining or leaving change scheduling only, not the batch size, learning rate or sample coverage.
- A disconnected node can finish work it already downloaded. It cannot produce later rounds without new weights.

## Install the owner and node

`pip install munet-nn` provides both commands on Linux:

- `munet-node` runs the packaged C++ worker.
- `munet-server` runs the owner.

Or use the standalone archives, which need no Python install (the server bundles CPython):

1. Download the matching `munet-node`, `munet-server` and `munet-sdk` archives and `SHA256SUMS-linux-ARCH` from
   [munet-releases](https://github.com/bryan-pakulski/munet-releases/releases).
2. Verify them: `sha256sum --check SHA256SUMS-linux-ARCH`. The check expects all three archives.
3. Extract them into a directory you own. Keep the directory structure: the node uses relative library paths and
   the server needs its `_internal` directory.
4. Add each extracted `bin` directory to `PATH`.

```bash
sha256sum --check SHA256SUMS-linux-x86_64
tar -xzf munet-node-VERSION-linux-x86_64.tar.gz
tar -xzf munet-server-VERSION-linux-x86_64.tar.gz
munet-node-VERSION-linux-x86_64/bin/munet-node --version
munet-server-VERSION-linux-x86_64/bin/munet-server --version
```

Requirements:

- Nodes need no Python, PyTorch, NumPy, ONNX or shader compiler. Vulkan nodes need the Vulkan loader and a driver.
- Preparing a job for Vulkan nodes needs `glslangValidator` on the preparing machine. Nodes receive compiled SPIR-V.

## Run a training job

1. Prepare a job directory with [`create_job`](#custom-models). The source tree has a demo:

   ```bash
   python examples/swarm_mlp.py prepare demo-job --epochs 10
   ```

   `--cpu-only` omits SPIR-V, for a machine without `glslangValidator`. Such a job accepts only CPU nodes. CPU nodes
   can also join a normal job.

2. Set the same `MUNET_SWARM_TOKEN` on the owner and on every node. It must have at least 32 characters. Generate one
   with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`. Pass it through the environment, never in a
   URL or command-line argument.

3. Start the owner and the nodes, each in its own terminal:

   ```bash
   munet-server demo-job --port 8765        # listens on 127.0.0.1 by default
   munet-node --owner http://127.0.0.1:8765 --state node-a --device vulkan:0
   munet-node --owner http://127.0.0.1:8765 --state node-b --device vulkan:1
   ```

   - Use a device index that exists on each host, or `--device cpu` for the CPU reference backend.
   - Give each node process its own `--state` directory.

4. Check progress, then export after training:

   ```bash
   munet-server demo-job --status           # reads the journal of a stopped owner
   python examples/swarm_mlp.py export demo-job --output trained.mnet --onnx
   ```

   In your own code, `Owner(job_dir).load_into(model)` loads the latest weights. Then use the normal `mu.compile`,
   `mu.save` and ONNX export.

## Custom models

```python
from munet.swarm import create_job

create_job(
    model, x, target, "my-job",  # finite float32 arrays [samples, ...]
    global_batch_size=32,       # defines the optimizer semantics
    micro_batch_size=4,         # smallest scheduling and memory unit
    epochs=10, lr=0.03, seed=42,
    include_vulkan=True,        # False: CPU nodes only, no glslangValidator needed
)
```

The model must meet this contract:

- The loss is MSE with matching prediction and target shapes.
- The model is deterministic and its outputs are independent across samples. Operations that couple samples in a
  batch break the contract, even when they trace.
- The model fits the [limits](#limits), including at most 100,000 parameter values.

## Authentication

- A node presents `MUNET_SWARM_TOKEN` only to `/v1/register`. The owner returns a node token, stores it hashed, and
  the node keeps it in `worker.json`. Registering again rotates it.
- Revoke a node on a stopped owner with `munet-server JOB --revoke WORKER_ID`. Its token then fails with 401,
  re-registering fails with 403, and its open leases expire.

Authorized nodes see the model weights and their assigned data. Tokens control admission; they do not prove what code
a node runs. The owner rejects malformed and duplicate results, but a malicious node could send a wrong gradient.
Public anonymous workers are out of scope.

## Remote connections

The owner listens on loopback by default. A non-loopback listener needs TLS or `--allow-http`.

- **HTTPS:** `munet-server JOB --bind 0.0.0.0 --cert server.crt --key server.key`, or an HTTPS reverse proxy to a
  loopback owner. TLS 1.2+ with forward-secret AEAD ciphers.
- **Certificates:** nodes verify certificates and hostnames and do not forward credentials on redirects.
  `MUNET_CA_BUNDLE` sets a private CA bundle; otherwise nodes use the host trust store.
- **mTLS:** `--client-ca ca.pem` requires client certificates signed by that CA.
- **Rate limits:** per client IP and per node, answered with 429 and `Retry-After`. `--rate` (default 20 requests/s;
  0 disables), `--burst` (default twice the rate), `--max-connections` (default 64 concurrent requests).
- **Plain HTTP:** `--allow-http` on both the owner and the node, only on a private network. Plaintext exposes tokens,
  weights and data.

## Operating the owner

The job directory holds:

| Item | Purpose |
|---|---|
| `job.json` | job identity, sample manifest, rounds, learning rate, profiles |
| `artifacts/` | programs, data chunks, checkpoints and accepted results, addressed by SHA-256 |
| `owner.sqlite` (+ WAL, SHM) | workers, leases, accepted chunks, round commits, current checkpoint |
| `owner.lock` | lock that allows one owner per directory |

- Restart with the same directory and options. Leases and completed chunks survive.
- Never run two owners on copies of one live job. The lock only protects one directory; there is no failover.
- Keep the job directory on a reliable local filesystem.
- Back up by stopping the owner and copying the directory, or with an SQLite backup.
- An epoch is complete when all its rounds are committed and every sample contributed exactly once.
- `GET /v1/status` (and `--status`) reports the checkpoint version, accepted samples, completed epochs, per-round
  losses, last-seen time per worker and throughput estimates.

Owner scheduling options:

| Option | Default | Effect |
|---|---|---|
| `--lease-seconds` | 120 | how long a node holds a bundle before it is reissued. Cover a bundle's download and compute time. |
| `--target-seconds` | 30 | bundle size target, from each node's measured throughput |
| `--max-bundle` | 4 | most chunks per bundle (1 to 64) |

## Operating nodes

The state directory holds the node identity, a verified artifact cache, pending assignments, a result outbox and
failed or rejected records.

- One state directory per process; a file lock enforces it.
- Restart with the same state directory and owner URL to continue.
- Cached work runs while the owner is offline. Results are fsynced before upload and kept until the owner
  acknowledges them.
- A failed assignment is retried after its lease expires. Leases use owner time, so clocks need no sync.
- The node sends no heartbeats during a long kernel. If that outlasts the lease, the chunk may be computed twice; only
  one result counts.

A node must fit the whole model plus its microbatch: the swarm does not shard models. Choose a microbatch that fits
the smallest worker.

| Node option | Default | Effect |
|---|---|---|
| `--device` | `vulkan:0` | `vulkan:N` or `cpu` |
| `--state` | `munet-node-state` | state directory |
| `--name` | `node` | name reported to the owner (part of its capabilities in the status) |
| `--memory-mib` | RAM/4; on Vulkan at most half the device heap | memory budget the owner schedules against |
| `--max-batch` | 1024 | largest microbatch the node accepts |
| `--samples-per-second` | measured | initial throughput estimate |
| `--poll-seconds` | 2 | wait between bundles (0.01 to 60) |
| `--once` | off | process one bundle and exit: 0 done, 1 compute failed, 2 work or results still pending |

## Limits

- Six endpoints, all bearer-authenticated: `POST /v1/register`, `/v1/lease`, `/v1/heartbeat`, `/v1/results`,
  `GET /v1/artifacts/{sha256}`, `/v1/status`.
- Protocol ABI `munet-swarm-13`: owner and node must be the same MuNet version, because generated kernels must match.
- float32 only.
- At most 100,000 parameter values per job, 256 MiB per artifact and 32 MiB per result upload.
- Data chunks and all epoch manifests are written ahead of time. Cached artifacts and results are not
  garbage-collected.
- Not benchmarked at scale.
