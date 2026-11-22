# Training pool

A pool trains one model on whatever machines are connected. One master holds the weights and the optimizer. Nodes join
with a token, take work, and may leave at any time.

- [How a step works](#how-a-step-works)
- [Train a model you already have](#train-a-model-you-already-have)
- [Start the master](#start-the-master)
- [Start nodes](#start-nodes)
- [Several GPUs in one machine](#several-gpus-in-one-machine)
- [Dashboard](#dashboard)
- [Native node](#native-node)
- [Memory limit](#memory-limit)
- [Automatic cuts](#automatic-cuts)
- [Stages](#stages)
- [Bundles](#bundles)
- [Slow links](#slow-links)
- [Row cache](#row-cache)
- [Local steps](#local-steps)
- [Nodes that leave or fail](#nodes-that-leave-or-fail)
- [BatchNorm](#batchnorm)
- [Security](#security)
- [Limits](#limits)

## How a step works

1. `pool.step(...)` cuts the batch into shards of `shard_rows` rows. Each shard is one task.
2. A node asks for work. It receives one or more shards, and the current weights when its copy is stale.
3. The node runs the gradient program on its own device and returns the losses, the gradients and the state the
   program updated (BatchNorm statistics).
4. When every shard is answered, the master averages the results weighted by shard size and applies the optimizer once.

Nodes pull work, so a faster node takes more shards.

A **row** is one training example: an image and its labels. A **shard** is `shard_rows` rows sent to a node as one
task. The dashboard hovers explain both.

The gradient program is a data-only `.mnet` graph. A node downloads it once, checks its hash, and builds kernels for its
own GPU. Different GPU vendors and CPU nodes can share a pool, and nodes need no model code.

With a loss that is a mean over rows and a model without BatchNorm, a pool step equals a single-machine step on the
whole batch, up to float rounding. [Local steps](#local-steps) give that up for throughput.

## Train a model you already have

No custom distributed code is needed. Build the model, the optimizer and a loss function as for
[single-machine training](training.md), and give them to a `pool.Pool`:

```python
model = MyNet()                                               # any model built from munet.nn
optimizer = mu.optim.AdamW(model.parameters(), lr=1e-3)

def loss_fn(*batch):                                          # the loss of one shard
    return my_loss(model(batch[0]), *batch[1:])

p = pool.Pool(loss_fn, model.parameters(), optimizer, token=token, shard_rows=8)
p.serve("0.0.0.0", 8770, context=context)                     # see Security
for batch in loader:
    loss = p.step(*batch)                                     # replaces the compiled train step
```

On each machine that should help: `python -m munet.pool https://master:8770 --device all`.

The pool traces `loss_fn` once on the master into a `.mnet` graph. Nodes download the graph, so they need no model
code and no copy of your project. They need the same MuNet version.

What your `loss_fn` must respect:

- Return the **mean** loss over the rows of the shard. For another reduction, use `p.step_shards(shards, weights)`.
- Do not read values on the host (`.item()`, `.numpy()`, Python `if` on a tensor). Tracing raises `RuntimeError`.
- Every batch argument has the same first dimension. The pool cuts all of them into shards of `shard_rows` rows.
- `uint8` and `bool` arguments travel as bytes. Everything else, including integer token ids, arrives as float32.
- Batch statistics (BatchNorm) are per shard, so pooled gradients differ slightly from a whole-batch step. See
  [BatchNorm](#batchnorm).
- Clipping, EMA and learning-rate changes go in a function passed in place of the optimizer; see
  [Start the master](#start-the-master).

A model that does not fit in one device is cut into [stages](#stages), or automatically with `memory_limit`. Each
node still runs whole programs, so the pool is data parallel with optional cuts. It does not shard one tensor across
machines.

## Start the master

```python
import os
import munet as mu
from munet import pool
from munet.swarm.server import tls_context

model = MyNet()
optimizer = mu.optim.AdamW(model.parameters(), lr=1e-3)

def loss_fn(images, labels):
    return mu.nn.functional.cross_entropy(model(images), labels)

p = pool.Pool(loss_fn, model.parameters(), optimizer, token=os.environ["MUNET_POOL_TOKEN"], shard_rows=8)
p.serve("0.0.0.0", 8770, context=tls_context("cert.pem", "key.pem"))
for images, labels in loader:
    loss = p.step(images, labels)        # blocks until the nodes have answered every shard
p.close()                                # tells the nodes the pool is finished
```

- `token`: the secret nodes join with, at least 32 characters.
- `shard_rows`: rows per task. Without it the whole batch is one task.
- `optimizer`: an optimizer, or a function to run in its place once the averaged gradients are set:

  ```python
  def update():
      mu.optim.clip_grad_norm_(model.parameters(), 0.1)
      optimizer.step()
      ema.update()
  ```

- `device`: where the master applies the update: `"vulkan"` (default), `"vulkan:N"` or `"cpu"`. It is checked when
  the pool is built. Use a GPU: the CPU executor is a single-threaded reference implementation.
- `lease_seconds`: how long a node may hold a task before another node gets it (default 600). It must cover the slowest
  node's first task, which includes building its kernels.
- `memory_limit`: bytes. The pool joins adjacent [stages](#stages) so that every program needs at most that many bytes
  of buffers. See [Memory limit for the pool](#memory-limit-for-the-pool).
- `local`: `True` makes every shard an optimizer step that a node runs itself. See [Local steps](#local-steps).
- `uint8` and `bool` arguments (images as bytes, masks) travel to nodes as one byte per value. `loss_fn` receives the
  same values as float32, so scale images there: `model(images * (1 / 255))`. Other arguments travel as float32.
- `p.step(..., timeout=seconds)` raises `TimeoutError` when the shards are not answered in time.
- `p.step_shards(shards, weights)` takes explicit shards (a list of argument lists) and one weight per shard. Use it
  when arguments are not all cut on the first dimension, or when the loss is not a mean over rows.
- `p.prefetch(images, labels)` queues a batch for a later `p.step()` called without a batch. Queued batches train in
  order. `p.prefetch_shards(shards, weights)` and `p.step_shards()` do the same for explicit shards.
  `p.steps(batches)` trains on every batch of an iterable, keeps the next one queued and yields each loss. See
  [Slow links](#slow-links).
- `p.status()` returns the weights version, queue lengths, and per node: device, tasks, rows, compute seconds,
  rejected results, `backups` (tasks it took while another node held them), `late` (results that arrived after
  another node's), `carried` (shards whose answer was late for a [local round](#local-steps)), `encoding`,
  `cache` and `hit` (see [Row cache](#row-cache)), and what the master has measured for it: `down` and `up`
  (link rate in bytes per second), `shard_seconds` (compute seconds per shard, by program) and `bundle` (the most
  shards it takes per request).
  `built` lists the gradient programs the node has built kernels for; `building` is the one it was last told to
  build, or `None`.
  `queued` is the number of batches waiting for their step. `stages` is how the stages are grouped into
  programs, as `[first, last]` pairs; it is `None` until the first batch has been cut.
  `cuts` lists the chains [automatic cuts](#automatic-cuts) built: `limit`, `programs` and `largest`, in bytes.
  `carried` is the number of late shards of local rounds whose answer has not arrived yet.
  `local` is `True` in a pool with [local steps](#local-steps).

The master keeps its state in memory. Save checkpoints from the training loop as usual.

## Start nodes

Install the same MuNet version as the master, then:

```bash
export MUNET_POOL_TOKEN=...                               # the pool's token
python -m munet.pool https://trainer.example:8770 --device vulkan:0
```

- `--device`: `vulkan:N` or `cpu`; several separated by commas, or `all` for every GPU of the machine. See
  [Several GPUs in one machine](#several-gpus-in-one-machine).
- `--precision`: `fp32` (default) or `mixed`.
- `--activations`: `fp32` (default) or `fp16`. `fp16` stores forward activations as half floats, so a program needs
  less memory and its gradients are rounded. It needs `--precision fp32` and a Vulkan device. See
  [Half-float activations](#half-float-activations).
- `--encoding`: how tensors travel to and from this node: `f32` (default, exact) or `f16` (half floats with one
  scale per tensor, half the bytes). With `f16` the node computes at weights rounded to about 3 decimal digits and
  returns rounded gradients. Packing costs time: use it on a slow link.
  `q8` is `f16`, except that what the node returns from [local steps](#local-steps) goes as 8-bit integers with one
  scale per tensor, a quarter of float32. It is for a slow uplink and costs accuracy; see [Slow links](#slow-links).
- `--bundle N`: take up to N shards per request (1 to 64, default 64) and return one result for all of them. N is a
  cap: the master chooses how many it hands out. See [Bundles](#bundles).
- `--cache DIR`: keep the rows of large batch arguments (images) in DIR, so a row this node has seen does not cross
  its link again. `--cache-size` caps the directory (default `4G`). See [Row cache](#row-cache).
- `--name`: the name shown in `status()` (default: the host name).
- `--memory`: limit on the buffers this node allocates, e.g. `2G` or `1500M`. Default: three quarters of the device
  memory on Vulkan, no limit on `cpu`. The node reports it when it joins. See [Memory limit](#memory-limit).
- `--allow-http`: allow plain HTTP to another machine. Use it only on a trusted network.
- `MUNET_CA_BUNDLE`: CA bundle file for a master with a private certificate.

A node can start before the master. It retries until the master answers, and exits when the master closes the pool.

From Python: `pool.Node(url, token, device="vulkan:0").run()`. It starts `munet-pool-node` as a child process.

## Several GPUs in one machine

Give the node a list of devices. Each device joins as its own node, and the master sizes the work of each on its
own.

```bash
python -m munet.pool https://trainer.example:8770 --device all
python -m munet.pool https://trainer.example:8770 --device vulkan:0,vulkan:1
```

- `all` is every Vulkan GPU of the machine. A software rasterizer (llvmpipe) is not one; name it to use it.
- `cpu` may be in the list.
- With several devices the nodes are named `<name>-<N>`, and `<name>-cpu` for the CPU executor.
- `--memory` is a limit per device. The other options apply to every device.
- The devices share one row cache (`--cache`).
- A device that fails is logged and the others work on. The command exits with an error once they have stopped.
- Ctrl-C stops every node: each finishes its task and leaves the pool.

From Python:

```python
from munet import pool

pool.Nodes("https://trainer.example:8770", token, "all").run()
```

`Nodes.start()`, `stop()`, `wait()` and `status()` run the nodes next to other work.

The [node page](#dashboard) has one checkbox per device.

## Dashboard

The master serves a dashboard on the address nodes connect to. Open that address in a browser and enter the pool
token. Or put the token in the address: `https://trainer.example:8770/#token=TOKEN`. The part after `#` is not sent to
the server, and the page removes it from the address bar.

The page shows:

- The step, the loss, rows per second over the last 10 steps, the connected nodes, and the shards running and waiting.
- A training chart of the loss and of every value you log. Switch each series on or off. Plot across steps, rows
  trained or minutes, on a linear or log axis, with a moving average over 10 or 50 values.
- A node chart with one line per node and one for the pool. Choose what to plot: rows per second, shards per step,
  compute seconds per shard, link speed to or from the node, late answers, answers carried to a later round, row
  cache hits, or seconds per step. Switch each node on or off.
- A table with one row per node: device, last seen, shards, rows, seconds per shard, link speeds, encoding, memory
  limit, and its late, backup, carried and rejected counts.
- **Add a node**: the one-line command that joins this pool. Choose the shell, the device,
  a name, the encoding, a memory limit and a row cache directory. The command on screen hides the token until you
  tick **Show the token**. **Copy command** copies it with the token.

The pointer, or the arrow keys when a chart has the focus, reads the values at one step. A value that is missing stays
a gap in its line.

Log your own values after a step:

```python
loss = p.step(images, labels)
p.log(held_out=evaluate(), lr=1e-4)      # named numbers for the step that just finished
```

- Each name becomes a series in the training chart. A step holds at most 32 names.
- The master keeps the last 5000 steps in memory. A master that is started again starts with an empty chart.
- The page asks the master for what changed every two seconds.

### Join from a page

On the machine that should join, run:

```
python -m munet.pool ui
```

1. It prints an address that carries the page's key and opens it in a browser. `--no-browser` only prints it.
   `--port N` changes the port (default 8771).
2. Paste the command from the dashboard, copied with its token. The form fills in. Or fill in the fields by hand.
3. Submit the form. The page then shows the node's state, the shards it computed, its compute time, the programs it
   built, a chart of shards per minute, and its log.

**Leave the pool** stops the node after its current task. Ctrl-C in the terminal leaves the pool and stops the page.

- The page serves `127.0.0.1` only. To use it on a machine without a screen, forward the port:
  `ssh -L 8771:127.0.0.1:8771 node-host`.
- It runs `munet-pool-node` as a child process.

## Native node

`munet-pool-node` is the node. `python -m munet.pool` and `pool.Node` start it as a child process, and the wheels
carry it in `munet/bin`. It needs no Python, so the same binary runs on phones and on machines that only have the C
library. It takes every option above. It is a POSIX program: there is no Windows node.

Each release ships it:

- Linux: `munet-pool-node-VERSION-linux-ARCH.tar.gz`, entry point `bin/munet-pool-node`.
- Android: `bin/<abi>/munet-pool-node` in `munet-android-VERSION.tar.gz`.

```bash
export MUNET_POOL_TOKEN=...
munet-pool-node https://trainer.example:8770 --device vulkan:0 --ca ca.pem
```

Or build it:

```bash
cmake -S . -B build/pool-node -DMUNET_PYTHON=OFF -DMUNET_C_API=ON -DMUNET_POOL_NODE=ON
cmake --build build/pool-node --target munet-pool-node
```

- It links the C API and a vendored TLS client (BearSSL, `cpp/third_party/bearssl`). It needs no system HTTP or TLS
  library.
- An `https` master needs `--ca FILE` or `MUNET_CA_BUNDLE`: PEM CA certificates. The system trust store is not read.
- TLS 1.2 only. The certificate must name the master's host as a DNS name; a certificate issued for an IP address is
  not accepted.
- Linux: a Vulkan device needs `glslangValidator` on `PATH` (or `MUNET_GLSLANG`) to build kernels. `--device cpu`
  does not.
- Android: push the binary to `/data/local/tmp` and run it from `adb shell`. It builds kernels in-process and needs
  only `libc`, `libm` and `libdl`. To build it yourself, add the NDK toolchain file, `-DMUNET_SHADERC=ON` and
  `-DMUNET_SHADERC_ROOT=...` as [android.md](android.md) describes. Over USB,
  `adb reverse tcp:8770 tcp:8770` lets the phone reach a master on the computer's loopback address.

## Memory limit

`--memory 2G`, or `pool.Node(url, token, memory_limit=2 * 2**30)`, caps the buffers a node allocates for gradient
programs.

- The node sizes a program before it allocates anything. A program over the limit is declined, and the master stops
  offering that program to that node.
- The size checked is what the program holds once loaded. On Vulkan that is its arena on the device, the weights kept
  in host memory, a staging buffer of at most 16 MiB for reads and writes, and two input buffers. On the CPU executor
  it is the arena and the weights.
- Programs that fit share the limit. The node drops the least recently used program to make room for another.
- The limit covers MuNet's buffers only. Memory the GPU driver maps on top is not counted, so leave headroom.
- `status()` shows each node's `memory_limit` and the programs it `declined`, with the bytes each one needs there.

### Joined shards

A node with memory to spare runs several shards as one. The master traces the program for 2, 4, 8, ... shards' rows
and gives each node the largest one that fits its limit. The node joins those shards' inputs and runs them once, so
it does more work per weight download. In [local steps](#local-steps) a joined run is one larger optimizer step.
A node runs one joined size and the single-shard program, so it never drops programs to make room.
`status()` shows each node's `run_rows`. Joining applies to a pool given one function, not to stages or cuts.

A pool given one function cuts the model for a node that cannot hold it; see [Automatic cuts](#automatic-cuts).

When no connected node can take a shard, the step waits and logs a warning every 10 seconds; with `timeout=` it
raises `TimeoutError` naming the size needed. That happens when a part of the model that cannot be cut is over every
node's limit, or in a pool given [stages](#stages), which the master does not cut smaller. Lower `shard_rows`, or
write smaller stages. `Pool(..., memory_limit=...)` joins stages to fit a limit; see
[Memory limit for the pool](#memory-limit-for-the-pool).

### Half-float activations

`--activations fp16`, or `pool.Node(url, token, activations="fp16")`, stores the forward activations of each gradient
program as half floats. The node sizes programs with it, so more of them fit under `--memory`.

- Gradients, indices, sampling positions, updated state and results stay float32.
- Stored activations round to 11 bits and saturate at 65504, so the node's gradients differ from float32 ones.
- It saves memory where forward activations make a program's peak, and little where gradients do.

The gradient differs from the float32 one, most in layers that predict sampling offsets. See
[training.md](training.md#half-float-activations).

## Automatic cuts

Give the pool one function and it cuts the model itself. You write no stages:

```python
p = pool.Pool(loss_fn, model.parameters(), optimizer, token=token, shard_rows=1)
```

1. Every shard starts as one program.
2. A node reports its memory limit when it joins.
3. When the whole program is over a node's limit, the master traces `loss_fn`, cuts its graph where few bytes cross,
   and sizes the pieces until each one fits that limit.
4. That node gets shards as a chain of the smaller programs, run like [stages](#stages). A node that holds the whole
   program keeps taking whole shards.

- The master cuts once per shard shape and node limit, on the training thread. Nodes keep working meanwhile.
- Sizes are for the master's device. When a node declines a program the master sized within its limit, the master
  lowers that node's limit by the same ratio and cuts again.
- `Pool(loss_fn, ..., memory_limit=bytes)` cuts every shard to that limit, whatever the nodes report. It raises
  `ValueError` when the model cannot be cut that small.
- Not cut: the operations from the first that draws random numbers on (dropout in training mode), and a `loss_fn`
  that takes gradients itself.
- A node whose limit is under a part that cannot be cut gets no work. The master logs the size.
- `p.status()["cuts"]` lists each chain built: the limit, the number of programs and the largest program's bytes.

A limit costs time. A node holds only the programs that fit its limit together, so it drops and rebuilds programs
from its kernel cache every step.

## Stages

A list of stage functions cuts the model where you choose. The pool runs the stages as written and does not cut
inside one:

```python
def backbone(images, labels):
    return model.backbone(images)            # a tensor, or a tuple of tensors

def head(features, images, labels):          # what the stage before returned, then the shard
    return mu.nn.functional.cross_entropy(model.head(features), labels)

p = pool.Pool([backbone, head], model.parameters(), optimizer, token=token, shard_rows=8)
```

- Each stage receives what the stage before it returned, followed by the shard. The last stage returns the loss.
- Each stage is its own program. A shard runs as a chain of tasks: the forward program of every stage but the last, the
  last stage (loss and gradients), then the backward program of each earlier stage, last to first.
- Any node may take any task of the chain. A node with a memory limit declines the programs that do not fit and takes
  the others.
- A node receives only the weights of the stages it runs.
- A pool step with stages equals the same step without them, up to float rounding.
- BatchNorm works in any stage.
- An earlier stage with nothing to train or update, and nothing trainable below it, has no backward program.
- A stage can be as small as one layer. When the loss reads what several stages computed, each stage returns those
  tensors with its own and the next stage passes them on.
- A node takes several shards of one program per request, as in a pool without stages; see [Bundles](#bundles).

Costs:

- Every stage but the last runs its forward pass twice: once on its own and once inside its backward program.
- The tasks of one shard run one after another, and the tensors passed between stages travel through the master.
- Stages before the last must not draw random numbers, such as dropout in training mode. The pool raises `ValueError`.

### Memory limit for the pool

Write small stages and let the pool choose where to cut:

```python
p = pool.Pool([stem, stage1, stage2, stage3, encoder, head], model.parameters(), optimizer, token=token,
              shard_rows=1, memory_limit=640 * 2**20)
```

- The pool sizes the programs of every run of adjacent stages, for the master's device, when it first cuts a shard of
  a given shape.
- It picks a grouping whose largest program is within the limit. Of those it takes the one that passes the fewest
  bytes between groups, then the one with the fewest groups.
- Each group runs as one stage. `p.status()["stages"]` shows the groups.
- When no grouping fits, the step raises `ValueError` and names the smallest size that can be reached.
- The size is the one a node's `--memory` limit is compared with (see [Memory limit](#memory-limit)). A node sizes
  each program for its own device, and its own limit still decides what it takes.

## Bundles

A node takes up to `--bundle` shards per request (default 64) and answers once for all of them. The master
chooses how many it hands out. `--bundle 1` answers every shard on its own.

A node keeps the weighted sum of the parameter gradients on its device, in every program that returns them: a pool's
one program, and the last and backward programs of [stages](#stages) and [automatic cuts](#automatic-cuts). Per shard
it reads back only the loss, the gradients the stage below needs, and the state the program updated. It reads the sums
once per task.

The sum is added on the device in float32. Single answers are averaged on the master in float64, so a pool with
bundles and one with a task per shard differ by float rounding.

A handout of a chain's program carries at most 256 MiB of the tensors passed between programs, each way.

## Slow links

Each step a node downloads the weights that changed, and each task uploads one gradient set. Either is the size of the
gradients: 4 bytes per parameter as float32. Two node options cut that:

- `--encoding f16` halves the bytes.
- `--encoding q8` does the same and halves again what a node returns from [local steps](#local-steps).
- `--bundle N` (default 64) takes up to N shards in one request and returns one gradient set for all of them, so N
  shards cost one upload. See [Bundles](#bundles).

Pass images to `p.step()` as `uint8`: a 640 px RGB row is 4.69 MiB as float32, 2.34 MiB as half floats and 1.17 MiB
as bytes. See [Start the master](#start-the-master).

A node started with `--cache DIR` downloads each image row once. See [Row cache](#row-cache).

A node asks for its next task while it computes the current one, and uploads a result while it computes the next.

Queue the next batch and its inputs travel before its step begins:

```python
for loss in p.steps(loader):             # each item of loader holds the arguments of p.step()
    print(loss)
```

A node with nothing to ask for then receives the inputs of the shards the master expects to hand it in the next step,
up to its `--bundle`. When that step begins, those shards cost its link only the new weights.

The master chooses how many shards a node gets, up to its `--bundle`. It measures each node's link rate in both
directions (on messages of at least 1 MiB) and its compute seconds per shard. A node gets what it can deliver by the
time the rest of the pool has finished the other shards, and nothing when it could not add a shard in that time.

Until a node has answered one task of a program, the master has nothing to size from. The node then gets its share of
the step's unanswered shards: their count divided by the number of nodes, rounded up, and at most `--bundle`. On a slow
link that first bundle arrives too late, so the first two steps are slower than without the node.

Pools with [stages](#stages) and [automatic cuts](#automatic-cuts) hand out bundles too:
a node gets its share of the unanswered tasks.

## Row cache

A node started with `--cache DIR` keeps the rows of large batch arguments on its disk. A row it has seen does not
cross its link again.

```bash
python -m munet.pool https://trainer.example:8770 --device vulkan:0 --cache ~/munet-rows --cache-size 20G
```

1. The master hashes every row of a batch argument whose rows are at least 64 KiB. Smaller arguments, such as labels
   and boxes, travel as before.
2. A node with a cache receives the row keys in place of those tensors.
3. The node reads the rows it holds from DIR, fetches the others from the master, and stores them.

- A key is the hash of a row's content. Shuffling between epochs does not matter: a row hits in whatever shard it
  lands.
- `--cache-size` is the most bytes DIR may hold (default `4G`). Over it, the least recently used rows go.
- A 640 px RGB row takes 1.17 MiB as `uint8`. Float rows are stored as the node's `--encoding` carried them, so a
  node that changes its encoding fetches them again.
- A later run on the same data finds its rows in DIR.
- `status()` reports `cache` and `hit` per node: the share of the bytes sent by key that the node did not fetch for
  its last task. The master counts only the rest against the node's link when it sizes the node's share.
- Rows the master changes every epoch (random augmentation) never hit.
- DIR holds training data unencrypted. See [Security](#security).
- From Python: `pool.Node(url, token, cache_dir=DIR, cache_bytes=20 * 2**30)`.

## Local steps

`Pool(..., local=True)` trades exactness for throughput. Nodes run the optimizer themselves and exchange weights with
the master once per round, not once per step.

```python
p = pool.Pool(loss_fn, model.parameters(), optimizer, token=token, shard_rows=4, local=True)
for images, labels in loader:            # 128 rows: 32 local steps of 4 rows, shared between the nodes
    loss = p.step(images, labels)        # one round
```

1. `p.step(...)` cuts the batch into shards of `shard_rows` rows. Each shard is one optimizer step.
2. A node takes several shards as one task. It starts from the master's weights and runs one training step per
   shard, each from where the one before ended.
3. The node returns how far the weights and the BatchNorm statistics moved.
4. The master adds the mean of what the tasks moved, weighted by rows. It runs no update itself.

- `shard_rows` is the batch size of a local step. Give a round many shards: a round costs each node one download
  and one upload of the weights, whatever its length.
- A node runs at most `--bundle` shards per task (default 64).
- A round does not wait for a late answer. A task is late once it has taken 1.1 times the seconds the master
  expected of its node, link included, and 0.25 s. When another node then has nothing to do, the round ends
  without that task. Its answer goes into the round in training when it arrives. No node computes a local step
  twice, and a round keeps at least one answer.
- A late answer started from older weights. The master adds it like the others, weighted by rows.
- A node on a slow link runs one task per round on the weights of the round before, which it already holds. The new
  weights download with its next request while it computes. Its answer counts as one round late.
- A late answer computed from weights more than two rounds old is dropped: its shards are trained again in the
  round in training, and the node's `late` count goes up.
- Late shards whose node went away are trained again in the round in training.
- `p.flush()` waits for the late answers still on their way and takes them in. Call it after the last `p.step`;
  `p.steps(batches)` calls it. It weighs them against the last batch, so they move the weights as far as they
  would have in a round. It returns their mean loss, or `None` when nothing was late. `close()` logs a warning
  when late shards were left out.
- `optimizer` is traced into the node's program, with gradient clipping or anything else an update function does.
  Learning rates and other optimizer settings go out with every round.
- Optimizer state (moments, step counts) and an EMA kept by the update function live on each node and are never
  sent. A node that joins later or reloads the program starts them from their initial values. The master's
  optimizer takes no steps.
- `p.step` returns the mean loss of the round's local steps, weighted by rows.
- It needs one loss function: a list of stages or `memory_limit` raises `ValueError`, and the training step is never
  cut.
- The result differs from single-machine training.

## Nodes that leave or fail

- A node that stops tells the master, and its task goes to the next node that asks.
- A node that vanishes loses its task when the lease ends, or at once when the master sees the connection drop.
- A node with nothing else to do also computes a task another node holds for too long: more than 1.5 times the
  seconds the master expected that node to need, its link included, and 0.25 s. For a node the master has not
  measured, the limit is twice the fastest answer seen for that program (at least 0.25 s). A slow node or a slow
  link then costs duplicate work instead of step time. Only a node that has built the task's program backs it up.
- A pool with [local steps](#local-steps) computes no task twice. A late answer leaves its round and goes into a
  later one.
- A node can join during training. The master names the gradient program it needs, the node builds the kernels and
  reports the program as built, and only then is it handed shards of it. Until then it holds no task, and the
  shares of the other nodes do not count on it. A node that joins between steps builds the programs of the last
  batch.
- A result with the wrong shapes or non-finite values is rejected, and the shard is computed again.
- With no node connected, a step waits and logs a warning every 10 seconds.
- `close()` keeps answering until every connected node has heard that the pool is finished, for at most 5 seconds
  (`close(wait=...)`). A node that was busy for longer keeps retrying, as it would for a restarting master: stop it
  yourself.

The first valid result for a task wins. Later answers for the same task are ignored.

## BatchNorm

Each shard uses its own batch statistics. When a batch is cut into several shards, the gradients therefore differ from
a single-machine step on the whole batch. The running statistics are averaged over the shards.

One shard per step reproduces single-machine training exactly.

## Security

- Serving on a non-loopback address needs TLS (`context=`), or `allow_http=True` on a trusted network. Plain HTTP
  exposes the token, the weights and the data.
- A node joins with the pool token and then uses its own node token.
- Every node receives the weights and the training data of its shards.
- A node started with `--cache DIR` keeps the rows it received in DIR, unencrypted, until the cache drops them or you
  delete DIR.
- The master checks shapes and finiteness of results, nothing more. Anyone holding the token can send wrong gradients.
  Give the token only to machines you trust.
- Anyone who can reach the master can load the dashboard's files. What the dashboard shows needs the pool token. The
  page keeps the token for the browser tab only, until the tab closes or you press **Lock**.
- A node's page serves `127.0.0.1` only, and acts only on requests that carry the key in its address. Whoever has that
  address can make the machine join a pool.
- There is no rate limiting.

## Limits

- Phones join with `munet-pool-node`. There is no Android app; the node runs from a shell.
- The node is a POSIX program, built for Linux and Android. Wheels for other systems have none, and `pool.Node`
  raises an error there.
- [Automatic cuts](#automatic-cuts) are sized for the master's device, and a node under a limit rebuilds programs
  every step.
- With a list of stages, the pool's `memory_limit` joins the stages you wrote and never cuts inside one, so the
  largest single stage is the floor.
- A program that keeps gradient sums stores them in its file as zeros and holds them on the device, so the file and
  its buffers grow by the size of the parameters it trains. In a chain every last and backward program holds the sums
  of the parameters it trains.
- A step waits for all of its shards. Idle nodes take over tasks that slow nodes hold (see
  [Nodes that leave or fail](#nodes-that-leave-or-fail)); when every node is busy, the slowest one sets the pace.
- A [local](#local-steps) round ends without a late answer only once another node has nothing to do. With one
  node, or while every node is busy, it waits.
- Tensors are not compressed beyond `--encoding f16` and `q8`.
- A node helps only when its weights and gradients cross its link in less time than the rest of the pool needs for
  the step. Otherwise the master gives it no work; see [Slow links](#slow-links).
- The master sizes bundles from measurements only for shards that run as one program, and only for a node that has
  answered one task of the program, in time or too late.
- A node that gets no work is not measured again. When its link gets faster, restart it.
- Link rates are measured on the master. Off Linux the download rate counts bytes still in the send buffer as sent,
  so it reads high.
- Inputs go ahead of a step only for shards that run as one program, only for the oldest queued batch, once per node,
  and only to a node that has answered one task of the program.
- A queued batch stays in the master's memory until its step.
- [Local steps](#local-steps) keep optimizer state on the nodes: it is not in a checkpoint saved on the master,
  and every node needs room for the whole training step.
- Every distinct shard shape is its own gradient program, built once per node. Keep batch sizes a multiple of
  `shard_rows`.
- Master and nodes must run the same MuNet version and the same pool protocol. The master refuses a node that
  differs when it joins.

The older [training swarm](swarm.md) runs fixed, precomputed jobs with native nodes. It supports MSE loss and plain
SGD only.
