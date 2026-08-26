---
title: RAY cluster/concepts basics
---

## Overview

In this article we describe the basics of RAY - what is it and how is this useful.

## Ray Core Concepts

Ray Core is a distributed computing framework built around a small set of primitives - **tasks**, **actors**, and **objects** - for turning ordinary Python functions/classes into distributed workloads.

1. **`ray.init()` - starting the cluster.** Every Ray program begins by importing and initializing Ray. If you skip this call, the first remote API call implicitly runs `ray.init()` with no arguments for you.

    ```python
    import ray

    ray.init()
    ```

2. **Tasks - stateless remote functions.** Decorate a function with `@ray.remote`, call it with `.remote()` instead of a normal call, and retrieve the result with `ray.get()`. Each `.remote()` call returns immediately with a future (an object reference); the four `square` calls below run in parallel.

    ```python
    # Define the square task.
    @ray.remote
    def square(x):
        return x * x

    # Launch four parallel square tasks.
    futures = [square.remote(i) for i in range(4)]

    # Retrieve results.
    print(ray.get(futures))
    # -> [0, 1, 4, 9]
    ```

3. **Actors - stateful remote workers.** Decorating a class with `@ray.remote` and instantiating it starts a dedicated worker process that keeps its own internal state between method calls. Calls to a given actor run serially, in submission order, so state updates stay consistent.

    ```python
    # Define the Counter actor.
    @ray.remote
    class Counter:
        def __init__(self):
            self.i = 0

        def get(self):
            return self.i

        def incr(self, value):
            self.i += value

    # Create a Counter actor.
    c = Counter.remote()

    for _ in range(10):
        c.incr.remote(1)

    # Retrieve final actor state.
    print(ray.get(c.get.remote()))
    # -> 10
    ```

4. **Futures / non-blocking, asynchronous execution.** `.remote()` never blocks - it hands back an object reference right away while the actual computation runs on a worker elsewhere in the cluster. This is what lets `[square.remote(i) for i in range(4)]` fan out four tasks before a single result is collected; `ray.get()` is the only point where the caller actually waits.

5. **Zero-copy object passing via `ray.put()`.** Lazy execution has ben around for multiple-decades at this point and Ray is no exception to that. Passing a large object by value re-serializes and copies it into every task that receives it. Passing an `ObjectRef` instead lets every task read the same entry out of the shared object store, avoiding unnecessary data copying and enabling lazy execution:

    ```python
    import numpy as np

    # Define a task that sums the values in a matrix.
    @ray.remote
    def sum_matrix(matrix):
        return np.sum(matrix)

    # Call the task with a literal argument value.
    print(ray.get(sum_matrix.remote(np.ones((100, 100)))))
    # -> 10000.0

    # Put a large array into the object store.
    matrix_ref = ray.put(np.ones((1000, 1000)))

    # Call the task with the object reference as an argument.
    print(ray.get(sum_matrix.remote(matrix_ref)))
    # -> 1000000.0
    ```


6. **Same code, laptop to cluster.** Ray Core's primitives are designed so the identical `@ray.remote` task/actor code that runs under a local `ray.init()` also runs unchanged on a real multi-node Ray cluster - Ray's scheduler decides where each task/actor actually executes. This is the property that lets the three simple primitives above compose into "virtually any distributed computation pattern".

7. **Compute graphs (task DAGs).** When one task's output (an `ObjectRef`) is passed as the input to another `.remote()` call, Ray doesn't run them independently - it links them by that data dependency and builds an implicit directed acyclic graph (DAG) of tasks, scheduling each one as soon as its inputs are ready. Chaining `double.remote(square.remote(x))` below never materializes an intermediate value on the caller; Ray schedules `square` first and feeds its result straight into `double` on the cluster.

    ```python
    @ray.remote
    def square(x):
        return x * x

    @ray.remote
    def double(x):
        return x * 2

    ref = double.remote(square.remote(3))
    print(ray.get(ref))
    # -> 18
    ```

    Ray also exposes this graph explicitly through the **Ray DAG API**: instead of firing remote calls eagerly, you `bind()` tasks/actor methods to an `InputNode` to describe the graph up front, then compile and execute it (repeatedly, with much lower per-call overhead than eager `.remote()` calls):

    ```python
    from ray.dag import InputNode

    with InputNode() as inp:
        dag = double.bind(square.bind(inp))

    compiled_dag = dag.experimental_compile()
    print(ray.get(compiled_dag.execute(3)))
    # -> 18
    ```
Now we can add arbitrary number of nodes in this graph and we can provide intercept methods to poke into them as processes are happening in Ray.

8. Ray can be integrated with many other library/frameworks as SLURM, verl, pytorch, pytorch lightning, vLLM and many more.



## Examples
### Problem 1: Multi-node multi-GPU linear model train using RAY

Let's say now we have two slurm managed gpu nodes. In each of nodes we have 4 gpus. We want to train this linear model on billion of data points, over 45 epochs. 

`submit_train_job.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=ray-train
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379

echo "Starting Ray head on $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port --num-gpus=4 --block &
sleep 10

echo "Starting Ray worker on ${nodes[1]}"
srun --nodes=1 --ntasks=1 -w "${nodes[1]}" \
    ray start --address "$head_node_ip:$port" --num-gpus=4 --block &
sleep 10

# driver runs on the head node, joins the cluster that's now up
srun --nodes=1 --ntasks=1 -w "$head_node" python train.py

```

`train.py`:

```python
import ray
import torch
import numpy as np

ray.init(address="auto")                 #attach to the SLURM-launched cluster instead of ray.init(num_gpus=..)

DIM = 128
NUM_WORKERS = 8                          # 2 nodes x 4 GPUs
BATCH_SIZE = 8192                        # mini-batches, full-shard grad is infeasible at this scale
EPOCHS = 45                              # explicit epoch target
LR = 0.01


@ray.remote(num_gpus=0, max_restarts=3)
class ParameterServer:
    def __init__(self, dim):
        self.weights = torch.zeros(dim, device="cuda")

    def apply_gradient(self, grad, lr=LR):
        self.weights -= lr * grad.to("cuda")
        return self.weights.cpu()

    def get_weights(self):
        return self.weights.cpu()


@ray.remote(num_gpus=1, max_restarts=3, max_task_retries=-1)
class DataWorker:
    def __init__(self, shard_id, shard_path, batch_size=BATCH_SIZE):   # takes a file path, not arrays
        data = np.load(shard_path, mmap_mode="r")     # memory-mapped, never fully loaded into RAM
        self.id = shard_id
        self.X, self.y = data["X"], data["y"]
        self.n = self.X.shape[0]
        self.batch_size = batch_size
        self.perm = np.random.permutation(self.n)
        self.cursor = 0
        self.epoch = 0
        print(f"[worker {shard_id}] GPU(s): "
              f"{ray.get_runtime_context().get_accelerator_ids()['GPU']}, rows: {self.n}")

    def _next_batch_idx(self):                        # epoch-aware batching
        end = self.cursor + self.batch_size
        if end > self.n:
            self.epoch += 1
            self.perm = np.random.permutation(self.n)
            self.cursor, end = 0, self.batch_size
        idx = self.perm[self.cursor:end]
        self.cursor = end
        return idx

    def compute_gradient(self, weights):
        idx = self._next_batch_idx()
        x = torch.tensor(self.X[idx], dtype=torch.float32, device="cuda")
        y = torch.tensor(self.y[idx], dtype=torch.float32, device="cuda")
        w = weights.to("cuda")
        error = x @ w - y
        grad = x.T @ error / len(y)
        return self.id, grad.cpu(), self.epoch        # report epoch back to driver


SHARD_PATHS = [f"/shared/data/shard_{i}.npz" for i in range(NUM_WORKERS)]   # pre-sharded on shared FS

ps = ParameterServer.remote(DIM)
workers = [DataWorker.remote(i, SHARD_PATHS[i]) for i in range(NUM_WORKERS)]

initial_weights = ray.get(ps.get_weights.remote())
in_flight = {w.compute_gradient.remote(initial_weights): w for w in workers}
worker_epochs = {w: 0 for w in workers}

step = 0
while min(worker_epochs.values()) < EPOCHS:            # run until every worker finishes 45 epochs
    ready, _ = ray.wait(list(in_flight.keys()), num_returns=1)
    done_ref = ready[0]
    worker = in_flight.pop(done_ref)

    try:
        worker_id, grad, epoch = ray.get(done_ref)
    except ray.exceptions.RayActorError:
        print(f"[step {step}] worker died, skipping update")
        in_flight[worker.compute_gradient.remote(initial_weights)] = worker
        continue

    worker_epochs[worker] = epoch
    new_weights_ref = ps.apply_gradient.remote(grad)
    if step % 500 == 0:
        print(f"[step {step}] worker {worker_id} epoch {epoch}")

    in_flight[worker.compute_gradient.remote(new_weights_ref)] = worker
    step += 1

final_weights = ray.get(ps.get_weights.remote()).numpy()
print("training complete, final weights:", final_weights)
ray.shutdown()

```

Now we have recipe for training models, we can use same pattern for pytorch model training. Ray can be used for any other tasks like inferenceing on LLM/FM, crawl webpages, get statistics of planet scale people, do weather simulation using data from multiple countries and aggregate them etc. I guess except peace, Ray can do all for us. 

