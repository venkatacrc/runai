# Phase 0 — CLI Login & Quota Check

Goal: at the end of this page you can run `runai project list` and see your
GPU quota, and you've picked a project to send jobs to.

Time: ~15 minutes (most of it is browser SSO).

---

## 0.1 Install the `runai` CLI v2

Run:AI's CLI is downloaded **from your own cluster**, because the installer
embeds the control-plane URL.

1. Open your Run:AI portal in a browser (the original lab used `https://runai.prod.walmart.com`; use your org's URL), sign in via SSO.
2. Click the **?** (Help) icon in the top-right corner.
3. Choose **Researcher Command Line Interface**.
4. Pick:
   - Cluster: the GB200 cluster you have access to
   - Operating system: **Mac (Apple Silicon)** — you're on `darwin/arm64`
5. Copy the `curl ... | bash` (or `curl + chmod + mv`) command shown.
6. Paste into a terminal and run it.

It typically looks like (substituting your portal hostname):

```bash
curl -L "${RUNAI_URL}/cli/darwin/arm64/runai" -o /tmp/runai \
  && chmod +x /tmp/runai \
  && sudo mv /tmp/runai /usr/local/bin/runai
```

If the portal hands you a slightly different command, **use that one** — it
already encodes the right control-plane URL.

### Verify install

```bash
runai version
which runai
```

Expect a version string like `v2.20.x` or newer.

---

## 0.2 Authenticate

```bash
runai login
```

This opens a browser tab pointing at your org's SSO. After signing in, the
terminal will say *"Login Succeeded"* and write a token under `~/.runai/`.

Sanity check:

```bash
ls ~/.runai/
runai whoami        # or: runai config view
```

---

## 0.3 Find the cluster, project, and GPU node pool

```bash
runai cluster list
```

You should see at least one cluster. Note the **name** (we'll need it later if
there's more than one).

```bash
runai project list
```

Run:AI projects are namespaces that own GPU quota. You'll likely see one or
more projects you belong to. Pick the one that owns the GB200 quota for today.

Set it as your default:

```bash
runai project set <YOUR_PROJECT_NAME>
runai config view                # verify "project" field
```

---

## 0.4 Verify GPU quota / availability

There are two angles: **what you're allowed to use** and **what is physically there**.

### What you're allowed to use (project quota)

```bash
runai project list -o yaml | grep -A6 <YOUR_PROJECT_NAME>
```

Look for `gpus` / `deserved` / `limit` fields.

In the **portal** (often easier): Projects → click your project → "Resources"
tab shows GPU quota numerically.

### What is physically there (the node pool)

```bash
runai node list
```

Look for nodes whose GPU type contains `GB200` or `Blackwell`. Note:
- node name (e.g. `gb200-node-01`)
- GPU count per node (8 if it's a standard GB200 NVL tray slice)
- the node-pool label (you'll pass this with `--node-pools` later)

If `runai node list` is empty for you (some projects can't list nodes), use:

```bash
kubectl get nodes -L nvidia.com/gpu.product,run.ai/node-pool
```

with the kube-context that matches the Run:AI cluster.

### Burn a 10-second pod to truly confirm

This is the most honest quota check — actually request a GPU and see if it
schedules.

```bash
runai workspace submit phase0-hello \
  --image nvcr.io/nvidia/pytorch:25.04-py3 \
  --gpu-devices-request 1 \
  --command -- bash -lc 'nvidia-smi -L; echo OK; sleep 5'

runai workload list
runai workload logs phase0-hello
runai workload delete phase0-hello
```

If you see `GPU 0: NVIDIA GB200 ...` in the logs — you're done with Phase 0.

---

## 0.5 Record the values we'll reuse

Fill these into `scripts/env.sh` (already stubbed in the repo) so every later
phase is one-liner-able:

```bash
export RUNAI_PROJECT=<your project>
export RUNAI_CLUSTER=<your cluster>
export RUNAI_NODE_POOL=<gb200 node pool name, if any>
export RUNAI_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3
# Optional: a PVC name you have write access to for checkpoints / datasets
export RUNAI_PVC=<pvc-name-or-empty>
# HuggingFace token for Phases 2-4 (gated models, dataset downloads)
export HF_TOKEN=hf_xxx
```

Source it once per shell:

```bash
source scripts/env.sh
```

---

## Troubleshooting

| Symptom | Likely fix |
|---|---|
| `runai: command not found` after install | `/usr/local/bin` not in PATH, or installer wrote elsewhere — `which -a runai`, then re-symlink. |
| `runai login` opens browser but terminal hangs | Local port 8000/8080 in use; close conflicting process or pass `--port` if supported. |
| `runai project list` returns empty | You may not be added to any project — ask your Run:AI admin to add you to the GB200 project. |
| `workspace submit` says `quota exceeded` | Someone else is using the GPUs; `runai workload list -A` to see who. |
| Pod stuck in `Pending` with `0/N nodes available: ... insufficient nvidia.com/gpu` | Wrong node pool, or you asked for more GPUs than the project quota allows. |

---

Once you've seen `nvidia-smi -L` report a GB200 from inside a job, move on to
[Phase 1](../phase1-sanity/notes.md).
