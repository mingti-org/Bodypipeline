# Embedded Viewer Gateway and User Execution Safety Design

## Context

The Server listens on port 8081 and owns authentication, workbench APIs, capture
uploads, conversion jobs, and the authenticated Slurm user session broker. The
Visualizer Gateway currently runs as a separate FastAPI service on port 8082,
while the Next.js Visualizer UI runs on a loopback-only port managed by
`ViewerManager`.

The split creates two browser origins for one workflow. It also causes
Visualizer proxy calls to authenticate to the Server with the shared machine
token. Every proxied action is therefore attributed to `service-account`, even
when the browser user authenticated with their own H800 SSH credentials.

The baseline examined for this design is branch
`Guoyunhan/instruction-trim-rebase` at commit `4b6ef7c`. The work must preserve
the existing workbench, capture, conversion, Visualizer, manual trim,
instruction editing, review export, recycle, and annotation behavior.

## Goal

Move the Visualizer transport and API surface behind port 8081 without exposing
Server routes to a Visualizer catch-all, then execute Visualizer compute tasks
with the submitting user's existing H800 ControlMaster session. Machine-token
and internal jobs continue to run as `service-account`.

## Safety Invariants

1. Authenticated identity comes only from
   `request.state.principal`, populated by the Server security middleware.
2. Usernames, UIDs, SSH execution handles, and login nodes are never accepted
   from request JSON, query parameters, or client-controlled headers.
3. A personal execution job stores `owner_user_id`, `owner_uid`,
   `submitted_as`, `slurm_user`, `slurm_execution_id`, and `submission_auth`
   from `Principal.job_metadata()`.
4. `slurm_execution_id` is an internal execution handle. API responses,
   browser state, UI code, logs, error messages, receipts, and checkout artifacts
   never expose it.
5. A missing or expired personal SSH session fails the task with a clear
   authentication error. Execution never falls back to the Server OS account,
   `service-account`, another user's session, or another user's ControlMaster.
6. Service-token, device-token, and internal jobs retain their existing
   `service-account` behavior. Legacy persisted jobs without
   `slurm_execution_id` remain resumable in that mode.
7. Remote workload execution uses the submitter's OpenSSH ControlMaster.
   Durable queue state, shared indexes, locks, and publication bookkeeping remain
   owned by the Server account. Any dataset content created by the remote
   workload must be made readable to the Server through the existing explicit
   publication step.
8. The same user, dataset, and operation deduplication key may reuse an active
   job. A different user may not inherit or mutate another user's active job
   through deduplication.
9. Cookie-authenticated mutating requests continue to require a same-origin
   `Origin` header.
10. Browser sessions remain HttpOnly and SameSite=Strict. Passwords are used
    once to create the SSH ControlMaster and are never persisted, logged, or
    placed in process arguments.
11. Embedded Visualizer routing uses explicit prefixes and route handlers.
    `/api/*`, `/assets/*`, uploads, workbench pages, and unknown paths are never
    sent to the Next.js process by a catch-all.
12. The existing Server root remains the workbench entry point. A Visualizer
    root redirect is available only in the standalone 8082 deployment.

## Architecture

The target request and execution boundaries are:

```text
browser
  |
  | same origin, HttpOnly pipeline_server cookie
  v
Server FastAPI :8081
  |-- workbench / assets / uploads / Server APIs
  |-- allowlisted Visualizer transport
  |     |-- /bp/<job>/...
  |     |-- /_next/...
  |     |-- /_data/...
  |     `-- /favicon.ico
  |-- native Server APIs for templates, instructions, review and execution
  |
  | request.state.principal
  v
queued task metadata
  |
  | bind_record_session(row)
  v
SlurmIdentityBroker -> submitter's OpenSSH ControlMaster
  |
  v
srun / LeTools / instruction and score workers
```

The Next.js Visualizer remains a loopback-only child process managed by
`ViewerManager`. Embedding the Gateway does not move Next.js into the Python
process and does not expose its internal port.

## Gateway Boundary

### Shared transport

Extract the transport routes currently implemented in
`pipeline/server/viewer.py` into a focused module. Both `create_gateway()` and
`create_app()` call the same registration function. The embedded application
reuses the Server's persisted `Store`, `ViewerManager.registry`,
`EpisodeScoreCache`, and `RecycleBin`; it must not open a second logical state
database or race a separate preview registry.

### Explicit route allowlist

The embedded Gateway may own these route families:

| Route | Responsibility |
| --- | --- |
| `/bp/{dataset_id}/trash` | Recycle page |
| `/bp/{dataset_id}/{path:path}` | Visualizer page and `/resolve/main/...` data |
| `/_next/{path:path}` | Audited Next.js static and RSC resources |
| `/_data/{path:path}` | Registered dataset files, videos, and Doctor output |
| `/favicon.ico` | Visualizer browser asset |
| `/bodypipeline-icon.png` | Visualizer browser asset |
| `/bodypipeline-mark.png` | Visualizer browser asset |

Next.js route proxying accepts only `GET` and `HEAD`. The proxy forwards the
existing audited range, cache, content negotiation, and Next.js RSC headers.
It does not forward `Authorization` or `Cookie` to the loopback Next.js
process.

The only remaining dynamic proxy prefix is `/bp/`. There is no
`/{path:path}` catch-all in the embedded application.

### API ownership

The Server already owns these Visualizer-used APIs:

- `/api/task-templates`
- `/api/instructions/...`
- `/api/subtask-review/...`

The embedded Server adds native handlers for the Gateway-only APIs:

- `/api/episode-scores/{dataset_id}`
- `/api/manual-trims/{dataset_id}`
- `/api/manual-trims/{dataset_id}/jobs/{job_id}`
- `/api/manual-trims/{dataset_id}/jobs/{job_id}/preview`
- `/api/recycle/{dataset_id}`
- `/api/recycle/{dataset_id}/{action}`

Duplicate route registration is prohibited. Server APIs resolve previews
through the persisted preview registry before accepting a dataset ID.

## Execution Identity

### Viewer actions that execute work

The following actions must preserve submitter identity:

| Action | Execution boundary |
| --- | --- |
| Manual trim | `TrimRunner` queued through `JobRunner` |
| Instruction apply | `InstructionApplyRunner` |
| Review export | `SubtaskExportRunner` queued through `JobRunner` |
| Episode score | `EpisodeScoreCache` |

Template edits, instruction drafts, review drafts, and status reads are state
operations without a compute subprocess. They retain existing authenticated
Server behavior and do not create execution ownership records.

Recycle operations mutate shared index and quarantine state under the Server
account. They require authenticated Server access but are not user Slurm jobs.
The UI must not describe them as running in the submitter's account.

### New task metadata

For a personal Principal, enqueue uses `Principal.job_metadata()` exactly:

```json
{
  "owner_user_id": "alice",
  "owner_uid": 1001,
  "submitted_as": "alice",
  "slurm_user": "alice",
  "slurm_execution_id": "opaque-internal-handle",
  "submission_auth": "h800-ssh"
}
```

For an admin or machine Principal, the existing service-account metadata is
used:

```json
{
  "owner_user_id": null,
  "owner_uid": null,
  "submitted_as": "service-account",
  "slurm_user": null,
  "slurm_execution_id": null,
  "submission_auth": "service-token"
}
```

### Session binding

`JobRunner.bind_record_session(row)` is the only supported binding path. It
looks up the private broker session by `row["slurm_execution_id"]` and binds it
to a context-local execution scope for one worker thread.

Instruction apply and episode scoring currently bypass this boundary. They
must run their worker processes through the same bound execution context.
Background tick methods must bind the persisted row before recovery, apply,
rollback, or score execution begins.

A missing session produces HTTP-style error text equivalent to:

```text
任务发起者的 H800 登录会话已过期；请重新登录并重新提交任务
```

No retry may switch to local `subprocess.run` after a missing personal session
has been detected.

### File permission preflight

Before a personal job is queued, the Server probes permissions through the
submitter's ControlMaster:

- Dataset traversal requires execute permission on each parent directory.
- Reading requires `meta/info.json` and dataset content to be readable.
- In-place writes require write permission on the dataset directory and
  `meta`.
- Work and cache roots require traversal and write permission for the remote
  user.

The probe uses an argv API, not shell interpolation. Permission failure returns
a 403-style domain error before queueing. Runtime permission changes after
preflight still fail the running task and are recorded as failed, not silently
executed under the Server account.

## State and Concurrency

Existing per-dataset locks remain the final serialization boundary for manual
trim and instruction apply. The identity fields are additive JSON fields, so
old rows remain readable.

Deduplication keys for new personal jobs include
`submitted_by = principal.username`. This prevents user B from receiving user
A's queued job while preserving deduplication for repeated clicks by the same
user.

Queue recovery follows these rules:

| Persisted state | Recovery behavior |
| --- | --- |
| Personal `QUEUED`, session still valid | Claim and execute |
| Personal `QUEUED`, session expired | Terminal `INTERRUPTED` with relogin guidance |
| Personal `RUNNING`, service restart | Recover through the same user session; fail closed if expired |
| Legacy row without execution ID | Existing service-account recovery |
| Committed instruction journal | Reconcile before any new write, using the same user session |

## Configuration and Rollout

Add a strict deployment setting:

```json
{
  "viewer_gateway": {
    "embedded": false
  },
  "viewer_execution_identity": {
    "enabled": false
  }
}
```

Both defaults are disabled. Unknown keys or non-boolean values fail
configuration loading.

When embedding is enabled, `viewer_origin` must identify the public 8081 origin,
for example `http://100.64.0.71:8081`. `ViewerManager` uses this value both when
constructing preview URLs and when building the UI's `/_data` origin.

Rollout order:

1. Deploy the shared transport and feature flag with `embedded=false`.
2. Verify standalone 8082 behavior is unchanged.
3. Deploy identity metadata and worker execution changes with existing
   service-token and legacy recovery tests green, while
   `viewer_execution_identity.enabled=false`.
4. Enable embedded routing on a staging or isolated Server instance.
5. Keep 8082 running while the browser uses 8081 for previews.
6. Enable submitter execution on the isolated instance and test with at least
   one non-admin H800 account that has independent Slurm
   and filesystem permissions.
7. Roll out to the production release only after both the gateway and identity
   acceptance checks pass.
8. Disable 8082 only after the observation window and an explicit rollback
   check.

## Rollback

Gateway rollback:

1. Set `viewer_gateway.embedded` to `false`.
2. Restore `viewer_origin` to the 8082 origin.
3. Restart only the Server release process during the approved maintenance
   window.
4. Confirm the workbench remains on 8081 and preview links again use 8082.

Identity rollback sets `viewer_execution_identity.enabled=false` while keeping
the new release running. New tasks return to service-account behavior. Existing
rows with personal execution metadata remain row-bound: they still require the
original user session or fail closed. They are never replayed by the Server
account merely because new submissions are disabled.

A code-only release rollback is permitted only after an operator confirms there
are no `QUEUED` or `RUNNING` personal jobs. Otherwise those rows must first be
left terminal `INTERRUPTED`; the old release must not start them as
`service-account`. Completed data and journals remain on disk.

Rollback does not delete, rewrite, or restore datasets automatically.

## Acceptance Criteria

Gateway:

- Port 8081 serves the workbench root, `/assets/*`, uploads, listings, and all
  existing Server APIs after embedded routing is enabled.
- A registered preview opens from a 8081 URL and loads Next.js pages, `/_next`
  resources, `/_data` files, video ranges, and all Visualizer APIs.
- Unauthenticated requests receive the Server authentication response.
- Unknown paths return 404 and are never proxied to Next.js.
- Path traversal, symlink escapes, unknown previews, and cross-dataset IDs are
  rejected by the existing transport validation.
- Standalone 8082 continues to pass its existing tests when embedding is
  disabled.

Identity:

- Browser cookie requests create personal identity metadata for manual trim,
  instruction apply, review export, and episode score.
- Machine-token requests remain explicit `service-account` jobs.
- Personal jobs invoke `srun` and worker processes through the submitter's
  ControlMaster.
- Expired, missing, forged, or cross-user execution IDs fail closed.
- Two users operating on the same dataset cannot exchange sessions or inherit
  each other's deduplicated tasks.
- In-place instruction apply preserves journals, commit receipts, rollback,
  restart recovery, and dataset locks.
- `slurm_execution_id` never appears in API responses or logs.

## Non-goals

- Replacing the Server process account for queue storage, indexes, locks, or
  service health checks.
- Allowing users to choose an arbitrary SSH host, port, username, or ControlPath.
- Moving Next.js into the Python process.
- Deleting the 8082 service in the same release as the initial cutover.
- Changing robot-specific capture, conversion, validation, or Viewer data
  semantics.
