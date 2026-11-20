# Viewer User Execution Cutover

Use this procedure to enable per-user Viewer workload execution on the embedded
8081 Server. The feature is independently gated by
`viewer_execution_identity.enabled` and starts disabled:

```json
"viewer_execution_identity": {
  "enabled": false
}
```

The embedded Viewer, user login, Server-owned queue state, and publication
path must already be healthy before this procedure starts. Standalone 8082 may
remain available for compatibility during the rollout, but the per-user
execution guarantee applies to the embedded 8081 surface.

## Preconditions

Record the release and deployed configuration digests before changing
`viewer_execution_identity.enabled`:

```bash
git rev-parse HEAD
sha256sum configs/server.json
sha256sum configs/server.deployed.example.json
```

If the deployed files are elsewhere, digest the files actually loaded by the
Server and record their paths. Record the release identifier, config digests,
test username, login node, rollback owner, operator, and start time in the
release ticket or the evidence bundle.

Confirm the existing embedded Viewer and login path are healthy. Do not begin
the cutover if the workbench is unavailable or existing previews cannot be
opened.

## Enable Sequence

Use this sequence exactly:

```text
1. Confirm viewer_gateway.embedded=true and the workbench smoke test passes.
2. Confirm user_sessions.enabled=true.
3. Deploy this release with viewer_execution_identity.enabled=false.
4. Verify a machine-token task still records service-account metadata.
5. Set viewer_execution_identity.enabled=true.
6. Restart the Server release process once.
7. Log in with a non-admin H800 test account.
8. Run one small manual trim, one instruction apply on a disposable dataset,
   one review export, and one episode score.
9. Confirm each new task records the test username, UID, login node, and
   non-empty execution handle; confirm the API response omits the handle.
10. Confirm `squeue -u <test-user>` shows the workload under that user.
11. Confirm the changed files are readable through the existing publication
    step and the Server can update queue state.
12. Roll the UI or account audience only after the real-data fixture passes.
```

### Step 1: Confirm Embedded Viewer Health

Verify that the root remains the workbench entry point on 8081, an existing
preview opens through the embedded Viewer, and the workbench smoke test passes
without redirecting APIs, assets, uploads, or unknown paths to the Viewer.
Keep the prior release identifier so rollback can identify it.

### Step 2: Confirm User Sessions

Confirm the deployed Server configuration has:

```json
"user_sessions": {
  "enabled": true
}
```

The approved login nodes must include the H800 test account's node. Do not
continue if login sessions cannot be established.

### Step 3: Deploy with Identity Disabled

Deploy the release with:

```json
"viewer_execution_identity": {
  "enabled": false
}
```

This safe default preserves existing service-account behavior for new
submissions while the deployment itself is verified.

### Step 4: Verify Machine-Token Baseline

Submit or inspect one machine-token workload and confirm its newly queued
metadata is:

```text
submitted_as: service-account
slurm_user: null
submission_auth: service-token
slurm_execution_id present: false
```

Record only the boolean for execution-handle presence. Never record a handle
value.

### Step 5: Enable the Identity Gate

Change the deployed configuration to:

```json
"viewer_execution_identity": {
  "enabled": true
}
```

Digest the deployed file after the edit and record the new digest.

### Step 6: Restart Once

Restart the Server release process once after the configuration write. Do not
restart unrelated services and do not start new personal workloads before the
new process is healthy.

### Steps 7 and 8: Authenticate and Run the Real-Data Fixture

Use a non-admin H800 test account. Prepare a disposable dataset large enough
to exercise real execution but small enough for a bounded smoke test. Run:

1. one small in-place manual trim;
2. one whole-dataset instruction apply;
3. one review export; and
4. one episode score.

Record the four task IDs. Do not submit these actions through standalone 8082
when validating the embedded identity guarantee.

### Step 9: Verify Attribution and API Redaction

For each task, verify the internal Store row contains:

- `owner_user_id`: the test username;
- `owner_uid`: the test UID;
- `submitted_as`: the test username;
- `slurm_user`: the test username;
- `submission_auth: h800-ssh`; and
- a non-empty `slurm_execution_id`.

Record only `slurm_execution_id present: true`; never copy the handle value.
Record the login node separately from sanitized user-session evidence because
the login node is not part of standard task metadata.

Retrieve the corresponding public task status or report endpoint. Verify:

- the response identifies the test user through allowed ownership and
  submission fields;
- the response contains no `slurm_execution_id` field;
- the response contains no handle value, ControlPath, session identifier,
  password, cookie, or token; and
- queue state and publication state are still updated by the Server.

If an API response cannot be sanitized safely, record only the HTTP status and
the names of fields that were verified absent.

### Step 10: Verify Slurm Attribution

While a test workload is active, run:

```bash
squeue -u <test-user>
```

Record the command time, job state, partition, and that the job appears under
the test user. Sanitize the job name, paths, account fields, and any execution
identifiers that the site considers private.

### Step 11: Verify Publication and Queue Ownership

Confirm the published files produced by the fixture are readable through the
existing publication step. Confirm the Server records the expected queue and
publication state transitions. The submitter owns remote compute; the Server
continues to own queue state, indexes, locks, publication, and bookkeeping.

### Step 12: Expand the Audience

Roll the UI or account audience out only after all four real-data fixture
actions, the API redaction check, the Slurm attribution check, and the
publication check pass.

Keep the previous release and the prior configuration available. Observe the
cutover through one complete work cycle before removing any compatibility
path. Stop expansion and use the rollback procedure if a personal session
expires, a task falls back to service-account execution, an API exposes a
private handle, or acceptance of a new personal job is not reflected in queue
state.

## Evidence

Record the following in the release ticket or under
`artifacts/viewer-user-execution-cutover/<date>/`:

- release identifier and deployment timestamp;
- pre-enable and post-enable configuration digests;
- test username, UID, and login node;
- the manual-trim, instruction-apply, review-export, and episode-score task IDs;
- sanitized submitter metadata for each task;
- presence-only evidence for each execution handle;
- sanitized API redaction evidence;
- `squeue` attribution evidence;
- queue, publication, and changed-file readability evidence;
- observation start/end time and any failure timestamps; and
- rollback owner and rollback decision, if one is made.

Before attaching any screenshot, response capture, log excerpt, or database
excerpt, sanitize it. Do not retain passwords, cookies, tokens, SSH
ControlPath values, private socket names, session identifiers,
`slurm_execution_id`, or other execution handles. A screenshot or capture that
cannot be fully sanitized must not be attached.

Never attach raw `/api/config` output. Confirm `/api/config`, user/bootstrap
payloads, task responses, logs, receipts, and checkout artifacts contain no
`slurm_execution_id` before evidence is retained.

## Rollback

Use this procedure exactly:

```text
1. Set viewer_execution_identity.enabled=false.
2. Leave the new release running.
3. Restart the Server release process once.
4. Verify new machine-token and browser tasks return to service-account mode.
5. Do not code-rollback while personal QUEUED or RUNNING rows remain.
6. For each personal queued row, allow the user session to finish or mark the
    row INTERRUPTED using the normal queue recovery procedure.
7. A code rollback is permitted only after no personal active rows remain.
8. Keep datasets and journals in place; do not attempt an automatic global
    rollback.
```

Disabling the gate affects new submissions only. Any persisted personal row
with an execution handle must still bind to that exact handle; it must never
be replayed as the Server service account or another user.

Queued personal jobs from a restarted process are expected to become
`INTERRUPTED` because the in-memory user session cannot be reconstructed.
Instruction-apply recovery must preserve its dataset, journal, staging, and
backup state and direct the user to log in and resubmit. The resubmission first
runs transaction recovery under the newly authenticated user, then applies any
new assignment.

Legacy and internal rows without `slurm_execution_id` continue through the
existing service-account execution and recovery path.

Record the rollback time, release identifier, config digest, rollback owner,
affected task IDs, final task states, and sanitized evidence that new
machine-token and browser submissions use service-account metadata again.
