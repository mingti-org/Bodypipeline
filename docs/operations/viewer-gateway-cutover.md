# Embedded Viewer Gateway Cutover

Use this procedure to move the Visualizer from the standalone 8082 surface to
the Server on 8081 while keeping 8082 available for compatibility and rollback.
The default remains off in both `configs/server.json` and
`configs/server.deployed.example.json`:

```json
"viewer_gateway": {
  "embedded": false
}
```

Keep the 8081 Server, the 8082 Viewer Gateway, and the configured storage
available throughout the cutover. Do not stop or disable 8082 during this
procedure.

## Record the Baseline

Before changing deployment configuration, record the deployed release
identifier:

```text
Pre-cutover release identifier:
Pre-cutover date and operator:
Last known good viewer_origin (8082):
Existing listing robot_type:
Existing preview job:
```

After the cutover restart, record the new release identifier in the same
evidence set. If the deployment tooling exposes both a release/build ID and a
git commit, record both.

Store the evidence in the release ticket or under
`artifacts/viewer-gateway-cutover/<date>/`. Use screenshots or response
captures that do not expose `PIPELINE_TOKEN`, cookies, or other credentials.

## Ordered Cutover Sequence

1. Deploy the new release with embedded=false.
2. Confirm 8081 workbench and 8082 preview are unchanged.
3. Stop starting new Visualizer jobs during the cutover window.
4. Set embedded=true.
5. Set viewer_origin to the public 8081 origin.
6. Restart the Server release process once.
7. Run all smoke checks while 8082 remains running.
8. Roll out to users only after a non-admin H800 browser test passes.
9. Observe for one full work cycle before disabling 8082.

### 1-3. Establish the Pre-Cutover State

Deploy and start the new release with this exact default:

```json
"viewer_gateway": {
  "embedded": false
}
```

Keep `viewer_origin` on the last known 8082 URL during this validation. Confirm
that the existing 8081 workbench and one existing 8082 preview still behave as
they did before the release. Then stop submitting new Visualizer jobs for the
duration of the cutover window so that the smoke checks exercise a stable set
of previews.

### 4-6. Enable and Restart Once

In the deployed Server configuration, make both changes before restarting:

```json
"viewer_origin": "<public 8081 origin>",
"viewer_gateway": {
  "embedded": true
}
```

`viewer_origin` must be the public origin that serves the 8081 Server, not the
loopback-only value used by the pre-cutover example. Restart the Server release
process exactly once after saving both values. Do not restart 8082.

### 7. Run Health, Route, and Browser Smoke Checks

Run these checks from the Server host and retain their HTTP status and output:

```bash
curl -fsS http://127.0.0.1:8081/healthz
curl -fsS -o /dev/null http://127.0.0.1:8081/
curl -fsS -o /dev/null http://127.0.0.1:8081/assets/app.js
curl -fsS -H "Authorization: Bearer $PIPELINE_TOKEN" \
  http://127.0.0.1:8081/api/config
```

The 8082 process must still be running for the entire check.

In an authenticated browser at the public 8081 origin, record evidence for
each of these checks:

```text
1. `/` renders the workbench.
2. `/assets/app.js` returns the workbench client bundle.
3. `/api/listings/status?robot_type=<existing>` returns the pre-cutover result.
4. Opening one existing preview produces a URL on 8081.
5. `/bp/<job>/episode_0` renders the Visualizer.
6. `/_next/static/...` resources return 200.
7. A video `/_data/...` request returns 206 for a byte range.
8. Doctor, episode score, recycle, manual trim, instruction template,
   instruction episode edit, and review pages load.
9. An unknown path returns 404 and no workbench response is replaced by the
   Visualizer startup page.
10. The browser has only the `pipeline_server` cookie; no second login is
    requested.
```

For the byte-range check, use an existing registered preview and retain the
request URL, `Range` header, `206` status, and `Content-Range` response. For
the cookie check, retain the browser's cookie inventory with values redacted.

### 8. Validate with a Non-Admin H800 Browser

Before exposing the cutover to users, authenticate from an H800 browser as a
non-admin user and repeat the browser checks above. Record the user class,
browser, timestamp, preview job, and results without recording credentials.
An administrator-only success is not sufficient evidence for this step.

### 9. Observe for One Full Work Cycle

Keep 8082 running while users exercise the embedded 8081 surface for at least
one complete work cycle. During the observation period:

- Monitor 8081 Server health and application errors.
- Confirm that newly opened previews continue to produce 8081 URLs.
- Confirm that Visualizer pages, static resources, byte ranges, Doctor,
  scoring, recycle, manual trim, instruction editing, and review remain usable.
- Confirm that unknown paths remain 404 and that the workbench root is not
  replaced by the Visualizer startup page.
- Keep a record of failures, timestamps, affected previews, and rollback
  decisions.

Do not disable 8082 until the observation period and the non-admin H800 test
have passed.

## Identity and Compatibility Caveat

During the observation period, standalone 8082 remains a compatibility route
for preview and token-authenticated automation. The authenticated per-user
execution guarantee applies to the embedded 8081 surface after the second plan
is deployed.

This cutover must not activate identity behavior that is not part of the
current release. Continue to treat authenticated automation separately from
the interactive 8081 browser surface.

## Required Evidence

Attach the following evidence to the release ticket:

- Pre-cutover and post-cutover release identifiers, with timestamps.
- The deployed `viewer_gateway.embedded` and `viewer_origin` values, with any
  hostname redacted only if required by the environment.
- Output and status for every health and route curl command.
- Browser evidence for all ten authenticated checks.
- Byte-range request and response evidence showing `206` and `Content-Range`.
- Browser cookie inventory showing the single `pipeline_server` cookie.
- Non-admin H800 browser evidence and the user class used.
- Observation start and end times, failures, and final decision.
- The rollback operator and trigger, if any rollback occurs.

## Rollback

Use this exact order if the embedded surface fails validation or the
observation period exposes a blocking regression:

1. Set viewer_gateway.embedded=false.
2. Restore viewer_origin to the last known 8082 URL.
3. Restart the Server process.
4. Verify `/` still renders the workbench.
5. Open a new preview and confirm its URL uses 8082.
6. Confirm no process is listening on an unexpected new public port.

Keep the 8082 Viewer Gateway running through rollback. After step 3, record the
same workbench, listing, preview, and unknown-path evidence used during
cutover. Record the post-rollback release identifier and attach the evidence to
the release ticket before investigating the failed cutover.
