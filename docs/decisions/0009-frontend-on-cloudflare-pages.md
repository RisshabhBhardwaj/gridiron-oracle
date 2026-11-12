# 0009 Frontend hosting moves to Cloudflare Pages

Date: 2026-11-12. Status: accepted (owner request, via the risshabh.com coordinator).

## Decision

Serve the frontend from Cloudflare Pages (project `gridiron-oracle`, owner's personal account) at `gridiron.risshabh.com`, next to the owner's other sites. The backend stays on Railway. Vercel stays live until the owner retires it.

The `/api` proxy becomes a Pages Function (`frontend/functions/api/[[path]].js`) with the same contract as the Vercel proxy. It reuses `frontend/api/proxy-policy.js` unchanged, so there is one access policy for both hosts.

## First deploy matches what is live

On 2026-11-12, Vercel and Railway both ran `989101b`. Local `main` was 94 commits ahead and changed both the frontend and the backend; for example, its frontend calls a backend `/odds` route that `989101b` does not have. So the first Cloudflare deploy is `989101b` plus the Pages Function, the policy, and a port of the Vercel odds handler (branch `agent/cloudflare-live`). It behaves like the live Vercel site, except that settings writes are refused (decision 0007). Redeploy from `main` once `main` is pushed and Railway runs it.

## Why

- One host for the owner's sites. Pages has the SPA fallback built in, and its catch-all Functions route `[[path]]` works, unlike Vercel's zero-config `/api` directory.
- A frontend must always be deployed against the backend it was built with.
