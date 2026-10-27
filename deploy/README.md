# Deployment

GitLab builds frontend and engine separately with Buildx registry caches. On the default branch, successful builds push commit-tagged images and the deployment job updates only those tags on `deploy`, watched by Argo CD. Configure a masked `GITOPS_TOKEN` with repository write access. `ARGO_WEBHOOK_URL` and `ARGO_WEBHOOK_SECRET` are optional; polling remains the fallback. Bootstrap `deploy` from the default branch before running CI.

Provision namespace `vizml` with label `shared-gateway-access=true` and a `regcred` image-pull secret; then apply `deploy/argocd/project.yaml` and `deploy/argocd/application.yaml`. The PVC is deliberately not pruned. The `/data` volume persists checkpoint imports and provides space for future persistent templates; bundled templates remain part of the application. The engine is single-replica because checkpoints are local to its volume. This application has no user authentication or tenant isolation; deploy behind an access control layer appropriate for the environment.

The deployment uses a CPU-specific lock in `docker/engine-deps`; update it when changing Python dependencies. The root lock is retained for local development. Changes to Kubernetes manifests on the default branch must also be promoted to `deploy`; the tag-bump job only updates image tags.
