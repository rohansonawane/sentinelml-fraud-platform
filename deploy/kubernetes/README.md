# Kubernetes

Placeholders for API + dashboard. Secrets are not real credentials.

Apply (when you have a cluster; not executed on the M4 smoke host):

```bash
kubectl apply -f deploy/kubernetes/configmap.yaml
kubectl apply -f deploy/kubernetes/secret.yaml
kubectl apply -f deploy/kubernetes/api.yaml
kubectl apply -f deploy/kubernetes/dashboard.yaml
```

Host Postgres :5432 and Redis :6379 are **not** targeted by these manifests. Point `AUDIT_DATABASE_URL` at a dedicated database.
