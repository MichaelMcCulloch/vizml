#!/bin/sh
set -eu
: "${IMAGE_TAG:?}"
awk -v built="${1:-}" -v tag="$IMAGE_TAG" '
  BEGIN { n=split(built,a," "); for(i=1;i<=n;i++) want[a[i]]=1 }
  $1 == "-" && $2 == "name:" { comp=$3; sub(/.*\//,"",comp); hit=(comp in want) }
  $1 == "newTag:" && hit { print "    newTag: \"" tag "\""; hit=0; next }
  { print }
' deploy/k8s/kustomization.yaml > deploy/k8s/kustomization.yaml.tmp
mv deploy/k8s/kustomization.yaml.tmp deploy/k8s/kustomization.yaml
