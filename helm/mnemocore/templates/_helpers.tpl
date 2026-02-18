{{/*
Expand the name of the chart.
*/}}
{{- define "mnemocore.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "mnemocore.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "mnemocore.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mnemocore.labels" -}}
helm.sh/chart: {{ include "mnemocore.chart" . }}
{{ include "mnemocore.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mnemocore.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mnemocore.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "mnemocore.serviceAccountName" -}}
{{- if .Values.mnemocore.serviceAccount.create }}
{{- default (include "mnemocore.fullname" .) .Values.mnemocore.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.mnemocore.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Redis fullname
*/}}
{{- define "mnemocore.redis.fullname" -}}
{{- printf "%s-redis" (include "mnemocore.fullname" .) }}
{{- end }}

{{/*
Qdrant fullname
*/}}
{{- define "mnemocore.qdrant.fullname" -}}
{{- printf "%s-qdrant" (include "mnemocore.fullname" .) }}
{{- end }}

{{/*
ConfigMap fullname
*/}}
{{- define "mnemocore.configmap.fullname" -}}
{{- printf "%s-config" (include "mnemocore.fullname" .) }}
{{- end }}

{{/*
Secret fullname
*/}}
{{- define "mnemocore.secret.fullname" -}}
{{- printf "%s-secret" (include "mnemocore.fullname" .) }}
{{- end }}

{{/*
PVC fullname
*/}}
{{- define "mnemocore.pvc.fullname" -}}
{{- printf "%s-data" (include "mnemocore.fullname" .) }}
{{- end }}

{{/*
HPA fullname
*/}}
{{- define "mnemocore.hpa.fullname" -}}
{{- printf "%s-hpa" (include "mnemocore.fullname" .) }}
{{- end }}

{{/*
Return the proper Storage Class
*/}}
{{- define "mnemocore.storageClass" -}}
{{- if .Values.global.storageClass }}
  {{- if (eq "-" .Values.global.storageClass) }}
  {{- else }}
storageClassName: "{{ .Values.global.storageClass }}"
  {{- end }}
{{- else if .Values.mnemocore.persistence.storageClass }}
  {{- if (eq "-" .Values.mnemocore.persistence.storageClass) }}
  {{- else }}
storageClassName: "{{ .Values.mnemocore.persistence.storageClass }}"
  {{- end }}
{{- end }}
{{- end }}
