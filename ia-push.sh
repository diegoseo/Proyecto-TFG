#!/bin/bash

RAMA="dev-Ricardo"
git checkout "$RAMA"
git add .

if [[ -z $(git diff --cached) ]]; then
  echo "⚠️ No hay cambios staged para commitear"
  exit 0
fi

FECHA=$(date +"%Y-%m-%d %H:%M")
RESUMEN=$(git diff --cached --stat | sed ':a;N;$!ba;s/\n/ | /g')

COMMIT_MSG="🤖 [Auto] $FECHA – Cambios detectados: $RESUMEN"

echo -e "\n💬 Mensaje de commit:\n$COMMIT_MSG"
read -p "¿Usar este mensaje? (s/n): " confirm
[[ "$confirm" != "s" ]] && echo "❌ Cancelado" && exit 0

git commit -m "$COMMIT_MSG"
git push origin "$RAMA"