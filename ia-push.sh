#!/bin/bash

DIFF=$(git diff --cached | head -c 1000)

RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d "{\"data\": [\"$DIFF\"]}" \
  https://hf.space/embed/ricklegac/commit-bot/+/api/predict)

echo -e "\n🔍 DIFF enviado al Space:"
echo "$DIFF"

echo -e "\n📦 Respuesta cruda de la API:"
echo "$RESPONSE"

COMMIT_MSG=$(echo "$RESPONSE" | jq -r '.data[0]')

echo -e "\n💬 Commit generado por tu Space:"
echo "$COMMIT_MSG"

if [[ -z "$COMMIT_MSG" || "$COMMIT_MSG" == "null" ]]; then
  echo "❌ El modelo no devolvió ningún texto (puede haber tardado mucho o fallado)."
  exit 1
fi