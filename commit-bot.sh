DIFF=$(git diff --cached | head -c 2000)

RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d "{\"data\": [\"$DIFF\"]}" \
  https://hf.space/embed/ricklegac/commit-bot/+/api/predict)

COMMIT_MSG=$(echo "$RESPONSE" | jq -r '.data[0]')
echo -e "\n📦 Respuesta cruda de la API:"
echo "$RESPONSE"
echo -e "\n💬 Commit generado por tu Space:"
echo "$COMMIT_MSG"