#!/bin/bash

# CONFIG
HF_TOKEN="$HF_TOKEN"  # Reemplazá con tu token de HuggingFace
HF_MODEL="mistralai/Mistral-7B-Instruct-v0.1"
RAMA="dev-Ricardo"

# Verificar si estás en un repositorio Git
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
  echo "❌ No estás en un repositorio Git"
  exit 1
fi

# Cambiar a la rama correcta
git checkout "$RAMA"

# Verificar cambios
if [[ -z $(git status --porcelain) ]]; then
  echo "⚠️ No hay cambios en el repositorio"
  exit 0
fi

# Agregar todos los cambios
git add .

# Verificar si hay cambios staged
if [[ -z $(git diff --cached) ]]; then
  echo "⚠️ No hay cambios staged para commitear"
  exit 0
fi

# Obtener el diff
DIFF=$(git diff --cached | head -c 2000)  # Limitamos a 2000 caracteres por el API

# Preparar prompt
PROMPT="Generá un mensaje de commit corto, útil y claro en español, basado en los siguientes cambios:\n$DIFF"

# Llamar a Hugging Face API
echo "🤖 Contactando a Hugging Face para generar el mensaje..."

RESPONSE=$(curl -s \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"inputs\": \"$PROMPT\"}" \
  https://api-inference.huggingface.co/models/$HF_MODEL)

# Extraer respuesta
COMMIT_MSG=$(echo "$RESPONSE" | jq -r '.[0].generated_text' | head -n 1)

echo -e "\n📦 Respuesta cruda de la API:"
echo "$RESPONSE"

# Validar resultado
if [[ -z "$COMMIT_MSG" || "$COMMIT_MSG" == "null" ]]; then
  echo "❌ No se pudo generar un mensaje válido"
  exit 1
fi

# Confirmación
echo -e "\n💬 Mensaje sugerido por IA:\n$COMMIT_MSG"
read -p "¿Usar este mensaje para commit? (s/n): " confirm
[[ "$confirm" != "s" ]] && echo "❌ Cancelado" && exit 0

# Commit y push
git commit -m "$COMMIT_MSG"
git push origin "$RAMA"

echo "✅ Commit y push realizados con éxito."