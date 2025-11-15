#!/bin/bash
# Test LLM-based ingestion with full logging

PDF_PATH="$1"
if [ -z "$PDF_PATH" ]; then
    echo "Usage: $0 <path-to-pdf>"
    exit 1
fi

if [ ! -f "$PDF_PATH" ]; then
    echo "Error: File not found: $PDF_PATH"
    exit 1
fi

echo "=================================================="
echo "LLM-Based Ingestion Test"
echo "=================================================="
echo "PDF: $PDF_PATH"
echo "File size: $(ls -lh "$PDF_PATH" | awk '{print $5}')"
echo ""

# Get auth token (assuming default user)
TOKEN="test-token-user-1"

# Upload PDF
echo "Uploading PDF..."
RESPONSE=$(curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@$PDF_PATH" \
  -F "title=Heat Transfer Test" \
  http://localhost:8000/api/resources/upload)

echo "Response: $RESPONSE"
echo ""

# Extract resource_id from response
RESOURCE_ID=$(echo $RESPONSE | grep -o '"resource_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$RESOURCE_ID" ]; then
    echo "Error: Failed to extract resource_id from response"
    exit 1
fi

echo "Resource ID: $RESOURCE_ID"
echo ""

# Create chunks for this resource (forces re-chunk if already exists)
echo "Creating chunks via API..."
curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/resources/${RESOURCE_ID}/chunk?force=true" \
  | sed -E 's/.{200}/&\n/g'

# Wait a bit for DB writes
echo "Waiting for chunk inserts..."
sleep 3

# Check chunks created
echo ""
echo "Checking chunks in database..."
docker exec btp_studyagent-postgres-1 psql -U postgres -d app -c "
SELECT 
    COUNT(*) as total_chunks,
    COUNT(CASE WHEN tags->>'pedagogy_role' IS NOT NULL THEN 1 END) as tagged_chunks,
    COUNT(CASE WHEN tags->>'domain' IS NOT NULL THEN 1 END) as has_domain,
    COUNT(CASE WHEN tags->>'topic' IS NOT NULL THEN 1 END) as has_topic
FROM chunk 
WHERE resource_id = '$RESOURCE_ID';
"

echo ""
echo "Sample chunks with tags:"
docker exec btp_studyagent-postgres-1 psql -U postgres -d app -c "
SELECT 
    LEFT(full_text, 60) as text_sample,
    tags->>'pedagogy_role' as pedagogy_role,
    tags->>'domain' as domain,
    tags->>'topic' as topic
FROM chunk 
WHERE resource_id = '$RESOURCE_ID'
LIMIT 5;
"

echo ""
echo "Pedagogy role distribution:"
docker exec btp_studyagent-postgres-1 psql -U postgres -d app -c "
SELECT 
    tags->>'pedagogy_role' as role,
    COUNT(*) as count
FROM chunk 
WHERE resource_id = '$RESOURCE_ID' AND tags->>'pedagogy_role' IS NOT NULL
GROUP BY tags->>'pedagogy_role'
ORDER BY count DESC;
"

echo ""
echo "=================================================="
echo "Ingestion complete!"
echo "=================================================="
