curl -X POST http://172.17.0.2:8080//generate-text \
-H 'Content-Type: application/json' \
-d '{"message": "Direktør i Sundhedsstyrelsen Søren", "max_length":200}'