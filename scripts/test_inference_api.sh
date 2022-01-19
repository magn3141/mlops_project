curl -X POST https://us-central1-mlops-project-338109.cloudfunctions.net/generate-covid-press \
-H 'Content-Type: application/json' \
-d '{"message": "Direktør i Sundhedsstyrelsen Søren", "max_length":200}'