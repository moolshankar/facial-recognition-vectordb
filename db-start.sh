# Start PostgreSQL with pgvector
docker-compose down
docker-compose up -d

# Wait for database to be ready (check logs)
docker-compose logs -f postgres