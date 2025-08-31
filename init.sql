-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create database if not exists
SELECT 'CREATE DATABASE facial_recognition'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'facial_recognition');