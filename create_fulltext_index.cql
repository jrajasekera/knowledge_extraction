// create_fulltext_index.cql
// Migration: Add fulltext index for hybrid search (vector + keyword)
//
// This enables keyword/fuzzy matching on fact embeddings to complement
// vector semantic search. The hybrid approach improves recall for:
// - Exact identifiers (project codes, error codes, user IDs)
// - Rare entities not well-represented in embedding model
// - Acronyms requiring exact matching
//
// Usage:
//   cat create_fulltext_index.cql | cypher-shell -a bolt://localhost:7687 -u neo4j -p 'test'

CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS
FOR (f:FactEmbedding)
ON EACH [f.text, f.person_name, f.fact_object];
