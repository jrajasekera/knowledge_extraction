// create_message_fulltext_index.cql
// Migration: Add fulltext index for message hybrid search (vector + keyword)
//
// This enables keyword/fuzzy matching on message embeddings to complement
// vector semantic search. The hybrid approach improves recall for:
// - Exact identifiers (error codes, issue trackers, URLs)
// - Username/channel searches (@user, #channel)
// - Exact phrases that vector search might miss
// - Typo tolerance via fuzzy matching
//
// Usage:
//   cat create_message_fulltext_index.cql | cypher-shell -a bolt://localhost:7687 -u neo4j -p 'test'

CREATE FULLTEXT INDEX message_fulltext IF NOT EXISTS
FOR (m:MessageEmbedding)
ON EACH [m.content, m.clean_content, m.author_name, m.channel_name, m.guild_name];
