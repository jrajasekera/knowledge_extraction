// ingest.cql
// Neo4j constraints, indexes, and GDS projection for the Discord social graph

// Constraints & indexes
CREATE CONSTRAINT person_id IF NOT EXISTS FOR (n:Person)   REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT guild_id  IF NOT EXISTS FOR (n:Guild)    REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT channel_id IF NOT EXISTS FOR (n:Channel) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT message_id IF NOT EXISTS FOR (n:Message) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT role_id    IF NOT EXISTS FOR (n:Role)    REQUIRE n.id IS UNIQUE;

CREATE INDEX topic_name IF NOT EXISTS FOR (n:Topic) ON (n.name);
CREATE INDEX org_name   IF NOT EXISTS FOR (n:Org)   ON (n.name);
CREATE INDEX place_lbl  IF NOT EXISTS FOR (n:Place) ON (n.label);

// Optional helper relationship projections will be created by the Python loader

// GDS projection for interaction graph (run after interactions are materialized)
CALL gds.graph.drop('peopleInteractions', false);
CALL gds.graph.project(
  'peopleInteractions',
  ['Person'],
  { INTERACTED_WITH: { type: 'INTERACTED_WITH', orientation: 'UNDIRECTED', properties: 'weight' } }
);
