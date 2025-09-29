# Name Disambiguation Strategy for Discord Knowledge Extraction

## Problem Statement

Discord users in your social graph have multiple identities that create ambiguity during fact extraction:

1. **Discord username** (e.g., "CoolGamer123")
2. **Server-specific nickname** (e.g., "Gaming God")
3. **Real/official name** used in conversations (e.g., "John Smith")
4. **Casual name variants** used by friends (e.g., "John", "Johnny", "J")

The LLM performing information extraction sees conversation text referring to "John" while the author metadata shows "CoolGamer123", leading to:
- Failed identity resolution
- Duplicate person nodes in facts
- Incorrect relationship attribution
- Lower confidence scores

## Recommended Solution: Multi-Layered Name Contextualization

### 1. Database Schema Extension

**Add `official_name` column to the `member` table:**

```sql
ALTER TABLE member ADD COLUMN official_name TEXT;
```

This approach is superior to a config file because:
- Data stays colocated with the source truth
- Survives database backups/migrations
- Can be queried efficiently during windowing
- Enables future features (privacy controls, profile generation)

**Populate manually for your friend group:**
```sql
UPDATE member SET official_name = 'John Smith' WHERE id = '123456789';
```

### 2. Enhanced Participant Context in IE Prompts

Modify `ie/prompts.py` to surface official names to the LLM. The participant section should become:

```
Participants:
- John Smith (Discord: CoolGamer123, author_id=123456789)
- Sarah Johnson (Discord: SGamer, author_id=987654321)
```

This gives the LLM three critical pieces of information:
1. The official name (primary identifier for conversation analysis)
2. The Discord display name (what appears in message metadata)
3. The stable author_id (what gets written to facts)

### 3. Enhanced System Instructions

Add explicit name disambiguation instructions to the IE prompt:

> **Important**: Participants may be referred to by their official names in conversation text (e.g., "John", "Johnny") even though message metadata shows their Discord usernames (e.g., "CoolGamer123"). When extracting facts:
>
> 1. Use the `author_id` values for `subject_id` and `object_id` fields
> 2. If someone mentions "John" in message content, infer they mean the participant whose official name is John Smith
> 3. Common name variations (nicknames, first names only) should resolve to the official name when context is clear

### 4. Implementation Changes

**Minimal code changes required:**

1. **Schema migration** (run once against existing database)

2. **Windowing query update** (`ie/windowing.py`):
   - Modify the SQL query to `SELECT` the `official_name` column
   - Add it to the `MessageRecord` dataclass
   - Thread it through to the window builder

3. **Prompt builder enhancement** (`ie/prompts.py`):
   - Update the participant formatting logic to show: `official_name (Discord: display_name, author_id=...)`
   - Add the name disambiguation instructions to the system prompt

4. **Optional**: Update `facts_to_graph.py` to use official names when creating Person nodes in Neo4j (set a `realName` property)

### 5. Optional Enhancement: Name Alias Table

For more sophisticated resolution, consider a separate alias table:

```sql
CREATE TABLE IF NOT EXISTS member_alias (
  member_id TEXT NOT NULL REFERENCES member(id) ON DELETE CASCADE,
  alias TEXT NOT NULL,
  alias_type TEXT CHECK (alias_type IN ('nickname', 'first_name', 'variation')),
  PRIMARY KEY (member_id, alias)
);
```

This would store common variations like:
- `('123456789', 'John', 'first_name')`
- `('123456789', 'Johnny', 'nickname')`
- `('123456789', 'J', 'nickname')`

The IE prompt could then include an "Aliases" section, though this may clutter the context window and isn't necessary for initial implementation.

### 6. Handling Edge Cases

**When official_name is NULL:**
- Fall back to display_name in participant list
- This maintains backward compatibility with members you haven't mapped yet

**Ambiguous first names:**
- If multiple friends share a first name (two Johns), include last names or disambiguating context in the prompt
- The LLM can use surrounding message context to infer which John is being discussed

**Pronouns and indirect references:**
- The LLM should be instructed that "he", "she", "them" might refer to participants
- Include context that the LLM should only extract facts when identity is clear and confidence is high

## Implementation Priority

1. **High Priority** (necessary for baseline improvement):
   - Add `official_name` column to schema
   - Manually populate for your friend group
   - Update windowing query to fetch it
   - Enhance prompt builder to display it

2. **Medium Priority** (polish):
   - Add name disambiguation instructions to system prompt
   - Handle NULL official_name gracefully

3. **Low Priority** (future enhancement):
   - Alias table for automatic nickname resolution
   - Pre-processing step to auto-suggest aliases based on @mention patterns

## Expected Outcomes

After implementing this strategy:

1. **Higher extraction accuracy**: LLM can correctly attribute facts to author_ids even when conversations use real names
2. **Better confidence scores**: Reduced ambiguity leads to more certain extractions
3. **Fewer false duplicates**: No more separate entities for "John" vs "CoolGamer123"
4. **Richer context**: Official names make facts more human-readable during review
5. **Privacy-ready**: Having official names in the database enables future redaction/anonymization features

## Example Before/After

**Before:**
```
Participants:
- CoolGamer123 (author_id=123456789)
- SGamer (author_id=987654321)

Conversation:
[2024-01-15T10:30:00] SGamer: Hey John, did you finish that project?
[2024-01-15T10:31:00] CoolGamer123: Yeah, wrapped it up last night
```
*LLM struggles to connect "John" with "CoolGamer123"*

**After:**
```
Participants:
- John Smith (Discord: CoolGamer123, author_id=123456789)
- Sarah Johnson (Discord: SGamer, author_id=987654321)

Conversation:
[2024-01-15T10:30:00] Sarah Johnson: Hey John, did you finish that project?
[2024-01-15T10:31:00] John Smith: Yeah, wrapped it up last night
```
*LLM clearly sees the connection and extracts facts accurately*
