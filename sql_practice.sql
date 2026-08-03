-- SQL Practice — FindMyGig AI schema
-- Run these with: docker exec -it findmygig-db psql -U postgres -d findmygig
-- (Create some sample gigs via /docs first, or run the INSERT statements below.)

-- ============================================
-- Sample data — run this first if your table is empty
-- ============================================
INSERT INTO gigs (title, description, required_skills, client_name) VALUES
('Build a React dashboard', 'Analytics dashboard for internal use', ARRAY['React', 'TypeScript'], 'Acme Corp'),
('FastAPI backend for a marketplace', 'Need REST APIs + Postgres', ARRAY['Python', 'FastAPI', 'SQL'], 'Beta Labs'),
('LangChain chatbot for docs', 'RAG-based Q&A over internal docs', ARRAY['Python', 'LangChain', 'RAG'], 'Acme Corp'),
('iOS app bug fixes', 'Fix crashes in existing Swift app', ARRAY['Swift', 'iOS'], 'Gamma Inc'),
('Data pipeline in Airflow', 'ETL pipeline for sales data', ARRAY['Python', 'SQL', 'Airflow'], 'Beta Labs'),
('ML model for churn prediction', 'Classification model, sklearn', ARRAY['Python', 'scikit-learn', 'ML'], 'Delta Co'),
('Docker + K8s deployment help', 'Containerize and deploy existing app', ARRAY['Docker', 'Kubernetes'], 'Gamma Inc'),
('Vector search for product catalog', 'Semantic search using embeddings', ARRAY['Python', 'Vector DB', 'RAG'], 'Delta Co');

-- 1. Basic SELECT — all gigs
SELECT * FROM gigs;

-- 2. SELECT specific columns
SELECT title, client_name FROM gigs;

-- 3. WHERE — gigs from a specific client
SELECT * FROM gigs WHERE client_name = 'Acme Corp';

-- 4. WHERE with LIKE — title contains "AI"-adjacent keyword
SELECT title FROM gigs WHERE description ILIKE '%RAG%';

-- 5. ORDER BY
SELECT title, client_name FROM gigs ORDER BY client_name ASC;

-- 6. LIMIT
SELECT * FROM gigs LIMIT 3;

-- 7. COUNT — how many gigs total
SELECT COUNT(*) AS total_gigs FROM gigs;

-- 8. GROUP BY + COUNT — gigs per client
SELECT client_name, COUNT(*) AS gig_count
FROM gigs
GROUP BY client_name;

-- 9. GROUP BY + HAVING — clients with more than 1 gig posted
SELECT client_name, COUNT(*) AS gig_count
FROM gigs
GROUP BY client_name
HAVING COUNT(*) > 1;

-- 10. Array contains — gigs requiring "Python"
SELECT title FROM gigs WHERE 'Python' = ANY(required_skills);

-- 11. Array contains — gigs requiring both "Python" AND "SQL"
SELECT title FROM gigs
WHERE required_skills @> ARRAY['Python', 'SQL'];

-- 12. Array length — gigs requiring 3+ skills
SELECT title, array_length(required_skills, 1) AS num_skills
FROM gigs
WHERE array_length(required_skills, 1) >= 3;

-- 13. UNNEST — flatten skills into rows (useful for a "most in-demand skills" report)
SELECT unnest(required_skills) AS skill, COUNT(*) AS demand
FROM gigs
GROUP BY skill
ORDER BY demand DESC;

-- 14. CASE WHEN — tag gigs as "AI-related" or not
SELECT title,
       CASE
           WHEN 'LangChain' = ANY(required_skills) OR 'RAG' = ANY(required_skills)
               THEN 'AI-related'
           ELSE 'Other'
       END AS category
FROM gigs;

-- 15. Subquery — clients whose gigs all require Python
SELECT DISTINCT client_name
FROM gigs
WHERE client_name NOT IN (
    SELECT client_name FROM gigs WHERE NOT ('Python' = ANY(required_skills))
);

-- 16. CTE (WITH clause) — same "most in-demand skills" query, written as a CTE
WITH skill_counts AS (
    SELECT unnest(required_skills) AS skill
    FROM gigs
)
SELECT skill, COUNT(*) AS demand
FROM skill_counts
GROUP BY skill
ORDER BY demand DESC
LIMIT 5;

-- 17. UPDATE — a client renamed
UPDATE gigs SET client_name = 'Acme Corporation' WHERE client_name = 'Acme Corp';

-- 18. DELETE — remove a gig by id (adjust id as needed)
-- DELETE FROM gigs WHERE id = 1;

-- ============================================
-- Self-check: can you explain, without looking, what @>, ANY(), and unnest()
-- do differently, and why UNNEST + GROUP BY is the pattern for "most common
-- item inside an array column across many rows"? This exact pattern is what
-- you'll reuse when you build the "most in-demand skills" feature for real.
-- ============================================
