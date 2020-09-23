-- Initialize the database.
-- Drop any existing data and create empty tables.

DROP TABLE IF EXISTS Url;

CREATE TABLE Url (
        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        original_url VARCHAR(512),
        short_url VARCHAR(6),
        visits INTEGER,
        created_at DATETIME NOT NULL,
        last_access DATETIME,
        PRIMARY KEY (id),
        UNIQUE (short_url)
);
