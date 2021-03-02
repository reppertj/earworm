BEGIN TRANSACTION;

INSERT
    INTO
        vectorsearch_artist
        (name)
    VALUES
        ('ThisArtist'),
        ('AnotherArtist');

INSERT
    INTO
        vectorsearch_album
        (name, artist_id)
    VALUES
        ('FunAlbum', 2),
        ('SuperAlbum', 3);

INSERT
    INTO
        vectorsearch_embedder
        (name, precision, vector_length)
    VALUES
        ('magicalembedder', 8, 50);

INSERT
    INTO
        vectorsearch_song
        (album_id, artist_id, external_url)
    VALUES
        (2, 2, "http://localhost/"),
        (3, 3, "http://localhost/");

INSERT 
    INTO
        vectorsearch_license
        (abbreviation, description, name)
    VALUES
        ("BY", "requires attribution", "attribution"),
        ("ND", "cannot put in videos", "no derivative");

INSERT
    INTO
        vectorsearch_song_licenses
        (song_id, license_id)
    VALUES
        (1, 2),
        (2, 2),
        (3, 2),
        (3, 3);

INSERT
    INTO
        vectorsearch_resultvector
        (vector, embedder_id, song_id)
    VALUES
        ("[3,5,6,8]", 1, 2),
        ("[5,2,3,6]", 1, 3),
        ("[1,2,3,4]", 1, 1),
        ("[1,3,5,3]", 1, 2),
        ("[1,3,5,3]", 1, 2),
        ("[1,2,3,4]", 1, 3),
        ("[5,68,3,2]", 1, 2);

COMMIT;