# Salami-dataset-used-in-music-structure-classification

1. This project reorganized the salami dataset.
2. Merge two project together.
3. All the data are through the test of youtube audio if it can downloads. And the task is test by youtube-dl.
4. Renew the labels that use in the "structure classification" task.
  
*  "New_salami_dataframe.csv" - Merging two files (salami_youtube_pairings.csv and SALAMI_iTunes_library.csv) into the new file and add the structure annotation into it.
*  "original_sum_of_categories.csv" - Sum of all categories of structure labels.
*  "new_sum_of_categories.csv" - Sum of the structure labels after selected. 
('Verse', 'Silence', 'Chorus', 'End', 'Outro', 'Intro', 'Bridge', 'Intrumental', 'Interlude', 'Fade-out', 'Solo', 'Pre-Verse', 'silence', 'Pre-Chorus', 'Head', 'Coda', 'Theme', 'Transition', 'Main_Theme', 'Development', 'Secondary_theme', 'Secondary_Theme', 'outro')
*  "new_sum_of_categories_2.csv" - Sum of the structure labels after selected.
('Verse', 'Chorus', 'Outro', 'Intro', 'Bridge', 'Intrumental', 'Pre-Verse', 'Pre-Chorus', 'outro')
*  "genre_original.csv" - Sum of all genre in the SALAMI dataset.

## Reference
*  https://github.com/DDMAL/salami-data-public 
*  https://github.com/jblsmith/matching-salami 
*  http://jblsmith.github.io/Getting-SALAMI-from-YouTube/ 
