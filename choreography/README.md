# choreography/

Builds the final routine from retargeted clips in `motions/k1_csv/`.

- **Feasibility filter:** flag clips that hit joint limits, slide the feet, or move too fast for the K1.
- **Segments:** cut the useful parts of each clip; have each one end near a neutral stance.
- **Transitions:** blend 0.5–1 s between segments.
- **Music:** time segment boundaries to the song's beats.

Output: one combined `.csv` for the routine, converted to `.npz` in `motions/k1_npz/` with booster_train's `csv_to_npz.py`.
