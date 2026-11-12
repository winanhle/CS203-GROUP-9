# SPAQ near-duplicate visual review

Reviewed UTC: 2026-11-08T20:08Z

Method: selected Q-Align JPEGs were extracted into ignored raw storage and opened at high detail. The review intentionally included non-byte-identical pairs at every 64-bit dHash Hamming distance from 0 through 4. This is a targeted validation of the grouping heuristic, not a claim that every edge was individually human-reviewed.

| dHash distance | Pair | Observed relationship |
|---:|---|---|
| 0 | `00895.jpg`, `05923.jpg` | Same mall scene and framing; exposure/rendering differs |
| 0 | `01896.jpg`, `01489.jpg` | Same blurred deck/pipe scene and framing |
| 0 | `02986.jpg`, `09218.jpg` | Same washroom scene; exposure/focus differs |
| 0 | `09641.jpg`, `03208.jpg` | Same tree display scene and framing |
| 1 | `06176.jpg`, `04613.jpg` | Same kiosk scene and framing |
| 2 | `03813.jpg`, `07137.jpg` | Same deck/pipe scene and framing; exposure differs |
| 3 | `07282.jpg`, `05642.jpg` | Same outdoor utility-cover scene; extreme exposure difference; MOS absolute difference 68.47 |
| 4 | `06566.jpg`, `02884.jpg` | Same blurred sign scene and framing |

Decision: use every edge in `evidence/q_align_duplicate_audit.json` as a conservative same-content grouping constraint. Connected components, not individual edges, are indivisible split groups. The resulting candidate manifest has 10,340 groups; 539 groups contain multiple images, cover 1,324 rows, and have a maximum component size of 22. This can merge a rare false-positive edge, but it prevents demonstrably repeated scenes from leaking across partitions and costs little effective sample size.

