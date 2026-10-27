# Audio data

Three short recordings are bundled with the repository. Every other WAV file in
this folder is ignored by git.

| File | What it is | Used by |
|---|---|---|
| `hyperbolic_chirp.wav` | Synthetic chirp falling from 4 kHz to 0.9 kHz over 15 s (8 kHz sample rate) | paper, `python -m benchmarks.run_real_audio` |
| `humpback_whale_song.wav` | 5 s of humpback whale song | paper |
| `snapping_shrimp.wav` | 5 s of snapping shrimp clicks | paper |

## Checksums

The full paper driver (`python -m benchmarks.run_paper`) checks these three
files before running:

| File | SHA-256 |
|---|---|
| `hyperbolic_chirp.wav` | `e95a075a4554a855536739d7eed11c9e36f777d6b6e40a4d8b043c134f94407c` |
| `humpback_whale_song.wav` | `261157108e59afcd64984d01ef8bf632dbdad8c88ba261622360c35c8ee7295f` |
| `snapping_shrimp.wav` | `98fc0a1db44be63076d1336b04fa32dc25d6d164652b744a805ae83f69eabdc5` |

## Sources

| File | Source | Original file name |
|---|---|---|
| `humpback_whale_song.wav` | SanctSound [1], Channel Islands site CI04, 2020-02-25 | `SanctSound_CI04_04_humpbackwhale_20200225T112447Z.wav` |
| `humpback_whale_and_dolphins.wav` (not bundled) | SanctSound [1], Hawaiian Islands site HI04, 2020-01-29 | `SanctSound_HI04_02_humpbackwhaleanddolphin_20200129T175046Z.wav` |
| `snapping_shrimp.wav` | SanctSound [1], Florida Keys site FK02, 2019-11-03 | `SanctSound_FK02_03_snappingshrimp_20191103T014035Z.wav` |
| `fish_chorus.wav` (not bundled) | SanctSound [1], Gray's Reef site GR01, 2019-09-02 | `SanctSound_GR01_02_fishchorus_20190902T004059Z.wav` |
| `english_speech_1.wav` (not bundled) | VCTK corpus [2], speaker p227 | `p227.wav` |
| `english_speech_2.wav` (not bundled) | VCTK corpus [2], speaker p228 | `p228.wav` |
| `hyperbolic_chirp.wav` | Synthetic, generated for this work | `test_hyperbolic_chirp.wav` |

[1] NOAA Office of National Marine Sanctuaries and U.S. Navy, *SanctSound Raw
Passive Acoustic Data*, NOAA National Centers for Environmental Information,
2020, doi:[10.25921/saca-sp25](https://doi.org/10.25921/saca-sp25). Cite this
collection when you reuse the excerpts. NOAA's disclaimer and use constraints
apply to them independently of this repository's license.

[2] J. Yamagishi, C. Veaux and K. MacDonald, *CSTR VCTK Corpus: English
Multi-speaker Corpus for CSTR Voice Cloning Toolkit* (version 0.92), University
of Edinburgh, Centre for Speech Technology Research, 2019,
doi:[10.7488/ds/2645](https://doi.org/10.7488/ds/2645).
