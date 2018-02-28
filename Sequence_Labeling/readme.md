# Sequence Labeling
* Project link: [link](https://www.csie.ntu.edu.tw/~yvchen/f106-adl/A1)
* Requirement: Given TIMIT dataset to train, predict the phone sequence by the given MFCC features
* Kaggle Competition [link](https://www.kaggle.com/c/hw1-timit/leaderboard): I got the rank 12th/191 in this competition

## Dataset
* TIMIT dataset
* Features: MFCC features (39 dim) and FBank features (69 dim) for each frame
* Labels: 48 kinds of phone
For example:
Feautres format: ID_frame, MFCC features
Labels format: ID_frame, phone label

## Pre-processing
1. Phone mapping (48->39)

| Original Phoneme(s) | Mapped Phoneme |
| :------------------  | :-------------------: |
| aa | aa |
| ae | ae |
| ah | ah |
| ao | aa |
| aw | aw |
| ax | ah |
| ay | ay |
| b | b |
| ch | ch |
| cl | sil |
| d | d |
| dh | dh |
| dx | dx |
| eh | eh |
| el | l |
| en | n |
| epi | sil |
| er | er |
| ey | ey |
| f | f |
| g | g |
| hh | hh |
| ih | ih |
| ix | ih |
| iy | iy |
| jh | jh |
| k | k |
| l | l |
| m | m |
| ng | ng |
| n | n |
| ow | ow |
| oy | oy |
| p | p |
| r | r |
| sh | sh |
| sil | sil |
| s | s |
| th | th |
| t | t |
| uh | uh |
| uw | uw |
| vcl | sil |
| v | v |
| w | w |
| y | y |
| zh | sh |
| z | z |

2. Concatenate the audio with same ID by frames, we get feature dim * audio frame(length)
3. Transform the label from phone to number

## Result
* [Kaggle competition](https://www.kaggle.com/c/hw1-timit/leaderboard): I got the rank 12th/191 