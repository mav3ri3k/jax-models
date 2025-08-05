[Input]
Board = b_BL
Moves = m_B

Batch means boards
Total = b_BL + m_B = x_BL
Procs = x_BLD
[10, 77, 256]
x_bld

[Auto-regressive Decoder]
batch means inputs
L means baords
x_BL -> x_Bbld
[10, 10, 77, 256] -> [10, 10 * 77, 256]
x_B(bl)d -> x_B(b*l)d


NO WRONG

L means list of (board + moves) showing states

x_BLD for decoder
[10, 65, 77] -> Embed -> [10, 65, 77, 64] -> LinearGeneral((77, 64), 256) -> [10, 65, 256]
  -> LinearGeneral(256, (77, 65)) -> [10, 65, 77, 64] -> Embed.call() -> [10, 65, 77]
[10, 65, 77] -> Linear -> [10, 65, 256] -> Linear -> [10, 65, 77]


# Moves in a game distribution
| Move Count Range | # Games | Cumulative % |
| ---------------- | ------: | -----------: |
| 0                |     160 |         0.02 |
| 1–10             |  54 767 |         5.24 |
| 11–20            | 144 064 |        18.97 |
| 21–30            | 272 261 |        45.03 |
| 31–40            | 260 886 |        69.94 |
| 41–50            | 163 490 |        85.49 |
| 51–60            |  89 925 |        94.08 |
| 61–70            |  41 296 |        98.03 |
| 71–80            |  14 762 |        99.46 |
| 81–90            |   4 564 |        99.78 |
| 91–100           |   1 433 |        99.92 |
| 101–110          |     500 |        99.96 |
| 111–120          |     205 |        99.99 |
| 121–130          |      82 |       100.00 |
| 131–140          |      34 |       100.00 |
| 141–150          |      10 |       100.00 |
| 151–160          |       1 |       100.00 |

