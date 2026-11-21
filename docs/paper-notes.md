---
type: fact
status: current
---

# Paper notes

Frej et al., "Finding Paths for Explainable MOOC Recommendation: A Learner Perspective", LAK '24. `paper.pdf` page refs below are PDF pages (1-12). Refs = p11-12, skip.

## Problem (p1-2)
- Path-based explainable rec for MOOCs: RL agent walks KG from learner to course, path = explanation.
- Prior: PGPR (Xian 2019) on Amazon only, needs manual path patterns. EKCRec = proprietary data, no user study.
- RQ1: performance + interpretability of path-based model. RQ2: user preference for explanations (approach, motivation, complexity).

## KG (p3, Table 1 p4)
- Entities: learner, school, course, category, teacher, concept.
- Relations: enrolled, teaches, has_concept, belongs_to, provides (school-course).
- COCO concepts not provided, extracted from descriptions with skillNER.
- Dropped learners with <10 courses.

| Dataset | Learners | Courses | Enrollments | KG entities | KG relations |
|---|---|---|---|---|---|
| COCO | 25,979 | 23,319 | 428,930 | 4,378 | 51,210 |
| Xuetang | 6,548 | 687 | 97,592 | 5,010 | 106,468 |

## Method (p3-5)
- PGPR: (1) train KG embeddings (TransE-style, maximize likelihood of triplets), (2) RL agent on KG, embeddings in state. Reward = 0 if path not in predefined patterns, else dot-product sim(learner, course).
- UPGPR (ours): no path patterns, any length. Binary reward R_0/1 = 1 if learner enrolled in end course AND n_hops > 1, else 0 (Eq. 1, p5).
  - n_hops > 1 stops trivial learner->course direct path (Fig 3, p5).
  - Reward ignores embeddings, but embeddings still in agent state.
- Training: agent may go +1 longer than eval max path length for exploration.

## Experiment setup (p5)
- Split 80/10/10 per user (train/test/val). 3 seeds.
- Baselines via RecBole: Pop, NeuMF, CFKG. Batch 512, Adam lr 0.001, 200 epochs, early stop patience 10.
- UPGPR: KG embedding hparams (batch, lr, dim) same as original PGPR. RL agent 50 epochs, Adam lr 0.001.
- Metrics: NDCG, Recall, HR, Precision. "Invalid users" = agent can't produce >=10 items.

## Results RQ1 (p6-8, Table 2 p8)
- NDCG per dataset (UPGPR best at path len 5):

| Dataset | Pop | NeuMF | CFKG | PGPR(3) | UPGPR 3 / 4 / 5 |
|---|---|---|---|---|---|
| Xuetang | 3.06 | 14.05 | 14.20 | 18.16 | 18.43 / 17.96 / **20.85** |
| COCO | 3.90 | 5.55 | 7.79 | 7.47 | 6.66 / 8.49 / **8.87** |

- Len 3: UPGPR ~ PGPR without manual patterns. Longer path = better, tested only up to 5 (users dislike long).
- Xuetang len 3/4 has ~5-7% invalid users, 0% at len 5.
- Patterns found on COCO (Fig 7, p8): len 3 mostly L-C-L-C (shared enrollment, 77%), then same teacher (15%), category (7%), concept (1%). Len 4 reuses len 3 patterns (KG structure: learner only connects to course). Len 5 has 18 patterns, top 5 cover 89%.
- Category/concept paths rare, because COCO KG is 89.3% enrolled, teaches dominant, category 4.8%, concept 2.1%.

## User study (p5, p7-10)
- 32 participants (7 pilot, 25 final). PhD students/postdocs, STEM, avg age 29.6. Semi-structured interviews.
- Conditions: motivation Learn (skills) vs Credits (fulfill requirements), random assign.
- 3 scenarios: (1) Algorithm: path vs CF vs popularity. (2) Motivation: teacher vs category vs learner relation. (3) Complexity: path len 2 vs 4 vs 6.
- Likert items (Explanation Satisfaction Scale): usefulness to goal, trust, sufficient details, irrelevant details. Stats: Kruskal-Wallis, Mann-Whitney, Benjamini-Hochberg.

## Results RQ2 (p7-10, Table 3 p10)
- Algorithm: path-based 13, CF 10, popularity 1. Path-based highest trust (4.00) and detail (4.00). Popularity lowest (biased, not personal). Usefulness: CF 3.96, path 3.80.
- Motivation: Credits group favored teacher (4.31 usefulness) then category. Learn group category ~ teacher (3.92), learner lowest (3.50). Preferred: category 12, teacher 8, learner 3. Learner path judged lacking info about the learner.
- Complexity: len 4 preferred (11), len 2 (10), len 6 (2). Len 6 seen as irrelevant detail by 17. Complaints: arbitrary steps, "too crowded".
- Learn group wants more detail than Credits.

## Discussion / future work (p10-11)
- Accuracy vs interpretability tradeoff: longer path = better metric, worse UX.
- Mitigations: reward shaping (smaller reward for longer paths), show only shortest path, diversify relations in training, enrich KG with categories/concepts, show multiple paths per course.
- Agent overuses learner-enrolled paths, which users trust/value least.
