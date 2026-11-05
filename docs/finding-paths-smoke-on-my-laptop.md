# Thesis feasibility: findings, spikes, risks

Cập nhật: 2026-10-29. Tổng hợp từ các session trước. Nguồn chi tiết: `docs/thesis-context.md`, `docs/init_spikes.md`, `docs/adr/0001-run-on-colab.md`.

## 1. Topic

**Explanation-Oriented Reward Shaping for RL-Based Explainable MOOC Recommendation.**

- Baseline: UPGPR (paper "Finding Paths for Explainable MOOC Recommendation: A Learner Perspective"). Framing ML/technical, user study chỉ là phụ.
- Ý tưởng: giữ UPGPR, sửa reward `R = R_rec + l1 R_length + l2 R_relation (+ l3 R_diversity)` để điều khiển thuộc tính cấu trúc của reasoning path, không làm giảm đáng kể recommendation.
- Main RQ: How does explanation-oriented reward shaping affect recommendation performance and structural properties of reasoning paths in RL-based MOOC recommendation?
- Sub-RQ: RQ1 thuộc tính nào đưa vào reward được; RQ2 ảnh hưởng accuracy; RQ3 ảnh hưởng length / relation composition / diversity / redundancy; RQ4 trade-off.
- Scope MSc: length + relation composition trước. Diversity, redundancy là metric đánh giá. Counterfactual relevance là mở rộng tùy chọn.
- Wording rule: dùng "explanation-oriented path properties". Không claim "explanation tốt hơn / dễ hiểu hơn / đáng tin hơn". Không claim chưa ai làm (gap xuất phát từ phân tích cơ chế UPGPR).

## 2. Đã xác minh (hệ thống chạy được)

Tất cả là smoke run 2 epoch, Xuetang (MOOCCube), Colab free T4, seed 23. **Chưa có run đủ epoch, chưa reproduce số nào của paper.**

| Việc | Kết quả |
|---|---|
| Deps Python 3.10 (uv venv) trên Colab | OK |
| Data MOOCCube trên Drive | OK, zip 1090.8 MB |
| preprocess + make_dataset | 6,507 users, 687 courses, 12,207 KG nodes, split 75,310 / 9,053 / 12,860 |
| TransE | chạy được, ~23 s/epoch, **chỉ 2 epoch**, chưa chạy đủ 30 |
| UPGPR len 3 train + test | chạy được, ~26 s/epoch, test ~400 s |
| PGPR(3) | chạy được, nhưng "path khớp pattern" **chưa xác minh** |
| Len 4, 5 | có timing, chưa có metric |
| Path-property tool | chạy được, số liệu chỉ trên beam candidates của model 2 epoch |
| GPU vs CPU | 88 s vs 97 s cho cùng run: agent bị CPU-bound, GPU gần như vô ích |

Smoke metrics (không so sánh được với paper): UPGPR NDCG 15.10, PGPR(3) NDCG 14.67.

## 3. Findings kỹ thuật quan trọng

1. **Protocol "+1" của paper:** train dài hơn eval 1 bước (p.5). `config/UPGPR/mooc.json` mặc định train 3 / eval 3, không theo paper. Cần ADR-0002 chọn protocol trước khi chạy full. Weights không phụ thuộc path length nên checkpoint train L+1 load được cho eval L.
2. **`topk` phải có đúng `max_path_len` phần tử** (beam search chạy `len(topk)` bước). Len 4 `[25,5,1,1]`, len 5 `[25,5,1,1,1]` là lựa chọn của mình, không phải của paper.
3. **Độ dài episode cố định = `max_path_len`.** Độ dài hiệu dụng chỉ đổi được qua self-loop (action 0). Self-loop dùng rất ít ở model 2 epoch (0.11% path). Đây là rủi ro lớn nhất cho `R_length`.
4. **Reward là hàm thuần của path** `[(relation, node_type, node_id), ...]`, tính ở mỗi bước (`_get_reward`, registry `_reward_fns` trong `kg_env.py`). Thêm R_length / R_relation vào đây là khả thi về code. R_diversity cần context theo batch (`_batch_get_reward`), khó hơn.
5. **`use_pattern` chỉ ảnh hưởng reward**, không giới hạn beam search. Candidate của PGPR(3) vẫn có path ngoài pattern. Pattern 4 trong config (`item_school` rồi `item_concept`) có vẻ không bao giờ khớp được (suy từ config, chưa xác nhận bằng run).
6. **Chi phí:** agent + test CPU-bound. Test tăng ~5x từ len 3 lên len 4 (nguyên nhân chưa điều tra). Ước tính (ngoại suy từ 2 epoch): 1 run UPGPR len 3 đủ epoch ~50 phút gồm TransE; ~4 h mỗi seed cho mọi cấu hình; ~12 h cho 3 seed.
7. **`train_agent`, `train_transe` không resume được.** Runtime Colab free có thể bị thu hồi (đã xảy ra 1 lần). Dùng marker file có epoch count, state trên Drive.
8. Baselines Pop / NeuMF / CFKG bỏ, trích số paper (Xuetang NDCG 3.06 / 14.05 / 14.20). PGPR(3) giữ. COCO chưa có, phải email tác giả.

## 4. Path properties quan sát được (model 2 epoch, mọi beam candidate)

| Property | UPGPR len 3 | PGPR(3) |
|---|---|---|
| Valid rate (kết thúc ở item) | 91.7% | 91.9% |
| Effective length mean | 2.998 | 3.000 |
| Path có self-loop | 0.11% | 0.01% |
| Unique pattern / entropy | 5 / 0.71 bits | 5 / 0.57 bits |
| `enroll > enroll > enroll` | 81.9% | 86.8% |
| `enroll > item_concept > item_concept` | 17.8% | 13.1% |
| Pattern có teacher/school | 0.15% | 0.08% |
| Path lặp node | 0% | 0% |
| Valid path kết thúc ở item đã enroll (train) | 18.3% | 19.2% |

**Cảnh báo:** đây là toàn bộ beam candidate (~60 path/user), không phải final explanation path mà `evaluate_paths` chọn cho từng cặp user-course. Model gần như chưa train. Chỉ dùng để định hướng, **không trích như kết quả**.

**Đọc sơ bộ:** relation composition rất lệch (gần 100% là enroll hoặc concept, teacher/school gần 0). Gap "reward không kiểm soát relation" có vẻ có thật, nhưng cũng có thể do KG thưa teacher/school (xem spike A).

## 5. Spike checks để chứng minh hướng đi có cơ sở

Nhóm này khác nhóm "chạy được": mỗi spike trả lời một câu hỏi về giá trị của thesis. Mỗi spike có kill criteria ghi trước khi chạy.

| # | Spike | Câu hỏi | Cần train? | Kill criteria / ý nghĩa |
|---|---|---|---|---|
| A | KG reachability | Với test pair (user, course đúng), bao nhiêu % có path 3 hop qua teacher / school / concept, so với chỉ enroll? | Không, chỉ graph | Path không-enroll < ~5%: R_relation không có chỗ tác động, bỏ hoặc đổi KG / dataset |
| B | Post-hoc baseline | Từ ~60 beam candidate mỗi user, chọn path ngắn nhất hoặc relation ưa thích sau khi train. Property đổi bao nhiêu, NDCG đổi bao nhiêu? | Không, chỉ test output | Câu hỏi reviewer chắc chắn hỏi. Nếu post-hoc đạt tương đương, motivation yếu, cần luận điểm khác (policy học được, không cần lọc) |
| C | Headroom baseline đủ epoch | Final explanation path của UPGPR đủ epoch: phân bố length, relation, self-loop rate | Cần full run | Nếu vẫn ~82% `enroll>enroll>enroll` thì gap relation có thật. Nếu self-loop ~0 thì R_length không có đòn bẩy |
| D | Length lever | Episode cố định độ dài, "ngắn hơn" thực hiện bằng cách nào? | Đọc code + run nhỏ | Nếu self-loop không dùng được, R_length phải thiết kế lại (thưởng self-loop, hoặc so sánh theo len 3/4/5 như biến độc lập) |
| E | Controllability | Đặt lambda cực lớn (thưởng mạnh path có teacher), train vài epoch: phân bố path có dịch chuyển không? | Smoke ngắn | Chứng minh plumbing reward hoạt động và agent phản ứng. Kiểm tra reward hacking (toàn self-loop) |
| F | Trade-off tồn tại | Paper: NDCG len 3/4/5 = 18.43 / 17.96 / 20.85. Reproduce có thấy accuracy đổi theo length / composition không? | Full run len 3 vs 5 | Nếu property không ảnh hưởng accuracy thì không có trade-off để nghiên cứu (RQ4 rỗng) |
| G | Seed variance | Std của NDCG qua 3 seed | Full runs | Biết hiệu ứng nhỏ nhất phát hiện được. Claim "không giảm accuracy" cần cái này |
| H | Novelty | Đã có paper shaping reward theo path length / relation cho PGPR-family chưa (CAFE, UCPR, ADAC, PLM-Rec, ...)? | Đọc paper | Đặt gap đúng, tránh claim sai |

**Thứ tự đề xuất:** A và B trước (dùng dữ liệu và output sẵn có, không tốn Colab) → D (đọc `kg_env.py`) → C + F + G cùng baseline full run → E → H song song bất kỳ lúc nào.

## 6. Rủi ro chính

1. **R_length yếu:** nếu agent hầu như không dùng self-loop, độ dài hiệu dụng gần như hằng số = `max_path_len`.
2. **Relation lệch do dữ liệu, không do reward:** teacher/school có thể hiếm trong KG nên shaping chỉ đổi accuracy mà không đổi composition.
3. **Post-hoc lọc là baseline mạnh** (spike B).
4. **Reproduce lệch paper:** protocol (L+1, epoch, seed) chưa chốt. Chưa có baseline tin cậy thì chưa tin được kết quả reward mới.
5. **Chi phí compute:** ~12 h / 3 seed / cấu hình, nhân thêm số biến thể reward và dataset. Colab free không đảm bảo. Cân nhắc Colab Pro hoặc máy nhiều core.
6. **Chỉ một dataset (Xuetang):** COCO phải xin tác giả. Kết luận khó khái quát.
7. **Metric offline không chứng minh perception.** Giữ wording rule.

## 7. Quyết định còn mở

- ADR-0002: protocol baseline (train L+1 như paper, hay 3/3 mặc định), số seed.
- Chạy TransE đủ 30 epoch (marker `.done_train_transe_e30`, ghi đè embedding 2 epoch). TransE chỉ cần 1 lần mỗi seed, dùng chung mọi cấu hình agent.
- Có chạy COCO không (phải email tác giả).
- Nơi lưu file: docs trong worktree `spikes-init` (chưa commit) vs main checkout; notebook, thesis-context, ADR đang untracked ở main.

## 8. Việc tiếp theo

1. Ghi spike A-H vào `docs/init_spikes.md` (hoặc giữ file này làm nguồn chính).
2. Spike A và B.
3. Chốt ADR-0002, rồi chạy full UPGPR len 3 seed 23, so NDCG với 18.43 của paper.
4. Đọc `kg_env.py` để thiết kế R_length / R_relation (spike D).
