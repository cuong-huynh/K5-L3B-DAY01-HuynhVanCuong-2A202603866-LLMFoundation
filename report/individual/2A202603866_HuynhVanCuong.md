# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603866
- **Họ và tên:** Huỳnh Văn Cường
- **SEED** (in ở ô đầu notebook): `3080`

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Bảng so sánh trên cùng tập dữ liệu (`SEED = 3080`, sau khi bỏ trùng còn `5.169` tin: `train = 3.619`, `val = 775`, `test = 775` với `99` spam, `SUB = 40` tin cân bằng 20 spam / 20 ham). Số liệu lấy từ lần **Run all** local trong `lab1_report.json`. **Phần 4 (LLM) chưa chạy được** vì thiếu API key — các ô gọi API bị bỏ qua; TODOs 4.x vẫn pass nhờ unit test mock.

| Mô hình / Phương án | Precision | Recall | F1 (test 775) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1. BoW + LogReg (tự code)** | 1.000 | 0.818 | 0.900 | 0.889 | 1.4 | 157,741 | 0.01 | 0.01 | 0.000 |
| **2. BoW + MLP (deep learning)** | 0.978 | 0.889 | 0.931 | 0.947 | 1.4 | 201,536 | 0.04 | 0.05 | 0.000 |
| **3a. Encoder (MiniLM) + kNN** | 0.832 | 0.949 | 0.887 | 0.976 | 2.7 | 311 | 14.72 | 210.84 | 0.000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.957 | 0.909 | 0.933 | 1.000 | 2.9 | 1,619 | 8.41 | 12.92 | 0.000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.958 | 0.919 | 0.938 | 1.000 | 3.1 | 1,667 | 6.76 | 7.46 | 0.000 |
| **4a–4c. LLM (zero/few-shot)** | — | — | — | — | — | — | — | — | — |

*Kết quả 5 phép tính tay (`HAND_CALC`):*
- `baseline_accuracy` = `4516 / 5169` = **`0.87`**
- `sigmoid_0` = $1 / (1 + e^0)$ = **`0.50`**
- `f1_example` (TP=80, FP=20, FN=40) → $P = 0.80, R = 2/3 \Rightarrow F_1 = \mathbf{0.73}$
- `cosine_example` (`[1, 2, 0]` và `[2, 1, 2]`) → $4 / (\sqrt{5}\cdot 3) \approx \mathbf{0.60}$
- `llm_cost_per_day` (1M SMS/ngày, 100 token in, 1 token out, giá \$0.10/\$0.50 mỗi 1M token) = **`10.50`** USD/ngày

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` có sự khác biệt thú vị nhất so với thực tế khi chạy thực nghiệm:

1. **`dl_curve` (MLP vs. Logistic Regression trên BoW):**
   - *Dự đoán:* Nghĩ rằng mạng nơ-ron nhiều tham số hơn (128.129 tham số so với 2.001 của LogReg) hoặc sẽ vượt trội hẳn, hoặc sẽ giảm val loss lâu hơn.
   - *Thực tế:* `val loss` chạm đáy rất sớm (khoảng epoch 5) rồi bắt đầu tăng ngược lên trong khi `train loss` tiến sát về `0.000`; F1 trên tập test chỉ nhỉnh hơn nhẹ (`0.931` so với `0.900`).
   - *Lý do lệch:* Đầu vào vẫn là vector Bag-of-Words nhị phân 2.000 chiều (mất thứ tự từ và ngữ nghĩa ngoài từ điển), cộng với tập train nhỏ (`3.619` mẫu) nên mạng MLP dư thừa dung lượng và nhanh chóng học thuộc lòng (overfit).

2. **`emb_tricky` & `3b vs 3c` (Thêm MLP head lên trên Embedding MiniLM):**
   - *Dự đoán:* Kỳ vọng thêm lớp ẩn MLP (`3c`) lên trên vector 384 chiều của MiniLM sẽ giúp tăng F1 rõ rệt giống như khi chuyển từ `1. BoW + LogReg` sang `2. BoW + MLP`.
   - *Thực tế:* Giữa `3b (Encoder + LogReg)` và `3c (Encoder + MLP head)`, điểm F1 gần như ngang nhau (`~0.933` vs `~0.938`, và cùng đạt `1.000` trên SUB 40 tin), trong khi trên 6 câu bẫy `tricky`, `Encoder + LogReg` thắng áp đảo `BoW + LogReg`.
   - *Lý do lệch:* Bản thân encoder `all-MiniLM-L6-v2` (6 lớp Transformer, ~22,7 triệu tham số) đã thực hiện các phép biến đổi phi tuyến sâu để gom các câu cùng ngữ nghĩa về hai cụm tách biệt tuyến tính sẵn trong không gian 384 chiều, nên chỉ cần 1 siêu phẳng tuyến tính (LogReg) là đủ.

3. **`hard_eval` (Độ sụt giảm trên bộ 50 case khó):**
   - *Dự đoán:* Dự đoán các mô hình đạt F1 `0.90–0.93` trên tập test chuẩn sẽ giữ được ít nhất `55–65%` F1 trên bộ 50 case khó.
   - *Thực tế:* Mô hình `1. BoW + LogReg` tụt F1 từ `0.900` xuống chỉ còn khoảng `0.378` (accuracy ~`54%`); `2. BoW + MLP` còn `0.450` / acc `56%`, trong khi Embedding giữ được F1 ~`0.67–0.72`.
   - *Lý do lệch:* Tập test chuẩn có cùng phân phối với tập train (nhiều mẫu quảng cáo lặp lại từ khoá `free`, `win`, `txt`, `call`), tạo cảm giác mô hình "hiểu" ngôn ngữ. Khi gặp văn bản cố tình viết lách từ khoá hoặc hội thoại đời thường chứa từ khoá spam, điểm mù của biểu diễn đếm từ mới bộc lộ hoàn toàn.

## 4. Phân tích 50 case khó

- **Mô hình tụt điểm mạnh nhất:** `1. BoW + LogReg` và `2. BoW + MLP` tụt mạnh nhất (F1 từ ~`0.90–0.93` xuống ~`0.38–0.45`). Nhóm case làm khó BoW nhất là:
  - **Spam giọng tự nhiên (`natural_spam`)** và **Thao túng / lừa đảo (`manipulation`)**: Hoàn toàn không chứa từ khoá quảng cáo truyền thống.
  - **Né từ khoá (`obfuscation`)**: Các từ khoá bị biến dạng thành token lạ nằm ngoài `vocab` 2.000 từ.
  - **Nói về spam (`talks_about_spam`)** và **Tưởng spam nhưng không phải (`ham_looks_spam`)**: Chứa dày đặc từ khoá spam khiến tổng trọng số dương vượt ngưỡng `0.5`.

- **Phân tích 2 case cụ thể và cơ chế gây sai:**
  1. **Case #21 (`manipulation`, nhãn thật = `spam`):**
     > *"Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account? I'll explain later x"*
     - **Cơ chế sai của BoW / MLP:** Hàm `tokenize` tách câu thành các từ giao tiếp gia đình rất phổ biến trong lớp `ham` (`hi`, `mum`, `i`, `my`, `phone`, `in`, `the`, `this`, `is`, `can`, `you`, `to`, `later`). Trong Logistic Regression, các đại từ `i`, `my`, `you` mang trọng số âm rất mạnh kéo về `ham`, trong khi không có từ khoá quảng cáo nào (`free`, `win`, `claim`). Do không hiểu mối quan hệ ngữ nghĩa giữa *"số điện thoại mới"* + *"chuyển £200 sang tài khoản bạn"* là kịch bản giả mạo người thân, mô hình tự tin đoán nhầm là `ham` (False Negative).
  2. **Case #29 (`obfuscation`, nhãn thật = `spam`):**
     > *"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m"*
     - **Cơ chế sai của BoW (và làm khó cả Embedding):** Bộ tách từ `re.findall(r"[a-z0-9£$€]+", text.lower())` giữ nguyên các chuỗi trộn chữ-số như `c0ngr4tulat10ns`, `w0n`, `cl4im`. Vì từ điển `vocab` chỉ chứa từ viết chuẩn (`congratulations`, `won`, `claim`), gần như toàn bộ các từ quan trọng trong câu này đều bị rơi vào trạng thái Out-Of-Vocabulary (vector `x` gần như toàn số `0`), dẫn đến xác suất $\sigma(b) < 0.5 \Rightarrow$ đoán `ham`. Ngay cả với MiniLM, việc thay số vào giữa từ làm bộ WordPiece tokenizer cắt vụn một từ thành nhiều sub-token vô nghĩa (`c0`, `##ng`, `##r4`, ...), làm biến dạng vector ngữ nghĩa của câu.

## 5. Thử thách cá nhân

- **Đề bài nhận được (`SEED = 3080` → `3080 % 6 = 2` → Đề #3):**
  > *"DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?"*

- **Cách làm:**
  Chạy vòng lặp huấn luyện `build_mlp(2000, hidden=h)` với `h ∈ {4, 64, 512}` trong cùng 30 epoch (`torch.manual_seed(3080)`), đo tổng số tham số `sum(p.numel() for p in m.parameters())`, thời gian huấn luyện và F1 trên tập test:

  ```python
  for h in [4, 64, 512]:
      torch.manual_seed(SEED)
      m_ch = build_mlp(X_tr.shape[1], hidden=h)
      n_params = sum(p.numel() for p in m_ch.parameters())
      t0 = time.perf_counter()
      hist_ch = train_mlp(m_ch, X_tr, y_tr, X_va, y_va, epochs=30)
      dt = time.perf_counter() - t0
      res_ch = metrics(y_te, predict_mlp(m_ch, X_te))
      print(f"hidden={h:>3} | tham số={n_params:>10,} | train={dt:.2f}s | TEST F1={res_ch['f1']:.3f}")
  ```

- **Kết quả thực nghiệm:**

  | Số nơ-ron lớp ẩn (`hidden`) | Công thức tính tham số | Tổng số tham số | Thời gian train (30 epochs) | F1 (test) |
  | ---: | --- | ---: | ---: | ---: |
  | **`4`** | $(2000 \times 4 + 4) + (4 \times 1 + 1)$ | **8,009** | ~0.7 s | **0.936** |
  | **`64`** (mặc định) | $(2000 \times 64 + 64) + (64 \times 1 + 1)$ | **128,129** | ~1.3 s | **0.931** |
  | **`512`** | $(2000 \times 512 + 512) + (512 \times 1 + 1)$ | **1,025,025** | ~3.8 s | **0.931** |

- **Giải thích:**
  - Tăng lớp ẩn từ `4` lên `512` làm số lượng tham số phình gấp **128 lần** (`8.009` $\rightarrow$ `1.025.025` tham số) và thời gian huấn luyện chậm hơn ~5 lần, nhưng **F1 trên tập test không tăng** (thậm chí `hidden=4` đạt `0.936`, nhỉnh hơn `0.931` của mạng lớn). Nguyên nhân là bài toán phân loại nhị phân trên biểu diễn Bag-of-Words có ranh giới quyết định khá đơn giản; mạng `512` nơ-ron có số tham số gấp gần 300 lần số mẫu train (`3.619`) nên nhanh chóng bão hoà và overfit.
  - **Khi nào nên chọn mạng nhỏ (`hidden = 4`)?**
    1. Khi tập dữ liệu huấn luyện nhỏ (vài nghìn mẫu) — mạng nhỏ đóng vai trò như một cơ chế điều chuẩn (*regularization*) tự nhiên chống overfitting.
    2. Khi đặc trưng đầu vào đã được trích xuất rõ ràng hoặc ranh giới phân lớp gần tuyến tính.
    3. Khi triển khai hệ thống cần tối ưu bộ nhớ (RAM/cache), giảm độ trễ suy luận (inference latency) hoặc chạy trên thiết bị biên (edge/mobile).

## 6. Phương án cho sản phẩm thật

**Bài toán:** Lọc **10 triệu SMS/ngày** (~116 tin/giây trung bình, giờ cao điểm ~500 tin/giây), yêu cầu SLA độ trễ **`p95 < 50 ms/tin`**.

1. **Đánh giá các phương án đơn lẻ theo số liệu thực nghiệm (và ước lượng chi phí LLM):**
   - **Nhóm LLM (`4a/4c`):** Chưa đo latency trên máy này (thiếu API key). Theo bài toán tính tay `llm_cost_per_day` và bảng giá trong notebook, gọi API tuần tự thường mất **hàng trăm ms/tin** (khó đạt SLA `p95 < 50 ms`); với 10 triệu tin/ngày, model nhỏ kiểu `gpt-6-luna` (~\$10.50 / 1M tin theo `HAND_CALC`) đã khoảng **\$105/ngày**, model lớn đắt hơn khoảng một bậc.
   - **Nhóm BoW (`1. BoW + LogReg`, `2. BoW + MLP`):** Cực nhanh trên máy local (`p95` ~`0.01–0.05 ms`, thông lượng rất cao, chi phí API = \$0), đạt F1 test chuẩn `0.90–0.93`, nhưng dễ bị qua mặt bởi spam lách từ khoá hoặc giọng tự nhiên (F1 trên 50 case khó tụt còn ~`0.35–0.45`).
   - **Phương án đơn lẻ tốt nhất đạt SLA (đã đo):** **`3b. Encoder (MiniLM) + LogReg`** (hoặc `3c`). Đạt `p95 ≈ 12.9 ms < 50 ms` (`3c` còn ~`7.5 ms`), thông lượng cả lô `~1.600 tin/s`, chi phí API = `0 USD`, F1 đạt `0.933–0.938` trên tập test thường và bền hơn BoW trên case khó. (`3a kNN` có `p95` cao khi đo từng tin nên không chọn làm mặc định.)

2. **Đề xuất kiến trúc lai (Cascading Router 3 tầng) cho Production:**
   - **Tầng 0 — Tiền xử lý chuẩn hoá (De-obfuscation < 0.05 ms):** Chuyển đổi ký tự thay thế (`0→o, 1→i, 4→a, €→e`), gom các chữ cái bị chèn khoảng trắng/dấu chấm (`F R E E` $\rightarrow$ `free`, `c.a.s.h` $\rightarrow$ `cash`).
   - **Tầng 1 — Bộ lọc nhanh `BoW + LogReg` (`p95 ~ 1 ms`, xử lý ~85% lưu lượng):** Chặn ngay các mẫu spam quảng cáo rõ ràng ($p > 0.95$) và cho qua ngay các tin nhắn hội thoại ngắn thông thường ($p < 0.05$).
   - **Tầng 2 — Bộ lọc ngữ nghĩa `Encoder MiniLM + LogReg` (`p95 ~ 12 ms`, xử lý ~14% lưu lượng có $p \in [0.05, 0.95]$):** Phân giải chính xác các tin nhắn bạn bè có chứa từ `"free"/"win"` và các tin spam viết giọng tự nhiên.
   - **Tầng 3 — `LLM nhỏ + Prompt phòng thủ <sms>` (chạy bất đồng bộ cho ~1% ca cực khó $p \in [0.35, 0.65]$):** Xử lý các chiến dịch lừa đảo mới (phishing/manipulation) và tự động gom nhãn để định kỳ huấn luyện lại Tầng 1 & 2.
   - **Đánh đổi chấp nhận:** Kiến trúc lai tăng nhẹ độ phức tạp vận hành (duy trì pipeline 2–3 model), nhưng đổi lại giúp giảm **~85% tải tính GPU** cho Encoder, giảm **99% chi phí API LLM** (chỉ còn ~\$30/tháng), đảm bảo `p95` toàn hệ thống luôn dưới **15 ms** (dư sức đạt SLA `< 50 ms`) mà vẫn chống được cả spam biến tướng lẫn prompt injection.

## 7. AI log và điều đã học

### AI log

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| **Phần 4 (TODO 4.1)** | Hỏi cú pháp gọi OpenAI SDK `chat.completions.create` kết hợp `**provider_kwargs(max_tokens)` để tương thích với model GPT-6 (`max_completion_tokens` và `reasoning_effort="none"`). | AI gợi ý khởi tạo `OpenAI()`, đo thời gian bằng `time.perf_counter()`, truyền `**provider_kwargs(max_tokens)` và lấy `resp.choices[0].message.content or ""`. | Kiểm tra với unit test mock `4.1 + 4.4`: xác nhận `max_tokens` không bị truyền trùng khi `token_param="max_completion_tokens"` và `reasoning_effort="none"` được gửi đúng. |
| **Phần 4 (TODO 4.2–4.4)** | Hỏi cách viết hàm `parse_label(response)` an toàn trước câu trả lời `"Not spam"` hoặc `"Label: spam"` và cách ước lượng chi phí token trong `llm_classify_batch`. | AI gợi ý chuẩn hoá chữ thường, xoá tiền tố `"label:"`, kiểm tra cụm `"not spam"` trước khi dùng regex `\bspam\b` / `\bham\b`, và tính token theo công thức `len(text.split()) / 0.75`. | Tự kiểm chứng trên toàn bộ các case biên (`" Spam."`, `"Label: spam"`, `"HAM\n"`, `"Not spam"`, `"I am not sure" → None`) và kiểm tra biến đếm `n_invalid` khi trả về `None`. |

### Điều đã học hôm nay

1. **Điểm F1 cao trên tập test cùng phân phối có thể gây ảo tưởng lớn:** Một mô hình Bag-of-Words đạt F1 `0.90` trên tập test chuẩn thực chất chỉ đang "nhớ mặt từ khoá" chứ chưa hiểu ý đồ câu chữ; khi gặp bộ stress test 50 case khó (lừa đảo mạo danh người thân, né từ khoá bằng số/dấu chấm), F1 lập tức sụp đổ xuống dưới `0.35`.
2. **Giá trị cốt lõi của Pre-trained Representation (Embedding):** Thêm lớp ẩn MLP lên trên Bag-of-Words (tăng gấp 64–512 lần tham số) không giúp mô hình hiểu được từ đồng nghĩa như *"complimentary"* hay *"at no cost"*, nhưng chỉ cần mượn một bộ Encoder đóng băng (MiniLM) cộng với bộ phân loại tuyến tính đơn giản là đã giải quyết được bài toán ngữ nghĩa với độ trễ chỉ ~12 ms.
3. **LLM mạnh ở khả năng Zero-shot nhưng cần phòng thủ và tối ưu chi phí:** Khi dùng các model thế hệ mới (reasoning models), nếu không tắt suy luận (`reasoning_effort="none"`) hoặc đặt `max_tokens` quá nhỏ thì toàn bộ ngân sách token sẽ bị tiêu hết vào bước suy nghĩ ẩn; đồng thời trong thực tế sản xuất, LLM nên được bọc bằng prompt tách biệt dữ liệu (`<sms>...</sms>`) và đặt sau một bộ định tuyến (Router) thay vì gọi trực tiếp cho 100% lưu lượng.
