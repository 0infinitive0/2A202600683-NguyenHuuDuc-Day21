# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Hữu Đức — 2A202600683
**Ngày nộp**: 2026-06-25
**Submission option**: A (lightweight)
**Link Google Drive (large files)**: https://drive.google.com/file/d/1HCV87kn3uZTk0nDobv19iDsB1quAySdm/view?usp=sharing

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (p95 = 562, rounded up)
- **GPU**: Tesla T4, 15.6 GB VRAM
- **Training cost**: $0.07 (~12.7 phút @ $0.35/hr)

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200       | 4.23 min   | 12.69 GB  | 1.557694  | 4.747861   |
| 16   | 3,686,400       | 4.22 min   | 10.61 GB  | 1.511347  | 4.532834   |
| 64   | 14,745,600      | 4.24 min   | 13.47 GB  | 1.476815  | 4.378976   |
| Base | -               | -          | -         | -         | -          |

## 3. Loss Curve Analysis
[Đính kèm loss_curve.png trong file nộp]
- Quan sát: Ở chế độ chạy trên Tesla T4, chức năng đánh giá trong quá trình huấn luyện (eval-during-training) đã được tắt đi để tiết kiệm VRAM, vì vậy notebook chỉ sinh ra được training loss curve. Nhìn vào quá trình huấn luyện, loss giảm đều và không có dấu hiệu phân kỳ. Việc kiểm tra final eval loss cho ra mức perplexity khá tốt cho thấy chưa có dấu hiệu overfitting nghiêm trọng trên 3 epochs này. 

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản...
**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI...
**Nhận xét**: Model fine-tuned trả lời có định hướng cấu trúc mạch lạc hơn một chút nhưng có hơi hướng dịch thuật (hallucinated từ "bộ môn công nghệ máy tính"), trong khi Base model giải thích tự nhiên hơn bằng cách dùng "phân khúc của trí tuệ nhân tạo".

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n bằng cách sử dụng vòng lặp. ```python def fibonacci(n): if n <= ...
**Fine-tuned (r=16)**: Để tính số Fibonacci thứ n, bạn có thể sử dụng đệ quy hoặc vòng lặp. Đây là một ví dụ về cách tính số Fibonacci thứ n bằng cách sử dụng đệ quy: ```python def fibonacci(n): if n == 0: return 0 elif n =...
**Nhận xét**: Cả 2 đều đưa ra cách giải quyết đúng. Base model chọn vòng lặp trong khi Fine-tuned model chọn thuật toán đệ quy.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: 1. Đơn giản: Nguyên tắc này đề xuất rằng thiết kế của sản phẩm nên đơn giản và dễ sử dụng, không có quá nhiều chi tiết không cần thiết. 2. Thích ứng: Nguyên tắc này khuyến khích thiết kế có thể thay đ...
**Fine-tuned (r=16)**: 1. Tạo trải nghiệm người dùng liền mạch: Trải nghiệm người dùng liên quan đến cách người dùng tương tác với sản phẩm. 2. Làm nổi bật chức năng quan trọng: Đặt trọng tâm của sản phẩm vào các chức năng ...
**Nhận xét**: Fine-tuned model có vẻ tập trung nhiều hơn về khía cạnh UX (trải nghiệm người dùng) thay vì UI cụ thể (giao diện).

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: LoRA và QLoRA là hai mô hình cải tiến được tạo ra để cải thiện tính khả dụng và hiệu suất của các mô hình NLP lớn. LoRA (Layer-wise Auto-Scaling) là một cách để thay đổi kích thước các màng nhầy của m...
**Fine-tuned (r=16)**: LoRA (Large Language Model Adapter) và QLoRA (Quantized LoRA) là hai phương pháp cải tiến được giới thiệu để cải thiện hiệu suất và khả năng ứng dụng của mô hình ngôn ngữ lớn. LoRA là phương pháp cải ...
**Nhận xét**: Cả hai đều cố gắng định nghĩa từ viết tắt. Base model có biểu hiện hallucinate ("màng nhầy", "Layer-wise Auto-scaling"). Fine-tuned model cải thiện độ ảo giác hơn, biết rằng QLoRA là Quantized LoRA mặc dù giải nghĩa từ LoRA chưa thật sự chính xác hoàn toàn (Low-Rank Adaptation).

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
**Base**: Prompt engineering, RAG và fine-tuning là những kỹ thuật khác nhau được sử dụng để cải thiện hiệu quả của hệ thống AI bằng cách cung cấp cho hệ thống một câu hỏi hoặc một tập tin dữ liệu để làm việc. ...
**Fine-tuned (r=16)**: Prompt Engineering, RAG (Reinforcement Aggregation Generation) và Fine-tuning là ba cách khác nhau để huấn luyện mô hình AI. 1. Prompt Engineering: Prompt Engineering là cách tạo ra prompt (một câu hỏ...
**Nhận xét**: Cả Base và Fine-tuned model đều nhận diện được đây là 3 kỹ thuật bổ trợ AI. Tuy nhiên Fine-tuned model giải nghĩa từ viết tắt sai cho RAG (Reinforcement Aggregation Generation thay vì Retrieval-Augmented Generation).

## 5. Conclusion về Rank Trade-off
- **Rank nào cho ROI tốt nhất trên dataset này? Tại sao?**
Rank 16 mang lại giá trị ROI (Return on Investment) tốt nhất. So với r=8, r=16 cho eval loss và perplexity giảm tốt hơn nhưng mức tiêu thụ peak VRAM lại tối ưu nhất (chỉ mất khoảng 10.61 GB so với 12.69 GB của r=8 hoặc 13.47 GB của r=64). Điều này cho thấy r=16 sử dụng tài nguyên vô cùng hiệu quả để mang lại kết quả chất lượng tốt.
- **Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?**
Khi chúng ta tăng rank từ 16 lên 64, số lượng parameters cần train tăng gấp 4 lần (từ 3.6 triệu lên 14.7 triệu), và VRAM đỉnh điểm tăng khoảng ~3 GB, nhưng perplexity chỉ cải thiện một biên độ rất nhỏ (từ 4.53 xuống 4.37). Điều này chính là dấu hiệu của diminishing returns — trả giá quá nhiều tài nguyên nhưng độ chính xác lại không được tăng lên tương xứng.
- **Recommendation: nếu deploy production, bạn chọn rank nào? Tại sao?**
Đối với dataset instruction đơn giản như này (khoảng 200 samples) và mục tiêu hiệu quả chi phí, lựa chọn ưu tiên khi deploy ra production là rank 16. Lựa chọn này giúp cân bằng tốt nhất giữa độ trễ sinh từ (do ít parameter cập nhật hơn r=64), chi phí VRAM thấp khi load model phục vụ nhiều user, nhưng vẫn giữ được chất lượng văn bản gen ra tốt và ít hallucination.

## 6. What I Learned
- Biết được cách thiết lập pipeline LoRA/QLoRA kết hợp Unsloth, SFTTrainer từ TRL với Hugging Face transformers. Giúp tiết kiệm lượng lớn VRAM mà vẫn đạt hiệu quả mong đợi trên phần cứng cấu hình thấp (GPU T4 16GB).
- Hiểu được cách tính toán kích thước mô hình theo Rank (r) trong cấu hình PEFT (Target modules `q_proj` và `v_proj`) và trực tiếp thấy hiệu suất/ VRAM scale như thế nào thông qua từng giá trị rank khác nhau.
- Thấy được tầm quan trọng của việc đánh giá Qualitative khi Fine-tuned models. Một số trường hợp Perplexity tốt hơn chưa chắc đã sinh ra đoạn văn phong phú và tự nhiên hơn so với Base Model.
