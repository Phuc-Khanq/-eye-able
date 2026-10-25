# EYEable – Frontend (bàn giao)

Frontend demo cho **EYEable**: phần mềm giúp **người già và người khuyết tật** điều khiển máy tính **không cần dùng tay**, bằng **mắt, mặt, đầu và giọng nói** (chỉ cần webcam và micro thường).

> ⚠️ Đây **chỉ là frontend / prototype giao diện**. Chưa có model nhận diện thật: ánh mắt được **mô phỏng bằng con trỏ chuột**, còn nháy mắt, gật đầu và lắc đầu được **mô phỏng bằng phím hoặc nút**.


> **Cập nhật theo engine của Khang (24/10, bản mới nhất, branch `eyes`):**
> - **Con trỏ đi theo mắt** (theo chốt cuối với Khang). Hiệu chỉnh **1 chấm ở giữa**, khoảng 2 s.
> - **Click trái = nhướn mày** hoặc nói "click", tự bám (snap) vào nút gần nhất trong ~150 px. **Click phải = há miệng rồi ngậm** (0.3–1.5 s) hoặc nói "right click".
> - **Ngẩng / cúi đầu = cuộn** lên / xuống (giữ 0.4 s), không dùng nghiêng đầu. **Lắc = no / huỷ**, **gật = yes** (chỉ khi đọc chính tả). Gật **không** còn là click. **Dwell tắt mặc định**, 1.0 s khi bật.
> - Giọng nói: luôn nghe, chỉ tính giọng người dùng (camera thấy miệng mở), bản fine-tune cho giọng khó nghe là **research build** (lỗi 41% → 19%), chỉ tiếng Anh, tiếng Việt là "future".
> - Lệnh rủi ro ("send it", "click <tên>", xoá dữ liệu) luôn hỏi trước, 12 s không trả lời = no.
> - Live demo đã đổi theo: phím **B** = nhướn mày, **M** = há miệng, **↑ / ↓** = ngẩng / cúi đầu (cuộn), **H** = lắc, **N** = gật; click chuột trái/phải trên màn hình = nhướn mày / há miệng. Có hộp hỏi xác nhận 12 s, hiệu chỉnh 1 chấm, biểu đồ brow / mouth / pitch / yaw (vẫn là **số mô phỏng**, engine chưa gửi yaw/pitch/roll qua WebSocket).
> - Khi nối WebSocket: dùng `ws://127.0.0.1:8765` (không dùng `localhost`), toạ độ là pixel vật lý. Hợp đồng message ở `engine/core/messages.py`, test không cần camera bằng `python -m tests.fake_engine`.

---

## 1. Chạy thử

- Thuần HTML + CSS + JS, **không cần build, không cần `npm install`**, không phụ thuộc thư viện nào:
  ```
  eyeable-frontend/
  ├── index.html        cấu trúc 2 màn hình (Introduction, Live demo)
  ├── css/style.css     toàn bộ giao diện
  ├── js/app.js         toàn bộ logic (Peep, gaze engine, demo Windows, voice, ...)
  ├── assets/bg-1..3.jpg  3 ảnh nền vân sáng
  └── README.md
  ```
- Cách mở:
  - Mở bằng VS Code, chuột phải chọn **Open with Live Server** (extension *Live Server*).
  - Hoặc nhấp đúp mở thẳng trong Chrome.
- Cần **có mạng** để tải font từ Google Fonts. Không có mạng thì trang tự dùng font dự phòng.
- **Camera và mic thật** chỉ chạy khi mở qua `http://localhost` (Live Server) hoặc `https`, và trình duyệt sẽ hỏi quyền. Trên link artifact của Claude, camera bị chặn, nên trang tự chuyển sang khuôn mặt mô phỏng.

---

## 2. Cấu trúc màn hình

### Màn 1 – Introduction (5 trang, mỗi trang full màn hình)
Chuyển trang bằng **thanh menu bo góc nổi phía trên**: logo EYEable bên trái, bên phải là các nút Home · How it works · Controls · Voice · Gestures · **Live demo** (nút trắng). Phím **← / →** (cả PageUp/PageDown) cũng chuyển trang. Có hiệu ứng trượt khi chuyển trang. Code nằm ở khối `intro pages` trong `js/app.js` (hàm `showPage`), class `.ipage`, `.top` và `.pagelinks` trong `css/style.css`.

| # | Trang | Nội dung |
|---|---|---|
| 1 | **Home** | Nhãn "Hands-free computing…", tiêu đề *Your eyes, face and voice are the mouse.*, 6 chip, nút **Allow camera**, nút **How it works →**, con mắt **Peep** (mục 4) |
| 2 | **How it works** | Con mắt SVG tương tác (iris theo chuột, hiện offset/tọa độ) + 4 bước: tìm mặt, đo mắt, hiệu chỉnh 9 điểm, làm mượt và hành động |
| 3 | **Controls** | 6 thẻ: Look, Blink, Nod, Shake, Tilt, Speak, kèm thông số |
| 4 | **Voice** | Nhận diện giọng nói ngọng/khó nghe + ví dụ "oh-peh yoo-choob" → *Open YouTube* |
| 5 | **Gestures** | Bảng gesture map + khối **Open the live demo →** |

### Màn 2 – Live demo (chia 2 bên)
**Cột trái (rail):**
- **Camera:** khuôn mặt cụ già mô phỏng vẽ bằng canvas. Mắt nhìn theo con trỏ, nháy/gật/lắc/nghiêng được, có landmark và tia nhìn. Góc phải hiện bảng số **yaw / pitch / roll / eyes**.
- **Biểu đồ tín hiệu 6 giây:** 4 làn eyes, pitch, yaw, roll, mỗi làn có vạch ngưỡng. Khi tín hiệu vượt ngưỡng thì cắm cột CLICK / R-CLICK / ignored.
- Nút **Simulate:** Blink, Nod, Shake, Tilt L, Tilt R, Quick blink.
- **Detected signals:** các ô sáng lên khi phát hiện cử chỉ.
- **Voice commands:** giữ mic (hoặc phím V), có waveform. Dùng Web Speech API nếu trình duyệt hỗ trợ, không thì mô phỏng. Có ô gõ thử câu nói ngọng và các câu mẫu theo ngữ cảnh.
- **Settings:** bật/tắt blink, nod, shake, tilt, nhìn giữ; chỉnh thời gian nhìn giữ và độ mượt con trỏ; **hiệu chỉnh 9 điểm**.

**Cột phải – màn hình Windows giả lập:**
- Desktop icons, taskbar, Start menu, đồng hồ.
- **Chrome:** trang Google (tìm kiếm, kết quả) và YouTube (lưới video, tìm kiếm, trang xem có play/pause/like), có back/forward.
- **Mail:** Inbox, đọc thư, Reply, soạn thư (To/Subject/Message), Send, Sent.
- **Bàn phím ảo** gõ bằng mắt, có gợi ý từ.
- **Menu chuột phải** (Back, Reload, Copy, Paste, Close).
- **3 Guided demos tự chạy:** ① mở YouTube, tìm và phát video; ② viết và gửi email bằng giọng nói; ③ Google search bằng gõ mắt.
- Nhật ký hoạt động (EYES / HEAD / VOICE / SYSTEM) và mascot SVG đưa mẹo.

### Phím tắt trong demo
| Phím / thao tác | Mô phỏng |
|---|---|
| Di chuột | Ánh mắt |
| Nhìn giữ ~0.9 s | Click (bật/tắt trong Settings) |
| `B` / click chuột trái | Nhướn mày = click (snap) |
| `M` / chuột phải | Há miệng = click phải |
| `N` | Gật đầu = yes |
| `H` | Lắc đầu = no / huỷ |
| `↑` / `↓` | Ngẩng / cúi đầu = cuộn lên / xuống |
| giữ `V` | Nói lệnh |
| `P` | Tạm dừng tracking |

---

## 3. Thiết kế hiện tại (đã chốt với chủ project)
- **Luôn nền tối:** đã khóa, không bị đổi theo chế độ sáng/tối của trình duyệt. Bản sáng chỉ bật khi bấm ◐ (class `.page-light` trên `<html>`).
- **Nền:** 3 ảnh vân sáng xám-đen (do chủ project gửi, đã xoay ngang) **chồng lên nhau, mờ dần sang nhau, trôi và zoom chậm**, vòng lặp 15 s. Ảnh đã giảm sáng (`brightness(.72)`) và có lớp phủ tối để chữ dễ đọc. Xem class `.bgimg` và `@keyframes bgflow`.
- **Màu:** trắng / đen / xám. Panel kiểu **kính mờ** (`backdrop-filter`), nút dạng viên thuốc, nút chính trắng chữ đen.
- **Font:** **Inter** cho toàn trang, **Rubik 800** riêng cho chữ EYEable (đậm, hơi tròn nhẹ, thẳng hàng).
- **Điểm màu duy nhất:** con mắt Peep giữ màu tròng mắt thật (xám xanh rêu + vòng hổ phách).

---

## 4. Con mắt Peep (hero)
Nằm trong JS, khối `const PEEP = (() => { … })()`.
- Vẽ tròng mắt chi tiết vào canvas ẩn (`paintIris`), rồi lấy mẫu thành khoảng 1.000 chấm. Chấm to nhỏ ngẫu nhiên, vị trí lệch nhẹ.
- Chấm bọc lên **hình cầu 3D**: iris và đồng tử ở mặt trước, lòng trắng là các chấm xám thưa. Có **lưới line** nối các chấm, ánh sáng từ góc trên trái.
- **Vòng lặp khoảng 15 s:** chấm rải khắp nền, **bay theo đường cong tụ lại** thành nhãn cầu (4.4 s), giữ nguyên (7.5 s), rồi **tản ra** (2.6 s).
- Nhãn cầu **xoay theo chuột**. Rê chuột vào thì **đồng tử giãn**, bấm vào thì nháy.
- **Nháy mắt:** mí trên kéo xuống thành đường cong ‿, có lông mi rủ. Chỉ còn **nháy nhanh** (đã bỏ nháy lâu theo yêu cầu).
- Thông số chỉnh nhanh: `T_DRIFT / T_GATHER / T_HOLD / T_SCATTER` (thời gian), `THETA_I` (cỡ iris trên cầu), `step` (mật độ chấm).

---

## 5. Bản đồ code (CSS ở `css/style.css`, HTML ở `index.html`, JS ở `js/app.js`)
| Vùng | Tìm theo comment |
|---|---|
| Biến màu / font (`:root`) | đầu thẻ `<style>` |
| CSS màn Intro / Demo / Windows giả lập | `INTRO`, `DEMO`, `simulated Windows world` |
| Lớp style kính mờ (đè lên style gốc) | `monochrome frosted glass` |
| HTML màn 1 / màn 2 | `SCREEN 1: INTRODUCTION`, `SCREEN 2: LIVE DEMO` |
| Chuyển tab, theme | `theme + navigation` |
| Xin quyền camera | `camera access` |
| Con mắt Peep | `Peep: 3D dot eyeball` |
| Thế giới Windows (Chrome/Google/YouTube/Mail, render, action) | `DEMO WORLD` |
| Con trỏ ánh mắt, nhìn giữ, cử chỉ | `gaze engine` |
| Nhận lệnh giọng nói (fuzzy match lời nói ngọng) | `voice` |
| Hiệu chỉnh 9 điểm | `calibration` |
| 3 demo tự chạy | `guided demos` |
| Camera mô phỏng + biểu đồ tín hiệu | `camera canvas` |
| Vòng lặp chính | `main loop` |

Ghi chú: khối `(unused) wavy halftone dot field` là nền chấm cũ, **không chạy**, giữ lại để tham khảo, có thể xóa.

---

## 6. Hạn chế và việc tiếp theo gợi ý
1. **Nối tracking thật:** dùng **MediaPipe Tasks – FaceLandmarker** (bản JS `@mediapipe/tasks-vision`) để lấy 478 landmark và blendshapes. Sau đó:
   - **Ánh mắt:** offset mống mắt (landmark 468/473) so với khóe mắt (33/133, 362/263) → hồi quy theo dữ liệu hiệu chỉnh 9 điểm → thay `G.target` trong *gaze engine*.
   - **Blink:** `eyeBlinkLeft/Right` ≥ 0.5 trong 0.3–1.2 s.
   - **Nod / shake / tilt:** pitch, yaw, roll từ ma trận biến đổi khuôn mặt (`outputFacialTransformationMatrixes`), dùng đúng các ngưỡng trong bảng Gesture map.
   - Gọi `gesture('blink' | 'nod' | 'shake' | 'tiltL' | 'tiltR')` là demo tự phản ứng.
   - Lưu ý: từ MediaPipe 0.10.30, API `mp.solutions` cũ đã bị bỏ, phải dùng **Tasks API**.
2. **Giọng nói ngọng:** hiện chỉ là fuzzy match từ vựng (Levenshtein) + regex intent. Bản thật cần model ASR có fine-tune hoặc voice profile riêng cho từng người.
3. **Điều khiển chuột thật của Windows:** trình duyệt không làm được. Đã có bản Python thử nghiệm (`eyeable_tracker.py`: MediaPipe + pyautogui, nằm trong `Desktop\eyeable` trên máy chủ project). Bản đó **chưa test với webcam thật**.
4. **Thương hiệu:** demo không dùng logo thật của Chrome, YouTube hay Windows, chỉ dùng icon chung chung.

---

## 7. Lịch sử thay đổi chính (tóm tắt)
- Bản đầu tiếng Việt → làm lại **tiếng Anh**, thêm các chức năng: blink, nod, shake, tilt, voice cho người nói ngọng, demo Windows (Chrome, Google, YouTube, Mail).
- Thêm mô phỏng cử chỉ + biểu đồ tín hiệu, thêm nghiêng đầu trái/phải để click trái/phải, quay lại kiểu **nhìn giữ để click**.
- Mascot: từ SVG, qua mắt đốm, khuôn mặt lưới, tròng mắt vẽ, rồi **nhãn cầu 3D đốm tụ/tản theo vòng lặp** (bản hiện tại).
- Thử nhiều phong cách (Warm kem, kính tối, kẹo pastel, kính trắng), rồi chốt **trắng–đen, nền vân sáng chuyển động**.
- Tên: thử logo vẽ tay, Fontdiner, Black Ops One, rồi chốt **EYEable – Rubik 800**, chữ thẳng hàng.

Link xem online (artifact, chỉ chủ project mở được trừ khi chia sẻ): https://claude.ai/artifact/9MSaXb4iyvFQ33C3w9fErV
