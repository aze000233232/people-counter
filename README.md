# 👥 people-counter

A Python project for **real-time people detection and counting** using [YOLOv5](https://github.com/ultralytics/yolov5) and [SORT](https://github.com/abewley/sort).

---

## 🚀 Features

- Real-time people detection from webcam or video file
- Unique person counting with tracking (SORT)
- Easy to use and extend

---

## 🛠️ Installation

1. **Clone this repository:**
   ```sh
   git clone https://github.com/Lian-Cunanan/people-counter.git
   cd people-counter
   ```

2. **Install all requirements:**
   ```sh
   pip install -r requirements.txt
   ```

   *(If you use Jupyter Notebooks, also run: `pip install notebook`)*

---

## ▶️ How to Run

- **For video file detection:**
  ```sh
  python humanupload.py
  ```

- **For webcam detection:**
  ```sh
  python humancount.py
  ```

- **For Jupyter Notebook:**
  1. Open `humancounter.ipynb` in Jupyter.
  2. Run the cells in order.

---

## 📁 Project Structure

```
people-counter/
│
├── humanupload.py
├── humancount.py
├── humancounter.ipynb
├── requirements.txt
├── video/
│   └── your_video_files.mp4
├── sort/
│   └── ... (SORT tracker files)
└── README.md
```

---

## 📝 Notes

- Make sure the `sort` folder is present in your project directory.
- For custom YOLOv5 weights, place your `.pt` file in the project and update the script accordingly.
- Press `q` to quit the detection window, or `s` to save the count and quit.

---

## 📷 Example Output

*(Add a screenshot or GIF here if you have one!)*

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT](LICENSE)
