import sys
import os
import cv2
import shutil
import json
import traceback
import threading
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from PyQt5.QtWidgets import (QApplication, QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFileDialog, QProgressBar, QLabel, QHeaderView, QAbstractItemView, QCheckBox, QSpinBox,
                             QDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PIL import Image
from nudenet import NudeClassifier

# --- Глобальные параметры ---
Image.MAX_IMAGE_PIXELS = None  # Отключает DecompressionBombWarning
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif", ".webp"}

# --- Класс для NSFW анализа ---
class NSFWPool:
    def __init__(self, num_threads=4):
        self.classifier = NudeClassifier()
        self.lock = threading.Lock()

    def analyze(self, fname):
        # NudeNet не потокобезопасен! Лочим!
        try:
            with self.lock:
                res = self.classifier.classify(fname)
            return {"result": res}
        except Exception as e:
            return {"error": str(e)}

# --- Галерея для свайпа/массового просмотра ---
class GalleryDialog(QDialog):
    def __init__(self, files, checked_indices, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Галерея фото")
        self.files = files
        self.checked_indices = checked_indices
        self.current_page = 0
        self.photos_per_page = 25
        self.grid_size = 5
        self.resize(1200, 800)
        self.init_ui()

    def init_ui(self):
        from PyQt5.QtWidgets import QGridLayout, QLabel, QCheckBox, QScrollArea, QPushButton
        layout = QVBoxLayout(self)
        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setSpacing(10)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.grid_widget)
        layout.addWidget(self.scroll)
        btn_layout = QHBoxLayout()
        self.btn_prev = QPushButton("<<")
        self.btn_next = QPushButton(">>")
        self.btn_prev.clicked.connect(self.prev_page)
        self.btn_next.clicked.connect(self.next_page)
        btn_layout.addWidget(self.btn_prev)
        btn_layout.addWidget(self.btn_next)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.update_grid()

    def update_grid(self):
        for i in reversed(range(self.grid.count())):
            widget = self.grid.itemAt(i).widget()
            if widget: widget.setParent(None)
        page_files = self.files[self.current_page*self.photos_per_page:(self.current_page+1)*self.photos_per_page]
        for idx, fname in enumerate(page_files):
            row, col = divmod(idx, self.grid_size)
            pix = QPixmap(fname)
            lbl = QLabel()
            lbl.setPixmap(pix.scaled(120,120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            chk = QCheckBox()
            global_idx = self.current_page*self.photos_per_page+idx
            chk.setChecked(global_idx in self.checked_indices)
            def update_check(state, idx=global_idx):
                if state:
                    self.checked_indices.add(idx)
                else:
                    self.checked_indices.discard(idx)
            chk.stateChanged.connect(update_check)
            self.grid.addWidget(lbl, row*2, col)
            self.grid.addWidget(chk, row*2+1, col)
        self.grid_widget.setMinimumSize(800, 600)

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_grid()

    def next_page(self):
        if (self.current_page+1)*self.photos_per_page < len(self.files):
            self.current_page += 1
            self.update_grid()

# --- Основной GUI класс ---
class NSFWScanner(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NSFW-Checker by Funti")
        self.pool = None
        self.files = []
        self.results = []
        self.threads = 4
        self.checked_indices = set()
        self.include_subfolders = False
        self.init_ui()

    def init_ui(self):
        v = QVBoxLayout(self)
        h = QHBoxLayout()
        self.btn_choose = QPushButton("Выбрать папку")
        self.btn_choose.clicked.connect(self.choose_folder)
        self.chk_subfolders = QCheckBox("Включать подпапки")
        self.chk_subfolders.stateChanged.connect(lambda s: self.set_include_subfolders(s == Qt.Checked))
        h.addWidget(self.btn_choose)
        h.addWidget(self.chk_subfolders)
        h.addWidget(QLabel("Ядер:"))
        self.spin_cores = QSpinBox()
        self.spin_cores.setRange(1, os.cpu_count() or 4)
        self.spin_cores.setValue(self.threads)
        self.spin_cores.valueChanged.connect(lambda v: setattr(self, "threads", v))
        h.addWidget(self.spin_cores)
        self.btn_scan = QPushButton("Анализировать NSFW")
        self.btn_scan.clicked.connect(self.start_scan)
        h.addWidget(self.btn_scan)
        v.addLayout(h)
        # Таблица
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["", "Миниатюра", "Файл", "Размер, KB", "Тип", "Папка", "NSFW %", "Удалить"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.cellDoubleClicked.connect(self.on_cell_double_click)
        v.addWidget(self.table)
        # Прогресс
        h2 = QHBoxLayout()
        self.progress = QProgressBar()
        self.status = QLabel("")
        self.btn_gallery = QPushButton("Галерея (F2)")
        self.btn_gallery.clicked.connect(self.show_gallery)
        self.btn_delete = QPushButton("Удалить выбранные")
        self.btn_delete.clicked.connect(self.delete_selected)
        h2.addWidget(self.progress)
        h2.addWidget(self.status)
        h2.addWidget(self.btn_gallery)
        h2.addWidget(self.btn_delete)
        v.addLayout(h2)
        self.setLayout(v)
        self.resize(1300, 700)

    def set_include_subfolders(self, val):
        self.include_subfolders = val

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбери папку для сканирования")
        if not folder: return
        self.folder = folder
        self.load_files()

    def load_files(self):
        self.files = []
        exts = tuple(IMAGE_EXTS | VIDEO_EXTS)
        if self.include_subfolders:
            for root, dirs, files in os.walk(self.folder):
                for f in files:
                    if f.lower().endswith(exts):
                        self.files.append(os.path.join(root, f))
        else:
            for f in os.listdir(self.folder):
                if f.lower().endswith(exts):
                    self.files.append(os.path.join(self.folder, f))
        self.populate_table()

    def populate_table(self):
        self.table.setRowCount(len(self.files))
        self.checked_indices = set()
        for i, fname in enumerate(self.files):
            # Чекбокс выбора
            cb = QCheckBox()
            cb.setFixedSize(24,24)
            cb.stateChanged.connect(partial(self.on_checkbox, i))
            self.table.setCellWidget(i, 0, cb)
            # Миниатюра
            pix = QPixmap(fname)
            if pix.isNull() and os.path.splitext(fname)[1].lower() in VIDEO_EXTS:
                cap = cv2.VideoCapture(fname)
                ret, frame = cap.read()
                if ret:
                    h, w, ch = frame.shape
                    qimg = QImage(frame.data, w, h, ch*w, QImage.Format_BGR888)
                    pix = QPixmap.fromImage(qimg)
                cap.release()
            icon = QTableWidgetItem()
            icon.setData(Qt.DecorationRole, pix.scaled(64,64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.table.setItem(i, 1, icon)
            # Имя файла
            self.table.setItem(i, 2, QTableWidgetItem(os.path.basename(fname)))
            # Размер
            size_kb = os.path.getsize(fname)//1024
            self.table.setItem(i, 3, QTableWidgetItem(str(size_kb)))
            # Тип
            ext = os.path.splitext(fname)[1].lower()
            self.table.setItem(i, 4, QTableWidgetItem(ext[1:].upper()))
            # Папка
            self.table.setItem(i, 5, QTableWidgetItem(os.path.dirname(fname)))
            # NSFW %
            self.table.setItem(i, 6, QTableWidgetItem("?"))
            # Удалить
            btn_del = QPushButton("🗑")
            btn_del.clicked.connect(partial(self.delete_row, i))
            self.table.setCellWidget(i, 7, btn_del)

    def on_checkbox(self, idx, state):
        if state: self.checked_indices.add(idx)
        else: self.checked_indices.discard(idx)

    def start_scan(self):
        if not self.files: return
        self.progress.setMaximum(len(self.files))
        self.progress.setValue(0)
        self.status.setText("Анализ...")
        QApplication.processEvents()
        self.pool = NSFWPool(num_threads=self.threads)
        self.results = [None]*len(self.files)
        self.errors = []
        # --- Многопоточный пул
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = []
            for idx, fname in enumerate(self.files):
                futures.append(executor.submit(self.worker_task, idx, fname))
            for fut in as_completed(futures):
                idx, res = fut.result()
                self.results[idx] = res
                nsfw_str = str(res) if isinstance(res, int) else "ERR"
                self.table.setItem(idx, 6, QTableWidgetItem(nsfw_str))
                self.progress.setValue(self.progress.value()+1)
        self.status.setText("Готово.")

    def worker_task(self, idx, fname):
        ext = os.path.splitext(fname)[1].lower()
        try:
            # Видео — извлекаем первый кадр
            if ext in VIDEO_EXTS:
                cap = cv2.VideoCapture(fname)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    tmpimg = fname + ".firstframe.jpg"
                    cv2.imwrite(tmpimg, frame)
                    res = self.pool.analyze(tmpimg)
                    try: os.remove(tmpimg)
                    except: pass
                else:
                    res = {"error": "Video read fail"}
            else:
                res = self.pool.analyze(fname)
            unsafe = 0
            if isinstance(res, dict) and "result" in res and res["result"]:
                img_key = list(res["result"].keys())[0]
                unsafe = int(res["result"][img_key].get("unsafe", 0)*100)
            else:
                unsafe = 0
            return idx, unsafe
        except Exception as e:
            with open("nsfw_error.log", "a", encoding="utf-8") as log:
                log.write(f"{fname}\t{str(e)}\n")
            return idx, "ERR"

    def delete_row(self, idx):
        fname = self.files[idx]
        try:
            # В корзину
            from send2trash import send2trash
            send2trash(fname)
            self.table.removeRow(idx)
            self.files.pop(idx)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Ошибка удаления: {e}")

    def delete_selected(self):
        for idx in sorted(self.checked_indices, reverse=True):
            self.delete_row(idx)
        self.checked_indices.clear()

    def on_cell_double_click(self, row, col):
        # Сортировка по столбцу (кроме миниатюры и кнопок)
        if col not in (0,1,7):
            self.table.sortItems(col, Qt.AscendingOrder if self.table.horizontalHeader().sortIndicatorOrder() == Qt.AscendingOrder else Qt.DescendingOrder)

    def show_gallery(self):
        dlg = GalleryDialog(self.files, self.checked_indices, self)
        dlg.exec_()
        # После закрытия обновить чекбоксы
        for idx in range(self.table.rowCount()):
            cb = self.table.cellWidget(idx,0)
            cb.setChecked(idx in dlg.checked_indices)
        self.checked_indices = dlg.checked_indices

# --- Запуск ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    wnd = NSFWScanner()
    wnd.show()
    sys.exit(app.exec_())
