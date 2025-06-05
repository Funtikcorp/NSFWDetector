def worker_task(self, idx, fname):
    ext = os.path.splitext(fname)[1].lower()
    # Для видео: извлекаем кадр
    if ext in VIDEO_EXTS:
        try:
            cap = cv2.VideoCapture(fname)
            ret, frame = cap.read()
            cap.release()
            if ret:
                tmpimg = fname + ".firstframe.jpg"
                cv2.imwrite(tmpimg, frame)
                res = self.pool.analyze(tmpimg)
                try: os.remove(tmpimg)
                except: pass
                unsafe = 0
                if isinstance(res, dict) and "result" in res and res["result"]:
                    # res["result"] - dict вида {'tmpimg': {'unsafe': 0.999, ...}}
                    img_key = list(res["result"].keys())[0]
                    unsafe = int(res["result"][img_key].get("unsafe", 0) * 100)
                return unsafe
        except Exception:
            return 0
    # Для фото:
    res = self.pool.analyze(fname)
    unsafe = 0
    if isinstance(res, dict) and "result" in res and res["result"]:
        img_key = list(res["result"].keys())[0]
        unsafe = int(res["result"][img_key].get("unsafe", 0) * 100)
    else:
        # Если ошибка — пропускаем (можно ещё писать в лог ошибки)
        pass
    return unsafe
