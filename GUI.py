import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from tkinter.ttk import Notebook, Frame, Button, Label
import sys
import os
import traceback

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and PyInstaller """
    try:
        base_path = sys._MEIPASS  # used by PyInstaller
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class ObjectDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Object Detection")
        master.geometry("600x400")
        master.resizable(True, True)

        try:
            cfg_path = resource_path("yolov3.cfg")
            weights_path = resource_path("yolov3.weights")
            names_path = resource_path("coco.names")

            self.net = cv2.dnn.readNet(weights_path, cfg_path)

            with open(names_path, "r") as f:
                self.classes = f.read().strip().split("\n")
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Could not load model files:\n{e}")
            with open("error.log", "a") as log:
                log.write(traceback.format_exc())
            master.destroy()
            return

        self.unsaved_changes = False

        self.welcome_label = tk.Label(
            master,
            text="Object Detection",
            font=("Helvetica", 36, "bold"),
            fg="#3498db",
        )
        self.welcome_label.pack(expand=True, anchor="center")

        master.after(1500, self.open_main_window)
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_menu(self):
        menubar = tk.Menu(self.master)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Exit", command=self.master.destroy)
        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="About Us", command=self.show_about_us)
        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="About", menu=about_menu)
        self.master.config(menu=menubar)

    def setup_toolbar(self):
        toolbar = tk.Frame(self.master)
        open_image_button = Button(toolbar, text="Open Image", command=self.open_image)
        open_image_button.pack(side=tk.LEFT, padx=5)
        realtime_button = Button(
            toolbar, text="Start Real-Time Detection", command=self.real_time_detection
        )
        realtime_button.pack(side=tk.LEFT, padx=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)

    def open_main_window(self):
        self.welcome_label.destroy()
        self.setup_menu()
        self.setup_toolbar()
        notebook = Notebook(self.master)
        notebook.pack(fill="both", expand=True)
        self.image_tab = Frame(notebook)
        notebook.add(self.image_tab, text="Image Detection")
        self.result_label = Label(self.image_tab, text="")
        self.result_label.pack(pady=10)
        realtime_tab = Frame(notebook)
        notebook.add(realtime_tab, text="Real-Time Detection")

    def open_image(self):
        try:
            file_path = filedialog.askopenfilename()
            if file_path:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Failed to read image file.")
                detected_image, detection_result = self.detect_objects(image)
                cv2.imshow("Detected Objects", detected_image)
                self.result_label.config(text=detection_result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not open/process the image:\n{e}")
            with open("error.log", "a") as log:
                log.write(traceback.format_exc())

    def detect_objects(self, frame):
        detection_result = ""
        try:
            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(
                frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
            )
            self.net.setInput(blob)
            outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
            detection_result = "Detected Objects:\n"
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        label = str(self.classes[class_id])
                        detection_result += f"{label}: {confidence:.2f}\n"
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(
                            frame,
                            f"{label} {confidence:.2f}",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )
        except Exception as e:
            print(f"Error during object detection: {e}")
            with open("error.log", "a") as log:
                log.write(traceback.format_exc())
        self.set_unsaved_changes()
        return frame, detection_result

    def real_time_detection(self):
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open webcam.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_with_labels, _ = self.detect_objects(frame)
                cv2.imshow("Real-Time Detection", frame_with_labels)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        except Exception as e:
            messagebox.showerror("Camera Error", f"Real-time detection failed:\n{e}")
            with open("error.log", "a") as log:
                log.write(traceback.format_exc())
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()

    def set_unsaved_changes(self):
        self.unsaved_changes = True

    def on_closing(self):
        if self.unsaved_changes:
            if messagebox.askyesno(
                "Unsaved Changes",
                "You have unsaved changes. Do you want to exit without saving?",
            ):
                self.master.destroy()
        else:
            self.master.destroy()

    def show_about_us(self):
        about_message = (
            "Object Detection App\n"
            "Version 1.0\n\n"
            "Developed by:\n"
            "Abdul Ahad and group members\n\n"
            "Description:\n"
            "This application performs object detection using YOLO (You Only Look Once).\n"
            "It supports static images and real-time webcam detection.\n"
            "Visit: www.obdetect.com for more info.\n"
        )
        messagebox.showinfo("About Us", about_message)

if __name__ == "__main__":
    try:
        window = tk.Tk()
        app = ObjectDetectionApp(window)
        window.mainloop()
    except Exception as e:
        with open("error.log", "a") as log:
            log.write("App crash:\n" + traceback.format_exc())
        messagebox.showerror("Fatal Error", f"The application encountered a fatal error:\n{e}")
