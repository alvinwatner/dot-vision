import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog,
                             QSizePolicy, QShortcut, QLineEdit, QMessageBox)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage,QBrush

from PyQt5.QtCore import Qt, QPoint
import pickle
import os

class ImageViewer(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.image = QImage()
        self.coordinates = []
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def resize_image(self, width, height):
        if not self.image.isNull():
            # Scale the image to the new size and display it
            self.setPixmap(QPixmap.fromImage(self.image.scaled(width, height, Qt.KeepAspectRatio)))


    def get_current_image(self):
        # Grab the current QPixmap from the label
        return self.pixmap()        

    def undo_last_coordinate(self):
        if self.coordinates:
            self.coordinates.pop()
            self.update()        

    def set_image(self, filepath):
        self.image.load(filepath)
        pixmap = QPixmap.fromImage(self.image)
        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio))

    def mouseMoveEvent(self, event):
        self.parent().display_hover_coords(event.pos(), self)

    def mousePressEvent(self, event):
        self.coordinates.append((event.pos().x(), event.pos().y()))
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # For smooth edges
        pen = QPen(Qt.red, 2)
        painter.setPen(pen)
        brush = QBrush(Qt.red)  # Set brush to fill the circle
        painter.setBrush(brush)  # Apply the brush to the painter
        for i, (x, y) in enumerate(self.coordinates, start=1):
            painter.drawEllipse(QPoint(x, y), 5, 5)  # This will draw a filled circle
            painter.drawText(QPoint(x + 5, y + 5), str(i))

class ImageApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image1_viewer = ImageViewer(self)
        self.image2_viewer = ImageViewer(self)
        self.hover_coords_label = QLabel("Hover Coordinates: ")  # Shared label for hover coordinates
         
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Coordinate Recorder')

        # Calculate maximum width and height based on screen size
        screen = QApplication.primaryScreen().availableGeometry()
        self.max_width = screen.width() // 2  # 50% of screen width
        self.max_height = screen.height() - 120  # Assuming 120px for other UI elements

        # Show the max width and height on the UI
        self.max_size_label = QLabel(f"Maximum Width: {self.max_width}px, Maximum Height: {self.max_height}px")

        # Create input fields and buttons for Image 1
        self.width_input1 = QLineEdit(self)
        self.width_input1.setPlaceholderText('Enter width')
        self.height_input1 = QLineEdit(self)
        self.height_input1.setPlaceholderText('Enter height')
        upload_button1 = QPushButton('Upload 3D Image')
        upload_button1.clicked.connect(lambda: self.load_image(self.image1_viewer, self.width_input1.text(), self.height_input1.text()))
        save_button1 = QPushButton('Save Coordinates 3D')
        save_button1.clicked.connect(lambda: self.save_coordinates(self.image1_viewer, '3D'))

        # Create input fields and buttons for Image 2
        self.width_input2 = QLineEdit(self)
        self.width_input2.setPlaceholderText('Enter width')
        self.height_input2 = QLineEdit(self)
        self.height_input2.setPlaceholderText('Enter height')

        # Connect textChanged signals to a slot for Image 1
        self.width_input1.textChanged.connect(lambda: self.check_and_resize_image(self.image1_viewer, self.width_input1.text(), self.height_input1.text()))
        self.height_input1.textChanged.connect(lambda: self.check_and_resize_image(self.image1_viewer, self.width_input1.text(), self.height_input1.text()))

        # Connect textChanged signals to a slot for Image 2
        self.width_input2.textChanged.connect(lambda: self.check_and_resize_image(self.image2_viewer, self.width_input2.text(), self.height_input2.text()))
        self.height_input2.textChanged.connect(lambda: self.check_and_resize_image(self.image2_viewer, self.width_input2.text(), self.height_input2.text()))


        upload_button2 = QPushButton('Upload 2D Image')
        upload_button2.clicked.connect(lambda: self.load_image(self.image2_viewer, self.width_input2.text(), self.height_input2.text()))
        save_button2 = QPushButton('Save Coordinates 2D')
        save_button2.clicked.connect(lambda: self.save_coordinates(self.image2_viewer, '2D'))

        # Set up layout for each column
        column1_layout = QVBoxLayout()
        column1_layout.addWidget(upload_button1)
        column1_layout.addWidget(self.width_input1)
        column1_layout.addWidget(self.height_input1)
        column1_layout.addWidget(self.image1_viewer)
        column1_layout.addWidget(save_button1)

        column2_layout = QVBoxLayout()
        column2_layout.addWidget(upload_button2)
        column2_layout.addWidget(self.width_input2)
        column2_layout.addWidget(self.height_input2)
        column2_layout.addWidget(self.image2_viewer)
        column2_layout.addWidget(save_button2)

        # Set up the main layout with two columns
        columns_layout = QHBoxLayout()
        columns_layout.addLayout(column1_layout)
        columns_layout.addLayout(column2_layout)

        # Combine the columns layout with the hover coordinates label
        main_layout = QVBoxLayout()
        main_layout.addLayout(columns_layout)
        main_layout.addWidget(self.hover_coords_label)
        main_layout.addWidget(self.max_size_label)        

        self.setLayout(main_layout)
        self.showMaximized()

    def check_and_resize_image(self, viewer, width_text, height_text):
        width, height = 0, 0
        try:
            width = int(width_text) if width_text.isdigit() else None
            height = int(height_text) if height_text.isdigit() else None
        except ValueError:
            pass  # Handle the case where text is not convertible to int

        if width is not None and height is not None:
            if width > self.max_width or height > self.max_height:
                QMessageBox.warning(self, 'Size Warning', 'The dimensions exceed the maximum allowed size. Resizing to maximum allowable dimensions.')
                width = min(width, self.max_width)
                height = min(height, self.max_height)
                self.width_input1.setText(str(width))
                self.height_input1.setText(str(height))
            viewer.resize_image(width, height)        

    def resize_image(self, viewer, width, height):
        if viewer.image.isNull():
            return  # No image loaded
        if width.isdigit() and height.isdigit():
            viewer.resize_image(int(width), int(height))
        else:
            # If one or both inputs are not valid (e.g., empty), show the image at its original size
            viewer.setPixmap(QPixmap.fromImage(viewer.image))        

    
    def load_image(self, viewer, width, height):
        filepath, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'Image files (*.jpg *.png)')
        if filepath:
            viewer.image.load(filepath)
            if width.isdigit() and height.isdigit():
                viewer.resize_image(int(width), int(height))
            else:
                viewer.set_image(filepath)  # Use original size if no input is given


    def display_hover_coords(self, point, viewer):
        # This method should be updated to indicate which image the hover coordinates are for
        image_label = 'Image 1' if viewer == self.image1_viewer else 'Image 2'
        self.hover_coords_label.setText(f"Hover Coordinates: {point.x()}, {point.y()} on {image_label}")

    def save_coordinates(self, viewer, type):
        # Prompt user to select the directory to save the data
        directory = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if directory:
            # Construct file paths
            coordinates_path = os.path.join(directory, f'coordinates_{type}.pkl')
            image_path = os.path.join(directory, f'image_{type}.png')

            # Save coordinates
            with open(coordinates_path, 'wb') as f:
                pickle.dump(viewer.coordinates, f)

            # Save the rescaled image
            if viewer.get_current_image():
                viewer.get_current_image().save(image_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageApp()
    ex.show()
    sys.exit(app.exec_())
