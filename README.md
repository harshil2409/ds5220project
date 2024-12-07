# ds5220project

### README.md

---

# **Real-Time Image Captioning Project**

## **Overview**

This project implements a **Real-Time Image Captioning System** using the **Flickr 8k dataset**. It employs deep learning techniques to generate descriptive captions for images. The system preprocesses image data, extracts features using a pre-trained ResNet50 model, and leverages those features to generate captions.

---

## **Features**
- **Automated Caption Generation**: Generates multiple captions per image using deep learning models.
- **Image Preprocessing**: Handles renaming, resizing, and normalizing of images for feature extraction.
- **Deep Learning Integration**: Utilizes ResNet50 for feature extraction and builds a pipeline for scalable neural network training.

---

## **Steps to Run the Project**

### **1. Setup**
- Ensure you have Python 3.x installed.
- Install the necessary dependencies using:
  ```bash
  pip install -r requirements.txt
  ```
  
### **2. Download Dataset**
- The dataset is automatically downloaded from Kaggle:
  ```python
  import kagglehub
  path = kagglehub.dataset_download("adityajn105/flickr8k")
  ```
- Define paths to images and captions file:
  ```python
  images_path = os.path.join(path, "Images")
  captions_file = os.path.join(path, "captions.txt")
  ```

### **3. Preprocess Data**
- Load captions:
  ```python
  df = load_captions(captions_file)
  ```
- Rename and organize image files for easier processing:
  ```python
  df = rename_images(df, images_path)
  ```
- Save processed data:
  ```python
  df.to_csv("processed_captions.csv", index=False)
  ```

### **4. Feature Extraction**
- Use ResNet50 to extract image features:
  ```python
  model = ResNet50(weights="imagenet", include_top=True)
  features = model.predict(preprocessed_images)
  ```

### **5. Real-Time Caption Generation**
- Generate captions for an image:
  ```python
  captions = generate_captions(image_path)
  print(captions)
  ```

---

## **Dependencies**
- Python 3.x
- TensorFlow
- Pandas
- Matplotlib
- KaggleHub

---

## **Future Enhancements**
- Integrate a Transformer-based model for improved captioning.
- Add a web interface for uploading images and viewing captions in real-time.
- Optimize feature extraction for faster processing.

---

## **Acknowledgments**
- **Dataset**: Flickr 8k Dataset ([Kaggle](https://www.kaggle.com/adityajn105/flickr8k))
- **Model**: ResNet50 (Pre-trained on ImageNet)

---
