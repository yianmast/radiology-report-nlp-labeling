# 🧾 Disease Label Extraction from Radiology Reports using NLP

### 📍 Course Project – *AI for Medical Treatment* by DeepLearning.AI on Coursera

This project focuses on extracting structured medical labels from unstructured radiology report text using a rules-based NLP pipeline.

---

## 🧠 Key Concepts Practiced

- ✅ Rule-based keyword matching for pathology detection
- ✅ Handling negations using simple heuristics
- ✅ Evaluating performance using **F1 score**
- ✅ Improving performance with **text cleanup and dependency parsing**
- ✅ Utilizing **NegBio** for negation detection via dependency graphs

---

## 📊 What It Does

- Parses 1,000+ radiologist-written chest X-ray reports
- Detects presence/absence of 14 conditions (e.g., edema, pneumothorax, pneumonia)
- Calculates label F1 scores to evaluate extractor accuracy
- Incorporates **NegBio** and **Stanford CoreNLP** for advanced dependency-based negation handling

---

## 🛠️ Tools and Libraries Used

- Python 3
- `pandas`, `nltk`, `matplotlib`, `tensorflow`, `transformers`
- **NegBio** & **Stanford CoreNLP**
- Custom helper functions via `util.py` (provided by Coursera)

---


## ⚠️ Disclaimer

> 📌 This notebook was completed as part of the Coursera course:  
> [AI for Medical Treatment – by DeepLearning.AI](https://www.coursera.org/learn/ai-for-medical-treatment)  
> It is shared here for **educational and portfolio purposes only**.  
> **Please do not copy or submit it as your own work**.
