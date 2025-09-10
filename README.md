# Smart-Health-Prediction-System
Many people search symptoms on Google but get unreliable results. A machine learning model can predict possible diseases based on user symptoms, helping them take the right step early.

---

## 🧰 Tech Stack
- **Python** (NumPy, Pandas, Scikit-learn)  
- **Streamlit** (Web app deployment)  
- **Matplotlib / Seaborn** (for visualization during EDA)  

---

## 📊 Dataset
- Default: A synthetic dataset is auto-generated with common diseases & symptoms.  
- Custom: You can provide your own dataset `symptoms_disease.csv` with columns:  
  - `disease`: Name of the disease (e.g., Flu, Migraine).  
  - `symptoms`: Comma-separated list of symptoms.  

Example:
```csv
disease,symptoms
Flu,"fever,cough,sore throat"
Migraine,"headache,nausea,light sensitivity"
