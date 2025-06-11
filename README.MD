# Wilson_Disease_Prediction
ML project to detect Wilson's disease using clinical and genetic dataset.



## 📊 Dataset Overview

The dataset contains 60,000 records with features like:
- Ceruloplasmin Level
- Copper in Serum and Urine
- Liver Enzymes (ALT, AST, GGT)
- Kayser-Fleischer Rings
- Genetic Mutations
- Cognitive Scores, Region, BMI, etc.

Target: `Is_Wilson_Disease` (1 = Yes, 0 = No)

## 🤖 Models Used

- Random Forest
- Support Vector Machine (SVM)
- XGBoost

All models are evaluated using accuracy and classification reports.

## 🧪 Project Structure
Wilson_Disease_Prediction/
│
├── main.py # Main script to run models
├── requirements.txt # Required Python libraries
├── README.md # Project description
└── data/
└── Wilson_disease_dataset.csv


## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt


2. Run the project:
    python main.py

---

### 🧑‍💻 Step 4: Upload Using GitHub Desktop

Now let’s upload this clean project to GitHub:

#### 1. **Open GitHub Desktop**
- If you haven’t already, [Download it](https://desktop.github.com)

#### 2. **Create a New Repository**
- Go to **File → New Repository**
- Fill in:
  - **Name**: `Wilson_Disease_Prediction`
  - **Description**: `ML project to detect Wilson’s Disease using clinical and genetic data.`
  - **Local Path**: Choose the folder you just made
  - Keep **.gitignore: None** and **License: None**
  - Click **Create Repository**

#### 3. **Commit Your Files**
- GitHub Desktop will auto-detect the files.
- At the bottom:
  - Write a commit message:  
    ```
    Initial commit - added full Wilson Disease ML project
    ```
  - Click **Commit to main**

#### 4. **Publish Repository**
- Click the blue **Publish repository** button on top.
- Set visibility: **Public** (so recruiters can see)
- Click **Publish Repository**

---

### ✅ Done!    