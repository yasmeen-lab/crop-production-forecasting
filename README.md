project\_3/  
├── app.py                      \# Interactive Streamlit Dashboard code  
├── requirements.txt            \# Python dependencies log  
├── crop\_forecasting\_deck.html  \# Executive Presentation HTML slide deck  
└── README.md                   \# Project documentation (This file)

## **Installation & Setup**

### **1\. Environment Configuration**

It is highly recommended to run this application within a dedicated virtual environment (`venv` or `conda`) to avoid dependency collisions.

\# Navigate to your project directory  
cd /home/dell/Documents/project\_3/

\# Create a virtual environment (optional but recommended)  
python3 \-m venv venv  
source venv/bin/activate

### **2\. Install Dependencies**

Install all required libraries, including data manipulation, visualization, machine learning, and dashboard rendering engines, in a single step:

pip install \-r requirements.txt

## **Running the Streamlit Dashboard**

To launch the interactive dashboard locally, execute the following command in your terminal:

streamlit run app.py

