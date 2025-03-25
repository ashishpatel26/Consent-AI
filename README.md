# Consent AI / Consent Management System - AI Case Assistant
---
[![Video Title](ashishpatel26/Consent-AI/demo.jpg)](ashishpatel26/Consent-AI/consentai-1742893099136.mp4)


This Streamlit application is a Consent Management System with an AI Case Assistant. It helps manage and assess cases related to consent violations, using an AI model to predict risk levels.

## Features

* **AI-Powered Assessment**: The core feature is the AI model that assesses new cases and predicts the risk level (High, Medium, Low).
* **Case Management**: The app allows you to view, add, update, and delete cases, providing a basic case management system.
* **Filtering**: Cases can be filtered by category, consent status, and date range for easier management.
* **Retrainable Model**: The AI model can be retrained with new data, improving its accuracy over time.
* **User-Friendly Interface**: Streamlit provides a simple and intuitive user interface.
* **Detailed Case View**: The details of each case are displayed in an expandable section for better organization.
* **Error Handling**: The app provides feedback for actions like updating or deleting non-existent cases.
* **Date Tracking**: Each case includes a date, and the application supports filtering by date ranges.
* **Probability Display**: Displays the probabilities for each risk level, providing more nuanced information about the AI's assessment.
* **Comprehensive Categories and Statuses**: Includes a wide range of categories and consent statuses to cover more scenarios.
* **Input Validation**: Basic input validation.
* **Model Retraining**: The model is retrained after adding, updating, or deleting cases to keep it up-to-date.

## How to Use

1.  **View Cases**:

    * Go to the "View Cases" section in the sidebar.
    * Filter cases by category, consent status, and date range using the dropdowns and date pickers.
    * Click on a case to expand and view its details.
2.  **Add Case**:

    * Go to the "Add Case" section in the sidebar.
    * Enter the case description, select the category and consent status.
    * Click "Add Case" to add the new case. The AI model will be retrained.
    
3.  **Assess Case**:

    * Go to the "Assess Case" section in the sidebar.
    * Enter the case description.
    * Click "Assess" to get the AI's risk assessment.
    * The application will display the predicted risk level (High, Medium, Low) and the probabilities for each level.
    
4.  **Update Case**:

    * Go to the "Update Case" section in the sidebar.
    * Enter the ID of the case to update.
    * Enter the new details for the case. Leave a field blank to keep the current value.
    * Click "Update" to update the case. The AI model will be retrained.
    
5.  **Delete Case**:

    * Go to the "Delete Case" section in the sidebar.
    * Enter the ID of the case to delete.
    * Click "Delete" to delete the case. The AI model will be retrained.
    
6.  **Train Model**:

    * Go to the "Train Model" section in the sidebar.
    * Click "Retrain Model" to retrain the AI model with the current data.

## Code Overview

The application is built using Python and the following libraries:

* [Streamlit](https://streamlit.io/): For the user interface.
* [pandas](https://pandas.pydata.org/): For data manipulation.
* [scikit-learn](https://scikit-learn.org/): For the AI model (SVM).

Here's a breakdown of the main functions:

* `train_model(data)`: Trains the SVM model on the case descriptions and AI assessments.
* `assess_case(model, description)`: Predicts the AI assessment for a new case description using the trained model.
* `add_case(data, description, category, consent_status)`: Adds a new case to the dataset.
* `update_case(data, case_id, updates)`: Updates an existing case.
* `delete_case(data, case_id)`: Deletes a case.
* `display_case_details(case)`: Displays the details of a case.
* `main()`: The main Streamlit app function that handles user interaction and displays the UI.

## Data Structure

The application uses a list of dictionaries to store case data. Each case dictionary has the following structure:

{"id": int,"description": str,"category": str,"consent_status": str,"ai_assessment": str,"resolution": str,"date": datetime.date}
## AI Model

The AI model used is a Support Vector Machine (SVM) with a linear kernel. The model is trained on the case descriptions to predict the risk level ("High Risk", "Medium Risk", "Low Risk"). TfidfVectorizer is used to convert the text descriptions into numerical features.

## Installation

1.  Clone the repository.
2.  Install the required Python packages:

    ```bash
    pip install streamlit pandas scikit-learn
    ```
3.  Run the Streamlit application:

    ```bash
    streamlit run your_script_name.py # e.g., streamlit run consent_management_system.py
    ```

## Example Cases

The application includes the following sample cases:

```python
cases = [
    {"id": 1, "description": "Unauthorized data sharing with third parties.", "category": "Data Sharing", "consent_status": "No Consent", "ai_assessment": "High Risk", "resolution": "Legal action initiated.", "date": datetime.date(2024, 1, 15)},
    {"id": 2, "description": "Use of data for unrelated purposes without consent.", "category": "Purpose Limitation", "consent_status": "Insufficient Consent", "ai_assessment": "Medium Risk", "resolution": "Rectification of data usage.", "date": datetime.date(2024, 2, 20)},
    {"id": 3, "description": "Failure to obtain consent for sensitive data processing.", "category": "Sensitive Data", "consent_status": "No Consent", "ai_assessment": "High Risk", "resolution": "Data deletion and policy review.", "date": datetime.date(2024, 3, 10)},
    {"id": 4, "description": "Consent obtained but not properly documented.", "category": "Documentation", "consent_status": "Consent Obtained", "ai_assessment": "Low Risk", "resolution": "Improved documentation process.", "date": datetime.date(2024, 4, 5)},
    {"id": 5, "description": "Use of pre-ticked boxes for consent.", "category": "Valid Consent", "consent_status": "Consent Obtained", "ai_assessment": "Medium Risk", "resolution": "Change consent mechanism.", "date": datetime.date(2024, 5, 12)},
    {"id": 6, "description": "Denial of service for users refusing consent.", "category": "Fairness", "consent_status": "No Consent", "ai_assessment": "High Risk", "resolution": "Service access restored.", "date": datetime.date(2024, 6, 18)},
    {"id": 7, "description": "Consent not specific enough for data processing.", "category": "Specificity", "consent_status": "Insufficient Consent", "ai_assessment": "Medium Risk", "resolution": "Obtain specific consent.", "date": datetime.date(2024, 7, 22)},
    {"id": 8, "description": "Data processed for a new purpose without re-obtaining consent", "category": "Purpose Limitation", "consent_status": "No Consent", "ai_assessment": "High Risk", "resolution": "Stop processing data for new purpose.", "date": datetime.date(2024, 8, 1)},
    {"id": 9, "description": "Sharing data with a subsidiary without explicit consent.", "category": "Data Sharing", "consent_status": "Insufficient Consent", "ai_assessment": "Medium Risk", "resolution": "Review data sharing agreement.", "date": datetime.date(2024, 9, 8)},
    {"id": 10, "description": "Failure to provide users with an option to withdraw consent.", "category": "Withdrawal of Consent", "consent_status": "Consent Obtained", "ai_assessment": "Medium Risk", "resolution": "Implement consent withdrawal mechanism.", "date": datetime.date(2024, 10, 15)},
    {"id": 11, "description": "Using data for profiling without explicit consent.", "category": "Profiling", "consent_status": "No Consent", "ai_assessment": "High Risk", "resolution": "Stop profiling and delete associated data.", "date": datetime.date(2024, 11, 2)},
    {"id": 12, "description": "Consent request buried in lengthy terms and conditions.", "category": "Valid Consent", "consent_status": "Consent Obtained", "ai_assessment": "Medium Risk", "resolution": "Make consent request prominent and clear.", "date": datetime.date(2024, 12, 9)},
    {"id": 13, "description": "Processing children's data without parental consent.", "category": "Children's Data", "consent_status": "No Consent", "ai_assessment": "High Risk", "resolution": "Delete data and obtain proper consent.", "date": datetime.date(2025, 1, 20)},
    {"id": 14, "description": "Storing data longer than necessary without explicit consent.", "category": "Data Retention", "consent_status": "Insufficient Consent", "ai_assessment": "Medium Risk", "resolution": "Implement data retention policy.", "date": datetime.date(2025, 2, 28)},
    {"id": 15, "description": "Transferring data to a country with inadequate privacy laws.", "category": "Cross-border Transfer", "consent_status": "No Consent", "ai_assessment": "High Risk", "resolution": "Halt data transfer and review legal basis.", "date": datetime.date(2025, 3, 18)},
    {"id": 16, "description": "Using collected data for targeted advertising without consent.", "category": "Purpose Limitation", "consent_status": "No Consent", "ai_assessment": "High Risk", "resolution": "Cease targeted advertising.", "date": datetime.date(2025, 4, 10)},
    {"id": 17, "description": "Failing to inform users about the purpose of data collection.", "category": "Transparency", "consent_status": "No Consent", "ai_assessment": "Medium Risk", "resolution": "Update privacy policy.", "date": datetime.date(2025, 5, 5)},
    {"id": 18, "description": "Obtaining consent through deceptive practices.", "category": "Valid Consent", "consent_status": "High Risk", "resolution": "Revise consent process.", "date": datetime.date(2025, 6, 12)},
    {"id": 19, "description": "Using consent for one purpose to justify another, unrelated purpose.", "category": "Purpose Limitation", "consent_status": "No Consent", "ai_assessment": "High Risk", "resolution": "Stop unauthorized data use.", "date": datetime.date(2025, 7, 1)},
    {"id": 20, "description": "Not providing a clear and easy way for users to access their data.", "category": "Data Access", "consent_status": "Insufficient Consent", "ai_assessment": "Medium Risk", "resolution": "Implement data access mechanism.", "date": datetime.date(2025, 8, 8)}
]

```
---
Consent AI as per DPDP 2023 India Law
