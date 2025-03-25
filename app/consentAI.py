# %%writefile consent.py
# Author : Ashish Patel
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import random
import datetime

# Data for demonstration
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
    {"id": 18, "description": "Obtaining consent through deceptive practices.", "category": "Valid Consent", "consent_status": "Consent Obtained", "ai_assessment": "High Risk", "resolution": "Revise consent process.", "date": datetime.date(2025, 6, 12)},
    {"id": 19, "description": "Using consent for one purpose to justify another, unrelated purpose.", "category": "Purpose Limitation", "consent_status": "No Consent", "ai_assessment": "High Risk", "resolution": "Stop unauthorized data use.", "date": datetime.date(2025, 7, 1)},
    {"id": 20, "description": "Not providing a clear and easy way for users to access their data.", "category": "Data Access", "consent_status": "Insufficient Consent", "ai_assessment": "Medium Risk", "resolution": "Implement data access mechanism.", "date": datetime.date(2025, 8, 8)}
]

# Function to train the AI model
def train_model(data):
    df = pd.DataFrame(data)
    X = df['description']
    y = df['ai_assessment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SVC(kernel='linear', probability=True))  # Use probability=True for predict_proba
    ])
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    return pipeline, accuracy

# Function to assess new cases using the AI model
def assess_case(model, description):
    prediction = model.predict([description])[0]
    probability = model.predict_proba([description])[0] # Get probabilities for each class
    # Get the probabilities for 'High Risk', 'Medium Risk', and 'Low Risk'
    high_risk_prob = probability[model.classes_.tolist().index('High Risk')]
    medium_risk_prob = probability[model.classes_.tolist().index('Medium Risk')]
    low_risk_prob = probability[model.classes_.tolist().index('Low Risk')]

    return prediction, high_risk_prob, medium_risk_prob, low_risk_prob # Return all 3 probabilities

# Function to add a new case to the dataset
def add_case(data, description, category, consent_status):
    new_id = max(case['id'] for case in data) + 1
    new_case = {
        "id": new_id,
        "description": description,
        "category": category,
        "consent_status": consent_status,
        "ai_assessment": "Pending",  # Initial assessment
        "resolution": "Pending",
        "date": datetime.date.today()
    }
    data.append(new_case)
    return new_case

# Function to update an existing case
def update_case(data, case_id, updates):
    for case in data:
        if case['id'] == case_id:
            case.update(updates)
            return True
    return False

# Function to delete a case
def delete_case(data, case_id):
    for i, case in enumerate(data):
        if case['id'] == case_id:
            del data[i]
            return True
    return False

# Function to display case details
def display_case_details(case):
    st.write(f"**ID:** {case['id']}")
    st.write(f"**Description:** {case['description']}")
    st.write(f"**Category:** {case['category']}")
    st.write(f"**Consent Status:** {case['consent_status']}")
    st.write(f"**AI Assessment:** {case['ai_assessment']}")
    st.write(f"**Resolution:** {case['resolution']}")
    st.write(f"**Date:** {case['date']}")

# Main Streamlit app
def main():
    st.title("Consent Management System - AI Case Assistant")
    global cases  # Use the global cases list

    # Train the AI model
    model, accuracy = train_model(cases)
    st.write(f"AI Model Accuracy: {accuracy:.2f}")

    # Sidebar for actions
    action = st.sidebar.selectbox("Choose an Action", ["View Cases", "Add Case", "Assess Case", "Update Case", "Delete Case", "Train Model"])

    if action == "View Cases":
        st.header("Cases")
        # Add a selectbox to filter cases by category
        categories = ["All"] + list(set(case["category"] for case in cases))
        selected_category = st.selectbox("Filter by Category", categories)

        # Add a selectbox to filter cases by consent status
        consent_statuses = ["All"] + list(set(case["consent_status"] for case in cases))
        selected_consent_status = st.selectbox("Filter by Consent Status", consent_statuses)

        # Add a date range filter
        start_date = st.date_input("Start Date", min(case["date"] for case in cases), key="start_date")
        end_date = st.date_input("End Date", max(case["date"] for case in cases), key="end_date")

        filtered_cases = cases
        if selected_category != "All":
            filtered_cases = [case for case in filtered_cases if case["category"] == selected_category]
        if selected_consent_status != "All":
            filtered_cases = [case for case in filtered_cases if case["consent_status"] == selected_consent_status]
        filtered_cases = [case for case in filtered_cases if start_date <= case["date"] <= end_date]

        if not filtered_cases:
            st.write("No cases found matching the criteria.")
        else:
            for case in filtered_cases:
                with st.expander(f"Case {case['id']}: {case['description'][:50]}..."):
                    display_case_details(case)

    elif action == "Add Case":
        st.header("Add New Case")
        description = st.text_area("Description of the case")
        category = st.selectbox("Category", ["Data Sharing", "Purpose Limitation", "Sensitive Data", "Documentation", "Valid Consent", "Fairness", "Specificity", "Withdrawal of Consent", "Profiling", "Children's Data", "Data Retention", "Cross-border Transfer", "Transparency", "Data Access"])
        consent_status = st.selectbox("Consent Status", ["No Consent", "Insufficient Consent", "Consent Obtained"])
        if st.button("Add Case"):
            new_case = add_case(cases, description, category, consent_status)
            st.success("Case added successfully!")
            st.write("New Case Details:")
            display_case_details(new_case)
            # Re-train the model after adding a new case
            model, accuracy = train_model(cases)
            st.write(f"AI Model re-trained. New Accuracy: {accuracy:.2f}")

    elif action == "Assess Case":
        st.header("Assess Case")
        description = st.text_area("Enter the case description to assess:")
        if st.button("Assess"):
            assessment, high_prob, medium_prob, low_prob = assess_case(model, description) # Get probabilities
            st.write(f"AI Assessment: {assessment}")
            st.write(f"Probability of High Risk: {high_prob:.2f}")
            st.write(f"Probability of Medium Risk: {medium_prob:.2f}")
            st.write(f"Probability of Low Risk: {low_prob:.2f}")
            #st.write(f"AI Assessment: {assessment} (High Risk: {high_prob:.2f}, Medium Risk: {medium_prob:.2f}, Low Risk: {low_prob:.2f})") # combined output
            #st.write(f"AI Assessment: {assessment}  (High Risk: {high_prob:.2f} , Medium Risk: {medium_prob:.2f} , Low Risk: {low_prob:.2f})")

    elif action == "Update Case":
        st.header("Update Case")
        case_id = st.number_input("Enter the ID of the case to update", min_value=1, step=1)
        updates = {}
        description = st.text_area("Description (leave blank to keep current)")
        if description:
            updates['description'] = description
        category = st.selectbox("Category (leave blank to keep current)", ["Data Sharing", "Purpose Limitation", "Sensitive Data", "Documentation", "Valid Consent", "Fairness", "Specificity", "Withdrawal of Consent", "Profiling", "Children's Data", "Data Retention", "Cross-border Transfer", "Transparency", "Data Access"], index=0) # added index
        if category != "Data Sharing": # added if
             updates['category'] = category
        consent_status = st.selectbox("Consent Status (leave blank to keep current)", ["No Consent", "Insufficient Consent", "Consent Obtained"], index=0) # added index
        if consent_status != "No Consent": # added if
            updates['consent_status'] = consent_status
        resolution = st.text_area("Resolution (leave blank to keep current)")
        if resolution:
            updates['resolution'] = resolution
        if st.button("Update"):
            updated = update_case(cases, case_id, updates)
            if updated:
                st.success("Case updated successfully!")
                # Re-train the model after updating a case
                model, accuracy = train_model(cases)
                st.write(f"AI Model re-trained. New Accuracy: {accuracy:.2f}")
            else:
                st.error("Case not found.")

    elif action == "Delete Case":
        st.header("Delete Case")
        case_id = st.number_input("Enter the ID of the case to delete", min_value=1, step=1)
        if st.button("Delete"):
            deleted = delete_case(cases, case_id)
            if deleted:
                st.success("Case deleted successfully!")
                 # Re-train the model after deleting a case
                model, accuracy = train_model(cases)
                st.write(f"AI Model re-trained. New Accuracy: {accuracy:.2f}")
            else:
                st.error("Case not found.")
    elif action == "Train Model":
        st.header("Train Model")
        st.write("Click the button below to retrain the AI model with the current data.")
        if st.button("Retrain Model"):
            model, accuracy = train_model(cases)
            st.write(f"AI Model re-trained. Accuracy: {accuracy:.2f}")
            st.success("Model has been successfully retrained!")

if __name__ == "__main__":
    main()