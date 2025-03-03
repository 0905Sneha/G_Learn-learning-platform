import google.generativeai as genai
import datetime
import json

api_key=AIzaSyCl2a9rsKx8Xir3_sc6jzpeDgft3AV5kBA

class GeminiTutor:
    def __init__(self, api_key, subject, student_name):
        """
        Initializes the Gemini Tutor.

        Args:
            api_key: Your Gemini API key.
            subject: The subject the tutor will teach (e.g., "Mathematics", "Science").
            student_name: The name of the student.
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.subject = subject
        self.student_name = student_name
        self.prompt_history = self._initialize_prompt_history(subject) # Initialize the prompt history
        self.interaction_data = [] # Store data about each interaction
        self.start_time = datetime.datetime.now() #Track when the session begins

    def _initialize_prompt_history(self, subject):
      """
      Initializes the prompt history with instructions for Gemini's role.
      """
      prompt_history = [
        {"role": "user", "parts": [f"You are an extremely patient, engaging, and knowledgeable AI tutor for {self.student_name} teaching the subject of {subject}. Your goal is to provide high-quality, interactive education.  At each step, ensure the student understands before moving on. Provide simple questions, then ask for more and more detail."]},
        {"role": "model", "parts": [f"Okay, I understand. I will be {self.student_name}'s AI tutor for {subject}. I will focus on engagement, testing understanding at every step, and providing a positive learning experience. Let's begin!"]}
      ]
      return prompt_history

    def get_gemini_response(self, student_input):
        """
        Sends the student's input to Gemini, gets the response, and updates the prompt history and interaction data.

        Args:
            student_input: The student's text input.

        Returns:
            The Gemini's response (text).
        """
        self.prompt_history.append({"role": "user", "parts": [student_input]})
        try:
            response = self.model.generate_content(self.prompt_history)
            response_text = response.text
        except Exception as e:
            response_text = f"Error generating response: {e}"  # Handle errors gracefully
            print(f"Gemini API Error: {e}") #Log the full error

        self.prompt_history.append({"role": "model", "parts": [response_text]})

        # Store interaction data
        interaction_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "student_input": student_input,
            "gemini_response": response_text
        }
        self.interaction_data.append(interaction_data)

        return response_text

    def analyze_response(self, student_input, gemini_response):
        """
        Analyzes the student's input and Gemini's response to determine difficulty level, accuracy, and engagement.
        (This is a placeholder - implement detailed logic here)
        """
        # Placeholder logic - Replace with more sophisticated analysis
        correct = "unsure"  # Default to "unsure" - requires actual analysis
        if "correct" in student_input.lower():  # Very basic keyword example
            correct = "correct"
        elif "wrong" in student_input.lower():
            correct = "incorrect"

        return {"correctness": correct}  # More metrics will be added here

    def generate_report(self):
        """
        Generates a detailed report of the student's learning session.

        Returns:
            A dictionary containing the report data.
        """

        end_time = datetime.datetime.now()
        session_duration = end_time - self.start_time

        report = {
            "student_name": self.student_name,
            "subject": self.subject,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "session_duration": str(session_duration),  # Format as a string
            "total_interactions": len(self.interaction_data),
            "summary": self._summarize_learning(),
            "interaction_details": self._get_detailed_interaction_data()
        }
        return report

    def _summarize_learning(self):
        """
        Summarizes the student's learning based on the interaction data.
        (This is a placeholder - implement detailed logic here)
        """
        # Placeholder logic - replace with real analysis
        num_correct = sum(1 for interaction in self.interaction_data if "correct" in interaction["gemini_response"].lower())
        num_incorrect = sum(1 for interaction in self.interaction_data if "incorrect" in interaction["gemini_response"].lower())
        summary = f"During this session, {self.student_name} attempted {len(self.interaction_data)} interactions. \n"
        summary += f"A preliminary assessment indicates {num_correct} correct responses and {num_incorrect} incorrect responses (Note: More sophisticated analysis needed)."
        return summary

    def _get_detailed_interaction_data(self):
        """
        Extracts and formats interaction data for the report.
        """
        detailed_data = []
        for interaction in self.interaction_data:
            analysis = self.analyze_response(interaction["student_input"], interaction["gemini_response"])  # Analyze each response
            detailed_data.append({
                "timestamp": interaction["timestamp"],
                "student_input": interaction["student_input"],
                "gemini_response": interaction["gemini_response"],
                "analysis": analysis  # Include the analysis results
            })
        return detailed_data

    def save_report(self, filename="learning_report.json"):
        """
        Saves the report to a JSON file.
        """
        report = self.generate_report()
        with open(filename, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Report saved to {filename}")

# Example Usage (outside the class definition):
if __name__ == '__main__':
    # Replace with your actual API key, subject, and student name
    api_key = "YOUR_GEMINI_API_KEY"
    subject = "Mathematics"
    student_name = "Alice"

    tutor = GeminiTutor(api_key, subject, student_name)

    # Simulate a few interactions
    student_input1 = "What is 2 + 2?"
    response1 = tutor.get_gemini_response(student_input1)
    print(f"Student: {student_input1}\nGemini: {response1}\n")

    student_input2 = "correct 4" # Include a correctness clue in the input (example)
    response2 = tutor.get_gemini_response(student_input2)
    print(f"Student: {student_input2}\nGemini: {response2}\n")

    student_input3 = "wrong"
    response3 = tutor.get_gemini_response(student_input3)
    print(f"Student: {student_input3}\nGemini: {response3}\n")


    # Generate and save the report
    report = tutor.generate_report()
    print(json.dumps(report, indent=4)) # Print report to console


def analyze_response(self, student_input, gemini_response):
    analysis_prompt = f"""
    You are an expert learning analyst.  Analyze the following interaction between a student and an AI tutor:"""

    Subject: {self.subject}
    Student Input: {student_input}
    AI Tutor Response: {gemini_response}

    try:
        analysis_response = self.model.generate_content(analysis_prompt)
        analysis_json = json.loads(analysis_response.text)
        return analysis_json
    except (json.JSONDecodeError, Exception) as e:
        print(f"Analysis Error: {e}, Raw Response: {analysis_response.text if 'analysis_response' in locals() else 'No response'}")
        return {
            "correctness": "unsure",
            "misconceptions": [],
            "difficulty": "unknown",
            "engagement": "unknown",
            "emotion": "unknown",
            "topic": []
        }

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (replace with your actual data)
data = {'Date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01']),
        'Score': [75, 80, 85, 78, 90],
        'Completion Rate': [0.8, 0.85, 0.9, 0.75, 0.95]}
df = pd.DataFrame(data)

st.subheader("Progress Over Time")

# Line chart for Score
fig, ax = plt.subplots()
sns.lineplot(x='Date', y='Score', data=df, ax=ax)
ax.set_title("Student Score Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Score")
st.pyplot(fig)

# Line chart for Completion Rate
fig, ax = plt.subplots()
sns.lineplot(x='Date', y='Completion Rate', data=df, ax=ax)
ax.set_title("Student Completion Rate Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Completion Rate")
st.pyplot(fig)

# Sample data
data = {'Answer Type': ['Correct', 'Incorrect', 'Partially Correct'],
        'Count': [50, 20, 10]}
df = pd.DataFrame(data)

st.subheader("Answer Type Distribution")

fig, ax = plt.subplots()
ax.pie(df['Count'], labels=df['Answer Type'], autopct='%1.1f%%', startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax.set_title("Distribution of Answer Types")
st.pyplot(fig)

# Sample data
data = {'Date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01']),
        'Time Spent (minutes)': [30, 45, 60, 50, 70],
        'Questions Asked': [2, 5, 8, 6, 10]}
df = pd.DataFrame(data)

st.subheader("Engagement Over Time")

# Area chart for Time Spent
fig, ax = plt.subplots()
ax.fill_between(df['Date'], df['Time Spent (minutes)'], color="skyblue", alpha=0.4)
ax.plot(df['Date'], df['Time Spent (minutes)'], color="steelblue", linewidth=1)
ax.set_title("Time Spent Learning Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Time Spent (minutes)")
st.pyplot(fig)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gemini_tutor import GeminiTutor  # Your Gemini tutor module

# Your API Key and student details
API_KEY = "YOUR_GEMINI_API_KEY"
STUDENT_NAME = "Example Student"

# Initialize GeminiTutor if not already in session state
if 'tutor' not in st.session_state:
    st.session_state.tutor = GeminiTutor(API_KEY, "Mathematics", STUDENT_NAME)

# --- Session History ---
st.header("Learning Session")
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

for role, content in st.session_state['chat_history']:
    with st.chat_message(role):
        st.write(content)

#Chat input
prompt = st.chat_input("Say something")

#Get response from gemini and display it
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state['chat_history'].append(("user", prompt))

    with st.chat_message("assistant"):
        response = st.session_state.tutor.get_gemini_response(prompt)
        st.write(response)
    st.session_state['chat_history'].append(("assistant", response))

# --- Student Performance Analysis ---
st.header("Student Performance Analysis")

# Check if there's interaction data
if st.session_state.tutor.interaction_data:
    # Convert interaction data to DataFrame for easier analysis
    df = pd.DataFrame(st.session_state.tutor.interaction_data)

    # Calculate correctness (simple example: counting "correct" in responses)
    df['correct'] = df['gemini_response'].str.lower().str.contains('correct').astype(int)

    # Calculate cumulative correct answers over time
    df['cumulative_correct'] = df['correct'].cumsum()

    # --- Visualizations ---
    st.subheader("Cumulative Progress")
    st.line_chart(df['cumulative_correct'])

    # Correctness distribution
    correctness_counts = df['correct'].value_counts()
    st.subheader("Correctness Distribution")
    fig1, ax1 = plt.subplots()
    ax1.pie(correctness_counts, labels=['Incorrect', 'Correct'], autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)

    # Detailed interactions
    with st.expander("Detailed Interactions"):
        st.dataframe(df)
else:
    st.write("No interaction data available yet.")

# Generate and display the report
if st.button("Generate Report"):
    report = st.session_state.tutor.generate_report()
    st.json(report)