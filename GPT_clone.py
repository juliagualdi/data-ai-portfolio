import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="MedAI Assistant", page_icon="💬")

st.title("🏥 MedAI - Virtual Medical Assistant")
st.write("Hi! I'm MedAI. Let's schedule your appointment step by step 😊")


client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Initialize session state ---
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo" 

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": """
                You are a virtual medical assistant for a clinic. 
                Your goal is to schedule appointments.

                Follow this exact flow:
                1. Ask which specialist the user is looking for.
                2. Then ask which health insurance plan they have.
                3. Then ask what is their preferred appointment date.
                4. After collecting all information, summarize it and confirm.

                Be polite, clear, and professional.
                """
        }
    ]

# --- Display chat history ---
for message in st.session_state["messages"]:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- User input ---
if prompt := st.chat_input("What’s up?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Stream the model’s response
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=st.session_state["messages"],
            stream=True,
        )

        # Write streamed text as it arrives
        response = st.write_stream(stream)

    # Save the assistant’s reply
    st.session_state["messages"].append({"role": "assistant", "content": response})