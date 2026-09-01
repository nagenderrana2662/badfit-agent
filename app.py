import streamlit as st

from badfit_agent import BadfitAgent, load_gym_data


st.set_page_config(page_title="Bad.fit Assistant", page_icon="💪", layout="centered")
data = load_gym_data()
st.title("💪 Bad.fit AI Assistant")
st.caption(data["gym"]["tagline"])

if "agent" not in st.session_state:
    try:
        st.session_state.agent = BadfitAgent()
        st.session_state.messages = []
    except RuntimeError as error:
        st.error(str(error))
        st.info("Copy `.env.example` to `.env`, add your API key, then restart Streamlit.")
        st.stop()

with st.sidebar:
    st.subheader("Bad.fit")
    st.write(data["gym"]["address"])
    st.write(f"Contact: {data['gym']['phone']}")
    st.write(f"Morning: {data['hours']['morning']}")
    st.write(f"Evening: {data['hours']['evening']}")
    if st.button("Start new chat", use_container_width=True):
        st.session_state.agent.reset()
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about fitness, membership, trainers, or gym timings")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.agent.reply(prompt)
            except Exception as error:
                answer = f"Sorry, I couldn't connect right now. Please try again. ({error})"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
