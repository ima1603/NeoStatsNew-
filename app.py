import streamlit as st
from models.llm import get_gemini_flash_model
from utils.rag_utils import load_scheme_data, create_vectorstore_from_documents, retrieve_relevant_chunks
from utils.websearch import get_web_search_tool
from config.config import GEMINI_API_KEY

def is_scheme_related(prompt):
    keywords = ["scheme", "benefit", "eligibility", "Karnataka", "government", "Sanjeevini", "NRLM"]
    return any(word.lower() in prompt.lower() for word in keywords)

def get_chat_response(chat_model, messages, system_prompt):
    try:
        user_messages = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
        full_prompt = f"{system_prompt}\n\nConversation:\n{user_messages}"
        return chat_model(full_prompt)
    except Exception as e:
        return f"Error: {str(e)}"

def format_web_result(search_result):
    if isinstance(search_result, dict) and "RelatedQnA" in search_result:
        formatted = ""
        for item in search_result["RelatedQnA"]:
            data = item.get("Data", {})
            question = data.get("question")
            answers = data.get("qnAAnswers", [])
            if question and answers:
                answer_text = answers[0].get("answer", "")
                link_data = answers[0].get("sourceLinks", [{}])[0]
                link = link_data.get("hyperLink", "")
                link_text = link_data.get("hyperLinkText", "Source")
                formatted += f"**{question}**\n{answer_text}\n🔗 [{link_text}]({link})\n\n"
        return f"🌐 Here's what I found online:\n\n{formatted}"
    else:
        return f"🌐 Here's what I found online:\n\n{str(search_result)}"

def chat_page():
    st.title("🤖 CIVIC HELP BOT ")
    st.markdown("_Helping you navigate Karnataka's government schemes and civic queries_")

    response_mode = st.radio("Response Mode", ["Concise", "Detailed"], horizontal=True)
    use_web_fallback = st.checkbox("Use web search fallback if needed", value=True)

    system_prompt = (
        f"You are a helpful assistant focused on Karnataka government schemes. "
        f"If the user asks something outside this domain, you may use external sources to help. "
        f"Respond in a {response_mode.lower()} manner."
    )

    chat_model = get_gemini_flash_model()
    documents = load_scheme_data()
    vectorstore = create_vectorstore_from_documents(documents)
    web_tool = get_web_search_tool()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about a scheme, eligibility, or benefit..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chunks = retrieve_relevant_chunks(vectorstore, prompt)
                context = "\n\n".join([doc.page_content for doc in chunks])

                if not is_scheme_related(prompt):
                    if use_web_fallback:
                        st.sidebar.markdown("🕸️ External query detected — using web search")
                        search_result = web_tool.run(prompt)
                        response = format_web_result(search_result)
                    else:
                        response = "Sorry, I can only answer questions related to Karnataka government schemes."
                else:
                    if chunks:
                        full_prompt = (
                            f"{system_prompt}\n\n"
                            f"You have access to the following official scheme data. "
                            f"Use this information to answer the user's question. "
                            f"If the answer is present in the data, do not say you don't know.\n\n"
                            f"Relevant Info:\n{context}\n\n"
                            f"User Query: {prompt}"
                        )
                        response = get_chat_response(chat_model, st.session_state.messages, full_prompt)

                        if use_web_fallback and (
                            "i don't know" in response.lower() or
                            "not sure" in response.lower() or
                            len(response.strip()) < 50
                        ):
                            st.sidebar.markdown("🕸️ Model uncertain — using web search")
                            search_result = web_tool.run(prompt)
                            response = (
                                "That does not belong in my stored data, but I can fetch you the results from the WEB.\n\n"
                                f"{response}\n\n🔍 Web Search Result:\n{format_web_result(search_result)}"
                            )
                    else:
                        if use_web_fallback:
                            st.sidebar.markdown("🕸️ No relevant RAG data — using web search")
                            search_result = web_tool.run(prompt)
                            response = format_web_result(search_result)
                        else:
                            response = "Sorry, I couldn’t find any relevant scheme data for your query."

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

def instructions_page():
    st.title("📘 Instructions")
    st.markdown("Follow setup steps in the sidebar. Ensure API keys are set in `config/config.py`.")

def main():
    st.set_page_config(page_title="Karnataka Scheme ChatBot", page_icon="🤖", layout="wide")

    # ✅ Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.title("Chat Controls")
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_button"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("### 🕘 Chat History")
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                if st.button(f"🔁 {msg['content'][:40]}", key=f"history_{i}"):
                    st.session_state.messages.append({"role": "user", "content": msg["content"]})
                    st.rerun()

    chat_page()

if __name__ == "__main__":
    main()