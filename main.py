import streamlit as st

def main():
    st.set_page_config(page_title="Chat With Your PDF(s)", page_icon=":sunglasses:")

    st.header("Chat With Your PDF(s) :sunglasses:")
    st.text_input("Ask a question to your document(s):")
    
    with st.sidebar:
        st.subheader("Your Documents")

if __name__ == '__main__':
    main()