from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from langchain_classic.chains import LLMMathChain,LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import Tool,initialize_agent
from langchain_classic.agents.agent_types import AgentType
from dotenv import load_dotenv
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

## set up the streamlit app

st.set_page_config(page_title="Text to math problem solver and data search assistant")
st.title("Text to Math Problem Solver using Google api")

google_api_key=st.sidebar.text_input("Enter your Google api key",type="password")

if not google_api_key:
    st.info("please add your google api key to continue")
    st.stop()

llm=ChatGoogleGenerativeAI(google_api_key=google_api_key,model="gemini-3.8-flash")

## intializing the tools

wiki_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name='Wikipedia',
    func=wiki_wrapper.run,
    description="A tool for searching the internet to find the various information on the tool"
)

## Initialize the math tool

math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name='calculator',
    func=math_chain.run,
    description="a tools fror answering math related questions . only input mathematical expression need to be provided"
)

prompt="""
You are a agent tasked for soling mathematical questions . Logically arrive at the solution arrive and provide a detailed expression and display it point wise
question:{question}
"""

prompt_template=PromptTemplate(input_variables=['question'],template=prompt)

##combine all the tools into chain
chain=LLMChain(llm=llm,prompt=prompt_template)
reasoning_tool=Tool(
    name="reasoning",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions"
)
## initialize the agent

assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state['messages']=[{"role":"assistant","content":"Hi, i'm a math chatbot who can answer all your math problems"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])


##lets start the interaction
question=st.text_area("enter your question:")
if st.button("find my answer"):
    if question:
        with st.spinner("generating response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({"role":"assistant","content":response})
            st.write("### Response:")
            st.success(response)

    else:
        st.warning("please enter the question:")

