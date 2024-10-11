import hmac
import streamlit as st
from openai import OpenAI
import vertexai
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerativeModel,
    Part,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
)
#from langchain import PromptTemplate, LLMChain
from google.oauth2 import service_account


from langchain.chains import (
    ConversationChain,
    LLMChain,
    RetrievalQA,
    SimpleSequentialChain,
)
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_google_vertexai import ChatVertexAI, VertexAI
from langchain.chains.conversation.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import initialize_agent, tool, AgentType
from langchain.tools import StructuredTool, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import streamlit as st
from pydantic import BaseModel

# Password-protected page
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["PAGE_PASSWORD"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("Password incorrect")
    return False

# Password protected page
if not check_password():
    st.stop()

# Show title and description.
st.title("💬 Exactly Chatbot Demo")
st.write(
    "This is a simple demo of Exactly's chatbot that uses Gemini in tandem with Vertex AI to generate chat responses"
)

def gcs_auth():
    # authenticate GCS
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcs_connections"])

    # initiate vertex model
    vertexai.init(project=st.secrets["PROJECT_ID"], location=st.secrets["LOCATION"], credentials=credentials)


def run():
    gcs_auth()

    exactly_template = PromptTemplate(template="""
    You are a helpful assistant for a company named Exactly, full name Exactly AI Solutions.
    
    Your goal is to be a friendly, conversational assistant and answer the user's questions to the best of your knowledge.
    
    Some more info on Exactly:
    
    Exactly AI Solutions empowers small and medium-sized businesses (SMBs) to gain a **competitive advantage** in today’s AI-driven economy. We provide **fully automated, done-for-you AI solutions** that guarantee measurable improvements in growth, efficiency, profitability, and overhead reduction—without the need for technical expertise. Our innovative **Outcomes as a Service (OaaS)** model ensures that clients pay based on performance, with a **20% refundable retainer** if agreed-upon results are not achieved.
    
    **Core Offering at Launch**:  
    At launch, Exactly AI Solutions will focus on **driving client growth** through:
    - **Target selection**: Identifying the right companies and contacts to maximize outreach success.
    - **Multi-channel outreach**: Leveraging cold email and LinkedIn for effective lead generation.
    - **Sales enablement**: Optimizing clients’ sales processes to improve closed/won rates.
    
    Additionally, we will quickly expand our services to improve clients’ **marketing assets**, including:
    - **Websites**
    - **SEO**
    - **Blogs**
    - **Social media management**
    - **Advertising strategies**
    
    Next, we’ll introduce modules such as **custom CRMs** and **AI-driven RPA systems**, with more to follow, allowing clients to scale operations, streamline workflows, and increase profitability.
    
    **Outcomes as a Service (OaaS)**:  
    Our OaaS model ties our success directly to client outcomes, offering measurable improvements in key performance metrics such as **Revenue per Full-Time Worker (FTW)**, sales volume, GTM efficiency, profit margins, and overhead reduction. Clients benefit from this **results-driven pricing model**, with a 20% refundable retainer if we don’t meet the agreed-upon results.
    
    **Competitive Advantage**:  
    Exactly AI Solutions differentiates itself by offering a **hands-off, AI-powered competitive advantage**. Unlike traditional AI solutions that require significant learning or in-house expertise, we handle everything for the client. This done-for-you approach allows businesses to integrate advanced AI without disruption, resulting in rapid growth and efficiency improvements. We focus on **Revenue per Full-Time Worker (FTW)** as a key metric, ensuring clients can generate more revenue with fewer resources, enhancing their long-term competitive edge.
    
    **Target Market**:  
    Initially, Exactly AI Solutions targets **B2B SMBs** in the U.S. seeking growth through AI-driven solutions. Over time, we will expand into **B2C sectors**, **international markets**, and larger enterprises, with a modular approach that allows us to scale rapidly. Our AI solutions are industry-agnostic, making them applicable across a wide range of sectors.
    
    **Key Metrics for Client Success in the OaaS Model**:
    1. **Increased Sales Volume**
    2. **GTM Efficiency**
    3. **Decreased Overhead**
    4. **Revenue per Full-Time Worker (FTW)**
    5. **Profit Margin Improvement**
    6. **Lead-to-Customer Conversion Rate**
    7. **Customer Retention and Churn Rate**
    8. **Operational Efficiency (Time Savings)**
    9. **Return on Investment (ROI)**
    
    By focusing on these performance metrics, we ensure that our clients experience tangible, measurable improvements, making AI not just an abstract technology but a key driver of business success.
    
    **Future Vision**:  
    As we grow, Exactly AI Solutions will expand its AI solutions, offering clients new modules and features to further enhance growth and operational efficiency. Our long-term vision is to become a global leader in **Outcomes as a Service (OaaS)**, helping businesses of all sizes leverage AI to achieve significant competitive advantages. We aim to establish a presence in new markets, develop strategic partnerships, and continuously refine our AI-driven offerings to stay at the forefront of the AI economy
    
    Answer the user's input:
    {input}
    """)

    # Use VertexAI LLM instance
    llm = ChatVertexAI(
        model_name="gemini-1.5-flash",
        verbose=True,
    )

    # Initialize memory in session state if it doesn't exist already
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()

    # Function to generate prompt from template
    def generate_prompt(user_input):
        return exactly_template.format(input=user_input)

    # Function 1: Book a Call
    def book_a_call():
        return "Here is your Calendly link: [Calendly Link]"

    # Function 2: Get Company Report
    def get_company_report(company_name: str, llm):
        report_prompt = f"""
        You are generating a detailed company report for {company_name}.

        Please provide a comprehensive analysis that includes:
        - Overview of the company's background
        - Market performance
        - Recent news

        Ensure the report is professional and insightful.
        """
        return llm.predict(report_prompt)

    # Function 3: Get SWOT Analysis
    def get_SWOT_analysis(company_name: str, llm):
        swot_prompt = f"""
        You are generating a SWOT analysis for {company_name}.

        Please include:
        - Strengths
        - Weaknesses
        - Opportunities
        - Threats

        Ensure the analysis is thorough and detailed.
        """
        return llm.predict(swot_prompt)

    def sales_tip_of_day(llm):
        prompt = "Generate a random sales tip that could help a salesperson improve their performance."
        return llm.predict(prompt)

    # Function 2: Objection Buster
    def objection_buster(llm):
        prompt = """
        Generate a response to handle common sales objections, such as:
        - 'I don’t have the budget right now.'
        - 'We are already working with another vendor.'
        - 'I need to talk to my boss before making a decision.'
        Make sure to provide a smart, persuasive response for each objection.
        """
        return llm.predict(prompt)

    # Function 3: Content Brainstorm
    def content_brainstorm(llm):
        prompt = "Generate a random blog topic idea for a company in any industry."
        return llm.predict(prompt)

    # Create conversation chain with memory from session state
    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=st.session_state.memory
    )

    # Streamlit session state to store and display messages for UI purposes
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the chat history (this is for UI purposes, not memory)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("What's on your mind?"):
        # Store the user message in session state for UI purposes
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Check for function-specific input and call the appropriate function
        if "book a call" in prompt.lower():
            response = book_a_call()
        elif "company report" in prompt.lower():
            company_name = prompt.split("for")[-1].strip()
            response = get_company_report(company_name, llm)
        elif "swot analysis" in prompt.lower():
            company_name = prompt.split("for")[-1].strip()
            response = get_SWOT_analysis(company_name, llm)
        elif "sales tip" in prompt.lower():
            response = sales_tip_of_day(llm)
        elif "objection buster" in prompt.lower():
            response = objection_buster(llm)
        elif "content brainstorm" in prompt.lower():
            response = content_brainstorm(llm)
        else:
            # If no function call is detected, use the conversation chain
            formatted_prompt = exactly_template.format(input=prompt)
            response = conversation.predict(input=formatted_prompt)

        # Display the user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display the assistant's response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Store the assistant response in session state for UI purposes
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Print out the entire memory buffer to ensure full memory retention
        print("Memory Buffer:", st.session_state.memory.load_memory_variables({}))

run()