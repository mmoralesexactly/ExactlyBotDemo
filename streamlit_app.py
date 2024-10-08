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
    Tool,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold,
)
#from langchain import PromptTemplate, LLMChain
from google.oauth2 import service_account

from langchain_google_vertexai import VertexAI

# Password protected page
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

# Feedback form area



# Initiate Vertex
system_instructions = """
You are a chatbot whose goal is to help users of the Exactly website with their questions about both the company
and anything else they might have a question with.

Accuracy is the top priority. If you are unsure of an answer regarding cost or capabilities, do not give an answer and instead respond with
 something like "I apologize, you will have to speak to an Exactly correspondent for more information on that".

Please be courteous but brief in your responses. You want to give short responses that respond to the user's input while
 also fitting within the confines of a chatbox window, which is not very large. At most, try to keep your response under 100 characters in most cases.

Your other top priority is to try and be as friendly and knowledgeable as you can be. We do not want the typical chatbot experience,
we want to user to feel like that are conversating with an actual person that is not just responding with canned responses.

If the user asks something that can be answered using one of the FunctionDeclarations provided, trigger that FunctionDeclaration.

Here is a sampling of info on Exactly you can use to answer the user
* Exactly is a "growth accelerator" that aims to provide clients with a done-for-you engine to accelerate their growth
* Services that Exactly Offers
    * AI Readiness Assessment
    * Company Report, covering a SWOT analysis of the given company
    * SEO Optimization
    * Chatbot creation and implementation
    * Website and Social Media overview to see if best practices are in place

If the user asks something that has nothing to do with Exactly's services:
* Answer their question to the best of your ability
* Add an interesting tidbit of trivia if possible
    * Example: If they ask 'When is Valentine's Day?', answer the question and then add a bit of trivia about when the first Valentines day
       was celebrated, or who it is named after, or any sort of interesting knowledge the user may not be aware of
* After answering their question, try to use a smooth segue back into Exactly's services
    * Example: 'Speaking of Valentines Day, I'm sure Exactly has some services that you'll love!'
    
Here is additional info on Exactly's Business Summary. Use this to answer questions about Exactly's services and future business plans:

**Business Summary: Exactly AI Solutions**



**Overview**:  

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

As we grow, Exactly AI Solutions will expand its AI solutions, offering clients new modules and features to further enhance growth and operational efficiency. Our long-term vision is to become a global leader in **Outcomes as a Service (OaaS)**, helping businesses of all sizes leverage AI to achieve significant competitive advantages. We aim to establish a presence in new markets, develop strategic partnerships, and continuously refine our AI-driven offerings to stay at the forefront of the AI economy.
"""

# Vertex FuncCalls
get_product_info = FunctionDeclaration(
    name="get_product_info",
    description="Get the stock amount and identifier for a given product",
    parameters={
        "type": "object",
        "properties": {
            "product_name": {"type": "string", "description": "Product name"}
        }
    }
)

book_a_call = FunctionDeclaration(
    name="book_a_call",
    description="Used to book a call or meeting or appointment via Calendly",
    parameters={
        "type": "object",
        #"properties": {
        #    "datetime": {"type": "string", "description": "Date and Time for the booking"}
        #}
    }
)

get_company_report = FunctionDeclaration(
    name="get_company_report",
    description="Generates a company report for a given company",
    parameters={
        "type": "object",
        "properties": {
            "company_name": {"type": "string", "description": "Name of company to generate a report for"}
        }
    }
)

get_SWOT_report = FunctionDeclaration(
    name="get_SWOT_report",
    description="Generates a SWOT analysis report for a given company",
    parameters={
        "type": "object",
        "properties": {
            "company_name": {"type": "string", "description": "Name of company to generate a report for"}
        }
    }
)

func_tools = Tool(
    function_declarations=[
        get_product_info,
        book_a_call,
        get_company_report,
        get_SWOT_report
    ]
)

# used to book a calendly meeting
def calendly_meeting():
    pass

def generate_SWOT_report(company_name, chat):
    prompt = f"""
    Please create a comprehensive SWOT analysis report for a company called {company_name}. Base your report on
    publicly available information that you have access to. The report should contain the following sections:
        - **Strengths**: Internal factors that give the company an advantage
        - **Weaknesses**: Internal factors that may hinder the company
        - **Opportunities**: External factors the company can leverage for growth
        - **Threats**: External factors that could pose challenges to the company
    """

    response = chat.send_message(prompt)
    output = response.candidates[0].content.parts[0]
    return output

def generate_company_report(company_name, chat):
    pass

# function lookup for function_calls
function_handler = {
    "book_a_call": calendly_meeting,
    "get_company_report": generate_company_report,
    "get_SWOT_report": generate_SWOT_report
}



@st.cache_resource(show_spinner=False)
def LLM_init():
    # authenticate GCS
    credentials = service_account.Credentials.from_service_account_info(st.secrets["gcs_connections"])

    # initiate vertex model
    vertexai.init(project=st.secrets["PROJECT_ID"], location=st.secrets["LOCATION"], credentials=credentials)
    model = GenerativeModel(
        st.secrets["MODEL"],
        system_instruction=system_instructions,
        generation_config=GenerationConfig(temperature=1.0),
        tools=[func_tools],
        safety_settings=[
                                SafetySetting(
                                    category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                ),
                                SafetySetting(
                                    category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                ),
                                SafetySetting(
                                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                ),
                                SafetySetting(
                                    category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                ),
                                SafetySetting(
                                    category=HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                )
                            ]
    )

    return model

def generate_response(chat, model, input):
    response = chat.send_message(input)

    #handle cases w/ multiple chained function calls
    function_calling_in_progress = True
    while function_calling_in_progress:
        # extract function call response
        function_call = response.candidates[0].content.parts[0].function_call
        print(function_call)

        if function_call.name in function_handler.keys():
            function_name = function_call.name
            print(f"[FunctionCall]: {function_name}")

            if function_name == 'book_a_call':
                #response = chat.send_message(
                #    Part.from_function_response(
                #        name=function_name,
                #        response={"content": "https://calendly.com/b2bcustomleads"}
                #    )
                #)
                return "Sure! Please use the following calendly link to schedule a call with us: https://calendly.com/b2bcustomleads"
            elif function_name == 'get_company_report':
                function_calling_in_process = False
                params = {k: v for k, v in function_call.args.items()}
                company_name = params['company_name']
                print(f"Sure, I can generate a company report for {company_name}")

                prompt = f"""
                            "Please create a comprehensive company report for {company_name} based on publicly available information. The report should include the following sections:
                            Company Overview:
                                - Company type (public, private, employee-owned)
                                - Primary Industry/Industries: The main industry or industries the company operates in.
                                - Year founded
                                - Headquarters location
                                - Number of employees (if available)
                                - Annual revenue (if available)
                                - Website URL
                             
                            Mission & Values:
                                Mission statement
                                Core values
                            Expertise & Offerings:
                                Core areas of expertise or products/services offered
                                Additional offerings or specializations
                            Key Differentiators:
                                Unique aspects of their approach, technology, or services that set them apart from competitors
                                
                            Target Audience/Customers:
                                Primary target market or customer segments
                                Secondary or niche markets they serve
                            Products/Services (or Project Portfolio):
                                Description of core products, services, or projects
                                Key features and benefits
                                Pricing models (if available)
                            Competitive Landscape:
                                Main competitors
                                Factors that differentiate the company from its competitors
                            Marketing & Sales Insights:
                                Key messaging and branding themes
                                Marketing channels utilized
                                Types of content created (case studies, white papers, etc.)
                                Presence of client testimonials or reviews
                            Social Media Presence:
                                List of social media channels the company is active on (e.g., LinkedIn, Twitter, Facebook)
                                URLs of the company's social media profiles (where available)
                                Brief analysis of their social media activity and engagement (optional)
                            SWOT Analysis Summary:
                                Strengths: Internal factors that give the company an advantage
                                Weaknesses: Internal factors that may hinder the company
                                Opportunities: External factors the company can leverage for growth
                                Threats: External factors that could pose challenges to the company
                            12. Financial Performance (If Available):
                                Brief overview of financial health, revenue, or growth trends, if accessible from public sources
                                
                            Please ensure the report is well-structured, concise, and provides actionable insights based on the available information."
                            """

                response = model.generate_content(
                    prompt,
                    safety_settings=[
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        )
                    ])

                return response.text
            elif function_name == 'get_SWOT_report':
                function_calling_in_process = False
                params = {k: v for k, v in function_call.args.items()}
                company_name = params['company_name']
                print(f"Sure, I can generate a SWOT analysis for {company_name}")

                prompt = f"""
                                    Please create a comprehensive SWOT (Strengths, Weaknesses, Opportunites, Threats) report for the company called {company_name}. Base your report on
                                    publicly available information that you have been trained on. The report should contain the following sections:
                                        - **Strengths**: Internal factors that give the company an advantage
                                        - **Weaknesses**: Internal factors that may hinder the company
                                        - **Opportunities**: External factors the company can leverage for growth
                                        - **Threats**: External factors that could pose challenges to the company
                                    """

                response = model.generate_content(
                    prompt,
                    safety_settings=[
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        ),
                        SafetySetting(
                            category=HarmCategory.HARM_CATEGORY_UNSPECIFIED,
                            threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                        )
                    ])

                return response.text
        else:
            function_calling_in_progress = False
            print(response.candidates[0].content.parts[0])
            return response.candidates[0].content.parts[0].text

def run():
    model = LLM_init()
    chat = model.start_chat()

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("What's on your mind?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using Gmodel
        msg = generate_response(chat, model, prompt)

        #msg = chat.send_message(prompt)
        #stream = msg.candidates[0].content.parts[0]
        #msg = response.candidates[0].content.parts[0]

        # Stream the response to the chat using `st.write_stream`, then store it in
        # session state.
        #with st.chat_message("assistant"):
        #    response = st.write_stream(stream)
        #st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

run()