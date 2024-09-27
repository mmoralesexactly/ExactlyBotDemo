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

from langchain_google_vertexai import VertexAI

# Show title and description.
st.title("💬 Exactly Chatbot Demo")
st.write(
    "This is a simple demo of Exactly's chatbot that uses Gemini in tandem with Vertex AI to generate chat responses"
)

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

# Initiate Vertex
system_instructions = """
You are a chatbot whose goal is to help users of the Exactly website with their questions about both the company
and anything else they might have a question with.

Accuracy is the top priority. If you are unsure of an answer regarding cost or capabilities, do not give an answer and instead respond with
 something like "I apologize, you will have to speak to an Exactly correspondent for more information on that".

Please be courteous but brief in your responses. You want to give short responses that respond to the user's input while
 also fitting within the confines of a chatbox window, which is not very large.

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
    description="Used for booking and scheduling calls and appointments via Calendly",
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

# function lookup for function_calls
function_handler = {
    "book_a_call": calendly_meeting,
    "get_SWOT_report": generate_SWOT_report
}


vertexai.init(project=st.secrets["PROJECT_ID"], location=st.secrets["LOCATION"])
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
    msg = chat.send_message(prompt)
    stream = msg.candidates[0].content.parts[0]
    # stream = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": m["role"], "content": m["content"]}
    #         for m in st.session_state.messages
    #     ],
    #     stream=True,
    # )

    #msg = response.candidates[0].content.parts[0]

    # Stream the response to the chat using `st.write_stream`, then store it in
    # session state.
    #with st.chat_message("assistant"):
    #    response = st.write_stream(stream)
    #st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.messages.append({"role": "assistant", "content": stream.text})
    st.chat_message("assistant").write(stream.text)