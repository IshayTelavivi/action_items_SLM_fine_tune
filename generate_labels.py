from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema.runnable import RunnableLambda, RunnableSequence, RunnableParallel, RunnableBranch
from tqdm.auto import tqdm
import json
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from .env
load_dotenv()

# Create a model
model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
# print(os.path.dirname(os.path.realpath(__file__)))
# print("-----")
# print(os.getcwd())
calls = json.load(open("C:/Users/IshayTA/OneDrive/Documents/src/langchain-crash-course/3_chains/call_links_with_details_10.json"))

n_calls = len(calls["transcript"])
print("Found {} calls".format(n_calls))


system_message_content = """
                          We have a call conversation transcription between an agent and a potential client.
                          The agent's main goal is to convert the client, meaning to make them deposit money for trading.
                          The transcriptions contain normally 2 speakers, Speaker A and Speaker B, or Speaker 0 and Speaker 1.
                          First thing we need to do is figure out who the agent is and who the customer is among The speakers.
                          Normally Speaker A (or Speaker 0) is the agent but this is not always the case, so we need to make sure.

                          You have several tasks:

                          Task #1: call summary.
                          Please provide a summary of one paragraph between 50 and 150 words that describes the main topics and insights
                          from conversation. No need for the whole conversation, just the main topics and insights.
                          Please stick to the conversation content without any further interpretations.

                          Taks #2: Lead Status Determination
                          Your task is to provide the most probable lead status that represent the client at the end of the call.
                          Only the lead status title, no definition nor any other explanation.
                          Also, state the second probable lead status.
                          For that, please use the following lead status definitions:
                          “Tried To Deposit”: A Failed deposit by the client.
                            “Deposit Arrangement”: The client is clearly intends to deposit soon, and had stated a
                                                    specific amount and timefrme where thay intend to make the deposit.
                            “Potential”: The client is positive, expresses interest, however there isn't yet any
                                                    concrete action or a due time for a client action.
                            “In Process”: The client a bit of a slow burner. Is not fully reluctant about the product,
                                                    and may ask some questions, but not enthusiastic.
                            “Not Interested”: Client clearly states they are not interested in Foretrade services.
                                                    They may use terms such as "not interested" or "I don't want" or
                                                    something equivalent.
                            “Being Trained”: Co-browsing training with the client. This means that the client is logged
                                                    into the system and the agent is guiding the client about the
                                                    various actions and views in it.
                            “Converted”: Client has made a deposit.
                            “Do Not Call”: The client is adamant that they do not want to receive calls from us. They
                                                    may use terms such as "do not call me", "stop calling me", "cancel
                                                    my subscription", "delete my account" or something equivalent.
                            “No English”: Client does not speak English
                            “Under 18”: Client is under 18
                            “Above 75 Years Old”: Client is above 75 years old
                            “Banned Country”: Client resides in a country that cannot trade with us

                        Important guidelines for lead status determination:
                            1. If one of the following lead status is a probable lead status, it should receive priority over other probable lead statuses. These relevant lead statuses are: “Do Not Call”, “Converted”, “Under 18”, “Banned Country”, “No English”.
                            2. The lead status should relate to the state of the client at the end of the call. Even if during the call the client had said things, but the call ended with a different approach, what matters is the end of the call.

                        Your answer to this task should look like the following:
                        "Most probable lead status: <add most probable>
                         Second probable lead status: <add second probable>"

                        Task #3: Trading Lead Info
                         Only if available, please detail any financial information about the customer or anything that 
                         indicates their trading interests. For example: customer's income, savings, any instruments they
                         are interested in, how much they wish to deposit and related information. 
                         If there is no such information, you may leave this section empty or write "N/A".
                         
                        Task #4: Action Items
                        Please detail all of the actions items that were agreed upon by the agent and the client.
                        These actions may include a followup call, an email to be sent, a deposit by the client, a reminder to
                        be set by the agent and any other action. Please add the due time for each action, only if available.
                        Note: no need for detailing what happened in the call, just a bulleted concise list of actions. 
                        Please separate the timeing from the action. Also, if no time was set to the action, write "N/A".
                        Examples:
                        "- Action: A follow-up call to the client. When: tomorrow at 13:00
                         - Action: Send an email to the client with a link to the documents. When: N/A"
                         If there are no action items, state only "No follow-up action items agreed upon."
                         
                        There is no need to detail which one is the agent or client.
                        The call transcription is following this message.
"""



answer_template = """
                          The following is a transcription of a phone call.
                          "
                          {conversation_transcription}
                          "
                          Your task is determine if the call was answered or if was taken by a voicemail. Common phrases 
                          that may indicate a phone call are, for example: 'The person you are calling is not available', 
                          'The person you called is not available', 'The person you have called is not available',
                          '..messaging services..',  'Welcome to the voicemail', 'please leave your message', 'to leave 
                          a message', 'Can't take your call right now' etc. (there may be additional indications)
                          
                          If the cal was taken to voicemail, please state in your response only one word: "voicemail".
                          If the call was actually answered, please state in your response only one word: "answered".
                    """


voicemail_content = """
                        We know that the call was answered by your voicemail. 
                        Please state your response as follows.
                        First, indicate that the cal was taken to voicemail:
                        "There was no answer. The call was answered by a voicemail."       
                        Then, if the agent had left a message, state that a message was left, and summarize the message.        
                        Lead Status Determination: please state that the lead status in "No Answer"           
                        There is no need to detail which one is the agent or client.
                        The call transcription is following this message.
                        
"""

# Define branch conditions
def is_voicemail(result):
    return result.strip().lower() == "voicemail"



calls["response"] = {}
# Chat loop
for i in tqdm(range(3)):
    conversation_transcription = calls["transcript"][str(i)]

    if i == 0:
        conversation_transcription = "\nSpeaker 1 The EE voicemail, I'm sorry but the person you've called is not available. Please leave your message after the tone. After you've finished your message, just hang up or to hear more options,"
    # First chain to check if voicemail
    answer_prompt = PromptTemplate(
        template=answer_template,
        input_variables=["conversation_transcription"]
    )
    answer_chain = answer_prompt | model | StrOutputParser()


    def process_full_conversation(x):
        messages = [
            SystemMessage(content=system_message_content),
            HumanMessage(content=conversation_transcription)
        ]
        return model.invoke(messages).content


    def process_voicemail(x):
        messages = [
            SystemMessage(content=voicemail_content),
            HumanMessage(content=conversation_transcription)
        ]
        return model.invoke(messages).content

    # Create the branching chain
    chain = (
            answer_chain |
            RunnableBranch(
                (is_voicemail, RunnableLambda(process_voicemail)),
                RunnableLambda(process_full_conversation)
            )
    )
    response = chain.invoke({"conversation_transcription": conversation_transcription})

    calls["response"][str(i)] = response
    print(f"AI: {response}")

with open("C:/Users/IshayTA/OneDrive - Leader Capital/Documents/src/langchain-crash-course/3_chains/samples_output_long_10.json", "w") as f:
    json.dump(calls, f)


