from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet.
               Always provide detailed recommendations, including requests for length, virality, style, etc.""",
        ),
        MessagesPlaceholder("messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a twitter techie influencer assistant tasked with writing excellent twitter posts.
               Generate the best twitter post possible for the user's request.
               If the user provides critique, respond with a revised version of your previous attempts.""",
        ),
        MessagesPlaceholder("messages"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm
