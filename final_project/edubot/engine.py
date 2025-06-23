import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class EduBotEngine:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="meta-llama/llama-4-scout-17b-16e-instruct"
        )

    def load_knowledge(self) -> str:
        """Combine all .md files from the knowledge/ folder into one text block."""
        knowledge_path = Path(__file__).parent / "knowledge"
        full_context = ""
        for file in knowledge_path.glob("*.md"):
            full_context += f"\n\n## {file.stem.replace('_', ' ').title()}\n"
            full_context += file.read_text(encoding="utf-8")
        return full_context

    def ask(self, user_question: str) -> str:
        context = self.load_knowledge()
        prompt = PromptTemplate.from_template(
            """
            ### CONTEXT:
            {knowledge}

            ### INSTRUCTION:
            You are EduBot, a helpful assistant integrated into a student dropout prediction web application.
            Based on the above CONTEXT, answer the USER QUESTION clearly and concisely.
            If the answer is not in the context, say "I'm not sure about that."

            ### USER QUESTION:
            {question}

            ### RESPONSE:
            """
        )

        chain = prompt | self.llm
        response = chain.invoke({"knowledge": context, "question": user_question})
        return response.content.strip()


# Singleton entry point
bot = EduBotEngine()

def ask_edubot(question: str) -> str:
    return bot.ask(question)
