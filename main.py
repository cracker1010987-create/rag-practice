# step 1: API 키
from dotenv import load_dotenv
load_dotenv()

# step 2: 문서 불러오기
from langchain_community.document_loaders import TextLoader

loader = TextLoader("doc.txt" , encoding="utf-8")
doc = loader.load()

# step 3: 문서 청킹
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

split_doc = splitter.split_documents(doc)

# step 4: 벡터DB 구축
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = Chroma.from_documents(
    documents=split_doc,
    embedding=embedding_model
)

# step 5: 검색기 생성

retriever = vector_db.as_retriever(search_kwargs={"k":3})

# step 6: RAG 체인 생성
from langchain_core.prompts import PromptTemplate

template = """
너는 아래에 있는 문서를 보고 답하는 llm이야 아래에 있는 질문에 답을 해줘

#문서:{document}

#질문:{question}
"""
prompt = PromptTemplate.from_template(template)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0
)
output_parser = StrOutputParser()

def formatter(docs):
    return "\n\n ".join(doc.page_content for doc in docs)

chain = {"document": retriever | formatter, "question":RunnablePassthrough()} | prompt | llm | output_parser

print(chain.invoke("회사의 전망을 알려줘"))