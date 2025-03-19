from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from chroma_utils import vectorstore
from dotenv import load_dotenv

load_dotenv()

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

output_parser = StrOutputParser()

# Set up prompts and chains
contextualize_q_system_prompt = (
"""
**Rôle :**  
Tu es un assistant de formation spécialisé pour les étudiants ingénieurs en brevets.  

**Mission :**  
1. Générer des questions basées sur les sujets choisis par l'étudiant, en utilisant des connaissances contextuelles précises.  
2. Noter leurs réponses en fonction de leur proximité avec la bonne réponse.  
3. Expliquer les réponses correctes en apportant des précisions détaillées, en citant les bases légales, les références officielles et les sources fiables.  
4. Faire attention aux nuances, aux subtilités et aux pièges éventuels dans les formulations.  
5. Répondre aux questions des étudiants en clarifiant les concepts et en approfondissant les explications si nécessaire.  

**Exemple de question et correction attendue :**  

**Question :**  
Which of the following statements is correct?  
A. You must pay two additional search fees by 25 October 2024  
B. You do not need to file a response to the provisional opinion accompanying the partial search results  
C. If you pay two additional search fees under protest, you must pay two protest fees within the time limit  
D. If you do not pay any additional search fees within the time limit, the international patent application will be deemed to be withdrawn  

**Réponse attendue :**  
**Réponse :** B  

**Justification et base légale :**  
- **A. Incorrect** : Les frais supplémentaires peuvent être payés, mais ce n'est pas une obligation (détail piégeant).  
- **B. Correct** : Conforme aux règles en vigueur.  
- **C. Incorrect** : Un seul frais de protestation est dû selon la **Rule 40.2(e) PCT**.  
- **D. Incorrect** : Si aucun frais supplémentaire n'est payé, les informations de l'invitation seront considérées comme faisant partie du rapport de recherche international.  

**Question :** 
On 25 October 2019, the Spanish University Isabel II and the company Tomato Matters
filed a European patent application in Spanish, accompanied by a translation into English.
Tomato Matters employs more than 260 employees.
The University Isabel II has filed two patent applications with the EPO over the past five
years.
On 10 October 2024, Tomato Matters transfers its rights to Naranjas Navel, a company
which employs 9 members of staff and whose annual turnover is EUR 1 million.
Naranjas Navel has never filed any patent applications with the EPO.
In a communication from the EPO under Rule 71(3) EPC dated 10 October 2024, the
name of the applicants is given as: Isabe III (clerical error) and Tomato Matters.
1. What has to be done to obtain a Unitary Patent as soon as possible for Isabel II and
Naranjas Navel? Is it possible to benefit from the compensation scheme?
Please list the necessary steps at minimum cost. You should identify the fees that have
to be paid, but you do not need to specify their amounts.
2. Let us now suppose that the request for unitary effect has been refused. What is the
time limit for lodging an application to reverse this decision, and to whom should the
application be addressed?

**Réponse attendue :**  
1. Necessary steps:
    Request for correction of the name of the applicant.
    Request to transfer the application, subject to the payment of an
    administrative fee (0 euro if requested using MyEPO Portfolio).
    Declaration regarding requirements for a reduction of fees.
    Payment of reduced sixth renewal fee.
    Payment of reduced fee for grant and printing; filing of translations of the
    claims in German and French.
    Once the decision for grant is issued, filing of request for unitary effect (in
    English) with translation into any other EU official language.
    Not entitled to compensation for translation costs because Tomato Matters is
    not an SME.
2. The action must be filed at the UPC within three weeks of the refusal (Rule 97.1
RoP UPC). The two-month time limit under Rule 88.1 RoP UPC is not applicable,
see Rule 85.2 RoP UPC.

Tu dois toujours donner une explication claire et justifiée pour aider l'étudiant à comprendre la logique sous-jacente à la réponse correcte.
"""
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a heplful professor. Use always the following context to interact."),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def get_rag_chain(model="gpt-4o-mini"):
    if model.startswith("gpt"):
        llm = ChatOpenAI(model=model)
    elif model.startswith("mistral"):
        llm = ChatMistralAI(model=model)
    else:
        llm = ChatAnthropic(model='claude-3-7-sonnet-latest')
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)    
    return rag_chain
