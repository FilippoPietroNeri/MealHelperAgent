import streamlit as st
import os
import re
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- CARICAMENTO VARIABILI D'AMBIENTE ---
load_dotenv() # Carica le chiavi dal file .env

# --- FIX IMPORT AGENT EXECUTOR ---
# Se l'import standard fallisce, LangChain lo ha spostato qui nelle versioni recenti
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents import create_tool_calling_agent

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

st.set_page_config(page_title="Chef AI Agent", layout="wide")

# --- DATA STRUCTURES (Pydantic) ---
class Ingredient(BaseModel):
    name: str = Field(description="Nome dell'ingrediente")
    quantity: str = Field(description="Quantità (es: 200g, 2 uova)")
    expiry: str = Field(description="Scadenza (es: tra 2 giorni, oggi)")

class PantryUpdate(BaseModel):
    ingredients: List[Ingredient]

# --- LOGICA DI ESTRAZIONE ---
def update_pantry_state(user_input: str):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    # Utilizzo di with_structured_output per garantire coerenza con la sidebar
    structured_llm = llm.with_structured_output(PantryUpdate)
    
    prompt = ChatPromptTemplate.from_template(
        "Analizza il messaggio dell'utente: '{input}'. "
        "Estrai gli ingredienti citati. Se mancano quantità o scadenze, scrivi 'Non specificato'."
    )
    chain = prompt | structured_llm
    
    try:
        new_data = chain.invoke({"input": user_input})
        for item in new_data.ingredients:
            # Update logica: evita duplicati semplici
            if not any(i['name'].lower() == item.name.lower() for i in st.session_state.pantry):
                st.session_state.pantry.append(item.dict())
    except Exception:
        pass

# --- AGENTE CHEF (Planning & Tools) ---
def get_chef_agent():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0) # Temperature 0 per maggior controllo
    
    search_tool = TavilySearchResults(max_results=3)
    tools = [search_tool]
    
    # Prompt molto più rigoroso basato sulla logica di Chip Huyen (Cap. 6)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """SEI UN AGENTE DI PIANIFICAZIONE CULINARIA. IL TUO OBIETTIVO NON È DARE RICETTE SUBITO, MA RACCOGLIERE INFORMAZIONI.

        STATO ATTUALE DISPENSA: {pantry_context}

        --- REGOLA ZERO (IL BLOCCO): ---
        PRIMA di usare qualsiasi tool o proporre ricette, DEVI avere queste 4 conferme:
        1. Numero di persone esatto.
        2. Presenza di allergie, intolleranze (es. celiachia) o regimi alimentari.
        3. Livello di abilità dell'utente in cucina.
        4. Presenza di condimenti base (olio, sale, pepe).

        SE MANCA ANCHE SOLO UNA DI QUESTE, NON USARE TAVILY. Fai una domanda mirata per ottenere ciò che manca. 
        Se l'utente ha solo 1 o 2 ingredienti (es. solo riso), chiedi cosa ha per condirlo o quali altri ingredienti ha.

        --- GESTIONE GRUPPI MISTI E PORZIONI: ---
        - Se nel gruppo c'è un celiaco e hai pasta di grano e riso, devi calcolare le porzioni separatamente.
        - Esempio logico: 500g di riso per 1 celiaco sono troppi, mentre 500g di pasta per 4 persone sono giusti. Proponi di cucinare entrambi usando i condimenti in comune ma in pentole separate.

        --- REGOLE PER L'USO DI TAVILY: ---
        Solo quando hai TUTTE le info, cerca ricette che usino il MASSIMO degli ingredienti in dispensa.
        Non ignorare ingredienti difficili. Dai priorità a quelli in scadenza.

        --- FORMATO DI OUTPUT (Solo se hai tutte le info): ---
        ### 🍴 [TITOLO RICETTA]
        **Target:** [Es: Sicura per celiaci / Variante dedicata]
        **Info:** [X] min | Difficoltà: [Principiante/Intermedio/Esperto] | Per [N] persone
        
        **🥣 Ingredienti dalla tua dispensa (Grammature precise):**
        - [Ingrediente]: [Quantità usata] (Specifica se lo finisci o ne avanza)
        
        **👨‍🍳 Procedura Personalizzata:**
        1. [Istruzioni dettagliate]
        
        **🔗 Fonte Originale:** [INSERISCI QUI IL LINK URL PRESO DAL TOOL]

        Rispondi sempre in modo naturale, professionale e in Italiano."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5 # Evita loop infiniti
    )

# --- INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pantry" not in st.session_state:
    st.session_state.pantry = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛒 Dispensa Live")
    st.info("Gli ingredienti rilevati appariranno qui automaticamente.")
    if st.session_state.pantry:
        st.table(st.session_state.pantry)
    else:
        st.write("Dispensa vuota.")
    
    if st.button("Reset Conversazione"):
        st.session_state.pantry = []
        st.session_state.messages = []
        st.rerun()

# --- CHAT INTERFACE ---
st.title("👨‍🍳 Chef AI Agent")
st.caption("AI Agent basato su Llama 3.1 & Tavily Search")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            st.markdown(m["content"])

if user_input := st.chat_input("Scrivi qui gli ingredienti o fai una domanda..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 1. Estrazione ingredienti
    update_pantry_state(user_input)

    # 2. Risposta Agente con dropdown Reasoning
    with st.chat_message("assistant"):
        # Dropdown per mostrare il ragionamento (Reasoning)
        with st.status("👨‍🍳 Lo Chef sta analizzando la richiesta...", expanded=True) as status:
            st.write("Verifica ingredienti e quantità...")
            context = str(st.session_state.pantry)
            
            st.write("Controllo preferenze e vincoli...")
            agent_exec = get_chef_agent()
            
            history = [
                HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            
            st.write("Decisione: Domanda di chiarimento o Ricerca via Tavily?")
            response = agent_exec.invoke({
                "input": user_input,
                "pantry_context": context,
                "chat_history": history
            })
            status.update(label="✅ Analisi completata", state="complete", expanded=False)
        
        st.markdown(response["output"])
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
    st.rerun()