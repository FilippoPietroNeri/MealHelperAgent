import streamlit as st
import os
from typing import List, Any, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- LANGCHAIN IMPORTS ---
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents import create_tool_calling_agent

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# --- CARICAMENTO VARIABILI ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- CONFIGURAZIONE LIMITI ---
TOKEN_LIMIT = 20000  # Limite fittizio per la demo (Groq free tier è più alto, ma serve per testare il blocco)
MAX_RETRIES = 2      # Quante volte il critico può rimandare indietro il piatto

st.set_page_config(page_title="Chef AI & Critic", layout="wide")

# --- CLASSE PER MONITORAGGIO TOKEN (Callback) ---
class TokenUsageTracker(BaseCallbackHandler):
    """Intercetta ogni chiamata LLM e conta i token usati."""
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            total = usage.get("total_tokens", 0)
            st.session_state.total_tokens += total
            # Debug (opzionale)
            # print(f"Tokens used in this call: {total}")

# --- DATA STRUCTURES ---
class Ingredient(BaseModel):
    name: str = Field(description="Nome dell'ingrediente")
    quantity: str = Field(description="Quantità (es: 200g, 2 uova)")
    expiry: str = Field(description="Scadenza (es: tra 2 giorni, oggi)")

class PantryUpdate(BaseModel):
    ingredients: List[Ingredient]

class CriticEvaluation(BaseModel):
    approved: bool = Field(description="True se la ricetta rispetta i vincoli, False altrimenti")
    feedback: str = Field(description="Spiegazione dell'errore o conferma positiva")

# --- INIZIALIZZAZIONE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pantry" not in st.session_state:
    st.session_state.pantry = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# --- SIDEBAR (MONITORAGGIO TOKEN) ---
with st.sidebar:
    st.title("🛒 Dispensa & Status")
    
    # Sezione Token
    st.subheader("🔋 Token Disponibili")
    tokens_used = st.session_state.total_tokens
    remaining = max(0, TOKEN_LIMIT - tokens_used)
    progress_val = min(tokens_used / TOKEN_LIMIT, 1.0)
    
    if progress_val < 0.7:
        bar_color = "green"
    elif progress_val < 0.9:
        bar_color = "orange"
    else:
        bar_color = "red"
    
    st.progress(progress_val)
    st.caption(f"Usati: {tokens_used} / {TOKEN_LIMIT}")
    st.caption(f"Rimanenti: {remaining}")

    if remaining == 0:
        st.error("🚫 LIMITE RAGGIUNTO. Sistema bloccato.")
        app_blocked = True
    else:
        app_blocked = False

    st.divider()
    
    st.info("Ingredienti rilevati:")
    if st.session_state.pantry:
        # Mostra solo nome e quantità per pulizia
        simple_pantry = [{"Ingrediente": i['name'], "Q.tà": i['quantity']} for i in st.session_state.pantry]
        st.table(simple_pantry)
    else:
        st.write("Dispensa vuota.")
    
    if st.button("Reset Conversazione"):
        st.session_state.pantry = []
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.rerun()

# --- AGENTE ESTRAZIONE DISPENSA ---
def update_pantry_state(user_input: str, callbacks):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, callbacks=callbacks)
    structured_llm = llm.with_structured_output(PantryUpdate)
    
    prompt = ChatPromptTemplate.from_template(
        "Analizza il messaggio: '{input}'. Estrai ingredienti. Se mancano info scrivi 'Non specificato'."
    )
    chain = prompt | structured_llm
    
    try:
        new_data = chain.invoke({"input": user_input})
        for item in new_data.ingredients:
            if not any(i['name'].lower() == item.name.lower() for i in st.session_state.pantry):
                st.session_state.pantry.append(item.dict())
    except Exception:
        pass

# --- AGENTE CHEF ---
def get_chef_agent(callbacks):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5, callbacks=callbacks)
    search_tool = TavilySearchResults(max_results=3)
    tools = [search_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """SEI UN AGENTE DI PIANIFICAZIONE CULINARIA.
        
        STATO ATTUALE DISPENSA: {pantry_context}

        REGOLE FONDAMENTALI:
        1. Raccogli informazioni (numero persone, intolleranze) prima di proporre ricette.
        2. Se hai tutte le info, proponi una ricetta usando TAVILY se serve.
        3. RISPETTA RIGOROSAMENTE LE INTOLLERANZE SEGNALATE DALL'UTENTE ORA O NELLA CRONOLOGIA.
        
        Se ricevi un feedback negativo dal "CRITICO", riscrivi la ricetta correggendo l'errore segnalato.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# --- AGENTE CRITICO ---
def critic_agent(last_chef_response: str, user_constraints: str, chat_history: List, callbacks):
    """
    Valuta se la risposta dello chef rispetta i vincoli.
    Usa un modello più leggero (8b) per risparmiare token, o 70b per precisione.
    """
    # Usiamo il modello 70b per essere sicuri che capisca bene le allergie, ma potremmo usare 8b
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, callbacks=callbacks)
    structured_llm = llm.with_structured_output(CriticEvaluation)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Sei un severo CRITICO GASTRONOMICO e RESPONSABILE SICUREZZA.
        Il tuo compito è analizzare l'ultima risposta dello Chef.
        
        CASO 1: Lo Chef sta facendo domande o conversando (NON ha proposto una ricetta).
        -> Rispondi APPROVED: True.

        CASO 2: Lo Chef ha proposto una ricetta.
        -> Controlla i vincoli utente: {user_constraints}.
        -> Controlla la cronologia se ci sono allergie menzionate prima: {history_summary}
        -> Se la ricetta viola un vincolo (es. contiene glutine per un celiaco, usa ingredienti non in dispensa senza permesso), BOCCIALA.
        
        Se bocci, spiega chiaramente l'errore nel campo feedback.
        """),
        ("human", "Risposta Chef da valutare: \n{chef_response}")
    ])
    
    # Serializziamo brevemente la history per il contesto
    history_text = "\n".join([m.content for m in chat_history if isinstance(m, HumanMessage)])
    
    chain = prompt | structured_llm
    return chain.invoke({
        "user_constraints": user_constraints, 
        "history_summary": history_text[-1000:], # Ultimi 1000 caratteri di contesto utente
        "chef_response": last_chef_response
    })

# --- INTERFACCIA CHAT ---
st.title("👨‍🍳 Chef AI Agent (con Controllo Qualità)")

# Visualizza messaggi precedenti
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Logica di Input
if user_input := st.chat_input("Scrivi qui...", disabled=app_blocked):
    
    if app_blocked:
        st.error("Hai esaurito i token disponibili per questa sessione.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Tracker Token inizializzato
    token_tracker = TokenUsageTracker()

    # 1. Aggiorna Dispensa
    update_pantry_state(user_input, [token_tracker])

    # 2. Generazione e Critica
    with st.chat_message("assistant"):
        with st.status("👨‍🍳 Lo Chef è al lavoro...", expanded=True) as status:
            
            # Preparazione dati
            pantry_ctx = str(st.session_state.pantry)
            history = [
                HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"])
                for m in st.session_state.messages[:-1]
            ]
            
            agent_exec = get_chef_agent([token_tracker])
            
            # --- LOOP DI GENERAZIONE (REFLEXION) ---
            attempt = 0
            approved = False
            current_input = user_input # Input iniziale
            final_output = ""

            while not approved and attempt <= MAX_RETRIES:
                attempt += 1
                
                if attempt > 1:
                    st.write(f"🔄 Tentativo di correzione {attempt-1}...")
                
                # Chiamata allo Chef
                response = agent_exec.invoke({
                    "input": current_input,
                    "pantry_context": pantry_ctx,
                    "chat_history": history
                })
                output_text = response["output"]
                
                st.write("🕵️ Il Critico sta valutando...")
                
                # Chiamata al Critico
                critic_res = critic_agent(output_text, user_input, history, [token_tracker])
                
                if critic_res.approved:
                    approved = True
                    final_output = output_text
                    st.write("✅ Risposta approvata!")
                else:
                    st.write(f"❌ Bocciata: {critic_res.feedback}")
                    # Modifichiamo l'input per lo chef includendo il feedback negativo
                    current_input = f"La tua risposta precedente è stata bocciata dal critico. Motivo: {critic_res.feedback}. Riscrivila correggendo l'errore. Richiesta originale: {user_input}"
                    
                    if attempt > MAX_RETRIES:
                        final_output = output_text + f"\n\n*(Nota: Questa ricetta potrebbe non soddisfare perfettamente i vincoli: {critic_res.feedback})*"
            
            status.update(label="✅ Completato", state="complete", expanded=False)

        st.markdown(final_output)
        st.session_state.messages.append({"role": "assistant", "content": final_output})
    
    st.rerun() # Ricarica per aggiornare la barra laterale